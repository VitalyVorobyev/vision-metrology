"""`POST /api/mosaic` — composite two calibrated cameras' rectified views of the
calibration's `z = 0` plane, nearest-camera-centre priority, no blending by default.

Uses a **synthetic** 2-camera `table_calibration.json`-shaped fixture built in this file
rather than the real dataset's `tests/fixtures/table_calibration.json` deliberately:
that fixture's `camera0` sits at the identity pose, which places it exactly *on* the
default `z = 0` plane -- degenerate for `plane_grid_map` (see `test_smoke.py`'s own note
on this, and `crates/vision-metrology/examples/birdseye_mosaic.rs`'s module docs for the
full story). This fixture instead gives both cameras a non-identity pose standing off from
a proper `z = 0` plane -- same camera parameters and geometry
`crates/vision-metrology/tests/mosaic.rs` uses, so the two are cross-checked against each
other.
"""

from __future__ import annotations

import io
import json

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from vm_lab.app import create_app
from vm_lab.store import store

import vision_metrology as vm

RAW_W, RAW_H = 320, 240
FX = FY = 450.0
CX, CY = 160.0, 120.0
STANDOFF_MM = 500.0
CAM_OFFSETS_MM = (-80.0, 80.0)
DISTORTION = [-0.06, 0.008, 0.0004, -0.0002, 0.0]  # OpenCV order: k1,k2,p1,p2,k3
DARK, BRIGHT = 40.0, 220.0
FIDUCIAL_RADIUS_MM = 12.0


def _camera_json(offset_mm: float) -> dict:
    # pose(p) = p - (offset, 0, -STANDOFF): reference point (offset, 0, 0) lands
    # on this camera's own axis at depth STANDOFF -- same convention
    # `tests/mosaic.rs`'s `camera_pose` uses.
    return {
        "intrinsic": {
            "matrix": [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]],
            "distortion": DISTORTION,
            "frame_cols": RAW_W,
            "frame_rows": RAW_H,
        },
        "extrinsic": {
            "sensor2camera": [
                [1.0, 0.0, 0.0, -offset_mm],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, STANDOFF_MM],
                [0.0, 0.0, 0.0, 1.0],
            ]
        },
    }


def _table_calibration_bytes() -> bytes:
    intrinsic = {}
    extrinsic = {}
    for i, offset in enumerate(CAM_OFFSETS_MM):
        entry = _camera_json(offset)
        intrinsic[f"camera{i}"] = entry["intrinsic"]
        extrinsic[f"camera{i}"] = entry["extrinsic"]
    return json.dumps({"intrinsic": intrinsic, "extrinsic": extrinsic}).encode()


def _pattern_mm(x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
    """A single antialiased dark disc at the plane origin -- smooth, so seam
    disparity stays small (unlike a hard-edged checkerboard)."""
    d = np.hypot(x_mm, y_mm) - FIDUCIAL_RADIUS_MM
    t = np.clip((-d + 1.0) / 2.0, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return BRIGHT + (DARK - BRIGHT) * t


def _render_raw_png(camera: vm.CameraModel, pose: np.ndarray) -> bytes:
    ys, xs = np.mgrid[0:RAW_H, 0:RAW_W].astype(np.float32)
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1)
    mm = vm.pixel_to_plane(camera, pose, vm.Plane3.xy(), pixels)
    value = _pattern_mm(mm[:, 0], mm[:, 1])
    valid = np.isfinite(mm).all(axis=1)
    value = np.where(valid, value, BRIGHT)
    pixels_u8 = value.reshape(RAW_H, RAW_W).round().clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(pixels_u8, mode="L").save(buf, "PNG")
    return buf.getvalue()


def _isolate_store(tmp_path, monkeypatch) -> None:
    from vm_lab.config import Settings

    isolated = Settings(data_dir=tmp_path)
    monkeypatch.setattr("vm_lab.store.settings", isolated)
    monkeypatch.setattr("vm_lab.media.settings", isolated)
    store.images.clear()
    store.models.clear()
    store.calibrations.clear()


def _upload_calibration_and_images(client: TestClient) -> tuple[str, list[str]]:
    cal_bytes = _table_calibration_bytes()
    resp = client.post(
        "/api/calibration", files={"file": ("calibration.json", cal_bytes, "application/json")}
    )
    assert resp.status_code == 200, resp.text
    calibration_id = resp.json()["id"]
    assert resp.json()["n_cameras"] == 2

    cams = vm.load_table_calibration(cal_bytes)
    image_ids: list[str] = []
    for camera, pose in cams:
        png = _render_raw_png(camera, pose)
        resp = client.post("/api/images", files={"file": ("frame.png", png, "image/png")})
        assert resp.status_code == 200, resp.text
        image_ids.append(resp.json()["id"])
    return calibration_id, image_ids


def test_mosaic_composites_two_cameras_with_overlap(tmp_path, monkeypatch) -> None:
    _isolate_store(tmp_path, monkeypatch)
    client = TestClient(create_app())
    calibration_id, image_ids = _upload_calibration_and_images(client)

    resp = client.post(
        "/api/mosaic",
        json={
            "calibration_id": calibration_id,
            "cameras": [
                {"camera_index": 0, "image_id": image_ids[0]},
                {"camera_index": 1, "image_id": image_ids[1]},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["width"] > 0 and body["height"] > 0
    assert body["mm_per_px"] > 0
    assert len(body["cameras"]) == 2
    for cam in body["cameras"]:
        assert cam["coverage_fraction"] > 0.0
    assert body["union_coverage_fraction"] > 0.0
    assert body["overlap_fraction"] > 0.0  # the plan's "produces nonzero overlap" bar
    assert body["seam_disparity_p50"] is not None
    assert body["seam_disparity_p95"] is not None
    assert body["seam_disparity_p50"] >= 0.0

    img_resp = client.get(body["image_url"])
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/png"
    mosaic_img = PILImage.open(io.BytesIO(img_resp.content))
    assert mosaic_img.size == (body["width"], body["height"])
    etag = img_resp.headers["etag"]
    cached = client.get(body["image_url"], headers={"If-None-Match": etag})
    assert cached.status_code == 304

    feather_resp = client.get(body["image_url"], params={"feather": "true"})
    assert feather_resp.status_code == 200
    feather_img = PILImage.open(io.BytesIO(feather_resp.content))
    assert feather_img.size == (body["width"], body["height"])
    # Feather is a genuinely different rendering from the priority composite
    # (display-only blend vs. no-blend priority) -- not required to differ
    # everywhere, but over a real overlap region it should differ somewhere.
    assert list(feather_resp.content) != list(img_resp.content)

    sid_resp = client.get(body["source_id_url"])
    assert sid_resp.status_code == 200
    assert sid_resp.headers["content-type"] == "image/png"
    sid_img = PILImage.open(io.BytesIO(sid_resp.content))
    assert sid_img.size == (body["width"], body["height"])
    assert sid_img.mode == "RGB"


def test_mosaic_explicit_grid_overrides_auto_fit(tmp_path, monkeypatch) -> None:
    _isolate_store(tmp_path, monkeypatch)
    client = TestClient(create_app())
    calibration_id, image_ids = _upload_calibration_and_images(client)

    resp = client.post(
        "/api/mosaic",
        json={
            "calibration_id": calibration_id,
            "cameras": [
                {"camera_index": 0, "image_id": image_ids[0]},
                {"camera_index": 1, "image_id": image_ids[1]},
            ],
            "grid": {"origin_mm": [-20.0, -20.0], "mm_per_px": 0.5, "width": 80, "height": 80},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["width"], body["height"]) == (80, 80)
    assert body["mm_per_px"] == 0.5
    assert body["origin_mm"] == [-20.0, -20.0]


def test_mosaic_rejects_unknown_calibration(tmp_path, monkeypatch) -> None:
    _isolate_store(tmp_path, monkeypatch)
    client = TestClient(create_app())
    resp = client.post(
        "/api/mosaic",
        json={
            "calibration_id": "cal-999",
            "cameras": [
                {"camera_index": 0, "image_id": "img-1"},
                {"camera_index": 1, "image_id": "img-2"},
            ],
        },
    )
    assert resp.status_code == 404


def test_mosaic_rejects_out_of_range_camera_index(tmp_path, monkeypatch) -> None:
    _isolate_store(tmp_path, monkeypatch)
    client = TestClient(create_app())
    calibration_id, image_ids = _upload_calibration_and_images(client)

    resp = client.post(
        "/api/mosaic",
        json={
            "calibration_id": calibration_id,
            "cameras": [
                {"camera_index": 0, "image_id": image_ids[0]},
                {"camera_index": 7, "image_id": image_ids[1]},
            ],
        },
    )
    assert resp.status_code == 400


def test_mosaic_requires_at_least_two_cameras(tmp_path, monkeypatch) -> None:
    _isolate_store(tmp_path, monkeypatch)
    client = TestClient(create_app())
    calibration_id, image_ids = _upload_calibration_and_images(client)

    resp = client.post(
        "/api/mosaic",
        json={
            "calibration_id": calibration_id,
            "cameras": [{"camera_index": 0, "image_id": image_ids[0]}],
        },
    )
    assert resp.status_code == 422


def test_mosaic_unknown_id_is_404(tmp_path, monkeypatch) -> None:
    _isolate_store(tmp_path, monkeypatch)
    client = TestClient(create_app())
    assert client.get("/api/mosaic/mosaic-does-not-exist/image").status_code == 404
    assert client.get("/api/mosaic/mosaic-does-not-exist/source_id").status_code == 404
