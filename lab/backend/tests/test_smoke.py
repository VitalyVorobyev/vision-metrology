"""End-to-end smoke test: upload -> teach -> find -> measure a synthetic disc.

Exercises the whole chain the lab exists to make visible, with a known ground truth
(radius 40 px, anti-aliased so the true subpixel edge is exactly at r=40) so the
assertions are about correctness, not just "it didn't crash".
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from vm_lab.app import create_app
from vm_lab.store import store

_TABLE_CALIBRATION_FIXTURE = (
    Path(__file__).resolve().parents[3] / "crates" / "vision-metrology" / "tests" / "fixtures" / "table_calibration.json"
)

DISC_CENTER = (100.0, 100.0)
DISC_RADIUS = 40.0
IMG_SIZE = 200


def _synthetic_disc_png() -> bytes:
    cx, cy = DISC_CENTER
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    dist = np.hypot(xs - cx, ys - cy)
    cover = np.clip(DISC_RADIUS + 0.5 - dist, 0.0, 1.0)
    pixels = (20.0 + 180.0 * cover).round().astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(pixels, mode="L").save(buf, "PNG")
    return buf.getvalue()


def test_teach_find_measure_circle(tmp_path, monkeypatch) -> None:
    from vm_lab.config import Settings

    isolated = Settings(data_dir=tmp_path)
    monkeypatch.setattr("vm_lab.store.settings", isolated)
    monkeypatch.setattr("vm_lab.media.settings", isolated)
    store.images.clear()
    store.models.clear()
    store.calibrations.clear()

    client = TestClient(create_app())

    png = _synthetic_disc_png()
    resp = client.post("/api/images", files={"file": ("disc.png", png, "image/png")})
    assert resp.status_code == 200, resp.text
    image = resp.json()
    assert image["width"] == IMG_SIZE
    assert image["height"] == IMG_SIZE

    resp = client.get(f"/api/images/{image['id']}/thumb")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/webp"
    etag = resp.headers["etag"]
    resp = client.get(f"/api/images/{image['id']}/thumb", headers={"If-None-Match": etag})
    assert resp.status_code == 304

    roi = (40.0, 40.0, 120.0, 120.0)
    resp = client.post("/api/models", json={"image_id": image["id"], "roi": roi, "min_contrast": 0.1})
    assert resp.status_code == 200, resp.text
    model = resp.json()
    assert model["num_levels_built"] >= 1
    assert sum(model["point_counts"]) > 0

    resp = client.post(
        "/api/find", json={"image_id": image["id"], "model_id": model["id"], "min_score": 0.5}
    )
    assert resp.status_code == 200, resp.text
    matches = resp.json()["matches"]
    assert len(matches) >= 1
    best = max(matches, key=lambda m: m["score"])
    assert abs(best["x"] - DISC_CENTER[0]) < 2.0
    assert abs(best["y"] - DISC_CENTER[1]) < 2.0
    assert abs(best["scale"] - 1.0) < 0.05

    resp = client.post(
        "/api/measure",
        json={
            "image_id": image["id"],
            "model_id": model["id"],
            "min_score": 0.5,
            "objects": [
                {
                    "kind": "circle",
                    "label": "outer edge",
                    "cx": DISC_CENTER[0],
                    "cy": DISC_CENTER[1],
                    "r": DISC_RADIUS,
                    "n_calipers": 16,
                    "caliper_len": 10.0,
                    "caliper_width": 5.0,
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["fixture_source"] == "auto_find"
    (result,) = body["objects"]
    assert result["kind"] == "circle"
    assert abs(result["circle_r"] - DISC_RADIUS) < 0.5
    assert result["rms"] < 0.5
    assert result["n_used"] >= 12
    assert len(result["overlay"]) > 0
    assert len(result["calipers"]) == 16
    assert any(c["status"] == "hit" for c in result["calipers"])
    hit = next(c for c in result["calipers"] if c["status"] == "hit")
    assert len(hit["profile"]["values"]) > 0
    assert hit["profile"]["step_px"] > 0
    assert len(hit["profile"]["edges"]) == 1

    # -- rectify: find + rectify each match into a canonical model-frame crop --------
    resp = client.post(
        "/api/rectify",
        json={
            "image_id": image["id"],
            "model_id": model["id"],
            "min_score": 0.5,
            "crop": {"rect": list(roi), "px_per_unit": 1.0},
        },
    )
    assert resp.status_code == 200, resp.text
    rectify_body = resp.json()
    assert rectify_body["width"] == int(roi[2])
    assert rectify_body["height"] == int(roi[3])
    assert len(rectify_body["matches"]) >= 1
    rbest = max(rectify_body["matches"], key=lambda m: m["score"])
    assert abs(rbest["x"] - DISC_CENTER[0]) < 2.0
    assert rbest["validity"] > 0.9
    assert rbest["width"] == rectify_body["width"]
    assert rbest["height"] == rectify_body["height"]

    crop_resp = client.get(rbest["crop_url"])
    assert crop_resp.status_code == 200, crop_resp.text
    assert crop_resp.headers["content-type"] == "image/png"
    crop_etag = crop_resp.headers["etag"]
    crop_img = PILImage.open(io.BytesIO(crop_resp.content))
    assert crop_img.size == (rectify_body["width"], rectify_body["height"])

    cached_resp = client.get(rbest["crop_url"], headers={"If-None-Match": crop_etag})
    assert cached_resp.status_code == 304

    missing_resp = client.get(f"/api/rectify/{image['id']}/{model['id']}/999")
    assert missing_resp.status_code == 404

    # -- calibration: upload a real table_calibration.json, measure with mm ----------
    resp = client.get("/api/calibration")
    assert resp.status_code == 200
    assert resp.json() == []

    cal_bytes = _TABLE_CALIBRATION_FIXTURE.read_bytes()
    resp = client.post(
        "/api/calibration",
        files={"file": ("calibration.json", cal_bytes, "application/json")},
    )
    assert resp.status_code == 200, resp.text
    calibration = resp.json()
    assert calibration["format"] == "table_calibration"
    assert calibration["n_cameras"] == 2

    resp = client.get("/api/calibration")
    assert resp.status_code == 200
    assert [c["id"] for c in resp.json()] == [calibration["id"]]

    resp = client.post(
        "/api/calibration",
        files={"file": ("garbage.json", b"not json", "application/json")},
    )
    assert resp.status_code == 400

    resp = client.post(
        "/api/measure",
        json={
            "image_id": image["id"],
            "model_id": model["id"],
            "min_score": 0.5,
            "calibration_id": calibration["id"],
            # camera1: camera0's own extrinsics are the identity, which sits
            # exactly *on* the default z=0 plane (degenerate); camera1's
            # principal ray also points away from z=0 in this real rig, so
            # the plane is offset to z=-100mm, which both cameras actually
            # look toward — this is `PlaneIn` used for something other than
            # its default, not a workaround for a library bug.
            "camera_index": 1,
            "plane": {"nx": 0.0, "ny": 0.0, "nz": 1.0, "d": -100.0},
            "objects": [
                {
                    "kind": "circle",
                    "label": "outer edge",
                    "cx": DISC_CENTER[0],
                    "cy": DISC_CENTER[1],
                    "r": DISC_RADIUS,
                    "n_calipers": 16,
                    "caliper_len": 10.0,
                    "caliper_width": 5.0,
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    mm_body = resp.json()
    (mm_result,) = mm_body["objects"]
    assert mm_result["circle_cx_mm"] is not None
    assert mm_result["circle_cy_mm"] is not None
    assert mm_result["circle_r_mm"] is not None
    assert mm_result["circle_r_mm"] > 0.0
    hit = next(c for c in mm_result["calipers"] if c["status"] == "hit")
    edge = hit["profile"]["edges"][0]
    assert edge["x_mm"] is not None
    assert edge["y_mm"] is not None

    resp = client.post(
        "/api/measure",
        json={
            "image_id": image["id"],
            "model_id": model["id"],
            "min_score": 0.5,
            "calibration_id": calibration["id"],
            "camera_index": 7,
            "objects": [{"kind": "circle", "cx": 100.0, "cy": 100.0, "r": 40.0}],
        },
    )
    assert resp.status_code == 400

    resp = client.post(
        "/api/measure",
        json={
            "image_id": image["id"],
            "model_id": model["id"],
            "min_score": 0.5,
            "calibration_id": "cal-999",
            "objects": [{"kind": "circle", "cx": 100.0, "cy": 100.0, "r": 40.0}],
        },
    )
    assert resp.status_code == 404
