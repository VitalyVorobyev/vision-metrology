"""`POST /api/displacement` — pairwise subpixel inter-frame shift over an ordered
image sequence. Uses a textured (non-flat) synthetic pattern so ZNCC has a real peak
to lock onto, unlike the disc fixture in `test_smoke.py`.
"""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from vm_lab.app import create_app
from vm_lab.store import store

IMG_SIZE = 160


def _value_noise_sampler(seed: int, cell: float = 9.0, grid_extent: int = 32):
    rng = np.random.default_rng(seed)
    grid = rng.random((grid_extent, grid_extent)).astype(np.float32)

    def sample(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gx = x / cell
        gy = y / cell
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        fx = gx - x0
        fy = gy - y0
        sx = fx * fx * (3.0 - 2.0 * fx)
        sy = fy * fy * (3.0 - 2.0 * fy)
        x0c = np.clip(x0, 0, grid_extent - 2)
        y0c = np.clip(y0, 0, grid_extent - 2)
        v00 = grid[y0c, x0c]
        v10 = grid[y0c, x0c + 1]
        v01 = grid[y0c + 1, x0c]
        v11 = grid[y0c + 1, x0c + 1]
        a = v00 * (1.0 - sx) + v10 * sx
        b = v01 * (1.0 - sx) + v11 * sx
        return a * (1.0 - sy) + b * sy

    return sample


def _frame_png(sample, dx: float, dy: float) -> bytes:
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    v = 128.0 + 100.0 * (sample(xs - dx, ys - dy) - 0.5)
    pixels = np.clip(np.round(v), 0.0, 255.0).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(pixels, mode="L").save(buf, "PNG")
    return buf.getvalue()


def test_displacement_over_a_frame_sequence(tmp_path, monkeypatch) -> None:
    from vm_lab.config import Settings

    isolated = Settings(data_dir=tmp_path)
    monkeypatch.setattr("vm_lab.store.settings", isolated)
    monkeypatch.setattr("vm_lab.media.settings", isolated)
    store.images.clear()
    store.models.clear()
    store.calibrations.clear()

    client = TestClient(create_app())
    sample = _value_noise_sampler(seed=7)

    # A steady drift: (0, 0), (2, -1), (5, -1.5), (9, -3) relative to frame 0 —
    # per-pair shifts are (2, -1), (3, -0.5), (4, -1.5).
    positions = [(0.0, 0.0), (2.0, -1.0), (5.0, -1.5), (9.0, -3.0)]
    image_ids: list[str] = []
    for dx, dy in positions:
        png = _frame_png(sample, dx, dy)
        resp = client.post("/api/images", files={"file": ("frame.png", png, "image/png")})
        assert resp.status_code == 200, resp.text
        image_ids.append(resp.json()["id"])

    req = {
        "image_ids": image_ids,
        "window": [30.0, 30.0, 100.0, 100.0],
        "search_x": 8,
        "search_y": 8,
        "refine": "lucas_kanade",
        "lk_iters": 5,
        "min_score": 0.3,
    }
    resp = client.post("/api/displacement", json=req)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["pairs"]) == 3
    expected_steps = [(2.0, -1.0), (3.0, -0.5), (4.0, -1.5)]
    for pair, (edx, edy) in zip(body["pairs"], expected_steps, strict=True):
        assert abs(pair["dx"] - edx) < 0.1
        assert abs(pair["dy"] - edy) < 0.1
        assert pair["score"] > 0.3

    assert len(body["cumulative_x"]) == 4
    assert body["cumulative_x"][0] == 0.0
    assert body["cumulative_y"][0] == 0.0
    assert abs(body["cumulative_x"][-1] - 9.0) < 0.2
    assert abs(body["cumulative_y"][-1] - (-3.0)) < 0.2


def test_displacement_rejects_a_missing_image() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/displacement",
        json={"image_ids": ["img-does-not-exist", "img-also-missing"], "window": [0.0, 0.0, 10.0, 10.0]},
    )
    assert resp.status_code == 404


def test_displacement_requires_at_least_two_images() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/displacement",
        json={"image_ids": ["img-1"], "window": [0.0, 0.0, 10.0, 10.0]},
    )
    assert resp.status_code == 422
