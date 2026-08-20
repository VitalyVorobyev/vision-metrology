"""Generate `lab/contract/fixtures/`: golden request/response pairs for the core
operations — teach, find, measure (with and without calibration/mm), rectify,
displacement — captured from small, deterministic synthetic PNGs.

## Why these exist

Two consumers replay them:

1. `lab/backend/tests/test_contract_fixtures.py` — runs the same operations against a
   fresh FastAPI backend and asserts the response matches the committed golden within a
   float tolerance. This is the anti-drift gate for the browser (HTTP) path.
2. `lab/frontend/src-tauri/tests/contract_parity.rs` — runs the same operations through
   the Tauri command handlers (native Rust, calling `vision-metrology` directly, no HTTP)
   and asserts the *same* numeric agreement. This is what makes the two backends
   verifiably equivalent rather than "believed to be" (plan decision 7).

## Normalization

A fixture's `request`/`response` bodies are captured verbatim from the API **except**
for identifiers the store assigns at runtime (`img-N`, `model-N`, `cal-N`) — these are
not reproducible across separate runs of the two backends (the Tauri store has its own
counters) and would make every diff noise. Each id-shaped string is replaced, everywhere
it appears (including embedded in a URL like `crop_url`), by a fixed placeholder:
`$IMAGE_ID`, `$FRAME_A_ID`, `$FRAME_B_ID`, `$MODEL_ID`, `$CALIBRATION_ID`. Nothing else is
normalized — this API has no timestamps, and every other field (geometry, scores, pixel
values) is deterministic given the same synthetic input.

A replay test performs the identical substitution on its own run's ids before comparing,
so `assert normalize(actual) == golden` up to float tolerance.

Run from anywhere:

    uv run --directory lab/backend python scripts/export_contract_fixtures.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image as PILImage

_BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_BACKEND_SRC))

import vm_lab.media as media_mod  # noqa: E402
import vm_lab.store as store_mod  # noqa: E402
from vm_lab.app import create_app  # noqa: E402
from vm_lab.config import Settings  # noqa: E402

_CONTRACT_DIR = Path(__file__).resolve().parents[2] / "contract"
_FIXTURES_DIR = _CONTRACT_DIR / "fixtures"
_CAL_SRC = (
    Path(__file__).resolve().parents[3]
    / "crates"
    / "vision-metrology"
    / "tests"
    / "fixtures"
    / "table_calibration.json"
)

# -- deterministic synthetic images -------------------------------------------------

IMG_SIZE = 128
DISC_CENTER = (64.0, 64.0)
DISC_RADIUS = 24.0
ROI = (24.0, 24.0, 80.0, 80.0)

TEXTURE_SIZE = 112
SHIFT_DX, SHIFT_DY = 4.0, 3.0  # frame_b = frame_a's *content* shifted by exactly this
DISPLACEMENT_WINDOW = (20.0, 20.0, 70.0, 70.0)  # (x, y, w, h) in frame_a


def _disc_array(w: int, h: int, cx: float, cy: float, r: float) -> np.ndarray:
    """Anti-aliased bright disc — the true subpixel edge sits exactly at `r`."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.hypot(xs - cx, ys - cy)
    cover = np.clip(r + 0.5 - dist, 0.0, 1.0)
    return (20.0 + 180.0 * cover).round().astype(np.uint8)


# Deterministic, RNG-free value noise (bilinear-interpolated Hermite-eased integer-grid
# hash), same construction as `crates/vision-metrology/tests/accuracy.rs`'s
# `corr_texture` (a different, independent hash — no need to bit-match Rust's, only to
# give `corr::displacement`'s ZNCC search enough local gradient in every direction, which
# a periodic pattern like a plain sinusoid starves at coarse pyramid levels). Evaluable at
# any real `(x, y)`, so a fractional shift is exact rather than a resampling artifact.
_MASK64 = (1 << 64) - 1


def _hash01(ix: int, iy: int) -> float:
    h = (ix * 374761393 + iy * 668265263) & _MASK64
    h = ((h ^ (h >> 13)) * 1274126177) & _MASK64
    h ^= h >> 16
    return (h & 0xFFFF) / 65535.0


def _value_noise(x: float, y: float, cell: float) -> float:
    gx, gy = x / cell, y / cell
    x0, y0 = int(np.floor(gx)), int(np.floor(gy))
    fx, fy = gx - x0, gy - y0
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)
    a = _hash01(x0, y0) * (1.0 - sx) + _hash01(x0 + 1, y0) * sx
    b = _hash01(x0, y0 + 1) * (1.0 - sx) + _hash01(x0 + 1, y0 + 1) * sx
    return a * (1.0 - sy) + b * sy


def _texture_value(x: float, y: float) -> float:
    return 128.0 + 90.0 * (_value_noise(x, y, 9.0) - 0.5) + 40.0 * (_value_noise(x, y, 2.5) - 0.5)


def _texture_frame(size: int, dx: float, dy: float) -> np.ndarray:
    """Renders `size x size` of `_texture_value`, content shifted by `(dx, dy)` —
    `corr::displacement` tracked from the `dx=dy=0` frame to this one should report
    exactly `(dx, dy)`."""
    out = np.empty((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            v = _texture_value(x - dx, y - dy)
            out[y, x] = int(round(min(255.0, max(0.0, v))))
    return out


def _png_bytes(pixels: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    PILImage.fromarray(pixels, mode="L").save(buf, "PNG")
    return buf.getvalue()


# -- normalization -------------------------------------------------------------------


def _normalize(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        out = value
        for real, placeholder in mapping.items():
            out = out.replace(real, placeholder)
        return out
    if isinstance(value, list):
        return [_normalize(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v, mapping) for k, v in value.items()}
    return value


def _fixture(name: str, description: str, request_body: Any, response_body: Any, mapping: dict[str, str]) -> dict[str, Any]:
    return {
        "operation": name,
        "description": description,
        "request": _normalize(request_body, mapping),
        "response": _normalize(response_body, mapping),
    }


class Run:
    """One full pass through the operation sequence against `client`.

    `fixtures[name]` is the normalized `{operation, description, request, response}`
    dict for each of the six operations; `images[filename]` and `crop_png` are the raw
    PNG bytes involved, for a caller that wants to write or compare them too. Both
    `export_contract_fixtures.main` (write mode) and
    `tests/test_contract_fixtures.py` (replay mode) build one of these and only differ
    in what they do with it — this is the one place the operation sequence itself is
    written, so the two can never silently diverge.
    """

    def __init__(self, client: TestClient) -> None:
        self.fixtures: dict[str, dict[str, Any]] = {}
        self.images: dict[str, bytes] = {}
        self.crop_png: bytes = b""
        self._run(client)

    def _run(self, client: TestClient) -> None:
        # -- images ------------------------------------------------------------------
        disc_bytes = _png_bytes(_disc_array(IMG_SIZE, IMG_SIZE, *DISC_CENTER, DISC_RADIUS))
        self.images["disc.png"] = disc_bytes
        image_id = client.post("/api/images", files={"file": ("disc.png", disc_bytes, "image/png")}).json()["id"]

        frame_a = _texture_frame(TEXTURE_SIZE, 0.0, 0.0)
        frame_b = _texture_frame(TEXTURE_SIZE, SHIFT_DX, SHIFT_DY)
        frame_a_bytes, frame_b_bytes = _png_bytes(frame_a), _png_bytes(frame_b)
        self.images["frame_a.png"] = frame_a_bytes
        self.images["frame_b.png"] = frame_b_bytes
        frame_a_id = client.post("/api/images", files={"file": ("frame_a.png", frame_a_bytes, "image/png")}).json()["id"]
        frame_b_id = client.post("/api/images", files={"file": ("frame_b.png", frame_b_bytes, "image/png")}).json()["id"]

        cal_bytes = _CAL_SRC.read_bytes()
        self.images["calibration.json"] = cal_bytes
        calibration_id = client.post(
            "/api/calibration", files={"file": ("calibration.json", cal_bytes, "application/json")}
        ).json()["id"]

        # -- teach ---------------------------------------------------------------------
        teach_req = {"image_id": image_id, "roi": list(ROI), "min_contrast": 0.15}
        model = client.post("/api/models", json=teach_req).json()
        model_id = model["id"]

        mapping = {
            image_id: "$IMAGE_ID",
            frame_a_id: "$FRAME_A_ID",
            frame_b_id: "$FRAME_B_ID",
            calibration_id: "$CALIBRATION_ID",
            model_id: "$MODEL_ID",
        }
        self.fixtures["teach"] = _fixture(
            "teach", "POST /api/models over the disc image's ROI -> ModelOut (id normalized).", teach_req, model, mapping
        )

        # -- find ----------------------------------------------------------------------
        find_req = {"image_id": image_id, "model_id": model_id, "min_score": 0.5}
        find_resp = client.post("/api/find", json=find_req).json()
        self.fixtures["find"] = _fixture(
            "find", "POST /api/find of the taught model against its own image.", find_req, find_resp, mapping
        )

        # -- measure (no calibration) ----------------------------------------------------
        measure_obj = {
            "kind": "circle",
            "label": "outer edge",
            "cx": DISC_CENTER[0],
            "cy": DISC_CENTER[1],
            "r": DISC_RADIUS,
            "n_calipers": 12,
            "caliper_len": 8.0,
            "caliper_width": 4.0,
        }
        measure_req = {"image_id": image_id, "model_id": model_id, "min_score": 0.5, "objects": [measure_obj]}
        measure_resp = client.post("/api/measure", json=measure_req).json()
        self.fixtures["measure"] = _fixture(
            "measure",
            "POST /api/measure (auto-find fixture, pixel units only) of the disc's outer edge.",
            measure_req,
            measure_resp,
            mapping,
        )

        # -- measure with calibration/mm ---------------------------------------------------
        measure_mm_req = {
            "image_id": image_id,
            "model_id": model_id,
            "min_score": 0.5,
            "calibration_id": calibration_id,
            "camera_index": 1,
            # camera0's own extrinsics sit exactly on z=0 (degenerate); offset the plane,
            # matching lab/backend/tests/test_smoke.py's own reasoning.
            "plane": {"nx": 0.0, "ny": 0.0, "nz": 1.0, "d": -100.0},
            "objects": [measure_obj],
        }
        measure_mm_resp = client.post("/api/measure", json=measure_mm_req).json()
        self.fixtures["measure_mm"] = _fixture(
            "measure_mm",
            "POST /api/measure with calibration_id/camera_index/plane -> mm fields populated.",
            measure_mm_req,
            measure_mm_resp,
            mapping,
        )

        # -- rectify ---------------------------------------------------------------------
        rectify_req = {
            "image_id": image_id,
            "model_id": model_id,
            "min_score": 0.5,
            "crop": {"rect": list(ROI), "px_per_unit": 1.0},
        }
        rectify_resp = client.post("/api/rectify", json=rectify_req).json()
        self.fixtures["rectify"] = _fixture(
            "rectify", "POST /api/rectify of the taught model's own ROI -> a canonical crop.", rectify_req, rectify_resp, mapping
        )
        self.crop_png = client.get(rectify_resp["matches"][0]["crop_url"]).content

        # -- displacement ------------------------------------------------------------------
        displacement_req = {
            "image_ids": [frame_a_id, frame_b_id],
            "window": list(DISPLACEMENT_WINDOW),
            "search_x": 10,
            "search_y": 10,
            "refine": "lucas_kanade",
        }
        displacement_resp = client.post("/api/displacement", json=displacement_req).json()
        self.fixtures["displacement"] = _fixture(
            "displacement",
            f"POST /api/displacement over frame_a/frame_b, shifted exactly ({SHIFT_DX}, {SHIFT_DY}) px.",
            displacement_req,
            displacement_resp,
            mapping,
        )


def isolated_client() -> tuple[TestClient, tempfile.TemporaryDirectory]:
    """A fresh `TestClient` over an isolated `data_dir` with reset store counters —
    what both the generator and the replay test need before calling `Run`."""
    tmp = tempfile.TemporaryDirectory()
    isolated = Settings(data_dir=Path(tmp.name))
    store_mod.settings = isolated
    media_mod.settings = isolated
    store_mod.store.__init__()  # fresh registries, counters back to 1
    return TestClient(create_app()), tmp


def main() -> None:
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    client, tmp = isolated_client()
    with tmp:
        run = Run(client)
        for filename, payload in run.images.items():
            (_FIXTURES_DIR / filename).write_bytes(payload)
        (_FIXTURES_DIR / "rectify_crop.png").write_bytes(run.crop_png)
        for name, fixture in run.fixtures.items():
            path = _FIXTURES_DIR / f"{name}.json"
            path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
            print(f"wrote {path}")

    print(f"\nFixtures written to {_FIXTURES_DIR}")


if __name__ == "__main__":
    main()
