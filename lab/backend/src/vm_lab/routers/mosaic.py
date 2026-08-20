"""POST /api/mosaic — composite N calibrated cameras' rectified views of the calibration's
own `z = 0` plane into one grid, using the roadmap plan's **no-blending, nearest-camera-
centre priority** rule: for each destination pixel, among the cameras whose validity mask
is set there, keep the one whose reprojection of that plane point lands closest to its own
principal point (`vm.project_plane_points` -- the exact forward geometry `plane_grid_map`
composes internally, exposed pointwise so every *candidate* camera's own reprojection is
available, not just the one the map already picked). Ties broken by camera order. A pixel
no camera covers gets `source_id = 255`.

This mirrors `crates/vision-metrology/tests/mosaic.rs` and `examples/birdseye_mosaic.rs`
exactly (same rule, same underlying `metric`/`warp` primitives) -- see those files for the
Rust-side derivation and the "why no library module" reasoning.

**Feathering is opt-in and display-only.** `GET .../image?feather=true` returns a linear,
inverse-distance-weighted blend across the overlap instead -- never the default, and never
what `source_id`/coverage numbers describe (a measurement should always trace to one
camera). Both variants are rendered once at `POST` time and cached, mirroring `rectify.py`'s
crop cache: in-memory only, keyed by mosaic id, overwritten by nothing (each `POST` gets a
fresh id) and never disk-persisted.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from PIL import Image as PILImage

import vision_metrology as vm

from vm_lab.schemas import (
    MosaicCameraCoverageOut,
    MosaicRequest,
    MosaicResponse,
)
from vm_lab.store import store

router = APIRouter(prefix="/api/mosaic", tags=["mosaic"])

# Flat fill for a grid pixel no camera covers.
_UNCOVERED_FILL = 30
# Border-sample density (per edge) for auto-fitting the grid from the
# requested cameras' own footprints on the plane.
_BORDER_SAMPLES = 40
# Target grid width (px) when auto-fitting -- a demo/lab-sized resolution,
# not necessarily the dataset's native GSD (mirrors `birdseye_mosaic.rs`'s
# `TARGET_GRID_W` reasoning).
_AUTO_TARGET_W = 800
# Palette for the `source_id` PNG; cycles if more cameras are requested.
_PALETTE = [
    (80, 140, 255),
    (255, 160, 60),
    (90, 220, 120),
    (230, 90, 200),
    (240, 220, 60),
    (120, 200, 230),
]

# mosaic_id -> {"mosaic": (h,w) u8, "feather": (h,w) u8, "source_id": (h,w) u8}
_mosaic_cache: dict[str, dict[str, np.ndarray]] = {}
_next_mosaic_id = 1


def _etag(payload: bytes) -> str:
    return f'"{hashlib.sha256(payload).hexdigest()[:16]}"'


def _resolve_cameras(req: MosaicRequest) -> list[tuple[vm.CameraModel, np.ndarray]]:
    if req.calibration_id not in store.calibrations:
        raise HTTPException(404, f"no such calibration: {req.calibration_id}")
    cameras = store.get_calibration(req.calibration_id)
    resolved: list[tuple[vm.CameraModel, np.ndarray]] = []
    for c in req.cameras:
        if not 0 <= c.camera_index < len(cameras):
            raise HTTPException(
                400,
                f"camera_index {c.camera_index} out of range "
                f"(calibration has {len(cameras)} cameras)",
            )
        if c.image_id not in store.images:
            raise HTTPException(404, f"no such image: {c.image_id}")
        resolved.append(cameras[c.camera_index])
    return resolved


def _auto_fit_grid(
    cams: list[tuple[vm.CameraModel, np.ndarray]], shapes: list[tuple[int, int]]
) -> tuple[tuple[float, float], float, int, int]:
    """`(origin_mm, mm_per_px, width, height)` covering the union of every camera's own
    image-border projection onto the plane, plus a small margin. `pixel_to_plane` already
    handles distortion and camera pose correctly; this needs no new binding."""
    plane = vm.Plane3.xy()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for (camera, pose), (w, h) in zip(cams, shapes, strict=True):
        n = _BORDER_SAMPLES
        top = np.stack([np.linspace(0, w - 1, n), np.zeros(n)], axis=1)
        bottom = np.stack([np.linspace(0, w - 1, n), np.full(n, h - 1)], axis=1)
        left = np.stack([np.zeros(n), np.linspace(0, h - 1, n)], axis=1)
        right = np.stack([np.full(n, w - 1), np.linspace(0, h - 1, n)], axis=1)
        pixels = np.concatenate([top, bottom, left, right], axis=0).astype(np.float32)
        mm = vm.pixel_to_plane(camera, pose, plane, pixels)
        valid = np.isfinite(mm).all(axis=1)
        if valid.any():
            xs.append(mm[valid, 0])
            ys.append(mm[valid, 1])
    if not xs:
        raise HTTPException(
            422,
            "none of the requested cameras' image borders project onto the calibration's "
            "z=0 plane -- pass an explicit grid, or check the calibration/plane",
        )
    all_x, all_y = np.concatenate(xs), np.concatenate(ys)
    x0, x1 = float(all_x.min()), float(all_x.max())
    y0, y1 = float(all_y.min()), float(all_y.max())
    margin_x = 0.02 * max(x1 - x0, 1.0)
    margin_y = 0.02 * max(y1 - y0, 1.0)
    x0, x1 = x0 - margin_x, x1 + margin_x
    y0, y1 = y0 - margin_y, y1 + margin_y

    width_mm = max(x1 - x0, 1e-6)
    mm_per_px = width_mm / _AUTO_TARGET_W
    w = _AUTO_TARGET_W
    h = max(1, round((y1 - y0) / mm_per_px))
    return (x0, y0), mm_per_px, w, h


def _resolve_grid(req: MosaicRequest, cams: list[tuple[vm.CameraModel, np.ndarray]], images: list[np.ndarray]) -> vm.PlaneGrid:
    shapes = [(img.shape[1], img.shape[0]) for img in images]  # (w, h)
    auto_origin, auto_mm_per_px, auto_w, auto_h = _auto_fit_grid(cams, shapes)
    g = req.grid
    origin = g.origin_mm if g.origin_mm is not None else auto_origin
    mm_per_px = g.mm_per_px if g.mm_per_px is not None else auto_mm_per_px
    w = g.width if g.width is not None else auto_w
    h = g.height if g.height is not None else auto_h
    return vm.PlaneGrid(origin, mm_per_px, w, h)


def _grid_points_mm(grid: vm.PlaneGrid) -> np.ndarray:
    xs = grid.origin_mm[0] + np.arange(grid.w, dtype=np.float32) * grid.mm_per_px
    ys = grid.origin_mm[1] + np.arange(grid.h, dtype=np.float32) * grid.mm_per_px
    gx, gy = np.meshgrid(xs, ys)  # each (h, w)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def _composite(
    grid: vm.PlaneGrid, cams: list[tuple[vm.CameraModel, np.ndarray]], images: list[np.ndarray]
) -> dict[str, np.ndarray]:
    h, w = grid.h, grid.w
    pts_mm = _grid_points_mm(grid)

    rectified: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    for (camera, pose), img in zip(cams, images, strict=True):
        m = vm.plane_grid_map(camera, pose, grid)
        dst, mask = m.apply_with_mask(
            img, border_mode="constant", border_constant=float(_UNCOVERED_FILL)
        )
        rectified.append(dst.astype(np.float64))
        masks.append(mask == 255)

        px = vm.project_plane_points(camera, pose, pts_mm)  # (h*w, 2) float64
        pp = np.array([camera.intrinsics.cx, camera.intrinsics.cy], dtype=np.float64)
        d = np.linalg.norm(px - pp, axis=1).reshape(h, w)
        dists.append(np.where(np.isfinite(d), d, np.inf))

    rectified_stack = np.stack(rectified, axis=0)  # (C, h, w) float64
    mask_stack = np.stack(masks, axis=0)  # (C, h, w) bool
    dist_stack = np.stack(dists, axis=0)  # (C, h, w) float64

    any_valid = mask_stack.any(axis=0)
    dist_masked = np.where(mask_stack, dist_stack, np.inf)
    best = np.argmin(dist_masked, axis=0)  # (h, w) -- arbitrary but deterministic where all inf

    mosaic = np.take_along_axis(rectified_stack, best[None, :, :], axis=0)[0]
    mosaic = np.where(any_valid, mosaic, float(_UNCOVERED_FILL))
    source_id = np.where(any_valid, best, 255).astype(np.uint8)

    # Feather (display-only): inverse-distance weighting among valid cameras.
    eps = 1.0
    weights = np.where(mask_stack, 1.0 / (dist_stack + eps), 0.0)
    weight_sum = weights.sum(axis=0)
    feather = np.where(
        weight_sum > 0,
        (weights * rectified_stack).sum(axis=0) / np.maximum(weight_sum, 1e-9),
        float(_UNCOVERED_FILL),
    )

    # max - min over *valid* cameras only, without `nanmax`/`nanmin` (which
    # warn on an all-invalid pixel -- exactly the pixels `overlap_mask`
    # excludes below anyway, so sentinel fills are simpler than suppressing
    # a warning for a value never read).
    n_valid = mask_stack.sum(axis=0)
    overlap_mask = n_valid >= 2
    masked_hi = np.where(mask_stack, rectified_stack, -np.inf).max(axis=0)
    masked_lo = np.where(mask_stack, rectified_stack, np.inf).min(axis=0)
    disparity = masked_hi - masked_lo

    return {
        "mosaic": np.clip(mosaic, 0, 255).astype(np.uint8),
        "feather": np.clip(feather, 0, 255).astype(np.uint8),
        "source_id": source_id,
        "mask_stack": mask_stack,
        "overlap_mask": overlap_mask,
        "disparity": disparity,
    }


def _stats(
    req: MosaicRequest, result: dict[str, np.ndarray]
) -> tuple[list[MosaicCameraCoverageOut], float, float, float | None, float | None]:
    mask_stack = result["mask_stack"]
    n = mask_stack.shape[1] * mask_stack.shape[2]
    coverage = [
        MosaicCameraCoverageOut(
            camera_index=c.camera_index,
            image_id=c.image_id,
            coverage_fraction=float(mask_stack[i].sum()) / n,
        )
        for i, c in enumerate(req.cameras)
    ]
    any_valid = mask_stack.any(axis=0)
    union = int(any_valid.sum())
    overlap = int(result["overlap_mask"].sum())
    union_fraction = union / n
    overlap_fraction = (overlap / union) if union > 0 else 0.0

    disparities = result["disparity"][result["overlap_mask"]]
    if disparities.size == 0:
        return coverage, union_fraction, overlap_fraction, None, None
    p50 = float(np.percentile(disparities, 50))
    p95 = float(np.percentile(disparities, 95))
    return coverage, union_fraction, overlap_fraction, p50, p95


def _colorize_source_id(source_id: np.ndarray, intensity: np.ndarray) -> bytes:
    h, w = source_id.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    k = (intensity.astype(np.float64) / 255.0)[:, :, None]
    for i, color in enumerate(_PALETTE):
        sel = source_id == i
        if sel.any():
            rgb[sel] = (np.array(color, dtype=np.float64) * k[sel]).round().clip(0, 255).astype(np.uint8)
    rgb[source_id == 255] = (24, 24, 28)
    buf = io.BytesIO()
    PILImage.fromarray(rgb, mode="RGB").save(buf, "PNG")
    return buf.getvalue()


def _gray_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


@router.post("", response_model=MosaicResponse)
async def mosaic(req: MosaicRequest) -> MosaicResponse:
    global _next_mosaic_id

    cams = _resolve_cameras(req)
    images = [store.load_array(c.image_id) for c in req.cameras]
    grid = _resolve_grid(req, cams, images)

    result = _composite(grid, cams, images)
    coverage, union_fraction, overlap_fraction, p50, p95 = _stats(req, result)

    mosaic_id = f"mosaic-{_next_mosaic_id}"
    _next_mosaic_id += 1
    _mosaic_cache[mosaic_id] = {
        "mosaic": result["mosaic"],
        "feather": result["feather"],
        "source_id": result["source_id"],
    }

    return MosaicResponse(
        id=mosaic_id,
        width=grid.w,
        height=grid.h,
        origin_mm=grid.origin_mm,
        mm_per_px=grid.mm_per_px,
        image_url=f"/api/mosaic/{mosaic_id}/image",
        source_id_url=f"/api/mosaic/{mosaic_id}/source_id",
        cameras=coverage,
        union_coverage_fraction=union_fraction,
        overlap_fraction=overlap_fraction,
        seam_disparity_p50=p50,
        seam_disparity_p95=p95,
    )


@router.get("/{mosaic_id}/image")
async def get_image(mosaic_id: str, request: Request, feather: bool = False) -> Response:
    cached = _mosaic_cache.get(mosaic_id)
    if cached is None:
        raise HTTPException(404, "no cached mosaic for this id -- POST /api/mosaic first")
    payload = _gray_png(cached["feather"] if feather else cached["mosaic"])
    tag = _etag(payload)
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304)
    return Response(content=payload, media_type="image/png", headers={"ETag": tag})


@router.get("/{mosaic_id}/source_id")
async def get_source_id(mosaic_id: str, request: Request) -> Response:
    cached = _mosaic_cache.get(mosaic_id)
    if cached is None:
        raise HTTPException(404, "no cached mosaic for this id -- POST /api/mosaic first")
    payload = _colorize_source_id(cached["source_id"], cached["mosaic"])
    tag = _etag(payload)
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304)
    return Response(content=payload, media_type="image/png", headers={"ETag": tag})
