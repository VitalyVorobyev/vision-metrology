"""POST /api/measure — teach → find → measure → judge, the last step.

Places a `MetrologyModel` at a fixture pose (explicit, or the top auto-found match) and
measures each requested object. Everything the frontend needs to draw is computed here,
in source-image pixel coordinates — the client never reconstructs geometry (see
lab/README.md's "source-frame" rule).

Two passes over the same calipers, deliberately:
1. `vm.MetrologyModel.apply` does the real measurement — robust fit, residuals.
2. This module's own `_place_calipers` (mirroring `MetrologyModel::measure_one`,
   `crates/vision-metrology/src/measure/model.rs`) re-derives each caliper and measures
   it again, only to report **why** a caliper was rejected and to expose its raw profile
   for `LineProfile` — detail `apply`'s binding does not surface. Both passes share the
   same `MeasureConfig`, so they agree on every caliper that succeeds.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

import vision_metrology as vm

from vm_lab.geometry import perp, to_scene
from vm_lab.routers.find import run_find
from vm_lab.schemas import (
    CaliperProfileOut,
    CaliperResultOut,
    EdgeMarkOut,
    FindRequest,
    FixtureIn,
    MeasureObjectIn,
    MeasureObjectResultOut,
    MeasureRequest,
    MeasureResponse,
    OverlayPrimitiveOut,
)
from vm_lab.store import store

router = APIRouter(prefix="/api/measure", tags=["measure"])


def _resolve_metric(req: MeasureRequest) -> tuple[vm.CameraModel, np.ndarray, vm.Plane3] | None:
    """`(camera, pose, plane)` when `req.calibration_id` is set, else `None` — the
    off switch for every mm computation below."""
    if req.calibration_id is None:
        return None
    if req.calibration_id not in store.calibrations:
        raise HTTPException(404, f"no such calibration: {req.calibration_id}")
    cameras = store.get_calibration(req.calibration_id)
    if not 0 <= req.camera_index < len(cameras):
        raise HTTPException(
            400, f"camera_index {req.camera_index} out of range (calibration has {len(cameras)} cameras)"
        )
    camera, pose = cameras[req.camera_index]
    plane = vm.Plane3((req.plane.nx, req.plane.ny, req.plane.nz), req.plane.d)
    return camera, pose, plane


def _pixel_to_plane_mm(metric: tuple[vm.CameraModel, np.ndarray, vm.Plane3], x: float, y: float) -> tuple[float, float] | None:
    camera, pose, plane = metric
    pixels = np.array([[x, y]], dtype=np.float32)
    out = vm.pixel_to_plane(camera, pose, plane, pixels)[0]
    if not np.isfinite(out).all():
        return None
    return float(out[0]), float(out[1])


def _measure_config(obj: MeasureObjectIn) -> vm.MeasureConfig:
    # `select="strongest"` matches `MetrologyObject::new`'s Rust default (model.rs) —
    # forced explicitly here rather than left to the binding's own default, so this
    # config is guaranteed identical to the one handed to `vm.MetrologyObject` below.
    return vm.MeasureConfig(
        sigma=obj.measure.sigma,
        threshold=obj.measure.threshold,
        polarity=obj.measure.polarity,
        select="strongest",
        max_obliquity_deg=obj.measure.max_obliquity_deg,
    )


def _fit_config(obj: MeasureObjectIn) -> vm.FitConfig:
    return vm.FitConfig(loss=obj.fit.loss, inlier_tol=obj.fit.inlier_tol)


def _vm_shape(obj: MeasureObjectIn) -> Any:
    if obj.kind == "circle":
        if obj.cx is None or obj.cy is None or obj.r is None:
            raise HTTPException(400, "circle object needs cx, cy, r")
        arc = None
        if obj.arc is not None:
            arc = (math.radians(obj.arc[0]), math.radians(obj.arc[1]))
        return vm.MetrologyShape.circle((obj.cx, obj.cy), obj.r, arc)
    if obj.ax is None or obj.ay is None or obj.bx is None or obj.by is None:
        raise HTTPException(400, "line object needs ax, ay, bx, by")
    return vm.MetrologyShape.line((obj.ax, obj.ay), (obj.bx, obj.by))


def _place_calipers(
    obj: MeasureObjectIn, origin: tuple[float, float], x: float, y: float, angle: float, scale: float
) -> list[tuple[str, dict[str, Any]]]:
    """Re-derives each caliper's placement, mirroring `MetrologyModel::measure_one`."""
    n = obj.n_calipers
    specs: list[tuple[str, dict[str, Any]]] = []
    half_len = obj.caliper_len * scale
    half_width = obj.caliper_width * scale
    if obj.kind == "line":
        assert obj.ax is not None and obj.ay is not None and obj.bx is not None and obj.by is not None
        ax, ay = to_scene(obj.ax, obj.ay, origin, x, y, angle, scale)
        bx, by = to_scene(obj.bx, obj.by, origin, x, y, angle, scale)
        length = math.hypot(bx - ax, by - ay)
        if length < 1e-6:
            raise HTTPException(400, "metrology line has zero length")
        alx, aly = (bx - ax) / length, (by - ay) / length
        nx, ny = perp(alx, aly)
        cal_angle = math.atan2(ny, nx)
        for i in range(n):
            f = i / (n - 1)
            cx, cy = ax + (bx - ax) * f, ay + (by - ay) * f
            specs.append(("rect", dict(center=(cx, cy), angle=cal_angle, half_len=half_len, half_width=half_width)))
    else:
        assert obj.cx is not None and obj.cy is not None and obj.r is not None
        cx, cy = to_scene(obj.cx, obj.cy, origin, x, y, angle, scale)
        radius = obj.r * scale
        if obj.arc is not None:
            start, extent = math.radians(obj.arc[0]), math.radians(obj.arc[1])
        else:
            start, extent = 0.0, 2 * math.pi
        start = start + angle
        closed = abs(abs(extent) - 2 * math.pi) < 1e-3
        denom = n if closed else (n - 1)
        for i in range(n):
            phi = start + extent * i / denom
            specs.append(
                ("radial", dict(center=(cx, cy), radius=radius, angle=phi, half_len=half_len, half_width=half_width))
            )
    return specs


def _measure_calipers(
    specs: list[tuple[str, dict[str, Any]]],
    config: vm.MeasureConfig,
    img: Any,
    metric: tuple[vm.CameraModel, np.ndarray, vm.Plane3] | None = None,
) -> tuple[list[CaliperResultOut], list[OverlayPrimitiveOut]]:
    results: list[CaliperResultOut] = []
    overlay: list[OverlayPrimitiveOut] = []
    step_px = config.step
    for i, (kind, kwargs) in enumerate(specs):
        cal = vm.Caliper.rect(kwargs["center"], kwargs["angle"], kwargs["half_len"], kwargs["half_width"], config) \
            if kind == "rect" else \
            vm.Caliper.radial(kwargs["center"], kwargs["radius"], kwargs["angle"], kwargs["half_len"], kwargs["half_width"], config)
        cx, cy = kwargs["center"]
        box_angle = kwargs["angle"]
        try:
            edges = cal.measure(img)
        except vm.MeasureRejected as exc:
            reason = str(exc.args[0]) if exc.args else "rejected"
            profile = CaliperProfileOut(values=list(cal.profile()), step_px=step_px, edges=[])
            results.append(CaliperResultOut(index=i, status="rejected", reason=reason, profile=profile))
            overlay.append(
                OverlayPrimitiveOut(
                    kind="caliper", tone="defect", cx=cx, cy=cy,
                    width=2 * kwargs["half_len"], height=2 * kwargs["half_width"], angle=box_angle,
                )
            )
            continue
        edge = edges[0]
        mm = _pixel_to_plane_mm(metric, edge.x, edge.y) if metric is not None else None
        profile = CaliperProfileOut(
            values=list(cal.profile()), step_px=step_px,
            edges=[
                EdgeMarkOut(
                    pos_px=edge.t,
                    polarity=edge.polarity,
                    x_mm=mm[0] if mm is not None else None,
                    y_mm=mm[1] if mm is not None else None,
                )
            ],
        )
        results.append(CaliperResultOut(index=i, status="hit", profile=profile))
        overlay.append(
            OverlayPrimitiveOut(
                kind="caliper", tone="signal", cx=cx, cy=cy,
                width=2 * kwargs["half_len"], height=2 * kwargs["half_width"], angle=box_angle,
            )
        )
        overlay.append(OverlayPrimitiveOut(kind="point", tone="signal", x=edge.x, y=edge.y, cross=True))
    return results, overlay


def _resolve_fixture(req: MeasureRequest) -> tuple[FixtureIn, str]:
    if req.fixture is not None:
        return req.fixture, "explicit"
    find_req = FindRequest(image_id=req.image_id, model_id=req.model_id, min_score=req.min_score, max_matches=1)
    matches = run_find(find_req)
    if not matches:
        raise HTTPException(422, "auto-find found no match at or above min_score")
    best = max(matches, key=lambda m: m.score)
    return FixtureIn(x=best.x, y=best.y, angle=best.angle, scale=best.scale), "auto_find"


@router.post("", response_model=MeasureResponse)
async def measure(req: MeasureRequest) -> MeasureResponse:
    if req.image_id not in store.images:
        raise HTTPException(404, f"no such image: {req.image_id}")
    if req.model_id not in store.models:
        raise HTTPException(404, f"no such model: {req.model_id}")
    if not req.objects:
        raise HTTPException(400, "at least one object is required")

    fixture, source = _resolve_fixture(req)
    img = store.load_array(req.image_id)
    model = store.get_model(req.model_id)
    origin = model.origin
    metric = _resolve_metric(req)

    metrology_model = vm.MetrologyModel()
    for obj in req.objects:
        if obj.n_calipers < 2:
            raise HTTPException(400, "n_calipers must be >= 2")
        metrology_model.add(
            vm.MetrologyObject(
                shape=_vm_shape(obj),
                n_calipers=obj.n_calipers,
                caliper_len=obj.caliper_len,
                caliper_width=obj.caliper_width,
                measure=_measure_config(obj),
                fit=_fit_config(obj),
            )
        )

    raw_results = metrology_model.apply(
        img, x=fixture.x, y=fixture.y, angle=fixture.angle, scale=fixture.scale, origin=origin
    )

    out_objects: list[MeasureObjectResultOut] = []
    for obj, raw in zip(req.objects, raw_results, strict=True):
        specs = _place_calipers(obj, origin, fixture.x, fixture.y, fixture.angle, fixture.scale)
        config = _measure_config(obj)
        calipers, cal_overlay = _measure_calipers(specs, config, img, metric)

        if isinstance(raw, vm.MetrologyError):
            out_objects.append(
                MeasureObjectResultOut(kind="error", label=obj.label, message=raw.message, calipers=calipers, overlay=cal_overlay)
            )
            continue

        overlay = list(cal_overlay)
        if raw.kind == "circle" and raw.circle is not None:
            overlay.append(OverlayPrimitiveOut(kind="circle", tone="normal", cx=raw.circle.cx, cy=raw.circle.cy, r=raw.circle.r))
        elif raw.kind == "line" and raw.line is not None:
            line = raw.line
            ts = [(e.x - line.px) * line.dx + (e.y - line.py) * line.dy for e in raw.hits]
            tmin, tmax = (min(ts), max(ts)) if ts else (-obj.caliper_len, obj.caliper_len)
            overlay.append(
                OverlayPrimitiveOut(
                    kind="segment", tone="normal",
                    x1=line.px + line.dx * tmin, y1=line.py + line.dy * tmin,
                    x2=line.px + line.dx * tmax, y2=line.py + line.dy * tmax,
                )
            )

        circle_cx_mm = circle_cy_mm = circle_r_mm = None
        if metric is not None and raw.kind == "circle" and raw.circle is not None:
            center_mm = _pixel_to_plane_mm(metric, raw.circle.cx, raw.circle.cy)
            # Radius via two diametral points (exact per point; the distance
            # between them is a fronto-parallel-view approximation of the
            # radius under a tilted camera, see MeasureObjectResultOut docs).
            p1_mm = _pixel_to_plane_mm(metric, raw.circle.cx + raw.circle.r, raw.circle.cy)
            p2_mm = _pixel_to_plane_mm(metric, raw.circle.cx - raw.circle.r, raw.circle.cy)
            if center_mm is not None:
                circle_cx_mm, circle_cy_mm = center_mm
            if p1_mm is not None and p2_mm is not None:
                circle_r_mm = math.hypot(p1_mm[0] - p2_mm[0], p1_mm[1] - p2_mm[1]) / 2.0

        out_objects.append(
            MeasureObjectResultOut(
                kind=raw.kind,
                label=obj.label,
                circle_cx=raw.circle.cx if raw.circle else None,
                circle_cy=raw.circle.cy if raw.circle else None,
                circle_r=raw.circle.r if raw.circle else None,
                line_px=raw.line.px if raw.line else None,
                line_py=raw.line.py if raw.line else None,
                line_dx=raw.line.dx if raw.line else None,
                line_dy=raw.line.dy if raw.line else None,
                rms=raw.rms,
                max_dev=raw.max_dev,
                n_used=raw.n_used,
                circle_cx_mm=circle_cx_mm,
                circle_cy_mm=circle_cy_mm,
                circle_r_mm=circle_r_mm,
                calipers=calipers,
                overlay=overlay,
            )
        )

    return MeasureResponse(fixture=fixture, fixture_source=source, objects=out_objects)
