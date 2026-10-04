"""POST /api/measure — teach → find → measure → judge, the last step.

Places a `MetrologyModel` at a fixture pose (explicit, or the top auto-found match) and
measures each requested object. Everything the frontend needs to draw is computed here,
in source-image pixel coordinates — the client never reconstructs geometry (see
lab/README.md's "source-frame" rule).

One pass: `vm.MetrologyModel.explain` measures every caliper once and returns, per
object, what `apply` returns (the robust fit and its residuals) together with each
caliper's placement and trace (`vision_metrology::measure::diagnostics::explain_model`).
The placements are the ones the measurement used, so an overlay can never show a caliper
somewhere the measurement did not look; the traces say **why** a caliper was rejected and
carry its raw profile for `LineProfile`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

import vision_metrology as vm

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


# The lab's polarity names, read along the scan, as `vm.MeasureConfig` spells them. The
# desktop command (`commands/measure.rs`, `polarity_from`) maps them the same way.
_POLARITY = {"bright_to_dark": "falling", "dark_to_bright": "rising", "either": "any", None: "any"}


def _measure_config(obj: MeasureObjectIn) -> vm.MeasureConfig:
    # `select="strongest"` matches `MetrologyObject::new`'s Rust default (model.rs): a
    # caliper on a nominal edge reports that edge, not every edge it crosses.
    return vm.MeasureConfig(
        sigma=obj.measure.sigma,
        threshold=obj.measure.threshold,
        polarity=_POLARITY[obj.measure.polarity],
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


def caliper_id(object_index: int, caliper_index: int) -> str:
    """The id a caliper's overlay primitives carry, so a list row can find them."""
    return f"caliper-{object_index}-{caliper_index}"


def _box_center(placement: vm.CaliperPlacement) -> tuple[float, float]:
    """Where a caliper's box sits. A radial placement's `center` is its circle's, so its box
    is `radius` out along the caliper's own axis."""
    cx, cy = placement.center
    if placement.radius is not None:
        cx += placement.radius * math.cos(placement.angle)
        cy += placement.radius * math.sin(placement.angle)
    return cx, cy


def _residual(fit: vm.MetrologyResult | None, x: float, y: float) -> float | None:
    """Signed distance from `(x, y)` to the fitted shape: the residual the fit minimised.
    Outside a circle is positive; for a line, the left of its direction."""
    if fit is None:
        return None
    if fit.kind == "circle" and fit.circle is not None:
        c = fit.circle
        return math.hypot(x - c.cx, y - c.cy) - c.r
    if fit.kind == "line" and fit.line is not None:
        line = fit.line
        return line.dx * (y - line.py) - line.dy * (x - line.px)
    return None


def _caliper_results(
    trace: vm.ObjectTrace,
    object_index: int,
    metric: tuple[vm.CameraModel, np.ndarray, vm.Plane3] | None = None,
) -> tuple[list[CaliperResultOut], list[OverlayPrimitiveOut]]:
    """One object's calipers, in caliper order, from its `explain` trace, and the
    matching overlay. Each caliper's box and edge mark carry `caliper_id`."""
    fit = None if isinstance(trace.result, vm.MetrologyError) else trace.result
    results: list[CaliperResultOut] = []
    overlay: list[OverlayPrimitiveOut] = []
    for placement, cal in zip(trace.placements, trace.calipers, strict=True):
        cx, cy = _box_center(placement)
        box_angle = placement.angle
        i = placement.caliper_index
        cid = caliper_id(object_index, i)
        values = [float(v) for v in cal.profile]
        # Rect and radial calipers sample `±half_len` about their centre, which is where
        # an edge's `t` is measured from.
        span = {"start_px": -placement.half_len, "end_px": placement.half_len}
        box = {
            "kind": "caliper", "id": cid, "cx": cx, "cy": cy,
            "width": 2 * placement.half_len, "height": 2 * placement.half_width, "angle": box_angle,
        }
        if cal.reject is not None:
            profile = CaliperProfileOut(values=values, step_px=cal.spacing, edges=[], **span)
            results.append(CaliperResultOut(index=i, status="rejected", reason=cal.reject, profile=profile))
            overlay.append(OverlayPrimitiveOut(tone="defect", **box))
            continue
        edge = cal.edges[0]
        mm = _pixel_to_plane_mm(metric, edge.x, edge.y) if metric is not None else None
        profile = CaliperProfileOut(
            values=values, step_px=cal.spacing,
            edges=[
                EdgeMarkOut(
                    pos_px=edge.t,
                    polarity=edge.polarity,
                    amplitude=edge.amplitude,
                    x_mm=mm[0] if mm is not None else None,
                    y_mm=mm[1] if mm is not None else None,
                )
            ],
            **span,
        )
        results.append(
            CaliperResultOut(
                index=i, status="hit", profile=profile, residual=_residual(fit, edge.x, edge.y)
            )
        )
        overlay.append(OverlayPrimitiveOut(tone="signal", **box))
        overlay.append(OverlayPrimitiveOut(kind="point", id=cid, tone="signal", x=edge.x, y=edge.y, cross=True))
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

    traces = metrology_model.explain(
        img, x=fixture.x, y=fixture.y, angle=fixture.angle, scale=fixture.scale, origin=origin
    )

    out_objects: list[MeasureObjectResultOut] = []
    for object_index, (obj, trace) in enumerate(zip(req.objects, traces, strict=True)):
        raw = trace.result
        calipers, cal_overlay = _caliper_results(trace, object_index, metric)

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
