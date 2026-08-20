//! `measure` — mirrors `lab/backend/src/vm_lab/routers/measure.py`.
//!
//! Two passes over the same calipers, same reasoning as the Python router: (1)
//! `MetrologyModel::apply` does the real measurement (robust fit, residuals); (2) this
//! module re-measures each caliper individually, using the *exact same placement*
//! `apply` used internally (`measure::diagnostics::layout` — the W6 Part A API), only to
//! report which caliper was rejected and why, and to expose its raw profile. Both passes
//! share the same `MeasureConfig`, so they agree on every caliper that succeeds.

use std::collections::HashMap;

use vision_metrology::fit::{FitConfig, RobustLoss};
use vision_metrology::measure::diagnostics::{CaliperShape, layout};
use vision_metrology::measure::{
    Caliper, EdgeSelect, MeasureConfig as NativeMeasureConfig, MetrologyFit, MetrologyModel,
    MetrologyObject, MetrologyShape, PolaritySelect, RejectReason,
};
use vision_metrology::metric::{CameraModel, Plane3, Pose3, pixel_to_plane};
use vm_primitives::{Point2f, Similarity2f, Vec2f, similarity_from_parts, wrap_angle};

use super::find::run_find;
use crate::error::{AppError, AppResult, not_found};
use crate::state::AppState;
use crate::types::{
    CaliperProfileOut, CaliperResultOut, EdgeMarkOut, FindRequest, FixtureIn, MeasureObjectIn,
    MeasureObjectResultOut, MeasureRequest, MeasureResponse, MeasureShapeKind, OverlayPrimitiveOut,
};

/// `Translation(position) ∘ sR ∘ Translation(−origin)`, as a similarity — identical to
/// `ShapeMatch::pose`'s own construction (mirrors `vm-python`'s `measure_py::pose_from`).
fn pose_from(position: Point2f, angle: f32, scale: f32, origin: Point2f) -> Similarity2f {
    let (sn, cs) = wrap_angle(angle).sin_cos();
    let t = Vec2f::new(
        position.x - scale * (cs * origin.x - sn * origin.y),
        position.y - scale * (sn * origin.x + cs * origin.y),
    );
    similarity_from_parts(t, wrap_angle(angle), scale)
}

fn polarity_from(s: Option<&str>) -> PolaritySelect {
    match s {
        Some("rising") | Some("dark_to_bright") => PolaritySelect::Rising,
        Some("falling") | Some("bright_to_dark") => PolaritySelect::Falling,
        _ => PolaritySelect::Any,
    }
}

/// `inlier_tol` doubles as the robust-loss radius here — the lab's own `FitConfigIn`
/// only exposes `loss`/`inlier_tol`, no separate loss-scale knob, and this is the more
/// useful reading of the pair for a caller who never touches RANSAC (which needs
/// `ransac_iters` set too, and this command exposes no such field).
fn fit_from(fit: Option<&crate::types::FitConfigIn>) -> FitConfig {
    let loss_str = fit.and_then(|f| f.loss.as_deref()).unwrap_or("l2");
    let radius = fit.and_then(|f| f.inlier_tol).unwrap_or(2.0);
    let loss = match loss_str {
        "huber" => RobustLoss::Huber { k: radius },
        "tukey" => RobustLoss::Tukey { c: radius },
        _ => RobustLoss::None,
    };
    FitConfig {
        loss,
        ..FitConfig::default()
    }
}

fn measure_config_from(m: Option<&crate::types::MeasureConfigIn>) -> NativeMeasureConfig {
    let base = NativeMeasureConfig {
        select: EdgeSelect::Strongest,
        ..NativeMeasureConfig::default()
    };
    NativeMeasureConfig {
        sigma: m.and_then(|c| c.sigma).unwrap_or(base.sigma),
        threshold: m.and_then(|c| c.threshold).unwrap_or(base.threshold),
        polarity: m
            .and_then(|c| c.polarity.as_deref())
            .map(|s| polarity_from(Some(s)))
            .unwrap_or(base.polarity),
        max_obliquity_deg: m
            .and_then(|c| c.max_obliquity_deg)
            .unwrap_or(base.max_obliquity_deg),
        ..base
    }
}

fn shape_from(obj: &MeasureObjectIn) -> AppResult<MetrologyShape> {
    match obj.kind {
        MeasureShapeKind::Circle => {
            let (cx, cy, r) = (
                obj.cx
                    .ok_or_else(|| AppError("circle object needs cx".into()))?,
                obj.cy
                    .ok_or_else(|| AppError("circle object needs cy".into()))?,
                obj.r
                    .ok_or_else(|| AppError("circle object needs r".into()))?,
            );
            let arc = obj.arc.map(|(a, b)| (a.to_radians(), b.to_radians()));
            Ok(MetrologyShape::Circle {
                center: Point2f::new(cx, cy),
                radius: r,
                arc,
            })
        }
        MeasureShapeKind::Line => {
            let (ax, ay, bx, by) = (
                obj.ax
                    .ok_or_else(|| AppError("line object needs ax".into()))?,
                obj.ay
                    .ok_or_else(|| AppError("line object needs ay".into()))?,
                obj.bx
                    .ok_or_else(|| AppError("line object needs bx".into()))?,
                obj.by
                    .ok_or_else(|| AppError("line object needs by".into()))?,
            );
            Ok(MetrologyShape::Line {
                a: Point2f::new(ax, ay),
                b: Point2f::new(bx, by),
            })
        }
    }
}

type Metric = (CameraModel, Pose3, Plane3);

fn resolve_metric(state: &AppState, req: &MeasureRequest) -> AppResult<Option<Metric>> {
    let Some(cal_id) = &req.calibration_id else {
        return Ok(None);
    };
    let calibrations = state
        .calibrations
        .lock()
        .expect("calibrations mutex poisoned");
    let entry = calibrations
        .get(cal_id)
        .ok_or_else(|| not_found("calibration", cal_id))?;
    let (camera, pose) = *entry.cameras.get(req.camera_index).ok_or_else(|| {
        AppError(format!(
            "camera_index {} out of range (calibration has {} cameras)",
            req.camera_index,
            entry.cameras.len()
        ))
    })?;
    let plane = Plane3 {
        n: vm_primitives::Vec3f::new(req.plane.nx, req.plane.ny, req.plane.nz),
        d: req.plane.d,
    };
    Ok(Some((camera, pose, plane)))
}

fn pixel_to_plane_mm(metric: &Metric, p: Point2f) -> Option<(f32, f32)> {
    let (camera, pose, plane) = metric;
    pixel_to_plane(camera, pose, plane, p).map(|q| (q.x, q.y))
}

fn resolve_fixture(state: &AppState, req: &MeasureRequest) -> AppResult<(FixtureIn, &'static str)> {
    if let Some(f) = req.fixture {
        return Ok((f, "explicit"));
    }
    let find_req = FindRequest {
        image_id: req.image_id.clone(),
        model_id: req.model_id.clone(),
        min_score: req.min_score,
        max_matches: Some(1),
        ..FindRequest::default()
    };
    let matches = run_find(state, &find_req)?;
    let best = matches
        .into_iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).expect("score is never NaN"))
        .ok_or_else(|| AppError("auto-find found no match at or above min_score".into()))?;
    Ok((
        FixtureIn {
            x: best.position.x,
            y: best.position.y,
            angle: best.angle(),
            scale: best.scale(),
        },
        "auto_find",
    ))
}

fn caliper_for(shape: &CaliperShape, config: NativeMeasureConfig) -> Caliper {
    match *shape {
        CaliperShape::Rect(r) => Caliper::rect(r, config),
        CaliperShape::Radial(r) => Caliper::radial(r, config),
    }
}

fn reject_reason_str(r: RejectReason) -> &'static str {
    match r {
        RejectReason::ProfileTooShort => "profile_too_short",
        RejectReason::NoEdge => "no_edge",
        RejectReason::WrongPolarity => "wrong_polarity",
        RejectReason::TooOblique => "too_oblique",
        RejectReason::OffImage => "off_image",
    }
}

fn placement_geometry(shape: &CaliperShape) -> (Point2f, f32, f32, f32) {
    match *shape {
        CaliperShape::Rect(r) => (r.center, r.angle, r.half_len, r.half_width),
        CaliperShape::Radial(r) => (r.center, r.angle, r.half_len, r.half_width),
    }
}

fn measure_calipers(
    shapes: &[CaliperShape],
    config: NativeMeasureConfig,
    img: &vm_primitives::ImageView<'_, u8>,
    metric: Option<&Metric>,
) -> (Vec<CaliperResultOut>, Vec<OverlayPrimitiveOut>) {
    let mut results = Vec::with_capacity(shapes.len());
    let mut overlay = Vec::new();
    let step_px = config.step;

    for (i, shape) in shapes.iter().enumerate() {
        let mut cal = caliper_for(shape, config);
        let (center, angle, half_len, half_width) = placement_geometry(shape);
        match cal.measure(img) {
            Err(reason) => {
                let profile = CaliperProfileOut {
                    values: cal.profile().to_vec(),
                    step_px,
                    edges: Vec::new(),
                };
                results.push(CaliperResultOut {
                    index: i,
                    status: "rejected",
                    reason: Some(reject_reason_str(reason).to_string()),
                    profile,
                });
                overlay.push(OverlayPrimitiveOut {
                    kind: "caliper",
                    tone: Some("defect"),
                    cx: Some(center.x),
                    cy: Some(center.y),
                    width: Some(2.0 * half_len),
                    height: Some(2.0 * half_width),
                    angle: Some(angle),
                    ..Default::default()
                });
            }
            Ok(edges) => {
                let edge = edges[0];
                let mm = metric.and_then(|m| pixel_to_plane_mm(m, edge.p));
                let profile = CaliperProfileOut {
                    values: cal.profile().to_vec(),
                    step_px,
                    edges: vec![EdgeMarkOut {
                        pos_px: edge.t,
                        polarity: format!("{:?}", edge.polarity).to_lowercase(),
                        x_mm: mm.map(|m| m.0),
                        y_mm: mm.map(|m| m.1),
                    }],
                };
                results.push(CaliperResultOut {
                    index: i,
                    status: "hit",
                    reason: None,
                    profile,
                });
                overlay.push(OverlayPrimitiveOut {
                    kind: "caliper",
                    tone: Some("signal"),
                    cx: Some(center.x),
                    cy: Some(center.y),
                    width: Some(2.0 * half_len),
                    height: Some(2.0 * half_width),
                    angle: Some(angle),
                    ..Default::default()
                });
                overlay.push(OverlayPrimitiveOut {
                    kind: "point",
                    tone: Some("signal"),
                    x: Some(edge.p.x),
                    y: Some(edge.p.y),
                    cross: Some(true),
                    ..Default::default()
                });
            }
        }
    }
    (results, overlay)
}

pub fn measure(state: &AppState, req: MeasureRequest) -> AppResult<MeasureResponse> {
    if req.objects.is_empty() {
        return Err("at least one object is required".into());
    }
    for obj in &req.objects {
        if obj.n_calipers < 2 {
            return Err("n_calipers must be >= 2".into());
        }
    }

    let (fixture, source) = resolve_fixture(state, &req)?;
    let metric = resolve_metric(state, &req)?;

    let origin = {
        let models = state.models.lock().expect("models mutex poisoned");
        models
            .get(&req.model_id)
            .ok_or_else(|| not_found("model", &req.model_id))?
            .model
            .origin()
    };
    // After the models lock is released: `decoded` takes the images lock.
    let image = state.decoded(&req.image_id)?;

    let pose = pose_from(
        Point2f::new(fixture.x, fixture.y),
        fixture.angle,
        fixture.scale,
        origin,
    );

    let mut metrology_model = MetrologyModel::new();
    for obj in &req.objects {
        let shape = shape_from(obj)?;
        metrology_model.add(MetrologyObject {
            shape,
            n_calipers: obj.n_calipers,
            caliper_len: obj.caliper_len,
            caliper_width: obj.caliper_width,
            measure: measure_config_from(obj.measure.as_ref()),
            fit: fit_from(obj.fit.as_ref()),
        });
    }

    let view = image.as_view();
    let raw_results = metrology_model.apply(&view, &pose);
    let placements = layout(&metrology_model, &pose);
    let mut by_object: HashMap<usize, Vec<CaliperShape>> = HashMap::new();
    for p in placements {
        by_object.entry(p.object_index).or_default().push(p.shape);
    }

    let mut out_objects = Vec::with_capacity(req.objects.len());
    for (i, (obj, raw)) in req.objects.iter().zip(raw_results).enumerate() {
        let config = measure_config_from(obj.measure.as_ref());
        let shapes = by_object.remove(&i).unwrap_or_default();
        let (calipers, cal_overlay) = measure_calipers(&shapes, config, &view, metric.as_ref());

        let raw = match raw {
            Err(e) => {
                out_objects.push(MeasureObjectResultOut {
                    kind: "error",
                    label: obj.label.clone(),
                    message: Some(e.to_string()),
                    circle_cx: None,
                    circle_cy: None,
                    circle_r: None,
                    line_px: None,
                    line_py: None,
                    line_dx: None,
                    line_dy: None,
                    rms: None,
                    max_dev: None,
                    n_used: None,
                    circle_cx_mm: None,
                    circle_cy_mm: None,
                    circle_r_mm: None,
                    calipers,
                    overlay: cal_overlay,
                });
                continue;
            }
            Ok(r) => r,
        };

        let mut overlay = cal_overlay;
        let (kind, circle_cx, circle_cy, circle_r, line_px, line_py, line_dx, line_dy) =
            match &raw.fit {
                MetrologyFit::Circle(f) => {
                    overlay.push(OverlayPrimitiveOut {
                        kind: "circle",
                        tone: Some("normal"),
                        cx: Some(f.model.center.x),
                        cy: Some(f.model.center.y),
                        r: Some(f.model.radius),
                        ..Default::default()
                    });
                    (
                        "circle",
                        Some(f.model.center.x),
                        Some(f.model.center.y),
                        Some(f.model.radius),
                        None,
                        None,
                        None,
                        None,
                    )
                }
                MetrologyFit::Line(f) => {
                    let ts: Vec<f32> = raw
                        .hits
                        .iter()
                        .map(|e| {
                            (e.p.x - f.model.p.x) * f.model.dir.x
                                + (e.p.y - f.model.p.y) * f.model.dir.y
                        })
                        .collect();
                    let (tmin, tmax) = if ts.is_empty() {
                        (-obj.caliper_len, obj.caliper_len)
                    } else {
                        (
                            ts.iter().cloned().fold(f32::INFINITY, f32::min),
                            ts.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                        )
                    };
                    overlay.push(OverlayPrimitiveOut {
                        kind: "segment",
                        tone: Some("normal"),
                        x1: Some(f.model.p.x + f.model.dir.x * tmin),
                        y1: Some(f.model.p.y + f.model.dir.y * tmin),
                        x2: Some(f.model.p.x + f.model.dir.x * tmax),
                        y2: Some(f.model.p.y + f.model.dir.y * tmax),
                        ..Default::default()
                    });
                    (
                        "line",
                        None,
                        None,
                        None,
                        Some(f.model.p.x),
                        Some(f.model.p.y),
                        Some(f.model.dir.x),
                        Some(f.model.dir.y),
                    )
                }
            };

        let (mut circle_cx_mm, mut circle_cy_mm, mut circle_r_mm) = (None, None, None);
        if let (Some(m), MetrologyFit::Circle(f)) = (&metric, &raw.fit) {
            let center_mm = pixel_to_plane_mm(m, f.model.center);
            let p1_mm = pixel_to_plane_mm(
                m,
                Point2f::new(f.model.center.x + f.model.radius, f.model.center.y),
            );
            let p2_mm = pixel_to_plane_mm(
                m,
                Point2f::new(f.model.center.x - f.model.radius, f.model.center.y),
            );
            if let Some((x, y)) = center_mm {
                circle_cx_mm = Some(x);
                circle_cy_mm = Some(y);
            }
            if let (Some(p1), Some(p2)) = (p1_mm, p2_mm) {
                circle_r_mm = Some(((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)).sqrt() / 2.0);
            }
        }

        out_objects.push(MeasureObjectResultOut {
            kind,
            label: obj.label.clone(),
            message: None,
            circle_cx,
            circle_cy,
            circle_r,
            line_px,
            line_py,
            line_dx,
            line_dy,
            rms: Some(raw.rms()),
            max_dev: Some(raw.max_dev()),
            n_used: Some(raw.n_used()),
            circle_cx_mm,
            circle_cy_mm,
            circle_r_mm,
            calipers,
            overlay,
        });
    }

    Ok(MeasureResponse {
        fixture,
        fixture_source: source,
        objects: out_objects,
    })
}
