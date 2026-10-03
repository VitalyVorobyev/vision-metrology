//! `measure` — mirrors `lab/backend/src/vm_lab/routers/measure.py`.
//!
//! One pass, as in the Python router: `measure::diagnostics::explain_model` measures every
//! caliper once and returns, per object, what `MetrologyModel::apply` returns (the robust
//! fit and its residuals) with each caliper's placement and trace. The traces say which
//! caliper was rejected and why, and carry its raw profile; the placements are the ones
//! the measurement used.

use vision_metrology::fit::{FitConfig, RobustLoss};
use vision_metrology::measure::diagnostics::{CaliperShape, ObjectTrace, explain_model};
use vision_metrology::measure::{
    EdgeSelect, MeasureConfig as NativeMeasureConfig, MetrologyFit, MetrologyModel,
    MetrologyObject, MetrologyShape, PolaritySelect, ProfileConfig, RejectReason,
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
        threshold: m.and_then(|c| c.threshold).unwrap_or(base.threshold),
        polarity: m
            .and_then(|c| c.polarity.as_deref())
            .map(|s| polarity_from(Some(s)))
            .unwrap_or(base.polarity),
        max_obliquity_deg: m
            .and_then(|c| c.max_obliquity_deg)
            .unwrap_or(base.max_obliquity_deg),
        profile: ProfileConfig {
            sigma: m.and_then(|c| c.sigma).unwrap_or(base.profile.sigma),
            ..base.profile
        },
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

fn reject_reason_str(r: RejectReason) -> &'static str {
    match r {
        RejectReason::ProfileTooShort => "profile_too_short",
        RejectReason::NoEdge => "no_edge",
        RejectReason::WrongPolarity => "wrong_polarity",
        RejectReason::TooOblique => "too_oblique",
        RejectReason::OffImage => "off_image",
        RejectReason::IncompleteSequence => "incomplete_sequence",
        RejectReason::LowContrast => "low_contrast",
        RejectReason::NoCrossing => "no_crossing",
    }
}

/// A caliper's box: centre, axis angle, half-length and half-width. A radial placement's
/// `center` is its circle's, so its box sits `radius` out along the caliper's own axis.
fn placement_geometry(shape: &CaliperShape) -> (Point2f, f32, f32, f32) {
    match *shape {
        CaliperShape::Rect(r) => (r.center, r.angle, r.half_len, r.half_width),
        CaliperShape::Radial(r) => {
            let (sn, cs) = r.angle.sin_cos();
            let center = Point2f::new(r.center.x + r.radius * cs, r.center.y + r.radius * sn);
            (center, r.angle, r.half_len, r.half_width)
        }
    }
}

/// The id a caliper's overlay primitives carry, so a list row can find them.
pub fn caliper_id(object_index: usize, caliper_index: usize) -> String {
    format!("caliper-{object_index}-{caliper_index}")
}

/// Signed distance from `p` to the fitted shape: the residual the fit minimised. Outside a
/// circle is positive; for a line, the left of its direction (mirrors the Python router).
fn residual(fit: &MetrologyFit, p: Point2f) -> f32 {
    match fit {
        MetrologyFit::Circle(f) => f.model.signed_distance(p),
        MetrologyFit::Line(f) => {
            let (l, d) = (&f.model, f.model.dir);
            d.x * (p.y - l.p.y) - d.y * (p.x - l.p.x)
        }
    }
}

/// One object's calipers, in caliper order, from its `explain_model` trace, and the
/// matching overlay. Each caliper's box and edge mark carry `caliper_id`.
fn caliper_results(
    trace: &ObjectTrace,
    object_index: usize,
    metric: Option<&Metric>,
) -> (Vec<CaliperResultOut>, Vec<OverlayPrimitiveOut>) {
    let fit = trace.result.as_ref().ok().map(|r| &r.fit);
    let mut results = Vec::with_capacity(trace.calipers.len());
    let mut overlay = Vec::new();

    for (i, (shape, cal)) in trace.placements.iter().zip(&trace.calipers).enumerate() {
        let (center, angle, half_len, half_width) = placement_geometry(shape);
        let id = caliper_id(object_index, i);
        let step_px = cal.spacing;
        // Rect and radial calipers sample `±half_len` about their centre, which is where an
        // edge's `t` is measured from.
        let (start_px, end_px) = (Some(-half_len), Some(half_len));
        let caliper_box = |tone: &'static str| OverlayPrimitiveOut {
            kind: "caliper",
            tone: Some(tone),
            id: Some(id.clone()),
            cx: Some(center.x),
            cy: Some(center.y),
            width: Some(2.0 * half_len),
            height: Some(2.0 * half_width),
            angle: Some(angle),
            ..Default::default()
        };
        match cal.reject {
            Some(reason) => {
                let profile = CaliperProfileOut {
                    values: cal.profile.clone(),
                    step_px,
                    edges: Vec::new(),
                    start_px,
                    end_px,
                };
                results.push(CaliperResultOut {
                    index: i,
                    status: "rejected",
                    reason: Some(reject_reason_str(reason).to_string()),
                    profile,
                    residual: None,
                });
                overlay.push(caliper_box("defect"));
            }
            None => {
                let edge = cal.edges[0];
                let mm = metric.and_then(|m| pixel_to_plane_mm(m, edge.p));
                let profile = CaliperProfileOut {
                    values: cal.profile.clone(),
                    step_px,
                    edges: vec![EdgeMarkOut {
                        pos_px: edge.t,
                        polarity: format!("{:?}", edge.polarity).to_lowercase(),
                        x_mm: mm.map(|m| m.0),
                        y_mm: mm.map(|m| m.1),
                        amplitude: Some(edge.amplitude),
                    }],
                    start_px,
                    end_px,
                };
                results.push(CaliperResultOut {
                    index: i,
                    status: "hit",
                    reason: None,
                    profile,
                    residual: fit.map(|f| residual(f, edge.p)),
                });
                overlay.push(caliper_box("signal"));
                overlay.push(OverlayPrimitiveOut {
                    kind: "point",
                    tone: Some("signal"),
                    id: Some(id.clone()),
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

    let traces = explain_model(&metrology_model, &image.as_view(), &pose);

    let mut out_objects = Vec::with_capacity(req.objects.len());
    for (object_index, (obj, trace)) in req.objects.iter().zip(traces).enumerate() {
        let (calipers, cal_overlay) = caliper_results(&trace, object_index, metric.as_ref());

        let raw = match trace.result {
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

#[cfg(test)]
mod tests {
    use super::*;
    use vm_primitives::Image;

    /// A disc of radius 30 at (64, 64) on a dark 128 × 128 image.
    fn disc() -> Image<u8> {
        let data = (0..128 * 128)
            .map(|i| {
                let (x, y) = ((i % 128) as f32 - 64.0, (i / 128) as f32 - 64.0);
                let cover = (30.5 - (x * x + y * y).sqrt()).clamp(0.0, 1.0);
                (20.0 + 180.0 * cover).round() as u8
            })
            .collect();
        Image::from_vec(128, 128, data).expect("valid image")
    }

    /// The caliper list comes from the one `explain_model` pass: hits with their edge on
    /// the rim, and rejections, with their reason, on flat ground.
    #[test]
    fn calipers_are_read_from_the_traces() {
        let mut model = MetrologyModel::new();
        let mut rim = MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(64.0, 64.0),
            radius: 30.0,
            arc: None,
        });
        rim.n_calipers = 8;
        model.add(rim);
        let mut flat = MetrologyObject::new(MetrologyShape::Line {
            a: Point2f::new(8.0, 10.0),
            b: Point2f::new(8.0, 40.0),
        });
        flat.n_calipers = 3;
        flat.caliper_len = 4.0;
        model.add(flat);
        let img = disc();
        let traces = explain_model(&model, &img.as_view(), &Similarity2f::identity());

        let (rim, rim_overlay) = caliper_results(&traces[0], 0, None);
        assert_eq!(rim.len(), 8);
        assert!(
            rim.iter()
                .all(|c| c.status == "hit" && c.profile.edges.len() == 1)
        );
        assert_eq!(rim_overlay.len(), 16, "a box and an edge point per hit");
        let fitted = traces[0].result.as_ref().expect("the rim fits");
        for (c, hit) in rim.iter().zip(&fitted.hits) {
            assert_eq!(c.profile.edges[0].pos_px, hit.t);
            assert_eq!(c.profile.edges[0].amplitude, Some(hit.amplitude));
            // The residual is the edge's distance from the fit, so none exceeds `max_dev`.
            let r = c.residual.expect("a hit on a fitted object has a residual");
            assert!(r.abs() <= fitted.max_dev() + 1e-5, "residual {r} > max_dev");
            // `MetrologyObject::new`'s caliper reaches 10 px either side of the rim.
            assert_eq!(
                (c.profile.start_px, c.profile.end_px),
                (Some(-10.0), Some(10.0))
            );
        }
        // Each box sits on the rim, where its caliper looked, not at the circle's centre,
        // and its box and edge mark share the caliper's id.
        for (i, pair) in rim_overlay.chunks(2).enumerate() {
            let (b, e) = (&pair[0], &pair[1]);
            assert_eq!((b.kind, e.kind), ("caliper", "point"));
            let id = format!("caliper-0-{i}");
            assert_eq!((b.id.as_deref(), e.id.as_deref()), (Some(&*id), Some(&*id)));
            let r = (b.cx.unwrap() - 64.0).hypot(b.cy.unwrap() - 64.0);
            assert!(
                (r - 30.0).abs() < 1e-3,
                "box {i} centred at radius {r}, not on the rim"
            );
        }

        let (flat, flat_overlay) = caliper_results(&traces[1], 1, None);
        assert!(traces[1].result.is_err(), "nothing to fit");
        assert_eq!(flat.len(), 3);
        for (i, c) in flat.iter().enumerate() {
            assert_eq!((c.index, c.status), (i, "rejected"));
            assert_eq!(c.reason.as_deref(), Some("no_edge"));
            assert_eq!(c.profile.values.len(), 9);
            assert_eq!(c.residual, None);
        }
        assert!(flat_overlay.iter().all(|o| o.tone == Some("defect")));
        assert_eq!(flat_overlay[2].id.as_deref(), Some("caliper-1-2"));
    }
}
