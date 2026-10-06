//! Seeing what a caliper did: [`explain`] traces one measurement, [`explain_model`] traces
//! every caliper of a [`MetrologyModel`] along with its fits, and [`layout`] places a
//! model's calipers at a fixture pose without measuring.
//!
//! ## Caliper layout
//!
//! [`layout`] returns exactly the placements [`MetrologyModel::apply`] measures at.
//! An overlay can draw each [`Caliper`] from them before an image is available, or for
//! calipers that found no edge.

use vm_primitives::{Error, ImageView, LevelEdge, Pixel, Similarity2f};

pub use super::model::CaliperShape;
use super::model::{caliper_placements, measure_placed, placeholder_rect};
use super::{Caliper, MeasureEdge, MetrologyModel, MetrologyObject, MetrologyResult, RejectReason};

/// Everything one caliper measurement computed, from the sampled profile to the result.
///
/// Built by [`explain`]. The edges carry image coordinates and `t` in pixels; a level's
/// `x` is a profile index in samples, which `spacing` converts to pixels.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct CaliperTrace {
    /// Distance between profile samples, in pixels, as [`Caliper::spacing`] reports it:
    /// a level's `x · spacing` is its distance along the scan from the first sample.
    pub spacing: f32,
    /// Number of profile samples.
    pub samples: usize,
    /// Number of lines averaged across the scan into each sample.
    pub across: usize,
    /// [`MeasureConfig::threshold`](super::MeasureConfig::threshold): the response an
    /// edge needed to be a candidate.
    pub threshold: f32,
    /// The averaged profile.
    pub profile: Vec<f32>,
    /// The profile after the Gaussian of
    /// [`ProfileConfig::sigma`](super::ProfileConfig::sigma), which the level methods read.
    pub smoothed: Vec<f32>,
    /// The derivative response, whose peaks are the gradient candidates.
    pub response: Vec<f32>,
    /// The edges that passed threshold, polarity and the obliquity gate, before
    /// [`MeasureConfig::select`](super::MeasureConfig::select); the one crossing of a
    /// [`Locate::MidpointCrossing`](super::Locate::MidpointCrossing).
    pub candidates: Vec<MeasureEdge>,
    /// The level crossings behind the edges, as [`Caliper::levels`] reports them.
    pub levels: Vec<LevelEdge>,
    /// The edges [`Caliper::measure`] returned; empty when it rejected.
    pub edges: Vec<MeasureEdge>,
    /// Why [`Caliper::measure`] rejected, if it did.
    pub reject: Option<RejectReason>,
}

/// Measure with `cal` and keep every intermediate: the profile, its smoothed version and
/// derivative, the candidates, the level crossings, the edges and the rejection.
///
/// The edges and the rejection are exactly what [`Caliper::measure`] returns for the same
/// caliper and image, and the call leaves the caliper's later measurements unchanged.
/// It allocates the trace and recomputes the smoothed profile and response, so it
/// belongs off the hot path: in a tool that shows why a caliper failed, not in the loop
/// that measures.
///
/// # Example
/// ```
/// use vision_metrology::measure::diagnostics::explain;
/// use vision_metrology::measure::{Caliper, MeasureConfig, MeasureRect, RejectReason};
/// use vision_metrology::{Image, Point2f};
///
/// let flat = Image::from_vec(64, 64, vec![128u8; 64 * 64]).unwrap();
/// let rect = MeasureRect {
///     center: Point2f::new(32.0, 32.0),
///     angle: 0.0,
///     half_len: 20.0,
///     half_width: 4.0,
/// };
/// let mut cal = Caliper::rect(rect, MeasureConfig::default());
/// let trace = explain(&mut cal, &flat.as_view());
/// assert_eq!(trace.reject, Some(RejectReason::NoEdge));
/// assert_eq!(trace.samples, 41);
/// // Nothing anywhere near the threshold: the response is flat.
/// assert!(trace.response.iter().all(|r| r.abs() < trace.threshold));
/// ```
pub fn explain<P: Pixel>(cal: &mut Caliper, img: &ImageView<'_, P>) -> CaliperTrace {
    let (edges, reject) = match cal.measure(img) {
        Ok(edges) => (edges.to_vec(), None),
        Err(reason) => (Vec::new(), Some(reason)),
    };
    let (smoothed, response) = cal.smoothed_and_response();
    CaliperTrace {
        spacing: cal.spacing(),
        samples: cal.profile().len(),
        across: cal.across(),
        threshold: cal.config().threshold,
        profile: cal.profile().to_vec(),
        smoothed,
        response,
        candidates: cal.candidates().collect(),
        levels: cal.levels().to_vec(),
        edges,
        reject,
    }
}

/// One object of a [`MetrologyModel`], measured and explained: its fit and, for each of
/// its calipers, where the caliper sat and what it computed.
///
/// Built by [`explain_model`].
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectTrace {
    /// What [`MetrologyModel::apply`] returns for this object: the fit and its hits, or
    /// why it could not be measured.
    pub result: Result<MetrologyResult, Error>,
    /// Where each caliper sat, in caliper order: what [`layout_object`] returns. Empty
    /// when the object cannot be placed (fewer than 2 calipers, a zero-length line).
    pub placements: Vec<CaliperShape>,
    /// Each caliper's trace, parallel to `placements`. A caliper hit when its trace has
    /// edges; its first edge is the one the fit used.
    pub calipers: Vec<CaliperTrace>,
}

/// Measure every object of `model` at `fixture` and keep each caliper's trace: one entry
/// per object, in [`MetrologyModel::objects`] order.
///
/// It is [`MetrologyModel::apply`] and [`explain`] in one pass. Each caliper is placed by
/// the same code and measured once, through [`explain`]; its first edge goes to the fit
/// exactly as in `apply`, so `result` is what `apply` returns for the same model, image
/// and fixture. Like `explain`, it allocates every trace, so it belongs in a tool that
/// shows why a model measured what it did, not in the inspection loop.
///
/// # Example
/// ```
/// use vision_metrology::measure::diagnostics::explain_model;
/// use vision_metrology::measure::{MetrologyModel, MetrologyObject, MetrologyShape};
/// use vision_metrology::{Image, Point2f, Similarity2f};
///
/// // A bright disc of radius 30, anti-aliased, on a dark 128 × 128 image.
/// let data: Vec<u8> = (0..128 * 128)
///     .map(|i| {
///         let (x, y) = ((i % 128) as f32 - 64.0, (i / 128) as f32 - 64.0);
///         let cover = (30.5 - (x * x + y * y).sqrt()).clamp(0.0, 1.0);
///         (20.0 + 180.0 * cover).round() as u8
///     })
///     .collect();
/// let img = Image::from_vec(128, 128, data).unwrap();
///
/// let mut model = MetrologyModel::new();
/// model.add(MetrologyObject::new(MetrologyShape::Circle {
///     center: Point2f::new(64.0, 64.0),
///     radius: 30.0,
///     arc: None,
/// }));
/// let traces = explain_model(&model, &img.as_view(), &Similarity2f::identity());
/// let object = &traces[0];
/// assert_eq!(object.calipers.len(), 32);
/// let hits = object.calipers.iter().filter(|t| t.reject.is_none()).count();
/// assert_eq!(hits, object.result.as_ref().unwrap().hits.len());
/// ```
pub fn explain_model<P: Pixel>(
    model: &MetrologyModel,
    img: &ImageView<'_, P>,
    fixture: &Similarity2f,
) -> Vec<ObjectTrace> {
    let mut cal: Option<Caliper> = None;
    let mut points = Vec::new();
    model
        .objects()
        .iter()
        .map(|obj| {
            let placements = match caliper_placements(obj, fixture) {
                Ok(placements) => placements,
                Err(e) => {
                    return ObjectTrace {
                        result: Err(e),
                        placements: Vec::new(),
                        calipers: Vec::new(),
                    };
                }
            };
            let cal = cal.get_or_insert_with(|| Caliper::rect(placeholder_rect(), obj.measure));
            let mut calipers = Vec::with_capacity(placements.len());
            let result = measure_placed(cal, obj, &placements, &mut points, |cal| {
                let trace = explain(cal, img);
                let hit = trace.edges.first().copied();
                calipers.push(trace);
                hit
            });
            ObjectTrace {
                result,
                placements,
                calipers,
            }
        })
        .collect()
}

/// One caliper's placement, addressed by which object and which caliper within
/// it produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaliperPlacement {
    /// Index into [`MetrologyModel::objects`] — matches [`MetrologyModel::apply`]'s
    /// result order.
    pub object_index: usize,
    /// Index of this caliper within its object, `0..object.n_calipers`.
    pub caliper_index: usize,
    /// The placed geometry — a rectangle (line objects) or a radial caliper
    /// (circle objects).
    pub shape: CaliperShape,
}

/// Every caliper placement for `model`'s objects, mapped through `fixture`.
///
/// `fixture` is normally `ShapeMatch::pose` — the same value
/// [`MetrologyModel::apply`] takes. An object whose placement cannot be
/// computed (fewer than 2 calipers, or a degenerate zero-length line)
/// contributes no entries rather than failing the whole call — same reasoning
/// as [`MetrologyModel::apply`] reporting per-object results, just without an
/// error channel here since layout has nothing to attach one to.
///
/// # Example
/// ```
/// use vision_metrology::measure::diagnostics::{layout, CaliperShape};
/// use vision_metrology::measure::{MetrologyModel, MetrologyObject, MetrologyShape};
/// use vision_metrology::{Point2f, Similarity2f};
///
/// let mut model = MetrologyModel::new();
/// model.add(MetrologyObject::new(MetrologyShape::Circle {
///     center: Point2f::new(0.0, 0.0),
///     radius: 20.0,
///     arc: None,
/// }));
///
/// let placements = layout(&model, &Similarity2f::identity());
/// assert_eq!(placements.len(), 32); // MetrologyObject::new's default n_calipers
/// assert!(matches!(placements[0].shape, CaliperShape::Radial(_)));
/// ```
pub fn layout(model: &MetrologyModel, fixture: &Similarity2f) -> Vec<CaliperPlacement> {
    let mut out = Vec::new();
    for (object_index, obj) in model.objects().iter().enumerate() {
        if let Ok(placements) = caliper_placements(obj, fixture) {
            out.extend(
                placements
                    .into_iter()
                    .enumerate()
                    .map(|(caliper_index, shape)| CaliperPlacement {
                        object_index,
                        caliper_index,
                        shape,
                    }),
            );
        }
    }
    out
}

/// [`layout`] for a single object. Preview one object's calipers before adding it to
/// a model.
pub fn layout_object(obj: &MetrologyObject, fixture: &Similarity2f) -> Vec<CaliperShape> {
    caliper_placements(obj, fixture).unwrap_or_default()
}

#[cfg(test)]
mod explain_tests {
    use std::num::NonZeroUsize;

    use super::{CaliperTrace, explain};
    use crate::measure::{
        Caliper, Derivative, Locate, MeasureArc, MeasureConfig, MeasureEdge, MeasureRadial,
        MeasureRect, MeasureStrip, OffImage, ProfileConfig, RejectReason,
    };
    use vm_primitives::{EdgePolarity, Image, ImageView, Point2f, SubpixRefine};

    /// A bright bar on columns 30..60 and an anti-aliased disc of radius 25 centred at
    /// (140, 48), on a dark 192 × 96 image.
    fn scene() -> Image<u8> {
        let (w, h) = (192usize, 96usize);
        let data = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let disc = (25.5 - (x - 140.0).hypot(y - 48.0)).clamp(0.0, 1.0);
                let bar = if (30.0..60.0).contains(&x) { 1.0 } else { 0.0 };
                (20.0 + 180.0 * f32::max(disc, bar)).round() as u8
            })
            .collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    /// Every placement, each over the scene's features, and two that reject.
    fn calipers(cfg: MeasureConfig) -> Vec<Caliper> {
        let strip = |start: (f32, f32), end: (f32, f32), samples, across| MeasureStrip {
            start: Point2f::new(start.0, start.1),
            end: Point2f::new(end.0, end.1),
            half_width: 3.0,
            samples: NonZeroUsize::new(samples),
            across: NonZeroUsize::new(across),
        };
        let rect = |cx, half_len| MeasureRect {
            center: Point2f::new(cx, 48.0),
            angle: 0.1,
            half_len,
            half_width: 6.0,
        };
        vec![
            Caliper::rect(rect(45.0, 30.0), cfg),
            Caliper::rect(rect(32.0, 12.0), cfg),
            Caliper::strip(strip((8.0, 40.0), (88.0, 44.0), 161, 7), cfg),
            Caliper::strip(strip((-4.0, 40.0), (50.0, 40.0), 55, 1), cfg),
            Caliper::radial(
                MeasureRadial {
                    center: Point2f::new(140.0, 48.0),
                    radius: 25.0,
                    angle: 0.7,
                    half_len: 12.0,
                    half_width: 4.0,
                },
                cfg,
            ),
            Caliper::arc(
                MeasureArc {
                    center: Point2f::new(45.0, 48.0),
                    radius: 30.0,
                    angle_start: -0.6,
                    angle_extent: 1.2,
                    half_width: 3.0,
                },
                cfg,
            ),
            Caliper::rect(rect(100.0, 8.0), cfg),
        ]
    }

    fn configs() -> Vec<MeasureConfig> {
        let near = |locate| MeasureConfig {
            locate,
            ..MeasureConfig::default()
        };
        let central = ProfileConfig {
            derivative: Derivative::SmoothThenCentral { radius_px: 3.0 },
            ..ProfileConfig::default()
        };
        let nz = |n| NonZeroUsize::new(n).expect("nonzero");
        let mut out = vec![
            near(Locate::default()),
            near(Locate::GradientPeak {
                refine: SubpixRefine::Gaussian3,
            }),
            near(Locate::GradientPeak {
                refine: SubpixRefine::Centroid { radius: 2 },
            }),
            near(Locate::MidpointCrossing {
                endpoint_samples: nz(3),
                min_contrast: 5.0,
            }),
            near(Locate::HalfContrast {
                flank_near_px: 3.0,
                flank_far_px: 8.0,
                tol_px: 0.01,
                max_iter: nz(5),
                min_contrast: 0.0,
            }),
        ];
        let more: Vec<MeasureConfig> = out
            .iter()
            .map(|c| MeasureConfig {
                profile: central,
                max_obliquity_deg: 40.0,
                ..*c
            })
            .chain(out.iter().map(|c| MeasureConfig {
                profile: ProfileConfig {
                    off_image: OffImage::Reject,
                    step: 0.5,
                    ..ProfileConfig::default()
                },
                ..*c
            }))
            .collect();
        out.extend(more);
        out
    }

    type Bits = (u32, u32, u32, u32, EdgePolarity);

    fn bits(edges: &[MeasureEdge]) -> Vec<Bits> {
        edges
            .iter()
            .map(|e| {
                let f = f32::to_bits;
                (f(e.p.x), f(e.p.y), f(e.t), f(e.amplitude), e.polarity)
            })
            .collect()
    }

    fn measured(cal: &mut Caliper, img: &ImageView<'_, u8>) -> Result<Vec<Bits>, RejectReason> {
        cal.measure(img).map(bits)
    }

    /// The trace's edges and rejection are `measure`'s, to the bit, for every placement
    /// and every way of locating an edge; and explaining changes no later measurement.
    #[test]
    fn the_trace_is_what_measure_returns() {
        let img = scene();
        let view = img.as_view();
        let (mut found, mut rejected) = (0, 0);
        for cfg in configs() {
            let fresh = calipers(cfg);
            let traced = calipers(cfg);
            for (mut a, mut b) in fresh.into_iter().zip(traced) {
                let want = measured(&mut a, &view);
                let trace = explain(&mut b, &view);
                let got = match trace.reject {
                    None => Ok(bits(&trace.edges)),
                    Some(reason) => Err(reason),
                };
                assert_eq!(got, want, "{cfg:?}");
                assert!(trace.reject.is_some() || !trace.edges.is_empty());
                assert_eq!(measured(&mut b, &view), want, "after explain: {cfg:?}");
                assert_eq!(explain(&mut b, &view), trace, "explaining twice: {cfg:?}");
                match want {
                    Ok(_) => found += 1,
                    Err(_) => rejected += 1,
                }
            }
        }
        assert!(
            found > 30 && rejected > 10,
            "found {found}, rejected {rejected}"
        );
    }

    fn rect_on_the_bar(locate: Locate) -> Caliper {
        Caliper::rect(
            MeasureRect {
                center: Point2f::new(45.0, 48.0),
                angle: 0.0,
                half_len: 30.0,
                half_width: 6.0,
            },
            MeasureConfig {
                locate,
                ..MeasureConfig::default()
            },
        )
    }

    /// Scanning x = 15..75 across the bar (columns 30..60): the response peaks at both
    /// sides, the candidates are those peaks, and nothing is level-located.
    #[test]
    fn the_trace_holds_the_profile_response_and_candidates() {
        let img = scene();
        let mut cal = rect_on_the_bar(Locate::default());
        let t: CaliperTrace = explain(&mut cal, &img.as_view());
        assert_eq!((t.samples, t.across, t.spacing), (61, 13, 1.0));
        assert_eq!(t.threshold, 5.0);
        assert_eq!(t.profile.len(), 61);
        assert_eq!(t.smoothed.len(), 61);
        assert_eq!(t.response.len(), 61);
        assert_eq!(t.profile[0], 20.0);
        assert_eq!(t.profile[30], 200.0);
        // Sample i is at x = 15 + i: the edges at 29.5 and 59.5 are samples 14.5 and 44.5.
        let (imax, _) = t
            .response
            .iter()
            .enumerate()
            .fold((0, f32::MIN), |b, (i, &r)| if r > b.1 { (i, r) } else { b });
        assert!(imax == 14 || imax == 15, "rising peak at {imax}");
        assert_eq!(t.candidates.len(), 2);
        assert_eq!(t.candidates, t.edges, "`All` keeps every candidate");
        assert!((t.edges[0].p.x - 29.5).abs() < 0.05 && (t.edges[1].p.x - 59.5).abs() < 0.05);
        assert!(t.levels.is_empty());
        assert_eq!(t.reject, None);

        let mut half = rect_on_the_bar(Locate::HalfContrast {
            flank_near_px: 3.0,
            flank_far_px: 8.0,
            tol_px: 0.01,
            max_iter: NonZeroUsize::new(5).expect("nonzero"),
            min_contrast: 0.0,
        });
        let t = explain(&mut half, &img.as_view());
        assert_eq!(t.levels.len(), 2);
        assert_eq!(t.levels[0].x * t.spacing + 15.0, t.edges[0].p.x);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_trace_serializes() {
        let img = scene();
        let mut cal = rect_on_the_bar(Locate::default());
        let t = explain(&mut cal, &img.as_view());
        let v = serde_json::to_value(&t).expect("serializable");
        assert_eq!(v["samples"], 61);
        assert_eq!(v["edges"][0]["polarity"], "Rising");
        assert_eq!(v["reject"], serde_json::Value::Null);
        assert_eq!(v["profile"].as_array().map(Vec::len), Some(61));

        let flat = Image::from_vec(96, 96, vec![128u8; 96 * 96]).expect("valid");
        let t = explain(&mut cal, &flat.as_view());
        assert_eq!(
            serde_json::to_value(&t).expect("serializable")["reject"],
            "NoEdge"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{CaliperShape, layout};
    use crate::measure::{MetrologyModel, MetrologyObject, MetrologyShape};
    use vm_primitives::{Point2f, Similarity2f, Vec2f};

    #[test]
    fn line_object_places_rect_calipers_along_the_segment() {
        let mut model = MetrologyModel::new();
        let mut obj = MetrologyObject::new(MetrologyShape::Line {
            a: Point2f::new(0.0, 0.0),
            b: Point2f::new(10.0, 0.0),
        });
        obj.n_calipers = 3;
        model.add(obj);

        let placements = layout(&model, &Similarity2f::identity());
        assert_eq!(placements.len(), 3);
        for (i, p) in placements.iter().enumerate() {
            assert_eq!(p.object_index, 0);
            assert_eq!(p.caliper_index, i);
            let CaliperShape::Rect(r) = p.shape else {
                panic!("expected a rect placement")
            };
            // Hand-computed: calipers at f = 0, 0.5, 1 along (0,0)-(10,0).
            let expected_x = 5.0 * i as f32;
            assert!((r.center.x - expected_x).abs() < 1e-4, "x = {}", r.center.x);
            assert!(r.center.y.abs() < 1e-4);
            // Scan axis is perpendicular to the segment: angle = +-pi/2.
            assert!((r.angle.abs() - core::f32::consts::FRAC_PI_2).abs() < 1e-4);
        }
    }

    #[test]
    fn circle_object_places_radial_calipers_around_it() {
        let mut model = MetrologyModel::new();
        let mut obj = MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(50.0, 50.0),
            radius: 20.0,
            arc: None,
        });
        obj.n_calipers = 4;
        model.add(obj);

        let placements = layout(&model, &Similarity2f::identity());
        assert_eq!(placements.len(), 4);
        for (i, p) in placements.iter().enumerate() {
            let CaliperShape::Radial(r) = p.shape else {
                panic!("expected a radial placement")
            };
            // `MeasureRadial::center` is the circle's own centre for every
            // caliper — the geometry the current overlay convention draws.
            assert!((r.center - Point2f::new(50.0, 50.0)).norm() < 1e-4);
            assert!((r.radius - 20.0).abs() < 1e-4);
            let expected_angle = core::f32::consts::TAU * i as f32 / 4.0;
            assert!((r.angle - expected_angle).abs() < 1e-4);
        }
    }

    #[test]
    fn a_fixture_translates_scales_and_rotates_the_layout() {
        let mut model = MetrologyModel::new();
        let mut obj = MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(0.0, 0.0),
            radius: 10.0,
            arc: None,
        });
        obj.n_calipers = 4;
        model.add(obj);

        let fixture = Similarity2f::new(Vec2f::new(100.0, 200.0), 0.0, 2.0);
        let placements = layout(&model, &fixture);
        let CaliperShape::Radial(r) = placements[0].shape else {
            panic!("expected a radial placement")
        };
        assert!((r.center - Point2f::new(100.0, 200.0)).norm() < 1e-4);
        assert!((r.radius - 20.0).abs() < 1e-4, "r = {}", r.radius);
    }

    #[test]
    fn an_unmeasurable_object_contributes_no_placements() {
        let mut model = MetrologyModel::new();
        let mut obj = MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(0.0, 0.0),
            radius: 10.0,
            arc: None,
        });
        obj.n_calipers = 1; // invalid: needs >= 2
        model.add(obj);

        let placements = layout(&model, &Similarity2f::identity());
        assert!(placements.is_empty());
    }
}

#[cfg(test)]
mod explain_model_tests {
    use super::{CaliperShape, explain, explain_model, layout_object};
    use crate::measure::{Caliper, MetrologyModel, MetrologyObject, MetrologyShape};
    use vm_primitives::{Image, Point2f, Similarity2f, Vec2f};

    /// An anti-aliased bright disc of radius 25 centred at (60, 60) and a bright bar on
    /// columns 130..160, on a dark 192 × 128 image.
    fn scene() -> Image<u8> {
        let (w, h) = (192usize, 128usize);
        let data = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let disc = (25.5 - (x - 60.0).hypot(y - 60.0)).clamp(0.0, 1.0);
                let bar = if (130.0..160.0).contains(&x) {
                    1.0
                } else {
                    0.0
                };
                (20.0 + 180.0 * f32::max(disc, bar)).round() as u8
            })
            .collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    /// Four objects, in model space at the fixture below: the disc's rim; the bar's left
    /// edge; two calipers on flat ground, which hit nothing and so cannot be fitted; and
    /// one caliper, which cannot be placed.
    fn model() -> MetrologyModel {
        let mut model = MetrologyModel::new();
        let mut rim = MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(50.0, 50.0),
            radius: 25.0,
            arc: None,
        });
        rim.n_calipers = 24;
        model.add(rim);
        let mut edge = MetrologyObject::new(MetrologyShape::Line {
            a: Point2f::new(119.5, 20.0),
            b: Point2f::new(119.5, 90.0),
        });
        edge.n_calipers = 8;
        model.add(edge);
        let mut flat = MetrologyObject::new(MetrologyShape::Line {
            a: Point2f::new(170.0, 10.0),
            b: Point2f::new(170.0, 100.0),
        });
        flat.n_calipers = 2;
        flat.caliper_len = 4.0;
        model.add(flat);
        let mut single = MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(50.0, 50.0),
            radius: 25.0,
            arc: None,
        });
        single.n_calipers = 1;
        model.add(single);
        model
    }

    fn fixture() -> Similarity2f {
        Similarity2f::new(Vec2f::new(10.0, 10.0), 0.0, 1.0)
    }

    /// The traced model fits exactly what `apply` does, object by object, failures
    /// included.
    #[test]
    fn explain_model_returns_what_apply_returns() {
        let img = scene();
        let traces = explain_model(&model(), &img.as_view(), &fixture());
        let applied = model().apply(&img.as_view(), &fixture());
        assert_eq!(traces.len(), 4);
        for (i, (trace, applied)) in traces.iter().zip(&applied).enumerate() {
            assert_eq!(&trace.result, applied, "object {i}");
        }
        assert!(traces[0].result.is_ok() && traces[1].result.is_ok());
        assert!(traces[2].result.is_err(), "nothing to fit on flat ground");
        assert!(traces[3].result.is_err(), "one caliper cannot be placed");
    }

    /// Each caliper's trace is `explain` of a caliper at the layout's placement, and the
    /// hits are the traces' first edges, in caliper order.
    #[test]
    fn every_caliper_is_traced_where_layout_puts_it() {
        let img = scene();
        let model = model();
        let traces = explain_model(&model, &img.as_view(), &fixture());
        for (obj, trace) in model.objects().iter().zip(&traces) {
            assert_eq!(trace.placements, layout_object(obj, &fixture()));
            assert_eq!(trace.calipers.len(), trace.placements.len());
            for (shape, caliper) in trace.placements.iter().zip(&trace.calipers) {
                let mut alone = match *shape {
                    CaliperShape::Rect(r) => Caliper::rect(r, obj.measure),
                    CaliperShape::Radial(r) => Caliper::radial(r, obj.measure),
                };
                assert_eq!(caliper, &explain(&mut alone, &img.as_view()));
            }
            if let Ok(result) = &trace.result {
                let firsts: Vec<_> = trace
                    .calipers
                    .iter()
                    .filter_map(|t| t.edges.first().copied())
                    .collect();
                assert_eq!(result.hits, firsts);
            }
        }
        // The rim and the bar's edge: every caliper hit.
        assert_eq!(traces[0].result.as_ref().map(|r| r.hits.len()), Ok(24));
        assert_eq!(traces[1].result.as_ref().map(|r| r.hits.len()), Ok(8));
        // Flat ground: traced, every caliper rejected, nothing to fit.
        assert_eq!(traces[2].calipers.len(), 2);
        assert!(traces[2].calipers.iter().all(|t| t.reject.is_some()));
        // Unplaceable: no calipers at all.
        assert!(traces[3].placements.is_empty() && traces[3].calipers.is_empty());
    }
}
