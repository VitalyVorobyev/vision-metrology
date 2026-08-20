//! Black-box tests for the shape-based matcher.
//!
//! Every fixture is rendered from a signed distance function through a
//! `smoothstep` band, so the synthetic edges have the same gradual profile a
//! real lens produces — and, more importantly, the ground-truth pose is exact
//! to arbitrary subpixel precision.

use vision_metrology::matching::{
    ContourOrientation, Polarity, Refinement, ShapeMatch, ShapeMatcher, ShapeModel,
    ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig, ShapeSearchTuning,
};
use vision_metrology::{Image, Point2f, Polyline2f, Rect2f, Vec2f, wrap_angle};

const W: usize = 512;
const H: usize = 512;

// ── fixtures ────────────────────────────────────────────────────────────────

/// Signed distance to an axis-aligned box centred on the origin. Negative
/// inside.
fn sdf_box(p: (f32, f32), half: (f32, f32)) -> f32 {
    let dx = p.0.abs() - half.0;
    let dy = p.1.abs() - half.1;
    let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
    outside + dx.max(dy).min(0.0)
}

/// F1: an L-bracket — asymmetric, with straight edges and corners, so both
/// translation and rotation are well constrained.
fn sdf_bracket(p: (f32, f32)) -> f32 {
    let a = sdf_box((p.0, p.1 + 45.0), (60.0, 15.0));
    let b = sdf_box((p.0 + 45.0, p.1), (15.0, 60.0));
    a.min(b)
}

/// F2: a ring — rotationally symmetric, so its rotation is unobservable.
fn sdf_ring(p: (f32, f32)) -> f32 {
    let r = (p.0 * p.0 + p.1 * p.1).sqrt();
    (r - 45.0).abs() - 8.0
}

/// F6: a comb — a bar with fine teeth (4 px pitch). The teeth are exactly the
/// high-frequency structure a 2×2 box-mean pyramid aliases away by level 2–3,
/// which is risk R3: the true match can die at the coarse level before the
/// search ever descends.
fn sdf_comb(p: (f32, f32)) -> f32 {
    // Spine: 120×12 bar.
    let spine = sdf_box((p.0, p.1 + 20.0), (60.0, 6.0));
    // Teeth: 2 px half-width, 24 px long, every 8 px along the spine.
    let k = (p.0 / 8.0).round().clamp(-7.0, 7.0);
    let tooth = sdf_box((p.0 - 8.0 * k, p.1 - 2.0), (2.0, 16.0));
    spine.min(tooth)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

struct Render {
    dark: f32,
    bright: f32,
}

impl Default for Render {
    fn default() -> Self {
        Self {
            dark: 40.0,
            bright: 210.0,
        }
    }
}

/// Rasterise `sdf` placed at `(cx, cy)` with rotation `angle` and scale `s`.
fn render(
    sdf: &dyn Fn((f32, f32)) -> f32,
    cx: f32,
    cy: f32,
    angle: f32,
    s: f32,
    r: &Render,
) -> Image<u8> {
    let (sn, cs) = angle.sin_cos();
    let mut data = vec![0u8; W * H];
    for y in 0..H {
        for x in 0..W {
            let dx = (x as f32 - cx) / s;
            let dy = (y as f32 - cy) / s;
            // Inverse rotation into the model frame.
            let mx = cs * dx + sn * dy;
            let my = -sn * dx + cs * dy;
            let t = smoothstep(-1.0, 1.0, -sdf((mx, my)));
            data[y * W + x] = (r.dark + (r.bright - r.dark) * t).round().clamp(0.0, 255.0) as u8;
        }
    }
    Image::from_vec(W, H, data).expect("valid image")
}

fn bracket_at(cx: f32, cy: f32, angle: f32, s: f32) -> Image<u8> {
    render(&sdf_bracket, cx, cy, angle, s, &Render::default())
}

/// The ROI that encloses the bracket rendered at the image centre.
fn bracket_roi() -> Rect2f {
    Rect2f {
        x: 170.0,
        y: 170.0,
        width: 172.0,
        height: 172.0,
    }
}

fn build_bracket_model(cfg: &ShapeModelConfig) -> ShapeModel {
    let reference = bracket_at(256.0, 256.0, 0.0, 1.0);
    ShapeModelBuilder::new()
        .build(&reference.as_view(), bracket_roi(), cfg)
        .expect("bracket model builds")
}

/// Where the model origin lands when the object is rendered at `(cx, cy, angle, s)`.
///
/// The renderer places the shape's own coordinate origin at `(cx, cy)`, and the
/// model's reference point is the point-cloud centroid, at some offset `o` from
/// that origin. Under the pose the centroid moves to `(cx, cy) + s·R·o`.
fn expected_position(model: &ShapeModel, cx: f32, cy: f32, angle: f32, s: f32) -> Point2f {
    let o = Vec2f::new(model.origin().x - 256.0, model.origin().y - 256.0);
    let (sn, cs) = angle.sin_cos();
    Point2f::new(
        cx + s * (cs * o.x - sn * o.y),
        cy + s * (sn * o.x + cs * o.y),
    )
}

fn find_one(scene: &Image<u8>, model: &ShapeModel, cfg: &ShapeSearchConfig) -> Option<ShapeMatch> {
    ShapeMatcher::new()
        .find(&scene.as_view(), model, cfg)
        .into_iter()
        .next()
}

// ── tests ───────────────────────────────────────────────────────────────────

#[test]
fn t4_the_reference_image_scores_almost_one_at_its_own_pose() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let scene = bracket_at(256.0, 256.0, 0.0, 1.0);
    let cfg = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };
    let m = find_one(&scene, &model, &cfg).expect("the model must find itself");

    assert!(m.score >= 0.95, "score {}", m.score);
    let want = expected_position(&model, 256.0, 256.0, 0.0, 1.0);
    assert!((m.position.x - want.x).abs() < 0.15, "x {:?}", m.position);
    assert!((m.position.y - want.y).abs() < 0.15, "y {:?}", m.position);
    assert!(m.angle().abs() < 0.01, "angle {}", m.angle());
}

#[test]
fn t1_translation_is_recovered_to_subpixel() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let cfg = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };

    for &(cx, cy) in &[(200.0f32, 300.0f32), (300.0, 210.0), (256.5, 256.25)] {
        let scene = bracket_at(cx, cy, 0.0, 1.0);
        let m = find_one(&scene, &model, &cfg).unwrap_or_else(|| panic!("not found at {cx},{cy}"));
        let want = expected_position(&model, cx, cy, 0.0, 1.0);
        assert!(
            (m.position.x - want.x).abs() < 0.2 && (m.position.y - want.y).abs() < 0.2,
            "at ({cx},{cy}): got {:?}, want {want:?}",
            m.position
        );
    }
}

#[test]
fn t2_rotation_is_recovered_over_the_full_circle() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let cfg = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };

    for deg in (0..360).step_by(30) {
        let angle = (deg as f32).to_radians();
        let truth = wrap_angle(angle);
        let scene = bracket_at(256.0, 256.0, angle, 1.0);
        let m = find_one(&scene, &model, &cfg).unwrap_or_else(|| panic!("not found at {deg} deg"));

        let err = wrap_angle(m.angle() - truth).abs().to_degrees();
        assert!(err < 1.0, "{deg} deg: angle error {err:.2} deg");
        let want = expected_position(&model, 256.0, 256.0, angle, 1.0);
        assert!(
            (m.position.x - want.x).abs() < 0.5 && (m.position.y - want.y).abs() < 0.5,
            "{deg} deg: got {:?}, want {want:?}",
            m.position
        );
    }
}

#[test]
fn t21_an_angle_range_may_straddle_pi() {
    // 178 degrees is inside (170, 190) only if the range is used unwrapped.
    let model = build_bracket_model(&ShapeModelConfig {
        angle_range: (170f32.to_radians(), 190f32.to_radians()),
        ..Default::default()
    });
    let truth = 178f32.to_radians();
    let scene = bracket_at(256.0, 256.0, truth, 1.0);
    let m = find_one(&scene, &model, &ShapeSearchConfig::default()).expect("found across pi");

    let err = wrap_angle(m.angle() - wrap_angle(truth)).abs().to_degrees();
    assert!(err < 1.0, "angle error {err:.2} deg, got {}", m.angle());
}

#[test]
fn t19_a_uniform_scene_yields_nothing() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let flat = Image::from_vec(W, H, vec![128u8; W * H]).unwrap();
    let out = ShapeMatcher::new().find(&flat.as_view(), &model, &ShapeSearchConfig::default());
    assert!(out.is_empty(), "{out:?}");
}

#[test]
fn t22_output_is_reproducible() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let scene = bracket_at(230.0, 280.0, 0.6, 1.0);
    let cfg = ShapeSearchConfig {
        max_matches: None,
        ..Default::default()
    };
    let a = ShapeMatcher::new().find(&scene.as_view(), &model, &cfg);
    let b = ShapeMatcher::new().find(&scene.as_view(), &model, &cfg);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x, y);
    }
}

#[test]
fn t18_degenerate_inputs_report_the_right_error() {
    use vision_metrology::Error;
    let img = bracket_at(256.0, 256.0, 0.0, 1.0);
    let mut b = ShapeModelBuilder::new();
    let cfg = ShapeModelConfig::default();

    let empty = Rect2f {
        x: 10.0,
        y: 10.0,
        width: 0.0,
        height: 20.0,
    };
    assert_eq!(
        b.build(&img.as_view(), empty, &cfg),
        Err(Error::InvalidConfig("roi must have positive extent"))
    );

    let outside = Rect2f {
        x: 400.0,
        y: 400.0,
        width: 200.0,
        height: 200.0,
    };
    assert_eq!(
        b.build(&img.as_view(), outside, &cfg),
        Err(Error::OutOfBounds)
    );

    let bad_scale = ShapeModelConfig {
        scale_range: (0.0, 2.0),
        ..Default::default()
    };
    assert_eq!(
        b.build(&img.as_view(), bracket_roi(), &bad_scale),
        Err(Error::InvalidConfig(
            "scale_range must be positive and ordered"
        ))
    );

    let pts = vec![Point2f::default(); 5];
    let dirs = vec![Vec2f::new(1.0, 0.0); 3];
    assert_eq!(
        ShapeModel::from_directed_points(&pts, &dirs, &cfg),
        Err(Error::SizeMismatch {
            expected: 5,
            actual: 3
        })
    );

    // A blank image has no edges at all.
    let flat = Image::from_vec(W, H, vec![90u8; W * H]).unwrap();
    assert!(matches!(
        b.build(&flat.as_view(), bracket_roi(), &cfg),
        Err(Error::InsufficientData { .. })
    ));
}

#[test]
fn t17_auto_level_count_shrinks_with_the_model() {
    // Radius drives the level count: below ~6 px a level cannot be searched.
    let big = build_bracket_model(&ShapeModelConfig::default());
    // The bracket's model radius is ~90 px, so level 4 would be 5.6 px — below
    // the 6 px floor at which one angle step sweeps the whole model past
    // itself. Four levels is the honest answer, not five.
    assert_eq!(big.num_levels(), 4, "levels: {:?}", level_radii(&big));
    assert!(big.levels()[3].radius() >= 6.0);

    let small = render(
        &|p| sdf_box(p, (12.0, 12.0)),
        256.0,
        256.0,
        0.0,
        1.0,
        &Render::default(),
    );
    let roi = Rect2f {
        x: 236.0,
        y: 236.0,
        width: 40.0,
        height: 40.0,
    };
    let m = ShapeModelBuilder::new()
        .build(&small.as_view(), roi, &ShapeModelConfig::default())
        .expect("small model builds");
    assert!(
        m.num_levels() >= 2 && m.num_levels() <= 3,
        "a 24 px box gave {} levels",
        m.num_levels()
    );
    assert!(m.num_levels() < big.num_levels());
}

fn level_radii(m: &ShapeModel) -> Vec<f32> {
    m.levels().iter().map(|l| l.radius()).collect()
}

#[test]
fn t7_t8_polarity_decides_whether_inverted_contrast_matches() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let inverted = render(
        &sdf_bracket,
        256.0,
        256.0,
        0.0,
        1.0,
        &Render {
            dark: 210.0,
            bright: 40.0,
        },
    );

    let strict = ShapeSearchConfig::default();
    assert!(
        find_one(&inverted, &model, &strict).is_none(),
        "Polarity::Match must reject inverted contrast"
    );

    let lenient_model = build_bracket_model(&ShapeModelConfig {
        polarity: Polarity::IgnoreGlobal,
        ..Default::default()
    });
    let m = find_one(&inverted, &lenient_model, &strict).expect("IgnoreGlobal accepts inversion");
    assert!(m.score >= 0.9, "score {}", m.score);
    let want = expected_position(&lenient_model, 256.0, 256.0, 0.0, 1.0);
    assert!((m.position.x - want.x).abs() < 0.5 && (m.position.y - want.y).abs() < 0.5);
}

#[test]
fn t15_a_symmetric_model_does_not_blow_up() {
    // A ring has no observable rotation. The refinement must degrade to fewer
    // degrees of freedom rather than divide by a singular matrix.
    let reference = render(&sdf_ring, 256.0, 256.0, 0.0, 1.0, &Render::default());
    let roi = Rect2f {
        x: 190.0,
        y: 190.0,
        width: 132.0,
        height: 132.0,
    };
    let model = ShapeModelBuilder::new()
        .build(&reference.as_view(), roi, &ShapeModelConfig::default())
        .expect("ring model builds");

    let scene = render(&sdf_ring, 240.0, 270.0, 0.0, 1.0, &Render::default());
    let cfg = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };
    let m = find_one(&scene, &model, &cfg).expect("ring found");

    assert!(m.angle().is_finite(), "angle must stay finite");
    assert!(m.scale().is_finite() && (m.scale() - 1.0).abs() < 0.05);
    let want = expected_position(&model, 240.0, 270.0, 0.0, 1.0);
    assert!(
        (m.position.x - want.x).abs() < 0.5 && (m.position.y - want.y).abs() < 0.5,
        "got {:?} want {want:?}",
        m.position
    );
}

#[test]
fn t16_a_contour_model_agrees_with_an_image_model() {
    // Same geometry, two very different construction paths.
    let image_model = build_bracket_model(&ShapeModelConfig::default());

    // Sample the bracket outline analytically, walking the SDF's zero set.
    let mut pts = Vec::new();
    let n = 400;
    for i in 0..n {
        let t = i as f32 / n as f32 * core::f32::consts::TAU;
        // Ray-march outward from the centroid until the SDF crosses zero.
        let (sn, cs) = t.sin_cos();
        let mut r = 1.0f32;
        let mut prev = sdf_bracket((-30.0 + cs, -30.0 + sn));
        while r < 200.0 {
            let v = sdf_bracket((-30.0 + r * cs, -30.0 + r * sn));
            if prev < 0.0 && v >= 0.0 {
                break;
            }
            prev = v;
            r += 0.25;
        }
        pts.push(Point2f::new(256.0 - 30.0 + r * cs, 256.0 - 30.0 + r * sn));
    }
    let contour_model = ShapeModel::from_polylines(
        &[Polyline2f { points: pts }],
        ContourOrientation::Outward,
        &ShapeModelConfig {
            polarity: Polarity::IgnoreLocal,
            ..Default::default()
        },
    )
    .expect("contour model builds");

    let scene = bracket_at(260.0, 240.0, 0.3, 1.0);
    let cfg = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        min_score: 0.4,
        ..Default::default()
    };
    let a = find_one(&scene, &image_model, &cfg).expect("image model finds it");
    let b = find_one(&scene, &contour_model, &cfg).expect("contour model finds it");

    // Both report where their own origin landed, and the two origins differ,
    // so compare where each maps the same reference-image point.
    let probe = nalgebra::Point2::new(256.0f32, 256.0);
    let pa = a.pose * probe;
    let pb = b.pose * probe;
    assert!(
        (pa.x - pb.x).abs() < 1.5 && (pa.y - pb.y).abs() < 1.5,
        "image model maps to {pa}, contour model to {pb}"
    );
    let da = wrap_angle(a.angle() - b.angle()).abs().to_degrees();
    assert!(da < 1.5, "angle disagreement {da:.2} deg");
}

#[test]
fn t3_uniform_scale_is_recovered() {
    let cfg = ShapeModelConfig {
        scale_range: (0.7, 1.4),
        ..Default::default()
    };
    let model = build_bracket_model(&cfg);
    let search = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        min_score: 0.4,
        ..Default::default()
    };

    for &s in &[0.75f32, 0.9, 1.0, 1.2, 1.35] {
        let scene = bracket_at(256.0, 256.0, 0.0, s);
        let m = find_one(&scene, &model, &search).unwrap_or_else(|| panic!("not found at s={s}"));
        let rel = (m.scale() - s).abs() / s;
        assert!(rel < 0.02, "s={s}: recovered {} (rel {rel:.4})", m.scale());
        let want = expected_position(&model, 256.0, 256.0, 0.0, s);
        assert!(
            (m.position.x - want.x).abs() < 1.0 && (m.position.y - want.y).abs() < 1.0,
            "s={s}: got {:?}, want {want:?}",
            m.position
        );
    }
}

/// Paint a background-coloured rectangle over part of the object.
fn occlude(img: &mut Image<u8>, r: Rect2f) {
    for y in (r.y as usize)..((r.y + r.height) as usize).min(H) {
        for x in (r.x as usize)..((r.x + r.width) as usize).min(W) {
            img.data_mut()[y * W + x] = 40;
        }
    }
}

#[test]
fn t5_the_score_tracks_the_visible_fraction() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let search = ShapeSearchConfig {
        min_score: 0.25,
        tuning: ShapeSearchTuning {
            greediness: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };

    // The bracket's vertical arm sits on the left of the ROI, so a narrow strip
    // already hides a substantial share of the contour.
    let mut last = 2.0f32;
    let mut deepest = 0.0f32;
    for &cut in &[0.0f32, 20.0, 35.0, 50.0] {
        let mut scene = bracket_at(256.0, 256.0, 0.0, 1.0);
        // Occlude a vertical strip on the left of the bracket.
        if cut > 0.0 {
            occlude(
                &mut scene,
                Rect2f {
                    x: 170.0,
                    y: 170.0,
                    width: cut,
                    height: 180.0,
                },
            );
        }

        // Ground truth: the fraction of level-0 model points that the occluder
        // covers. Counting the actual points is exact, where guessing from the
        // occluded *area* would not be.
        let hidden = model
            .reference_points()
            .iter()
            .filter(|p| {
                cut > 0.0 && p.x >= 170.0 && p.x < 170.0 + cut && p.y >= 170.0 && p.y < 350.0
            })
            .count() as f32
            / model.point_count(0) as f32;

        let m = find_one(&scene, &model, &search)
            .unwrap_or_else(|| panic!("lost the object at cut={cut} (hidden {hidden:.2})"));
        assert!(
            (m.score - (1.0 - hidden)).abs() < 0.12,
            "cut={cut}: score {:.3}, 1 - hidden = {:.3}",
            m.score,
            1.0 - hidden
        );
        assert!(m.score <= last + 1e-3, "score must not rise with occlusion");
        last = m.score;
        deepest = deepest.max(hidden);
    }
    assert!(
        deepest > 0.25,
        "the sweep must actually occlude something, deepest was {deepest:.2}"
    );
    assert!(
        last < 0.8,
        "the last case must be visibly degraded, got {last}"
    );
}

#[test]
fn t6_clutter_does_not_produce_a_false_positive() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let mut scene = bracket_at(180.0, 200.0, 0.5, 1.0);

    // Bars and a disc that share edge orientations with the bracket but not its
    // shape. Drawn straight into the buffer so they have hard, high-contrast
    // edges — the least forgiving kind of clutter.
    for (x0, y0, w, h) in [(300usize, 60usize, 150usize, 20usize), (380, 300, 20, 160)] {
        for y in y0..(y0 + h) {
            for x in x0..(x0 + w) {
                scene.data_mut()[y * W + x] = 210;
            }
        }
    }
    for y in 330..430 {
        for x in 60..160 {
            let (dx, dy) = (x as f32 - 110.0, y as f32 - 380.0);
            if dx * dx + dy * dy < 45.0 * 45.0 {
                scene.data_mut()[y * W + x] = 210;
            }
        }
    }

    let cfg = ShapeSearchConfig {
        max_matches: None,
        min_score: 0.6,
        ..Default::default()
    };
    let out = ShapeMatcher::new().find(&scene.as_view(), &model, &cfg);
    assert_eq!(
        out.len(),
        1,
        "clutter produced {} matches: {out:?}",
        out.len()
    );
    let want = expected_position(&model, 180.0, 200.0, 0.5, 1.0);
    assert!(
        (out[0].position.x - want.x).abs() < 1.0 && (out[0].position.y - want.y).abs() < 1.0,
        "got {:?} want {want:?}",
        out[0].position
    );
}

#[test]
fn t9_ignore_local_accepts_a_half_inverted_object() {
    // Left half of the image keeps the model's contrast, right half is
    // inverted. `Match` sees the two halves cancel; `IgnoreLocal` does not.
    let normal = bracket_at(256.0, 256.0, 0.0, 1.0);
    let inverted = render(
        &sdf_bracket,
        256.0,
        256.0,
        0.0,
        1.0,
        &Render {
            dark: 210.0,
            bright: 40.0,
        },
    );
    let mut mixed = normal.clone();
    for y in 0..H {
        for x in 256..W {
            mixed.data_mut()[y * W + x] = inverted.data()[y * W + x];
        }
    }

    let strict = build_bracket_model(&ShapeModelConfig::default());
    let lenient = build_bracket_model(&ShapeModelConfig {
        polarity: Polarity::IgnoreLocal,
        ..Default::default()
    });
    let cfg = ShapeSearchConfig {
        min_score: 0.2,
        tuning: ShapeSearchTuning {
            greediness: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };

    let a = find_one(&mixed, &strict, &cfg).expect("Match still finds a partial signal");
    let b = find_one(&mixed, &lenient, &cfg).expect("IgnoreLocal finds it");
    assert!(
        a.score < 0.75,
        "Match scored {:.3}, expected roughly half",
        a.score
    );
    assert!(b.score > 0.9, "IgnoreLocal scored {:.3}", b.score);
}

#[test]
fn t10_t11_two_instances_are_reported_and_duplicates_are_not() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let a = bracket_at(150.0, 150.0, 0.0, 1.0);
    let b = bracket_at(360.0, 350.0, 1.2, 1.0);
    let mut scene = a.clone();
    for i in 0..W * H {
        scene.data_mut()[i] = a.data()[i].max(b.data()[i]);
    }

    let cfg = ShapeSearchConfig {
        max_matches: None,
        min_score: 0.7,
        ..Default::default()
    };
    let out = ShapeMatcher::new().find(&scene.as_view(), &model, &cfg);
    assert_eq!(out.len(), 2, "{out:?}");

    for &(cx, cy, angle) in &[(150.0f32, 150.0f32, 0.0f32), (360.0, 350.0, 1.2)] {
        let want = expected_position(&model, cx, cy, angle, 1.0);
        let hit = out.iter().any(|m| {
            (m.position.x - want.x).abs() < 1.0
                && (m.position.y - want.y).abs() < 1.0
                && wrap_angle(m.angle() - angle).abs() < 0.03
        });
        assert!(hit, "no instance at ({cx}, {cy}, {angle}): {out:?}");
    }

    // The same scene with the overlap rule tightened to zero still yields two
    // instances, because they genuinely share no model points.
    let tight = ShapeSearchConfig {
        max_overlap: 0.0,
        ..cfg.clone()
    };
    assert_eq!(
        ShapeMatcher::new()
            .find(&scene.as_view(), &model, &tight)
            .len(),
        2
    );
}

#[test]
fn t12_greediness_does_not_move_the_answer() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let scene = bracket_at(210.0, 290.0, -0.8, 1.0);
    let exhaustive = ShapeSearchConfig {
        tuning: ShapeSearchTuning {
            greediness: 0.0,
            ..Default::default()
        },
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };
    let greedy = ShapeSearchConfig {
        tuning: ShapeSearchTuning {
            greediness: 0.9,
            ..Default::default()
        },
        ..exhaustive.clone()
    };

    let a = find_one(&scene, &model, &exhaustive).expect("exhaustive finds it");
    let b = find_one(&scene, &model, &greedy).expect("greedy finds it");
    assert!((a.position.x - b.position.x).abs() < 1e-3);
    assert!((a.position.y - b.position.y).abs() < 1e-3);
    assert!((a.angle() - b.angle()).abs() < 1e-4);
    assert!((a.score - b.score).abs() < 1e-4);
}

#[test]
fn t14_subpixel_translation_is_tracked_across_the_pixel_grid() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let cfg = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };

    let mut errors = Vec::new();
    for k in 0..8 {
        let tx = 200.0 + k as f32 / 8.0;
        let scene = bracket_at(tx, 256.0, 0.0, 1.0);
        let m = find_one(&scene, &model, &cfg).unwrap_or_else(|| panic!("not found at tx={tx}"));
        let want = expected_position(&model, tx, 256.0, 0.0, 1.0);
        errors.push((m.position.x - want.x).abs());
    }
    let mean = errors.iter().sum::<f32>() / errors.len() as f32;
    let max = errors.iter().copied().fold(0.0f32, f32::max);
    assert!(mean < 0.06, "mean subpixel error {mean:.4} px: {errors:?}");
    assert!(max < 0.12, "max subpixel error {max:.4} px: {errors:?}");
}

#[test]
fn t20_the_score_survives_a_change_of_illumination() {
    // The measure compares gradient *directions*, so any monotonic intensity
    // change leaves it alone as long as the edges stay above the contrast floor.
    let model = build_bracket_model(&ShapeModelConfig::default());
    let cfg = ShapeSearchConfig::default();

    let normal = bracket_at(256.0, 256.0, 0.0, 1.0);
    let base = find_one(&normal, &model, &cfg).expect("found").score;

    let dimmed = render(
        &sdf_bracket,
        256.0,
        256.0,
        0.0,
        1.0,
        &Render {
            dark: 40.0 * 0.4 + 40.0,
            bright: 210.0 * 0.4 + 40.0,
        },
    );
    let m = find_one(&dimmed, &model, &cfg).expect("found under 40% contrast");
    assert!(
        base - m.score < 0.05,
        "score fell from {base:.3} to {:.3}",
        m.score
    );
}

#[test]
fn r3_fine_toothed_model_survives_the_pyramid() {
    // R3 probe: the comb's 8 px tooth pitch is 2 px at level 2 and 1 px at
    // level 3 — right at the box-mean pyramid's aliasing limit. The model
    // build drops aliased-away points level by level (that part is safe by
    // construction); the risk is the *scene* side losing the teeth so the
    // coarse score falls under the descent threshold and the object is never
    // found.
    //
    // Verdict encoded by this test: detection survives because the auto level
    // count stops where the model runs out of stable coarse points, and the
    // spine still anchors the coarse levels. If a future pyramid or model
    // change breaks this, the failure is THIS test, not a customer part.
    let reference = render(&sdf_comb, 256.0, 256.0, 0.0, 1.0, &Render::default());
    let roi = Rect2f {
        x: 180.0,
        y: 220.0,
        width: 152.0,
        height: 72.0,
    };
    let model = ShapeModelBuilder::new()
        .build(&reference.as_view(), roi, &ShapeModelConfig::default())
        .expect("comb model builds");

    // The comb must yield a genuinely multi-level model for the probe to mean
    // anything.
    assert!(
        model.num_levels() >= 3,
        "comb model collapsed to {} levels — fixture no longer probes R3",
        model.num_levels()
    );

    let angle = 25f32.to_radians();
    let scene = render(&sdf_comb, 310.0, 240.0, angle, 1.0, &Render::default());
    let m = find_one(&scene, &model, &ShapeSearchConfig::default())
        .expect("R3: comb lost at coarse level — box-mean aliasing killed the descent");

    assert!(m.score > 0.8, "score {}", m.score);
    let expected = expected_position(&model, 310.0, 240.0, angle, 1.0);
    assert!(
        (m.position.x - expected.x).abs() < 0.5 && (m.position.y - expected.y).abs() < 0.5,
        "position ({}, {}) vs expected ({}, {})",
        m.position.x,
        m.position.y,
        expected.x,
        expected.y
    );
    assert!((wrap_angle(m.angle() - angle)).abs() < 0.02);
}

/// Serialization round-trip: a persisted model must find the object at the
/// identical pose and score as the original (`serde` feature).
#[cfg(feature = "serde")]
#[test]
fn a_persisted_model_matches_identically() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    let bytes = model.to_bytes().expect("serializes");
    let restored = ShapeModel::from_bytes(&bytes).expect("deserializes");

    let scene = bracket_at(300.0, 220.0, 0.6, 1.0);
    let a = find_one(&scene, &model, &ShapeSearchConfig::default()).expect("original finds");
    let b = find_one(&scene, &restored, &ShapeSearchConfig::default()).expect("restored finds");
    assert_eq!(a.score, b.score);
    assert_eq!(a.position, b.position);
    assert_eq!(a.angle(), b.angle());

    // Version gate: a document from another format version must be refused,
    // not mis-read. The encoding is opaque, so this pokes at the *only* thing
    // the format promises — that a version mismatch is an error — by finding
    // whatever version number the writer emitted and changing it.
    let text = String::from_utf8(bytes.clone()).expect("the encoding is text today");
    let needle = text
        .split_once("\"format_version\":")
        .map(|(_, tail)| {
            let n: String = tail.chars().take_while(char::is_ascii_digit).collect();
            format!("\"format_version\":{n}")
        })
        .expect("envelope carries a version");
    let bad = text.replacen(&needle, "\"format_version\":999", 1);
    assert_ne!(bad, text, "version substitution did not fire");
    assert!(ShapeModel::from_bytes(bad.as_bytes()).is_err());
    assert!(ShapeModel::from_bytes(b"{}").is_err());
    assert!(ShapeModel::from_bytes(b"not a model at all").is_err());
}

/// Format 4 (roadmap W7) added `teach_points` — the pre-decimation level-0
/// edge set `resample_at` needs. A format-3 document (no such field) must
/// still load, bit-identically for everything `resample_at` does not touch,
/// with `resample_at` itself reporting a clear error rather than resampling
/// from already-decimated (scale-1.0-shaped) points.
#[cfg(feature = "serde")]
#[test]
fn a_format_3_document_loads_without_teach_data_and_resample_at_errors() {
    let model = build_bracket_model(&ShapeModelConfig::default());
    assert!(
        model.teach_point_count() > 0,
        "a fresh build carries teach data"
    );
    let bytes = model.to_bytes().expect("serializes");

    let mut doc: serde_json::Value = serde_json::from_slice(&bytes).expect("valid json");
    doc["format_version"] = serde_json::json!(3);
    doc["model"]
        .as_object_mut()
        .expect("model is a json object")
        .remove("teach_points");
    let v3_bytes = serde_json::to_vec(&doc).expect("re-serializes");

    let restored = ShapeModel::from_bytes(&v3_bytes).expect("a format-3 document still loads");
    assert_eq!(restored.teach_point_count(), 0);
    assert!(restored.resample_at(1.2).is_err());

    let scene = bracket_at(300.0, 220.0, 0.6, 1.0);
    let a = find_one(&scene, &model, &ShapeSearchConfig::default()).expect("original finds");
    let b = find_one(&scene, &restored, &ShapeSearchConfig::default())
        .expect("restored (format-3) finds");
    assert_eq!(a.score, b.score);
    assert_eq!(a.position, b.position);
    assert_eq!(a.angle(), b.angle());
}

/// The pyramid pre-filter travels with the model, because invariant 3 says the
/// scene must be decimated with the same kernel. A model taught with
/// `Binomial121` and searched by a matcher that knows nothing about it would
/// otherwise score against a differently band-limited scene.
#[cfg(feature = "serde")]
#[test]
fn the_pyramid_kernel_survives_persistence() {
    use vm_primitives::PreSmooth;

    let cfg = ShapeModelConfig {
        pre_smooth: PreSmooth::Binomial121,
        ..Default::default()
    };
    let model = build_bracket_model(&cfg);
    assert_eq!(model.pre_smooth(), PreSmooth::Binomial121);

    let restored =
        ShapeModel::from_bytes(&model.to_bytes().expect("serializes")).expect("deserializes");
    assert_eq!(restored.pre_smooth(), PreSmooth::Binomial121);

    let scene = bracket_at(300.0, 220.0, 0.6, 1.0);
    let a = find_one(&scene, &model, &ShapeSearchConfig::default()).expect("original finds");
    let b = find_one(&scene, &restored, &ShapeSearchConfig::default()).expect("restored finds");
    assert_eq!(a.score, b.score);
}
