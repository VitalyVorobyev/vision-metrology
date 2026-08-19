//! Pins the coordinate and angle conventions of the corrmatch bridge used by
//! `examples/pose_audit`.
//!
//! Convention mismatches (top-left vs centre anchoring, angle sign) are the
//! classic silent bug in cross-library comparison: everything runs, the
//! numbers are plausibly small, and the comparison is meaningless. This test
//! renders a scene with exact ground truth, runs BOTH corrmatch's own search
//! and the shape matcher, and checks that the bridge maps them onto each
//! other and onto the truth.

#[path = "../examples/common/corr.rs"]
mod corr;

use corrmatch::{
    CompileConfig, MatchConfig as CorrMatchConfig, Matcher as CorrMatcher, RotationMode,
    Template as CorrTemplate,
};
use vision_metrology::{
    Image, Point2f, Rect2f, ShapeMatcher, ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig,
};

const W: usize = 480;
const H: usize = 360;

/// Anti-aliased L-bracket (same construction as the shape-matching tests).
fn bracket(cx: f32, cy: f32, angle: f32) -> Image<u8> {
    fn sdf_box(p: (f32, f32), half: (f32, f32)) -> f32 {
        let dx = p.0.abs() - half.0;
        let dy = p.1.abs() - half.1;
        let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
        outside + dx.max(dy).min(0.0)
    }
    fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
    let (sn, cs) = angle.sin_cos();
    let mut data = vec![0u8; W * H];
    for y in 0..H {
        for x in 0..W {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let mx = cs * dx + sn * dy;
            let my = -sn * dx + cs * dy;
            let sdf =
                sdf_box((mx, my + 22.0), (30.0, 8.0)).min(sdf_box((mx + 22.0, my), (8.0, 30.0)));
            let t = smoothstep(-1.0, 1.0, -sdf);
            data[y * W + x] = (40.0 + 170.0 * t).round() as u8;
        }
    }
    Image::from_vec(W, H, data).expect("valid image")
}

const REF_C: (f32, f32) = (140.0, 120.0);
const ROI: Rect2f = Rect2f {
    x: 96.0,
    y: 76.0,
    width: 88.0,
    height: 88.0,
};

fn corr_search(scene: &Image<u8>, reference: &Image<u8>) -> corrmatch::Match {
    let (tpl, tw, th) = corr::roi_patch(reference, ROI);
    let template = CorrTemplate::new(tpl, tw, th).expect("template");
    let compiled = template.compile(CompileConfig::default()).expect("compile");
    let mut cfg = CorrMatchConfig::default();
    cfg.rotation = RotationMode::Enabled;
    let matcher = CorrMatcher::new(compiled).with_config(cfg);
    let view = corrmatch::ImageView::from_slice(scene.data(), scene.width(), scene.height())
        .expect("scene view");
    matcher.match_image(view).expect("corrmatch finds")
}

#[test]
fn the_two_matchers_agree_through_the_bridge() {
    let reference = bracket(REF_C.0, REF_C.1, 0.0);
    let roi_center = Point2f {
        x: ROI.x + 0.5 * ROI.width,
        y: ROI.y + 0.5 * ROI.height,
    };
    let (tw, th) = (ROI.width as usize, ROI.height as usize);

    // Object translated and rotated by a known amount.
    let truth_angle = 18f32.to_radians();
    let (tx, ty) = (261.0, 173.0);
    let scene = bracket(tx, ty, truth_angle);

    // Shape matcher.
    let model = ShapeModelBuilder::new()
        .build(&reference.as_view(), ROI, &ShapeModelConfig::default())
        .expect("model");
    let m = ShapeMatcher::new()
        .find(&scene.as_view(), &model, &ShapeSearchConfig::default())
        .into_iter()
        .next()
        .expect("shape match");
    let shape_center = m.pose * nalgebra::Point2::new(roi_center.x, roi_center.y);

    // corrmatch search on the same scene.
    let cm = corr_search(&scene, &reference);
    let corr_center = corr::corr_to_center(cm.x, cm.y, tw, th);

    // 1. Position agreement through the bridge.
    let dp = ((shape_center.x - corr_center.x).powi(2) + (shape_center.y - corr_center.y).powi(2))
        .sqrt();
    assert!(
        dp < 1.5,
        "bridge position mismatch: shape ({:.2}, {:.2}) vs corr ({:.2}, {:.2})",
        shape_center.x,
        shape_center.y,
        corr_center.x,
        corr_center.y
    );

    // 2. Angle sign and magnitude agreement (corrmatch reports degrees).
    let da = (cm.angle_deg - truth_angle.to_degrees()).abs();
    assert!(
        da < 3.0,
        "corrmatch angle {:.2} deg vs truth {:.2} deg — sign convention broken?",
        cm.angle_deg,
        truth_angle.to_degrees()
    );
    assert!((m.angle() - truth_angle).abs() < 0.02);

    // 3. Round trip of the top-left conversion.
    let tl = corr::center_to_corr_topleft(corr_center, tw, th);
    assert!((tl.x - cm.x).abs() < 1e-4 && (tl.y - cm.y).abs() < 1e-4);
}

#[test]
fn zncc_scores_the_true_pose_high_and_a_wrong_pose_low() {
    let reference = bracket(REF_C.0, REF_C.1, 0.0);
    let truth_angle = 25f32.to_radians();
    let scene = bracket(230.0, 190.0, truth_angle);

    let model = ShapeModelBuilder::new()
        .build(&reference.as_view(), ROI, &ShapeModelConfig::default())
        .expect("model");
    let m = ShapeMatcher::new()
        .find(&scene.as_view(), &model, &ShapeSearchConfig::default())
        .into_iter()
        .next()
        .expect("shape match");

    let roi_center = Point2f {
        x: ROI.x + 0.5 * ROI.width,
        y: ROI.y + 0.5 * ROI.height,
    };
    let c = m.pose * nalgebra::Point2::new(roi_center.x, roi_center.y);
    let center = Point2f { x: c.x, y: c.y };

    let good = corr::zncc_at_pose(&scene, &reference, ROI, center, m.angle().to_degrees())
        .expect("zncc at pose");
    assert!(good > 0.9, "true pose zncc {good}");

    // Same position, wrong angle: the independent score must collapse.
    let bad = corr::zncc_at_pose(
        &scene,
        &reference,
        ROI,
        center,
        m.angle().to_degrees() + 90.0,
    )
    .expect("zncc at wrong pose");
    assert!(
        bad < good - 0.3,
        "wrong-angle zncc {bad} not clearly below {good}"
    );

    // Wrong position: also collapses.
    let off = corr::zncc_at_pose(
        &scene,
        &reference,
        ROI,
        Point2f {
            x: center.x + 25.0,
            y: center.y,
        },
        m.angle().to_degrees(),
    )
    .expect("zncc offset");
    assert!(
        off < good - 0.3,
        "offset zncc {off} not clearly below {good}"
    );
}
