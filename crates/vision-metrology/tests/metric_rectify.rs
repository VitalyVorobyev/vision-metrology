//! Rectify-first 3-D acceptance (roadmap B5, plan decision 10).
//!
//! A planar, distinctively-shaped target is imaged through a **calibrated,
//! tilted** camera (pinhole + Brown-Conrady, tilt `0..=40°`), using the
//! *forward* model: for every raw-image pixel, [`pixel_to_plane`] finds
//! where that pixel's ray meets the reference frame's `z = 0` plane, and the
//! plane's own synthetic pattern is sampled there. This is the honest
//! inverse of [`plane_grid_map`] (which goes `plane grid → raw image`) —
//! rendering through `pixel_to_plane` rather than `plane_grid_map` keeps the
//! render path independent of the rectify path under test, so a bug shared
//! between the two could not silently cancel out.
//!
//! Each tilted raw image is then **rectified** back to the plane grid with
//! [`plane_grid_map`] + [`Map::apply_with_mask`]. A [`ShapeModel`] taught on
//! the 0° rectified image is searched for in every tilt's rectified image.
//! If rectify-first genuinely closes the planar 3-D case (roadmap decision
//! 10), the model must be found at every tilt, at (near enough) the same
//! grid position — proving that a fronto-parallel rectification turns an
//! out-of-plane pose into an in-plane 2-D similarity problem, which is what
//! the rest of this crate already solves well.
//!
//! **Measured (2026-08-20, M4 Pro, release):**
//!
//! | tilt | found | score | position error (px) |
//! |---|---|---|---|
//! | 0° | yes | 1.0000 | 0.0004 |
//! | 10° | yes | 1.0000 | 0.0037 |
//! | 20° | yes | 1.0000 | 0.0023 |
//! | 30° | yes | 1.0000 | 0.0120 |
//! | 40° | yes | 1.0000 | 0.0029 |
//!
//! Found at every one of the five tilts, score 1.0000 throughout (the
//! render and the rectify are both noise-free and geometrically exact, so
//! there is no occlusion/clutter for the score to discount), max position
//! error **0.012 px** — an order of magnitude inside the plan's 0.1 px
//! acceptance target, and small enough that the residual is almost
//! certainly the shape matcher's own subpixel-refinement floor rather than
//! anything rectify-specific. This is the number that answers "does
//! rectify-first close the planar 3-D case": yes, to well under a hundredth
//! of a pixel, for a target imaged up to 40° off fronto-parallel. It also
//! quantifies how little is left for homography refinement (roadmap
//! decision 10, next session) to buy on a genuinely planar target — its
//! value is for the *non*-planar case this test does not (and cannot)
//! exercise.

#![cfg(all(feature = "matching", feature = "metric"))]

use nalgebra::{Translation3, UnitQuaternion, Vector3};

use vision_metrology::matching::{
    Refinement, ShapeMatcher, ShapeModel, ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig,
};
use vision_metrology::metric::{
    CameraModel, Plane3, PlaneGrid, Pose3, homography_plane_to_image, pixel_to_plane,
    plane_grid_map,
};
use vision_metrology::warp::{Interp, Map};
use vision_metrology::{BorderMode, Image, Point2f, Rect2f};

const DARK: f32 = 40.0;
const BRIGHT: f32 = 220.0;

// ── Plane pattern: an SDF L-shape in millimetres, mirroring the fixture ──
// `tests/accuracy.rs` uses for `ShapeMatcher`, just evaluated in mm rather
// than pixels.

fn sdf_box(p: (f32, f32), half: (f32, f32)) -> f32 {
    let dx = p.0.abs() - half.0;
    let dy = p.1.abs() - half.1;
    let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
    outside + dx.max(dy).min(0.0)
}

fn sdf_l_shape_mm(p: (f32, f32)) -> f32 {
    let a = sdf_box((p.0, p.1 + 25.0), (35.0, 10.0));
    let b = sdf_box((p.0 + 25.0, p.1), (10.0, 35.0));
    a.min(b)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Intensity of the plane pattern at plane-frame `(x_mm, y_mm)` — the L-shape
/// centered at the plane origin, anti-aliased over roughly 1mm.
fn pattern_mm(x_mm: f32, y_mm: f32) -> f32 {
    let d = sdf_l_shape_mm((x_mm, y_mm));
    let t = smoothstep(-1.0, 1.0, -d);
    DARK + (BRIGHT - DARK) * t
}

// ── Camera / poses ──────────────────────────────────────────────────────

fn camera() -> CameraModel {
    use vision_metrology::metric::{BrownConrady5, PinholeIntrinsics};
    CameraModel {
        intrinsics: PinholeIntrinsics {
            fx: 1200.0,
            fy: 1200.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        },
        distortion: BrownConrady5 {
            k1: -0.08,
            k2: 0.01,
            k3: 0.0,
            p1: 0.0005,
            p2: -0.0003,
        },
    }
}

const RAW_W: usize = 640;
const RAW_H: usize = 480;
const STANDOFF_MM: f32 = 700.0;

/// `pose` for a camera tilted `tilt_deg` about the reference frame's own
/// x-axis, standing off `STANDOFF_MM` from the plane origin along its own
/// optical axis. Camera-from-reference (this module's convention).
fn tilted_pose(tilt_deg: f32) -> Pose3 {
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), tilt_deg.to_radians());
    Pose3::from_parts(Translation3::new(0.0, 0.0, STANDOFF_MM), rotation)
}

/// Render the raw camera image at `pose` via the forward model:
/// `pixel_to_plane` for every raw pixel, then sample the plane pattern.
fn render_raw(camera: &CameraModel, pose: &Pose3) -> Image<u8> {
    let plane = Plane3::xy();
    let mut data = vec![0u8; RAW_W * RAW_H];
    for y in 0..RAW_H {
        for x in 0..RAW_W {
            let pixel = Point2f::new(x as f32, y as f32);
            let value = match pixel_to_plane(camera, pose, &plane, pixel) {
                Some(p) => pattern_mm(p.x, p.y),
                None => DARK,
            };
            data[y * RAW_W + x] = value.round().clamp(0.0, 255.0) as u8;
        }
    }
    Image::from_vec(RAW_W, RAW_H, data).expect("valid image")
}

fn grid() -> PlaneGrid {
    PlaneGrid {
        origin_mm: Point2f::new(-80.0, -80.0),
        mm_per_px: 1.0,
        w: 160,
        h: 160,
    }
}

/// Rectify `raw` (shot at `pose`) to the canonical plane grid.
fn rectify(camera: &CameraModel, pose: &Pose3, raw: &Image<u8>) -> (Image<u8>, Vec<u8>) {
    let g = grid();
    let map: Map = plane_grid_map(camera, pose, &g);
    let mut dst = vec![0u8; g.w * g.h];
    let mut mask = vec![0u8; g.w * g.h];
    map.apply_with_mask(
        &raw.as_view(),
        &mut dst,
        &mut mask,
        Interp::Bilinear,
        BorderMode::Constant(DARK as u8),
    )
    .expect("apply_with_mask succeeds");
    (Image::from_vec(g.w, g.h, dst).expect("valid image"), mask)
}

fn model_roi() -> Rect2f {
    Rect2f {
        x: 15.0,
        y: 15.0,
        width: 130.0,
        height: 130.0,
    }
}

fn build_model(rectified_0deg: &Image<u8>) -> ShapeModel {
    ShapeModelBuilder::new()
        .build(
            &rectified_0deg.as_view(),
            model_roi(),
            &ShapeModelConfig::default(),
        )
        .expect("model builds on the 0deg rectified crop")
}

#[test]
fn rectify_first_finds_the_model_at_every_tilt() {
    let cam = camera();
    let tilts = [0.0f32, 10.0, 20.0, 30.0, 40.0];

    // Teach on the 0deg rectified crop.
    let pose0 = tilted_pose(0.0);
    let raw0 = render_raw(&cam, &pose0);
    let (rectified0, mask0) = rectify(&cam, &pose0, &raw0);
    let valid0 = mask0.iter().filter(|&&m| m == 255).count();
    assert!(
        valid0 > grid().w * grid().h / 2,
        "0deg rectified crop should be mostly valid, got {valid0}/{}",
        grid().w * grid().h
    );
    let model = build_model(&rectified0);
    // The model's own reference point in the 0deg rectified crop's
    // coordinates — since rectification maps every tilt's raw image back to
    // the *same* plane grid (the whole premise this test checks), a found
    // match's `position` should land here at every tilt, not just at 0deg.
    let expected = model.origin();

    let search_cfg = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };

    eprintln!("tilt_deg,found,score,pos_err_px");
    let mut all_found = true;
    let mut max_pos_err = 0.0f32;
    for &tilt in &tilts {
        let pose = tilted_pose(tilt);
        let raw = render_raw(&cam, &pose);
        let (rectified, _mask) = rectify(&cam, &pose, &raw);

        let matches = ShapeMatcher::new().find(&rectified.as_view(), &model, &search_cfg);
        match matches.into_iter().next() {
            Some(m) => {
                assert!(m.score > 0.95, "tilt {tilt}deg: score {} too low", m.score);
                let err = ((m.position.x - expected.x).powi(2)
                    + (m.position.y - expected.y).powi(2))
                .sqrt();
                max_pos_err = max_pos_err.max(err);
                eprintln!("{tilt},true,{:.4},{:.4}", m.score, err);
                // Envelope pinned at ~4x the measured worst case (see the
                // per-tilt table in this test's doc comment) — comfortably
                // inside the plan's 0.1 px acceptance target.
                assert!(
                    err < 0.05,
                    "tilt {tilt}deg: position error {err} px too large (found {:?}, expected {:?})",
                    m.position,
                    expected
                );
            }
            None => {
                all_found = false;
                eprintln!("{tilt},false,,");
            }
        }
    }
    assert!(all_found, "model must be found at every tilt 0..=40deg");
    eprintln!("max position error across all tilts: {max_pos_err:.4} px");

    // Sanity: `homography_plane_to_image` and `plane_grid_map` describe the
    // same geometry at zero distortion-free tilt (compile-time reachability
    // check that both public entry points this test exercises stay linked).
    let _ = homography_plane_to_image(&cam, &pose0, &grid());
}
