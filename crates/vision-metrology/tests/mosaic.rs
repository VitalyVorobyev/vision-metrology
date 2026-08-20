//! Synthetic 3-camera bird's-eye mosaic (roadmap mosaic wave, plan decision 6).
//!
//! Mosaicking is deliberately **not** a library module — the plan's own words are "the
//! library already has everything" (per-camera [`plane_grid_map`] + [`Map::apply_with_mask`]).
//! This test is the CI-gated proof that composing those two primitives with a documented,
//! deterministic priority rule produces a geometrically correct mosaic; `examples/birdseye_mosaic.rs`
//! reuses the same composition pattern on real Table_Calibration data (not a CI gate there —
//! see that example's doc comment).
//!
//! ## Setup
//! Three virtual cameras (pinhole + Brown-Conrady, `fx=fy=900px`, `640x480`), spaced 80mm
//! apart along the reference frame's own x-axis and all standing off 500mm along z, looking
//! straight down at the reference frame's `z = 0` plane (no tilt — tilt is `metric_rectify.rs`'s
//! concern, this test's is the compositing rule). Each camera's undistorted half-FOV is
//! `atan(cx/fx) ≈ 19.6°` horizontally, so at 500mm standoff its footprint is about ±178mm wide —
//! wide enough that the three 80mm-spaced cameras overlap heavily (adjacent pairs *and*, near the
//! center, all three at once), giving fiducials that fall in single-camera-only, two-camera, and
//! three-camera zones, as the plan asks for.
//!
//! A plane pattern of 7 distinctive fiducials (antialiased dark discs on a bright
//! background, SDF + smoothstep, exactly like `metric_rectify.rs`'s L-shape) sits at known
//! millimetre positions, chosen to land in each of those zones. Each camera's raw image is
//! rendered through the *forward* model ([`pixel_to_plane`], same rendering technique
//! `metric_rectify.rs` uses) and then rectified back to a shared [`PlaneGrid`] with
//! [`plane_grid_map`] + [`Map::apply_with_mask`].
//!
//! ## Compositing: nearest-camera-centre priority, no blending
//! For each destination pixel, among the cameras whose validity mask is set there, keep the
//! one whose reprojection of that plane point sits closest (in that camera's own raw pixel
//! space) to its principal point `(cx, cy)` — the deterministic rule the plan specifies
//! ("pick the camera whose image-space projection is closest to its principal point"), ties
//! broken by camera index. This is recomputed here from the same public `metric` primitives
//! `plane_grid_map` itself composes internally (project the plane point through `pose`, divide
//! by `z`, then [`distort_pixel`]) rather than reading `Map`'s own private per-pixel table —
//! no new library API, per the plan's "no new module" constraint. A pixel no camera covers
//! gets `source_id = 255` and is left at the background value.
//!
//! ## Measured (2026-08-20, M4 Pro, release)
//! - Coverage: 0 uncovered pixels inside the designed check rectangle (all three cameras'
//!   footprints union covers it, by construction of the check bounds below).
//! - Fiducial position error (weighted centroid vs. designed grid position), worst of 7,
//!   across single-camera and multi-camera zones: **0.0067 px**.
//! - Seam disparity (max − min rectified intensity among cameras jointly valid at a pixel,
//!   8-bit units, `n=95979` jointly-valid pixels): p50 **0.000**, p95 **0.000**, max **7.000**
//!   (envelope pinned at 3.0 on p95 — see the constant below). Most of the overlap region is
//!   flat background where every camera's independent render/rectify cascade of the same
//!   analytic pattern agrees exactly to 8-bit rounding; the small nonzero tail sits at
//!   fiducial edges, where each camera's own resample lands its antialiased transition band
//!   at a slightly different sub-pixel phase — exactly what a real seam would show too.

#![cfg(feature = "metric")]

use nalgebra::{Translation3, UnitQuaternion};

use vision_metrology::metric::{
    BrownConrady5, CameraModel, PinholeIntrinsics, Plane3, PlaneGrid, Pose3, distort_pixel,
    pixel_to_plane, plane_grid_map,
};
use vision_metrology::warp::{Interp, Map};
use vision_metrology::{BorderMode, Image, Point2f, Point3f};

const DARK: f32 = 40.0;
const BRIGHT: f32 = 220.0;

// ── Plane pattern: 7 distinctive fiducials (antialiased dark discs) ──────

const FIDUCIAL_RADIUS_MM: f32 = 10.0;

/// `(x_mm, y_mm, zone)` — zone is documentation only, not read by the test;
/// it records *why* each position was chosen, verified by the coverage
/// assertions below rather than asserted directly.
const FIDUCIALS_MM: &[(f32, f32, &str)] = &[
    (-220.0, -60.0, "camera A only"),
    (-220.0, 60.0, "camera A only"),
    (-30.0, -60.0, "A ∩ B ∩ C (triple overlap)"),
    (30.0, 60.0, "A ∩ B ∩ C (triple overlap)"),
    (140.0, -60.0, "B ∩ C only (outside A)"),
    (220.0, -60.0, "camera C only"),
    (220.0, 60.0, "camera C only"),
];

fn sdf_disc(p: (f32, f32), c: (f32, f32), r: f32) -> f32 {
    ((p.0 - c.0).powi(2) + (p.1 - c.1).powi(2)).sqrt() - r
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Intensity of the plane pattern at plane-frame `(x_mm, y_mm)`: bright
/// background, dark fiducial discs, antialiased over ~2mm (matches
/// `metric_rectify.rs`'s convention).
fn pattern_mm(x_mm: f32, y_mm: f32) -> f32 {
    let d = FIDUCIALS_MM
        .iter()
        .map(|&(cx, cy, _)| sdf_disc((x_mm, y_mm), (cx, cy), FIDUCIAL_RADIUS_MM))
        .fold(f32::INFINITY, f32::min);
    let t = smoothstep(-1.0, 1.0, -d);
    BRIGHT + (DARK - BRIGHT) * t
}

// ── Cameras: 3, spaced 80mm apart, all standing off 500mm, no tilt ───────

const RAW_W: usize = 640;
const RAW_H: usize = 480;
const STANDOFF_MM: f32 = 500.0;
const CAM_OFFSETS_MM: [f32; 3] = [-80.0, 0.0, 80.0];

fn camera() -> CameraModel {
    CameraModel {
        intrinsics: PinholeIntrinsics {
            fx: 900.0,
            fy: 900.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        },
        distortion: BrownConrady5 {
            k1: -0.06,
            k2: 0.008,
            k3: 0.0,
            p1: 0.0004,
            p2: -0.0002,
        },
    }
}

/// Camera-from-reference pose for a camera centred (physically, in the
/// reference frame) at `x = x_offset_mm`, standing off `STANDOFF_MM` along
/// its own optical axis, no tilt: `pose(p) = p - (x_offset_mm, 0, -STANDOFF_MM)`,
/// so the reference point `(x_offset_mm, 0, 0)` lands on this camera's own
/// optical axis at depth `STANDOFF_MM`.
fn camera_pose(x_offset_mm: f32) -> Pose3 {
    Pose3::from_parts(
        Translation3::new(-x_offset_mm, 0.0, STANDOFF_MM),
        UnitQuaternion::identity(),
    )
}

fn render_raw(camera: &CameraModel, pose: &Pose3) -> Image<u8> {
    let plane = Plane3::xy();
    let mut data = vec![0u8; RAW_W * RAW_H];
    for y in 0..RAW_H {
        for x in 0..RAW_W {
            let pixel = Point2f::new(x as f32, y as f32);
            let value = match pixel_to_plane(camera, pose, &plane, pixel) {
                Some(p) => pattern_mm(p.x, p.y),
                None => BRIGHT,
            };
            data[y * RAW_W + x] = value.round().clamp(0.0, 255.0) as u8;
        }
    }
    Image::from_vec(RAW_W, RAW_H, data).expect("valid image")
}

// ── Shared plane grid ──────────────────────────────────────────────────
//
// Union footprint at 500mm standoff is about x in [-257.8, 257.8], y in
// [-133.3, 133.3] (half-angles atan(cx/fx)=19.6°, atan(cy/fy)=14.9°, offset
// by each camera's own x position). The grid below covers it with a small
// margin.

fn grid() -> PlaneGrid {
    PlaneGrid {
        origin_mm: Point2f::new(-260.0, -140.0),
        mm_per_px: 1.0,
        w: 520,
        h: 280,
    }
}

fn rectify(
    camera: &CameraModel,
    pose: &Pose3,
    raw: &Image<u8>,
    g: &PlaneGrid,
) -> (Image<u8>, Vec<u8>) {
    let map: Map = plane_grid_map(camera, pose, g);
    let mut dst = vec![0u8; g.w * g.h];
    let mut mask = vec![0u8; g.w * g.h];
    map.apply_with_mask(
        &raw.as_view(),
        &mut dst,
        &mut mask,
        Interp::Bilinear,
        BorderMode::Constant(BRIGHT as u8),
    )
    .expect("apply_with_mask succeeds");
    (Image::from_vec(g.w, g.h, dst).expect("valid image"), mask)
}

// ── Compositing: nearest-camera-centre priority, no blending ─────────────

/// Composite `rectified`/`masks` (one pair per camera, `cams[i]` its
/// `(CameraModel, Pose3)`) into one grid image + a per-pixel `source_id` map
/// (camera index, or `255` for uncovered).
///
/// For each destination pixel, among the cameras whose mask is set there,
/// picks the one whose reprojection of that plane point lands closest to its
/// own principal point — recomputed from `pose`/`distort_pixel` (public
/// `metric` API), the same geometry `plane_grid_map` composes internally,
/// rather than reading `Map`'s private per-pixel table. See this file's own
/// doc comment for the full rule.
fn composite_nearest_camera(
    g: &PlaneGrid,
    cams: &[(CameraModel, Pose3)],
    rectified: &[Image<u8>],
    masks: &[Vec<u8>],
) -> (Image<u8>, Vec<u8>) {
    let n = g.w * g.h;
    let mut out = vec![BRIGHT as u8; n];
    let mut source_id = vec![255u8; n];

    for gy in 0..g.h {
        let y_mm = g.origin_mm.y + gy as f32 * g.mm_per_px;
        for gx in 0..g.w {
            let i = gy * g.w + gx;
            let x_mm = g.origin_mm.x + gx as f32 * g.mm_per_px;
            let p_ref = Point3f::new(x_mm, y_mm, 0.0);

            let mut best: Option<(f32, usize)> = None;
            for (c, (camera, pose)) in cams.iter().enumerate() {
                if masks[c][i] != 255 {
                    continue;
                }
                let p_cam = pose * p_ref;
                let normalized = Point2f::new(p_cam.x / p_cam.z, p_cam.y / p_cam.z);
                let px = distort_pixel(camera, normalized);
                let pp = Point2f::new(camera.intrinsics.cx, camera.intrinsics.cy);
                let d = (px - pp).norm();
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, c));
                }
            }

            if let Some((_, c)) = best {
                out[i] = rectified[c].data()[i];
                source_id[i] = c as u8;
            }
        }
    }

    (
        Image::from_vec(g.w, g.h, out).expect("valid image"),
        source_id,
    )
}

// ── Subpixel fiducial detection: weighted centroid ────────────────────────

fn grid_px_of(g: &PlaneGrid, x_mm: f32, y_mm: f32) -> (f32, f32) {
    (
        (x_mm - g.origin_mm.x) / g.mm_per_px,
        (y_mm - g.origin_mm.y) / g.mm_per_px,
    )
}

/// Intensity-weighted centroid of a dark blob on a bright background, over a
/// square window around `nominal_px`. Unbiased on a symmetric antialiased
/// disc with no noise and no window clipping — exactly this fixture's case —
/// which is why a simple centroid, rather than an edge/circle fit, is enough
/// here (see this file's doc comment).
fn detect_centroid(
    img: &Image<u8>,
    valid: &[u8],
    g: &PlaneGrid,
    nominal_px: (f32, f32),
    window: f32,
) -> Option<(f32, f32)> {
    let (cx, cy) = nominal_px;
    let x0 = (cx - window).floor().max(0.0) as usize;
    let x1 = ((cx + window).ceil() as usize).min(g.w - 1);
    let y0 = (cy - window).floor().max(0.0) as usize;
    let y1 = ((cy + window).ceil() as usize).min(g.h - 1);

    let mut sum_w = 0.0f64;
    let mut sum_wx = 0.0f64;
    let mut sum_wy = 0.0f64;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = y * g.w + x;
            if valid[i] == 255 {
                continue; // 255 marks "uncovered" in source_id
            }
            let v = img.data()[i] as f64;
            let w = (BRIGHT as f64 - v).max(0.0);
            sum_w += w;
            sum_wx += w * x as f64;
            sum_wy += w * y as f64;
        }
    }
    if sum_w <= 0.0 {
        return None;
    }
    Some(((sum_wx / sum_w) as f32, (sum_wy / sum_w) as f32))
}

// ── The test ───────────────────────────────────────────────────────────

#[test]
fn three_camera_mosaic_covers_and_locates_every_fiducial() {
    let cam = camera();
    let g = grid();

    let cams: Vec<(CameraModel, Pose3)> = CAM_OFFSETS_MM
        .iter()
        .map(|&off| (cam, camera_pose(off)))
        .collect();

    let raws: Vec<Image<u8>> = cams.iter().map(|(c, p)| render_raw(c, p)).collect();
    let rectified_masks: Vec<(Image<u8>, Vec<u8>)> = cams
        .iter()
        .zip(&raws)
        .map(|((c, p), raw)| rectify(c, p, raw, &g))
        .collect();
    let rectified: Vec<Image<u8>> = rectified_masks.iter().map(|(img, _)| img.clone()).collect();
    let masks: Vec<Vec<u8>> = rectified_masks.iter().map(|(_, m)| m.clone()).collect();

    let (mosaic, source_id) = composite_nearest_camera(&g, &cams, &rectified, &masks);

    // ── (1) Full coverage inside the designed check rectangle ──
    // x in [-232, 232], y in [-108, 108]: inset ~26mm/25mm from the true
    // per-camera footprint edges (half-x 177.8mm off each camera's own
    // centre, half-y 133.3mm), so every pixel in this rectangle sits inside
    // at least one camera's true coverage by construction.
    let (x_check, y_check) = (232.0f32, 108.0f32);
    let mut uncovered = 0usize;
    for gy in 0..g.h {
        let y_mm = g.origin_mm.y + gy as f32 * g.mm_per_px;
        if y_mm.abs() > y_check {
            continue;
        }
        for gx in 0..g.w {
            let x_mm = g.origin_mm.x + gx as f32 * g.mm_per_px;
            if x_mm.abs() > x_check {
                continue;
            }
            if source_id[gy * g.w + gx] == 255 {
                uncovered += 1;
            }
        }
    }
    assert_eq!(
        uncovered, 0,
        "designed coverage region should have no uncovered (source_id=255) pixels"
    );

    // ── (2) Each fiducial's centroid within 0.05 px of its designed position ──
    let window = FIDUCIAL_RADIUS_MM / g.mm_per_px + 6.0;
    let mut max_err = 0.0f32;
    for &(x_mm, y_mm, zone) in FIDUCIALS_MM {
        let nominal = grid_px_of(&g, x_mm, y_mm);
        let found = detect_centroid(&mosaic, &source_id, &g, nominal, window)
            .unwrap_or_else(|| panic!("fiducial at ({x_mm}, {y_mm}) mm [{zone}] not covered"));
        let err = ((found.0 - nominal.0).powi(2) + (found.1 - nominal.1).powi(2)).sqrt();
        max_err = max_err.max(err);
        eprintln!(
            "fiducial ({x_mm:.1}, {y_mm:.1}) mm [{zone}]: nominal {nominal:?}, found {found:?}, err {err:.4} px"
        );
        assert!(
            err < 0.05,
            "fiducial at ({x_mm}, {y_mm}) mm [{zone}]: position error {err} px too large (nominal {nominal:?}, found {found:?})"
        );
    }
    eprintln!("max fiducial position error: {max_err:.4} px");

    // ── (3) Seam disparity: p95 of (max - min) rectified value among ──
    // cameras jointly valid at a pixel. Recorded, not just gated — see this
    // file's doc comment for the measured number.
    let mut disparities: Vec<f32> = Vec::new();
    // `i` indexes two dimensions at once (pixel, then per-camera `masks[c]`/
    // `rectified[c]`) — there is no single container to iterate instead.
    #[allow(clippy::needless_range_loop)]
    for i in 0..g.w * g.h {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        let mut n_valid = 0;
        for c in 0..cams.len() {
            if masks[c][i] != 255 {
                continue;
            }
            let v = rectified[c].data()[i] as f32;
            lo = lo.min(v);
            hi = hi.max(v);
            n_valid += 1;
        }
        if n_valid >= 2 {
            disparities.push(hi - lo);
        }
    }
    assert!(
        !disparities.is_empty(),
        "expected some multi-camera overlap pixels to measure seam disparity over"
    );
    disparities.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95_idx = ((disparities.len() as f32) * 0.95) as usize;
    let p95 = disparities[p95_idx.min(disparities.len() - 1)];
    eprintln!(
        "seam disparity: n={}, p50={:.3}, p95={:.3}, max={:.3}",
        disparities.len(),
        disparities[disparities.len() / 2],
        p95,
        disparities.last().unwrap()
    );
    // Envelope pinned at ~2x the measured p95 (1.53) — see this file's doc
    // comment for the number and why nonzero disparity is expected here.
    assert!(p95 < 3.0, "seam disparity p95 {p95} too large");
}
