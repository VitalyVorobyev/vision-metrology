//! Bird's-eye mosaic on real calibrated data (roadmap mosaic wave, plan decision 6).
//!
//! Composes two calibrated cameras' rectified views of a shared plane into one grid, with
//! the same **no-blending, nearest-camera-centre priority** rule `tests/mosaic.rs` verifies
//! on a synthetic 3-camera fixture (see that file's doc comment for the exact rule and why
//! it is deliberately duplicated here rather than factored into a library module — the plan's
//! decision is that mosaicking is *not* a library module, so this composition helper is
//! intentionally re-derived per consumer, not shared code).
//!
//! ## Dataset: `~/vision/data/25_09_17_Table_Calibration/`
//! `calibration.json` (the `table_calibration` format `metric::io::import_table_calibration`
//! parses) plus `cam1/`/`cam2/` PNG folders, each holding the *same* 25 filenames
//! (`CamN_<Rx>Rx<Ry>Ry<X>X<Y>Y<Z>Z.png`) — a robot-swept calibration-target capture, not a
//! sequence. This example picks the nominal `+0.0Rx+0.0Ry+0.0X+0.0Y+0.0Z` frame from each
//! camera folder: identical filename suffix pairs the two cameras' simultaneous shots (the
//! "if the naming allows pairing" case the plan anticipates), and the all-zero offset is the
//! target's flat, centred pose — each camera's least-foreshortened view.
//!
//! **Camera-index judgment call.** `calibration.json` names its two cameras `camera0`/
//! `camera1` (0-indexed); the image folders are `cam1`/`cam2` (1-indexed). Nothing in the
//! dataset states the correspondence explicitly, so this example takes the only sensible
//! reading — `import_table_calibration`'s index-sorted `camera0` ↔ `cam1/`, `camera1` ↔
//! `cam2/` — documented here rather than asserted as fact, the same posture
//! `metric::io::table` itself takes for this dataset's unit and reference-frame calls.
//!
//! ## The table plane isn't `z = 0` of the calibration's own reference frame
//! `import_table_calibration` fixes `camera0`'s own frame as the reference frame (its
//! `sensor2camera` is the identity — see that importer's docs). `z = 0` there sits *at
//! camera0's own optical center* — a homography through it is singular, and physically
//! there is no "table" there at all. What the dataset actually gives two fixed cameras
//! calibrated together for is a **shared calibration target**, which they were normally
//! mounted to jointly converge on. [`common_standoff`] recovers that convergence point —
//! the closest-approach distance between the two cameras' own optical axes, computed purely
//! from the calibration's extrinsics — and this example shifts the reference frame along
//! camera0's axis to sit there before calling [`plane_grid_map`], i.e. it composes a
//! **new** camera-from-(shifted-reference) pose `pose ∘ Translation3(0, 0, standoff)` per
//! camera and hands that to the existing, unmodified `metric`/`warp` API — no library change,
//! per the plan's "the library already has everything" constraint.
//!
//! ## Measured (2026-08-20, this dataset, this frame pair, M4 Pro, release)
//! Run with `WRITE_ASSETS=1` to also regenerate `docs/assets/birdseye-mosaic.png` (both
//! camera choices are deterministic, so the asset is reproducible run to run).
//!
//! - Common standoff: **273.90 mm** along camera0's own optical axis, with the two axes
//!   passing within **0.406 mm** of each other there — a tight convergence for a ~111mm
//!   baseline, evidence this dataset really is a verged pair aimed at one target rather
//!   than two independently pointed cameras.
//! - Grid: 900x675 px at 0.0378 mm/px (the chosen gallery-sized resolution — coarser than
//!   the dataset's native ~0.013 mm/px GSD; this is a coverage/visual demo, not a precision
//!   measurement).
//! - Coverage: camera0 **59.0%**, camera1 **65.0%** of the grid; union **66.5%**; overlap
//!   (≥ 2 cameras valid) **86.6% of the covered area**. This is a **converged stereo pair**,
//!   not a wide-table array of mostly-disjoint cameras — both cameras see nearly the same
//!   patch of the calibration target (consistent with the checkerboard filling both raw
//!   frames almost identically), so most of what each camera covers, the other does too.
//!   The grid's own margin (`FOOTPRINT_MARGIN`) intentionally reaches a little past that
//!   shared patch, which is most of why coverage reads under 100%.
//! - Seam disparity (max − min rectified intensity among jointly-valid cameras, 8-bit
//!   units, `n=349877`): p50 **47.00**, p95 **251.00**, max **255.00**. This is large in
//!   absolute terms and expected: the target is a **hard-edged checkerboard**, not an
//!   antialiased pattern like `tests/mosaic.rs`'s synthetic fiducials, so once any real
//!   source of sub-pixel disagreement is present — the ~0.5-0.6px reprojection error
//!   `calibration.json` itself reports, this example's approximate (not per-frame-exact)
//!   standoff estimate, or ordinary cascaded-bilinear-resample rounding — a pixel that lands
//!   near a checker edge in one camera's rectified view but not quite the other's swings the
//!   *whole* 0-255 range rather than a few intensity units. `docs/assets/birdseye-mosaic.png`
//!   (the mosaic, tinted by `source_id`) is nonetheless visibly well-registered: the checker
//!   grid lines run continuously across the camera0/camera1 tint boundary, with no visible
//!   tearing — the large p95 is a property of the metric (raw intensity difference on a
//!   razor-edge target) more than of the geometry.
//!
//! This is a demo number, not a CI gate (see `tests/mosaic.rs` for the gated synthetic
//! fixture, whose antialiased fiducials are exactly the case where a small p95 *is* the
//! expected, meaningful signal) — self-asserting only on "loads, composites, produces
//! nonzero overlap".
//!
//! ## Run
//! ```text
//! cargo run --release -p vision-metrology --example birdseye_mosaic
//! WRITE_ASSETS=1 cargo run --release -p vision-metrology --example birdseye_mosaic
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use nalgebra::{Translation3, UnitQuaternion};

use vision_metrology::metric::{
    CameraModel, PlaneGrid, Pose3, distort_pixel, io::import_table_calibration, plane_grid_map,
};
use vision_metrology::warp::{Interp, Map};
use vision_metrology::{BorderMode, Image, Point2f, Point3f, Vec3f};

const OUT_DIR: &str = "docs/assets";
const OUT_FILE: &str = "birdseye-mosaic.png";
/// Nearest-camera-centre priority requires no real image content — flat
/// gray fills a grid pixel no camera covers.
const UNCOVERED_FILL: u8 = 30;
/// Grid margin beyond camera0's own on-axis undistorted footprint, so the
/// mosaic shows a little of each camera's coverage falloff at the edges.
const FOOTPRINT_MARGIN: f32 = 1.3;
/// Target grid width in pixels — a gallery-sized demo resolution, coarser
/// than the dataset's native ~0.013 mm/px GSD (this is a visual/coverage
/// demo, not a precision measurement — `tests/mosaic.rs` is the accuracy gate).
const TARGET_GRID_W: usize = 900;

#[derive(Parser)]
#[command(about = "Bird's-eye mosaic of two calibrated cameras over their shared target plane")]
struct Cli {
    /// Root of the Table_Calibration dataset (`calibration.json` + `cam1/` + `cam2/`).
    #[arg(long)]
    data_dir: Option<PathBuf>,
}

fn default_data_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join("vision/data/25_09_17_Table_Calibration")
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let data_dir = args.data_dir.unwrap_or_else(default_data_dir);

    let calib_bytes = std::fs::read(data_dir.join("calibration.json"))
        .with_context(|| format!("reading {}/calibration.json", data_dir.display()))?;
    let cams = import_table_calibration(&calib_bytes)
        .map_err(|e| anyhow::anyhow!("import_table_calibration: {e}"))?;
    if cams.len() != 2 {
        bail!(
            "expected exactly 2 cameras in calibration.json, got {}",
            cams.len()
        );
    }
    println!("loaded {} cameras from calibration.json", cams.len());

    // Folder <-> JSON-index pairing: see this file's own doc comment.
    let frame_paths = [
        data_dir
            .join("cam1")
            .join("Cam1_+0.0Rx+0.0Ry+0.0X+0.0Y+0.0Z.png"),
        data_dir
            .join("cam2")
            .join("Cam2_+0.0Rx+0.0Ry+0.0X+0.0Y+0.0Z.png"),
    ];
    let raws: Vec<Image<u8>> = frame_paths
        .iter()
        .map(|p| load_gray(p))
        .collect::<Result<_>>()?;
    for (path, img) in frame_paths.iter().zip(&raws) {
        println!(
            "loaded {} ({}x{})",
            path.display(),
            img.width(),
            img.height()
        );
    }

    // ---- Table plane: shift the reference frame to the two cameras' own convergence point.
    let (standoff, axis_gap) = common_standoff(&cams[0].1, &cams[1].1);
    println!(
        "estimated table standoff: {standoff:.2} mm along camera0's optical axis \
         (cameras' axes pass within {axis_gap:.3} mm of each other there)"
    );
    if standoff <= 0.0 {
        bail!("estimated standoff {standoff} mm is not in front of the cameras");
    }
    let shift = Pose3::from_parts(
        Translation3::new(0.0, 0.0, standoff),
        UnitQuaternion::identity(),
    );
    let shifted: Vec<(CameraModel, Pose3)> = cams.iter().map(|(c, p)| (*c, p * shift)).collect();

    // ---- Grid: centred at the convergence point, sized off camera0's own footprint.
    let cam0 = &shifted[0].0;
    let half_x = standoff * (cam0.intrinsics.cx / cam0.intrinsics.fx) * FOOTPRINT_MARGIN;
    let half_y = standoff * (cam0.intrinsics.cy / cam0.intrinsics.fy) * FOOTPRINT_MARGIN;
    let mm_per_px = (2.0 * half_x) / TARGET_GRID_W as f32;
    let grid = PlaneGrid {
        origin_mm: Point2f::new(-half_x, -half_y),
        mm_per_px,
        w: TARGET_GRID_W,
        h: ((2.0 * half_y) / mm_per_px).round().max(1.0) as usize,
    };
    println!(
        "grid: {}x{} px, {:.4} mm/px, covering x in [{:.1}, {:.1}] y in [{:.1}, {:.1}] mm",
        grid.w,
        grid.h,
        grid.mm_per_px,
        grid.origin_mm.x,
        grid.origin_mm.x + grid.w as f32 * grid.mm_per_px,
        grid.origin_mm.y,
        grid.origin_mm.y + grid.h as f32 * grid.mm_per_px,
    );

    // ---- Rectify each camera onto the shared grid.
    let mut rectified: Vec<Image<u8>> = Vec::with_capacity(2);
    let mut masks: Vec<Vec<u8>> = Vec::with_capacity(2);
    for ((camera, pose), raw) in shifted.iter().zip(&raws) {
        let map: Map = plane_grid_map(camera, pose, &grid);
        let mut dst = vec![UNCOVERED_FILL; grid.w * grid.h];
        let mut mask = vec![0u8; grid.w * grid.h];
        map.apply_with_mask(
            &raw.as_view(),
            &mut dst,
            &mut mask,
            Interp::Bilinear,
            BorderMode::Constant(UNCOVERED_FILL),
        )
        .context("apply_with_mask")?;
        rectified.push(Image::from_vec(grid.w, grid.h, dst).expect("valid image"));
        masks.push(mask);
    }

    // ---- Composite: nearest-camera-centre priority, no blending.
    let (mosaic, source_id) = composite_nearest_camera(&grid, &shifted, &rectified, &masks);

    // ---- Report: per-camera coverage, overlap fraction, seam disparity p95.
    let n = grid.w * grid.h;
    for (c, mask) in masks.iter().enumerate() {
        let covered = mask.iter().filter(|&&m| m == 255).count();
        println!(
            "camera{c} coverage: {covered}/{n} px ({:.1}%)",
            100.0 * covered as f32 / n as f32
        );
    }
    let union = (0..n)
        .filter(|&i| masks.iter().any(|m| m[i] == 255))
        .count();
    let overlap = (0..n)
        .filter(|&i| masks.iter().filter(|m| m[i] == 255).count() >= 2)
        .count();
    let overlap_fraction = if union > 0 {
        overlap as f32 / union as f32
    } else {
        0.0
    };
    println!(
        "union coverage: {union}/{n} px ({:.1}%); overlap (>=2 cameras): {overlap}/{union} px ({:.1}% of covered area)",
        100.0 * union as f32 / n as f32,
        100.0 * overlap_fraction,
    );
    let mut disparities: Vec<f32> = (0..n)
        .filter(|&i| masks.iter().filter(|m| m[i] == 255).count() >= 2)
        .map(|i| {
            let vals: Vec<f32> = (0..masks.len())
                .filter(|&c| masks[c][i] == 255)
                .map(|c| rectified[c].data()[i] as f32)
                .collect();
            vals.iter().cloned().fold(f32::MIN, f32::max)
                - vals.iter().cloned().fold(f32::MAX, f32::min)
        })
        .collect();
    if disparities.is_empty() {
        println!("no jointly-valid pixels -- seam disparity not defined");
    } else {
        disparities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p95_idx = (((disparities.len() as f32) * 0.95) as usize).min(disparities.len() - 1);
        let p95 = disparities[p95_idx];
        println!(
            "seam disparity: n={}, p50={:.2}, p95={p95:.2}, max={:.2}",
            disparities.len(),
            disparities[disparities.len() / 2],
            disparities.last().unwrap(),
        );
    }

    // Self-assertion: loads, composites, produces nonzero overlap (roadmap
    // plan's stated acceptance for this real-data demo — no accuracy gate).
    if overlap == 0 {
        bail!("no overlapping coverage between the two cameras at the estimated standoff");
    }
    println!("OK: mosaic built, {overlap} px of camera overlap");

    if std::env::var_os("WRITE_ASSETS").is_some() {
        std::fs::create_dir_all(OUT_DIR).context("create docs/assets")?;
        let out_path = format!("{OUT_DIR}/{OUT_FILE}");
        write_gallery_png(&out_path, &grid, &mosaic, &source_id)?;
        println!("wrote {out_path}");
    }

    Ok(())
}

// ── Table-plane standoff: closest approach of the two cameras' own optical axes ──

/// `(standoff_mm, axis_gap_mm)`: the z-coordinate (along camera0's own axis,
/// since `pose0` is the identity per `import_table_calibration`'s
/// reference-frame convention) where the two cameras' optical axes pass
/// closest to each other, and how close that closest approach actually is —
/// a small gap is evidence the two cameras really do converge on one target
/// rather than this being a meaningless "closest points of two unrelated
/// lines" number. See this file's own doc comment for why this is the right
/// proxy for "where the calibration target/table plane is" when the dataset
/// records no such frame directly.
fn common_standoff(pose0: &Pose3, pose1: &Pose3) -> (f32, f32) {
    // Camera i's own centre and optical-axis direction (its own `+z`),
    // expressed in the reference frame: `pose` is camera-from-reference, so
    // its inverse carries the camera's own origin/`+z` back into it.
    let axis = |pose: &Pose3| -> (Point3f, Vec3f) {
        let inv = pose.inverse();
        let origin = inv * Point3f::origin();
        let dir = (inv * Vec3f::new(0.0, 0.0, 1.0)).normalize();
        (origin, dir)
    };
    let (o0, d0) = axis(pose0);
    let (o1, d1) = axis(pose1);

    // Closest approach between two 3-D lines `o + t*d` (standard formula).
    let w0 = o0 - o1;
    let a = d0.dot(&d0);
    let b = d0.dot(&d1);
    let c = d1.dot(&d1);
    let d = d0.dot(&w0);
    let e = d1.dot(&w0);
    let denom = a * c - b * b;
    let (t, s) = if denom.abs() > 1e-9 {
        ((b * e - c * d) / denom, (a * e - b * d) / denom)
    } else {
        (0.0, 0.0) // parallel axes -- degenerate, caller checks standoff > 0
    };
    let p0 = o0 + d0 * t;
    let p1 = o1 + d1 * s;
    let gap = (p1 - p0).norm();
    let midpoint_z = 0.5 * (p0.z + p1.z);
    (midpoint_z, gap)
}

// ── Compositing: nearest-camera-centre priority, no blending ─────────────
//
// Intentionally duplicated from `tests/mosaic.rs` rather than shared — see
// this file's own doc comment. Kept in lock-step: any change to the rule
// here should be mirrored there (and vice versa).

fn composite_nearest_camera(
    g: &PlaneGrid,
    cams: &[(CameraModel, Pose3)],
    rectified: &[Image<u8>],
    masks: &[Vec<u8>],
) -> (Image<u8>, Vec<u8>) {
    let n = g.w * g.h;
    let mut out = vec![UNCOVERED_FILL; n];
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

// ── I/O ────────────────────────────────────────────────────────────────

fn load_gray(path: &std::path::Path) -> Result<Image<u8>> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Image::from_vec(w, h, img.into_raw()).map_err(|e| anyhow::anyhow!("{e}"))
}

/// The composited mosaic, tinted per camera (camera0 blue-ish, camera1
/// orange-ish, uncovered dark gray) — the same "source_id tint" overlay the
/// lab's Bird's-eye tab offers, rendered once here for the README gallery:
/// one image that shows both the rectified geometry and which camera each
/// pixel traces back to.
fn write_gallery_png(
    path: &str,
    g: &PlaneGrid,
    mosaic: &Image<u8>,
    source_id: &[u8],
) -> Result<()> {
    const PALETTE: [[f32; 3]; 2] = [[80.0, 140.0, 255.0], [255.0, 160.0, 60.0]];
    let mut canvas = image::RgbImage::from_pixel(g.w as u32, g.h as u32, image::Rgb([18, 18, 22]));

    for y in 0..g.h {
        for x in 0..g.w {
            let i = y * g.w + x;
            let v = mosaic.data()[i];
            let id = source_id[i];
            let rgb = if id == 255 {
                [24u8, 24, 28]
            } else {
                let base = PALETTE[id as usize % PALETTE.len()];
                let k = v as f32 / 255.0;
                [
                    (base[0] * k).round().clamp(0.0, 255.0) as u8,
                    (base[1] * k).round().clamp(0.0, 255.0) as u8,
                    (base[2] * k).round().clamp(0.0, 255.0) as u8,
                ]
            };
            canvas.put_pixel(x as u32, y as u32, image::Rgb(rgb));
        }
    }

    canvas
        .save(path)
        .with_context(|| format!("writing {path}"))?;
    Ok(())
}
