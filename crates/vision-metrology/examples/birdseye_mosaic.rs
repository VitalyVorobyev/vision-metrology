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
//! target's nominal, centred pose.
//!
//! **Camera-index judgment call.** `calibration.json` names its two cameras `camera0`/
//! `camera1` (0-indexed); the image folders are `cam1`/`cam2` (1-indexed). Nothing in the
//! dataset states the correspondence explicitly, so this example takes the only sensible
//! reading — `import_table_calibration`'s index-sorted `camera0` ↔ `cam1/`, `camera1` ↔
//! `cam2/` — documented here rather than asserted as fact, the same posture
//! `metric::io::table` itself takes for this dataset's unit and reference-frame calls.
//!
//! ## The target plane has to be *measured*, not assumed
//! `import_table_calibration` fixes `camera0`'s own frame as the reference frame (its
//! `sensor2camera` is the identity — see that importer's docs). `z = 0` there sits *at
//! camera0's own optical center* — a homography through it is singular, and physically
//! there is no "table" there at all. `calibration.json` carries intrinsics, the
//! camera-to-camera extrinsics and a hand-eye transform, but **no target pose**, so the
//! plane the two cameras share has to be recovered from the images themselves.
//!
//! An earlier version of this example guessed it: it took [`common_standoff`] — where the
//! two cameras' optical axes pass closest — and shifted the reference frame there by a
//! **pure translation** along camera0's `z`, i.e. it assumed the target is perpendicular to
//! camera0's optical axis. The distance was about right, the orientation was never measured,
//! and the result was not a mosaic at all: each camera picked up its own projective error
//! from being rectified onto the wrong plane, so the two halves of the composite came out at
//! visibly different scales with the checker grid stepping across the seam.
//!
//! What this example does instead ([`estimate_target_plane`]), in two stages:
//!
//! **Stage 1 — find the orientation by direct search ([`sweep_tilt`]).** Nothing in the
//! calibration constrains it, and the tracker cannot bootstrap itself into it: under a plane
//! this wrong the two rectified views disagree by a *differential* warp that smears every
//! usable correlation window (see [`SWEEP_GRID_W`]'s docs for the arithmetic). So candidate
//! orientations are scored directly — rectify both cameras onto the candidate, track a few
//! windows, keep the best median ZNCC. The score is translation-invariant *because* it
//! tracks: a wrong plane **distance** shifts one view almost rigidly against the other,
//! which the tracker absorbs, while a wrong **tilt** warps it, which the score reports. That
//! is what keeps this a 2-parameter search.
//!
//! **Stage 2 — solve for the plane, coarse to fine ([`ROUNDS`]).** Per round:
//!
//! 1. Rectify both cameras onto the current plane estimate.
//! 2. Track a lattice of windows from camera0's rectified view into camera1's with
//!    [`displacement`] (corrmatch ZNCC + Lucas-Kanade). Both views are already on a *shared*
//!    grid, so once the estimate is close the residual is a small local translation —
//!    exactly what a translation-only tracker can measure.
//! 3. Turn every accepted match into a correspondence between the two cameras' **normalized
//!    (undistorted) ray directions**. This step is exact and independent of how wrong the
//!    current plane guess still is: rectified pixel `g` in camera0 and rectified pixel
//!    `g + shift` in camera1 showing the same feature means those two cameras' rays through
//!    those two grid points meet at one physical point, whatever plane the grid was laid on.
//! 4. Solve for the plane. With `R`/`t` (camera1-from-camera0) known from the calibration,
//!    the plane-induced map is `x1 ~ (R + t·vᵀ) x0` with `v = n/d` — **linear in the three
//!    unknowns**, so a RANSAC + least-squares solve recovers `n` and `d` outright, with none
//!    of the two-solution ambiguity a homography *decomposition* would carry.
//!
//! Rounds run on progressively finer grids with shrinking search radii; the tracked residual
//! ends well under a grid pixel at this dataset's native GSD.
//!
//! The recovered plane then becomes the mosaic's reference frame: this example composes a
//! **new** camera-from-(plane) pose `pose ∘ shift` per camera, where `shift` is a full
//! isometry whose `z = 0` plane *is* the measured target plane, and hands that to the
//! existing, unmodified `metric`/`warp` API — no library change, per the plan's "the library
//! already has everything" constraint.
//!
//! ## The seam metric is ZNCC, and it gates the run
//! Registration quality is reported as **ZNCC between the two rectified images over their
//! jointly-valid overlap**, and the example fails below [`MIN_OVERLAP_ZNCC`]. The obvious
//! alternative — `max − min` raw intensity across the cameras valid at a pixel — is useless
//! on this target: it is a hard-edged coded checkerboard, so *any* sub-pixel disagreement
//! swings a near-edge pixel over the whole 0-255 range and the statistic saturates whether
//! the mosaic is registered or not. ZNCC is a whole-region agreement measure instead: it
//! reads ~1 when the two views line up and collapses toward 0 when they do not.
//!
//! ## Measured (2026-08-20, this dataset, this frame pair, M4 Pro, release, ~16 s)
//! Run with `WRITE_ASSETS=1` to also regenerate `docs/assets/birdseye-mosaic.png` (every
//! stage is deterministic — the sweep is exhaustive and the RANSAC uses a fixed-seed
//! xorshift — so the asset is reproducible run to run).
//!
//! - Seed: common standoff **273.90 mm** along camera0's own optical axis, the two axes
//!   passing within **0.406 mm** of each other there — a tight convergence for a ~111mm
//!   baseline, evidence this dataset really is a verged pair aimed at one target.
//! - Tilt sweep: 729 candidates on a 320x240 grid at 0.0613 mm/px, best median window ZNCC
//!   **0.943**, landing on **39.87°** (the refinement rounds then pull it to 37.81°). A
//!   finer 0.05-step probe of the same landscape scores 0.969 at the peak, ~0.90 one step
//!   away, and under 0.2 anywhere outside a ±0.1 neighbourhood in tangent — which is what
//!   [`SWEEP_TANGENT_STEP`] is set from.
//! - Estimated target plane, in camera0's frame: `n = (0.1584, 0.5923, 0.7900)`, i.e.
//!   **37.81° from camera0's optical axis**, piercing that axis at **276.35 mm**. That tilt
//!   is the whole story: it is far too large for a pure-translation guess to absorb, which is
//!   why the old version's composite was not a mosaic.
//! - Convergence: round 1 tracks 44/44 windows and keeps 26 as inliers; rounds 2 and 3 keep
//!   **all** of them (46/46, 47/47). Last round, on a 2800x2848 grid at 0.0128 mm/px (this
//!   dataset's own GSD): tracked residual median **0.12 grid px**, fit reprojection p50
//!   **0.11 px** in camera1 pixels — below the **0.63 px** `calibration.json` itself reports
//!   for camera1's extrinsic.
//! - Grid: 900x916 px at 0.0397 mm/px (gallery-sized — a deliberate ~3x downsample of the
//!   native GSD; this is a coverage/visual demo, not a precision measurement).
//! - Coverage: camera0 **50.9%**, camera1 **51.0%** of the grid; union **55.9%**; overlap
//!   (≥ 2 cameras valid) **82.4% of the covered area**. This is a **converged stereo pair**,
//!   not a wide-table array of mostly-disjoint cameras — both cameras see nearly the same
//!   patch of the calibration target, so most of what each camera covers, the other does too.
//!   The grid is the bounding box of camera0's footprint *quadrilateral* on a plane tilted
//!   38° away from it, plus `FOOTPRINT_MARGIN`, so a good part of that box was never going to
//!   be covered — which is most of why coverage reads around half.
//! - **Overlap ZNCC: 0.9927** (`n=379487` jointly-valid pixels), against the
//!   [`MIN_OVERLAP_ZNCC`] `= 0.75` gate. The remaining gap from 1.0 is resampling, not
//!   geometry: the gallery grid is a ~3x downsample of a target whose finest features are a
//!   few pixels wide, and each camera aliases it at its own sub-pixel phase.
//!   For scale: the plane the old version *assumed* scores **0.0656** on these same two
//!   frames.
//!
//! `tests/mosaic.rs` remains the CI-gated accuracy fixture (synthetic, exact geometry); this
//! example is a real-data demo that now refuses to publish a mosaic it cannot measure as
//! registered.
//!
//! ## Run
//! ```text
//! cargo run --release -p vision-metrology --example birdseye_mosaic
//! WRITE_ASSETS=1 cargo run --release -p vision-metrology --example birdseye_mosaic
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use nalgebra::{Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};

use vision_metrology::corr::{DisplacementConfig, Refine, displacement};
use vision_metrology::metric::{
    CameraModel, PlaneGrid, Pose3, distort_pixel, io::import_table_calibration, plane_grid_map,
    undistort_pixel,
};
use vision_metrology::warp::{Interp, Map};
use vision_metrology::{BorderMode, Image, Point2f, Point3f, Rect2f, Vec3f};

const OUT_DIR: &str = "docs/assets";
const OUT_FILE: &str = "birdseye-mosaic.png";
/// Nearest-camera-centre priority requires no real image content — flat
/// gray fills a grid pixel no camera covers.
const UNCOVERED_FILL: u8 = 30;
/// Grid margin beyond camera0's own footprint on the estimated plane, so the
/// mosaic shows a little of each camera's coverage falloff at the edges.
const FOOTPRINT_MARGIN: f32 = 1.3;
/// Target grid width in pixels — a gallery-sized demo resolution, coarser
/// than the dataset's native ~0.013 mm/px GSD (this is a visual/coverage
/// demo, not a precision measurement — `tests/mosaic.rs` is the accuracy gate).
const TARGET_GRID_W: usize = 900;

/// Coarse tilt sweep (the bootstrap), all in one place.
///
/// The sweep exists because the tracker cannot bootstrap itself here. Under a
/// wrong plane the two rectified views disagree by a *differential* warp, and
/// the fraction of a window that warp smears is `≈ 0.42 · tilt_error ·
/// window_mm / feature_mm` — with this target's 0.87 mm checker pitch and a
/// ~38° seed error, every usable window is smeared past recognition, whatever
/// resolution or window size it is tried at. So the tilt is found first by
/// direct search: rectify both cameras onto a candidate plane, track a few
/// windows, and keep the candidate whose windows agree best.
///
/// The score has to be **translation-invariant**, and tracking gives that for
/// free: a wrong plane *distance* shifts one rectified view almost rigidly
/// against the other, which a tracker absorbs into its own search, while a
/// wrong plane *tilt* warps it, which the tracker's ZNCC score reports. That
/// is what makes this a 2-parameter search rather than a 3-parameter one.
const SWEEP_GRID_W: usize = 320;
/// Sweep grid extent, as a fraction of camera0's *on-axis* footprint
/// (`standoff · cx/fx`). Deliberately not [`footprint_grid`]: a fixed extent
/// centred on camera0's optical axis keeps every candidate's cost and
/// sampling identical, and stays inside the real footprint for any tilt this
/// sweep considers.
const SWEEP_EXTENT: f32 = 0.75;
/// Sweep tracker window and search radius, in sweep-grid pixels — about
/// 4 mm and ±5 mm of target at this grid's ~0.06 mm/px. The sweep never reads
/// the tracked *shift*, only its score, so the search radius only has to be
/// wide enough that a candidate near the answer can find its match at all.
const SWEEP_WINDOW_PX: f32 = 64.0;
const SWEEP_SEARCH_PX: i32 = 80;
/// Tilt search range and step, as `n.x/n.z` / `n.y/n.z` tangents. `1.0` is
/// 45° from camera0's optical axis — past what a calibration target this pair
/// both fills its frame with could plausibly be at. The step is set by the
/// score's own capture range (measured: the peak falls off over ~0.1 of
/// tangent), so `0.08` cannot step over the peak.
const SWEEP_TANGENT_MAX: f32 = 1.0;
const SWEEP_TANGENT_STEP: f32 = 0.08;
/// Sweep windows that must be jointly covered for a candidate to be scored,
/// and the score the winner must beat. A tilt that is merely *near* the
/// answer already scores ~0.5 here, so this gate only catches a sweep that
/// found nothing at all.
const SWEEP_MIN_WINDOWS: usize = 3;
const SWEEP_MIN_SCORE: f32 = 0.5;

/// Refinement schedule: `(grid width, tracker window, search radius)` per
/// round, in grid pixels.
///
/// Round 1 has to cover the whole error the sweep leaves behind — the plane's
/// *distance* is still the seed's, and that shows up as a bulk shift of a
/// couple of millimetres — hence the wide `±200 px` search. Later rounds
/// tighten the search and refine the grid toward this dataset's own
/// ~0.013 mm/px GSD, which is what bounds how precisely a window centre can be
/// located and hence the fit.
///
/// **None of these grids is much coarser than that GSD, and that is
/// deliberate.** A coarse grid looks tempting (a small pixel window spans more
/// target, so it is more unique) but it is a trap on this target: the grid is
/// resampled with plain bilinear interpolation, so downsampling aliases away
/// the unique glyph inside each white square and leaves the bare, *periodic*
/// checker — against which a wide search happily locks onto the wrong period.
/// Uniqueness here comes from resolving the glyphs, not from spanning more
/// squares. (Tried: a 600-px round-1 grid at 0.059 mm/px tracked 22/22 windows
/// and the fit kept 4 of them.)
const ROUNDS: [(usize, f32, i32); 3] = [(1800, 192.0, 200), (2400, 256.0, 80), (2800, 256.0, 30)];
/// Window lattice over the jointly-covered part of the grid, `(cols, rows)`.
const LATTICE: (usize, usize) = (11, 11);
/// Minimum corrmatch ZNCC for a tracked window to become a correspondence.
/// Low on purpose: it is only a floor against tracking pure fill or noise —
/// RANSAC, not this threshold, is what rejects a confident wrong match. Round
/// 1 still tracks across the sweep's leftover tilt quantization and does throw
/// mismatches (measured: 26 of 44 windows kept as inliers); by round 2 every
/// window is an inlier.
const MIN_WINDOW_SCORE: f32 = 0.3;
/// RANSAC sample count and sample size for the `v = n/d` fit. Three
/// correspondences over-determine the 3 unknowns (2 independent equations
/// each), which keeps a minimal sample from being a near-degenerate pair.
const RANSAC_ROUNDS: usize = 400;
const RANSAC_SAMPLE: usize = 3;
/// RANSAC inlier threshold, in camera1 pixels. `calibration.json` reports
/// 0.63 px reprojection error for camera1's own extrinsic, and the tracked
/// window centres carry the estimation grid's own resampling error on top,
/// so anything under a few pixels is consistent with a correct plane.
const RANSAC_THRESH_PX: f32 = 4.0;
/// Minimum correspondences, and minimum inliers among them, for a round's fit
/// to be trusted. The tightest round measured here keeps 26; a fit resting on
/// fewer than this is reporting a coincidence, not a plane.
const MIN_INLIERS: usize = 15;
/// Registration gate on the composited overlap (see this file's doc comment
/// on why ZNCC and not `max − min` intensity). The measured value on this
/// dataset is **0.9927**; the pure-translation plane this example used to
/// assume scores **0.0656** on the very same two frames. The gate sits far
/// below the good value and far above the bad one, so it cannot be tripped by
/// ordinary resampling noise but catches any regression that loses the plane.
const MIN_OVERLAP_ZNCC: f32 = 0.75;

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

    // ---- Seed: the two cameras' own convergence point, distance only.
    let (standoff, axis_gap) = common_standoff(&cams[0].1, &cams[1].1);
    println!(
        "seed standoff: {standoff:.2} mm along camera0's optical axis \
         (cameras' axes pass within {axis_gap:.3} mm of each other there)"
    );
    if standoff <= 0.0 {
        bail!("seed standoff {standoff} mm is not in front of the cameras");
    }
    let seed = TargetPlane {
        n: Vec3f::new(0.0, 0.0, 1.0),
        d: standoff,
    };

    // ---- Measure the plane the two cameras actually share.
    let plane = estimate_target_plane(&cams, &raws, seed)?;
    println!(
        "estimated target plane (camera0 frame): n = ({:.4}, {:.4}, {:.4}), \
         {:.2}° from camera0's optical axis, piercing it at {:.2} mm",
        plane.n.x,
        plane.n.y,
        plane.n.z,
        plane.tilt_deg(),
        plane.axis_standoff(),
    );

    // ---- Grid: camera0's footprint on the measured plane, plus a margin.
    let shift = plane_shift(&cams[0].1, &plane);
    let shifted: Vec<(CameraModel, Pose3)> = cams.iter().map(|(c, p)| (*c, p * shift)).collect();
    let grid = footprint_grid(
        &shifted[0].0,
        &shifted[0].1,
        raws[0].width(),
        raws[0].height(),
        TARGET_GRID_W,
    )?;
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
    let (rectified, masks) = rectify_all(&shifted, &raws, &grid)?;

    // ---- Composite: nearest-camera-centre priority, no blending.
    let (mosaic, source_id) = composite_nearest_camera(&grid, &shifted, &rectified, &masks);

    // ---- Report: per-camera coverage, overlap fraction, overlap ZNCC.
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
    if overlap == 0 {
        bail!("no overlapping coverage between the two cameras on the estimated plane");
    }

    let (zncc, zncc_n) = overlap_zncc(&rectified[0], &rectified[1], &masks[0], &masks[1])
        .context("overlap ZNCC is undefined: one camera's overlap region has no contrast")?;
    println!("overlap ZNCC: {zncc:.4} (n={zncc_n} jointly-valid px, gate {MIN_OVERLAP_ZNCC})");
    if zncc < MIN_OVERLAP_ZNCC {
        bail!(
            "overlap ZNCC {zncc:.4} is below the {MIN_OVERLAP_ZNCC} registration gate — the two \
             cameras' rectified views do not agree, so this is not a mosaic"
        );
    }
    println!("OK: mosaic built and registered, {overlap} px of camera overlap");

    if std::env::var_os("WRITE_ASSETS").is_some() {
        std::fs::create_dir_all(OUT_DIR).context("create docs/assets")?;
        let out_path = format!("{OUT_DIR}/{OUT_FILE}");
        write_gallery_png(&out_path, &grid, &mosaic, &source_id)?;
        println!("wrote {out_path}");
    }

    Ok(())
}

// ── The target plane ────────────────────────────────────────────────────

/// The target plane in **camera0's own frame**: unit normal `n` pointing from
/// the camera toward the plane, and `d > 0` such that plane points satisfy
/// `n · X = d`.
///
/// Not [`Plane3`](vision_metrology::metric::Plane3) (which is `n · X + d = 0`
/// in the *reference* frame) because everything this example solves is
/// naturally posed in camera0's frame — the linear unknown of the fit is
/// `v = n/d` there. [`plane_shift`] is the one place the two meet.
#[derive(Clone, Copy, Debug)]
struct TargetPlane {
    n: Vec3f,
    d: f32,
}

impl TargetPlane {
    /// Recovers `(n, d)` from the fit's linear unknown `v = n/d`.
    ///
    /// `d = 1/‖v‖` is positive by construction, so the sign check that
    /// matters is `n.z > 0`: a plane camera0's own optical axis can actually
    /// pierce in front of it.
    fn from_v(v: Vector3<f64>) -> Result<Self> {
        let norm = v.norm();
        if !norm.is_finite() || norm <= 0.0 {
            bail!("plane fit returned a degenerate v = n/d (plane at infinity)");
        }
        let n = (v / norm).cast::<f32>();
        let plane = Self {
            n: Vec3f::new(n.x, n.y, n.z),
            d: (1.0 / norm) as f32,
        };
        if plane.n.z <= 0.1 {
            bail!(
                "plane fit returned a normal {:.3} away from camera0's optical axis — \
                 that is not a target this pair can both see",
                plane.tilt_deg()
            );
        }
        Ok(plane)
    }

    /// Angle between the plane normal and camera0's optical axis, degrees.
    fn tilt_deg(&self) -> f32 {
        self.n.z.clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// Where camera0's own optical axis pierces the plane, in mm. Comparable
    /// to [`common_standoff`]'s seed (`d` itself is the *perpendicular*
    /// distance, which is shorter whenever the plane is tilted).
    fn axis_standoff(&self) -> f32 {
        self.d / self.n.z
    }
}

/// `reference-from-plane`: the isometry to compose onto each camera pose so
/// that the [`PlaneGrid`]'s `z = 0` plane *is* `plane`.
///
/// Composed the same way the old pure-translation guess was — `pose ∘ shift`
/// per camera, feeding the unmodified `metric`/`warp` API — but with the
/// measured orientation rather than camera0's inherited one.
///
/// The frame's own axes are pinned deterministically: `z` is the plane
/// normal, `x` is camera0's own `x` projected into the plane, `y = z × x`.
/// Seeding from camera0 keeps the bird's-eye view in roughly the orientation
/// camera0 sees, so the gallery asset does not flip or spin when the estimate
/// shifts slightly.
fn plane_shift(pose0: &Pose3, plane: &TargetPlane) -> Pose3 {
    let ez = plane.n.normalize();
    let seed = Vec3f::new(1.0, 0.0, 0.0);
    let ex = (seed - ez * seed.dot(&ez)).normalize();
    let ey = ez.cross(&ex);
    // Origin: where camera0's optical axis pierces the plane, so the grid
    // stays centred on what camera0 is actually looking at.
    let origin = Vec3f::new(0.0, 0.0, plane.axis_standoff());
    let rot = Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[ex, ey, ez]));
    let cam0_from_plane = Pose3::from_parts(
        Translation3::from(origin),
        UnitQuaternion::from_rotation_matrix(&rot),
    );
    pose0.inverse() * cam0_from_plane
}

/// `(standoff_mm, axis_gap_mm)`: the z-coordinate (along camera0's own axis,
/// since `pose0` is the identity per `import_table_calibration`'s
/// reference-frame convention) where the two cameras' optical axes pass
/// closest to each other, and how close that closest approach actually is —
/// a small gap is evidence the two cameras really do converge on one target
/// rather than this being a meaningless "closest points of two unrelated
/// lines" number.
///
/// This is the **seed** for [`estimate_target_plane`], not the answer: it
/// fixes a distance and says nothing at all about the plane's orientation
/// (see this file's doc comment for what assuming that orientation cost).
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

// ── Plane estimation: track, correspond, solve ──────────────────────────

/// One tracked window turned into a ray-to-ray correspondence: normalized
/// (undistorted) camera coordinates `(x, y, 1)` in camera0 and camera1 that
/// see the same physical point on the target.
struct RayCorr {
    x0: Vector3<f64>,
    x1: Vector3<f64>,
}

/// Estimates the shared target plane from the two frames, seeded at `seed`.
///
/// Two stages: [`sweep_tilt`] to find the orientation (which nothing in the
/// calibration constrains and the tracker cannot bootstrap), then
/// [`ROUNDS`] coarse-to-fine rounds of track → correspond → linear fit for
/// the precise plane. See this file's doc comment for why each step is exact
/// rather than approximate.
///
/// Prints one line per round, so a run on a new dataset shows whether it
/// converged rather than silently returning whatever the last least-squares
/// solve produced.
fn estimate_target_plane(
    cams: &[(CameraModel, Pose3)],
    raws: &[Image<u8>],
    seed: TargetPlane,
) -> Result<TargetPlane> {
    // camera1-from-camera0: the calibration's own relative pose, the `R`/`t`
    // that make the plane fit linear (no homography decomposition needed).
    let rel = cams[1].1 * cams[0].1.inverse();
    let r: Matrix3<f64> = rel.rotation.to_rotation_matrix().matrix().cast::<f64>();
    let t: Vector3<f64> = rel.translation.vector.cast::<f64>();

    let mut plane = sweep_tilt(cams, raws, seed)?;

    for (round, &(grid_w, window, search)) in ROUNDS.iter().enumerate() {
        let shift = plane_shift(&cams[0].1, &plane);
        let shifted: Vec<(CameraModel, Pose3)> =
            cams.iter().map(|(c, p)| (*c, p * shift)).collect();
        let grid = footprint_grid(
            &shifted[0].0,
            &shifted[0].1,
            raws[0].width(),
            raws[0].height(),
            grid_w,
        )?;
        let (rect, masks) = rectify_all(&shifted, raws, &grid)?;

        let (corrs, tracked, shift_p50) =
            collect_correspondences(&shifted, &grid, &rect, &masks, window, search);
        if corrs.len() < MIN_INLIERS {
            bail!(
                "round {}: only {} of {tracked} lattice windows tracked between the two \
                 rectified views — not enough to fit a plane",
                round + 1,
                corrs.len(),
            );
        }

        let (v, inliers, reproj_p50) = fit_plane_ransac(&corrs, &r, &t)?;
        if inliers < MIN_INLIERS {
            bail!(
                "round {}: plane fit kept only {inliers} of {} correspondences as inliers",
                round + 1,
                corrs.len(),
            );
        }
        plane = TargetPlane::from_v(v)?;
        println!(
            "  round {}: grid {}x{} @ {:.4} mm/px, window {window} px, search ±{search} px | \
             {}/{tracked} windows tracked, residual p50 {shift_p50:.2} grid px | \
             tilt {:.2}°, axis standoff {:.2} mm, {inliers}/{} inliers, \
             reprojection p50 {reproj_p50:.2} px",
            round + 1,
            grid.w,
            grid.h,
            grid.mm_per_px,
            corrs.len(),
            plane.tilt_deg(),
            plane.axis_standoff(),
            corrs.len(),
        );
    }
    Ok(plane)
}

/// Coarse 2-D search for the plane's **orientation**, at the seed distance.
///
/// See [`SWEEP_GRID_W`]'s docs for why this stage has to exist and why two
/// parameters are enough. Returns the best-scoring plane; the distance it
/// carries is still the seed's, which the refinement rounds then correct.
fn sweep_tilt(
    cams: &[(CameraModel, Pose3)],
    raws: &[Image<u8>],
    seed: TargetPlane,
) -> Result<TargetPlane> {
    let standoff = seed.axis_standoff();
    let cam0 = &cams[0].0;
    // Fixed grid geometry, shared by every candidate: camera0's on-axis
    // footprint at the seed distance, scaled by `SWEEP_EXTENT`.
    let half_x = SWEEP_EXTENT * standoff * (cam0.intrinsics.cx / cam0.intrinsics.fx);
    let half_y = SWEEP_EXTENT * standoff * (cam0.intrinsics.cy / cam0.intrinsics.fy);
    let mm_per_px = 2.0 * half_x / SWEEP_GRID_W as f32;
    let grid = PlaneGrid {
        origin_mm: Point2f::new(-half_x, -half_y),
        mm_per_px,
        w: SWEEP_GRID_W,
        h: ((2.0 * half_y) / mm_per_px).round().max(1.0) as usize,
    };

    let steps = (SWEEP_TANGENT_MAX / SWEEP_TANGENT_STEP).round() as i32;
    let mut best: Option<(f32, TargetPlane)> = None;
    let mut scored = 0usize;
    for ia in -steps..=steps {
        for ib in -steps..=steps {
            let a = ia as f32 * SWEEP_TANGENT_STEP;
            let b = ib as f32 * SWEEP_TANGENT_STEP;
            let n = Vec3f::new(a, b, 1.0).normalize();
            // Every candidate plane passes through the seed's axis-pierce
            // point, so the sweep only ever changes orientation.
            let candidate = TargetPlane {
                n,
                d: standoff * n.z,
            };
            let shift = plane_shift(&cams[0].1, &candidate);
            let shifted: Vec<(CameraModel, Pose3)> =
                cams.iter().map(|(c, p)| (*c, p * shift)).collect();
            let Ok((rect, masks)) = rectify_all(&shifted, raws, &grid) else {
                continue;
            };
            let Some(score) = sweep_score(&grid, &rect, &masks) else {
                continue;
            };
            scored += 1;
            if best.as_ref().is_none_or(|&(b, _)| score > b) {
                best = Some((score, candidate));
            }
        }
    }

    let (score, plane) = best.context("tilt sweep scored no candidate at all")?;
    println!(
        "tilt sweep: {scored} candidates scored on a {}x{} grid @ {:.4} mm/px; \
         best ZNCC {score:.3} at tilt {:.2}° (n = ({:.4}, {:.4}, {:.4}))",
        grid.w,
        grid.h,
        grid.mm_per_px,
        plane.tilt_deg(),
        plane.n.x,
        plane.n.y,
        plane.n.z,
    );
    if score < SWEEP_MIN_SCORE {
        bail!(
            "tilt sweep's best candidate only scores {score:.3} (gate {SWEEP_MIN_SCORE}) — \
             the two cameras do not appear to be looking at one shared plane"
        );
    }
    Ok(plane)
}

/// One sweep candidate's score: the median tracked ZNCC over a fixed 3x3
/// lattice of windows, or `None` when too few of them are jointly covered.
///
/// Median, not mean: a window that lands on the target's blank border tracks
/// badly no matter how right the plane is, and should not drag a good
/// candidate down.
fn sweep_score(grid: &PlaneGrid, rect: &[Image<u8>], masks: &[Vec<u8>]) -> Option<f32> {
    let half = 0.5 * SWEEP_WINDOW_PX;
    let mut scores = Vec::with_capacity(9);
    for gy in 1..4 {
        for gx in 1..4 {
            let cx = gx as f32 * grid.w as f32 / 4.0;
            let cy = gy as f32 * grid.h as f32 / 4.0;
            if !window_covered(grid, &masks[0], cx, cy, half)
                || !window_covered(grid, &masks[1], cx, cy, half)
            {
                continue;
            }
            let cfg = DisplacementConfig {
                window: Rect2f {
                    x: cx - half,
                    y: cy - half,
                    width: SWEEP_WINDOW_PX,
                    height: SWEEP_WINDOW_PX,
                },
                search: (SWEEP_SEARCH_PX, SWEEP_SEARCH_PX),
                // Subpixel refinement is wasted here: the sweep only reads the
                // stage-1 score, never the shift.
                refine: Refine::None,
                min_score: f32::NEG_INFINITY,
            };
            if let Ok(d) = displacement(&rect[0].as_view(), &rect[1].as_view(), &cfg) {
                scores.push(d.score);
            }
        }
    }
    if scores.len() < SWEEP_MIN_WINDOWS {
        return None;
    }
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(scores[scores.len() / 2])
}

/// Tracks a lattice of windows from `rect[0]` into `rect[1]` and turns each
/// accepted match into a [`RayCorr`].
///
/// Returns `(correspondences, windows_attempted, median_tracked_shift_px)`.
/// The median shift is the honest convergence read-out: it is how far the two
/// rectified views still disagree under the current plane, in grid pixels.
fn collect_correspondences(
    shifted: &[(CameraModel, Pose3)],
    grid: &PlaneGrid,
    rect: &[Image<u8>],
    masks: &[Vec<u8>],
    window: f32,
    search: i32,
) -> (Vec<RayCorr>, usize, f32) {
    let half = 0.5 * window;
    // Only the window itself has to fit — `displacement` clips its search
    // region to the destination image, so a lattice inset by the search
    // radius as well would squeeze every window into the middle of the grid
    // and leave the fit with almost no spread to constrain the tilt.
    let margin = half + 1.0;
    let (cols, rows) = LATTICE;
    let mut corrs = Vec::new();
    let mut shifts = Vec::new();
    let mut attempted = 0usize;

    for row in 0..rows {
        for col in 0..cols {
            let fx = (col as f32 + 0.5) / cols as f32;
            let fy = (row as f32 + 0.5) / rows as f32;
            let cx = margin + fx * (grid.w as f32 - 2.0 * margin);
            let cy = margin + fy * (grid.h as f32 - 2.0 * margin);
            if !(cx > margin - 1.0 && cy > margin - 1.0) {
                continue; // grid too small for this lattice at this search radius
            }
            // Both cameras must actually cover the window; a window half
            // outside camera0's footprint is mostly `UNCOVERED_FILL` and
            // would correlate against the fill, not the target.
            if !window_covered(grid, &masks[0], cx, cy, half)
                || !window_covered(grid, &masks[1], cx, cy, half)
            {
                continue;
            }
            attempted += 1;

            let cfg = DisplacementConfig {
                window: Rect2f {
                    x: cx - half,
                    y: cy - half,
                    width: window,
                    height: window,
                },
                search: (search, search),
                refine: Refine::LucasKanade { iters: 5 },
                min_score: MIN_WINDOW_SCORE,
            };
            let Ok(d) = displacement(&rect[0].as_view(), &rect[1].as_view(), &cfg) else {
                continue;
            };

            // Grid pixel `g` in camera0's rectified view and `g + shift` in
            // camera1's show the same physical point, so the two cameras'
            // rays through those two *grid* points meet there — true whatever
            // plane the grid was laid on, which is what makes this usable
            // while the plane estimate is still wrong.
            let g0 = Point2f::new(cx, cy);
            let g1 = g0 + d.shift;
            shifts.push(d.shift.norm());
            corrs.push(RayCorr {
                x0: grid_ray(&shifted[0].1, grid, g0),
                x1: grid_ray(&shifted[1].1, grid, g1),
            });
        }
    }

    shifts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = shifts.get(shifts.len() / 2).copied().unwrap_or(f32::NAN);
    (corrs, attempted, p50)
}

/// Is the whole `2*half`-square window around `(cx, cy)` inside `mask`?
/// Corners plus centre is enough — the masks are single connected footprints,
/// not speckle.
fn window_covered(grid: &PlaneGrid, mask: &[u8], cx: f32, cy: f32, half: f32) -> bool {
    let probes = [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx - half, cy + half),
        (cx + half, cy + half),
        (cx, cy),
    ];
    probes.iter().all(|&(x, y)| {
        let (ix, iy) = (x.round(), y.round());
        if ix < 0.0 || iy < 0.0 || ix >= grid.w as f32 || iy >= grid.h as f32 {
            return false;
        }
        mask[iy as usize * grid.w + ix as usize] == 255
    })
}

/// The normalized (undistorted) camera-frame ray `(x, y, 1)` through the
/// plane point a (possibly fractional) grid coordinate names.
///
/// Distortion never enters: the grid → raw-pixel map already applied it
/// during rectification, so a grid coordinate is by construction an
/// *undistorted* quantity.
fn grid_ray(pose: &Pose3, grid: &PlaneGrid, g: Point2f) -> Vector3<f64> {
    let p = Point3f::new(
        grid.origin_mm.x + g.x * grid.mm_per_px,
        grid.origin_mm.y + g.y * grid.mm_per_px,
        0.0,
    );
    let c = pose * p;
    Vector3::new((c.x / c.z) as f64, (c.y / c.z) as f64, 1.0)
}

/// Deterministic xorshift64. RANSAC has to give the same answer run to run:
/// `docs/assets/birdseye-mosaic.png` is committed, and a gallery asset that
/// changes with the wind is not reproducible.
struct Rng(u64);

impl Rng {
    fn next_below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// RANSAC + least squares for `v = n/d` in `x1 ~ (R + t·vᵀ) x0`.
///
/// Returns `(v, inlier_count, inlier_reprojection_p50_px)`. The residual is
/// measured where it means something — camera1 **pixels** — not in the
/// algebraic cross-product units the solve itself minimizes.
fn fit_plane_ransac(
    corrs: &[RayCorr],
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
) -> Result<(Vector3<f64>, usize, f32)> {
    // Residuals are compared in camera1 pixels; with `fx ≈ fy` on this
    // dataset a single focal scale is an honest conversion, and it keeps the
    // threshold one number rather than an ellipse.
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let thresh = RANSAC_THRESH_PX as f64;

    let mut best: Option<(Vector3<f64>, Vec<usize>)> = None;
    for _ in 0..RANSAC_ROUNDS {
        let sample: Vec<usize> = (0..RANSAC_SAMPLE)
            .map(|_| rng.next_below(corrs.len()))
            .collect();
        let Some(v) = solve_v(corrs, r, t, &sample) else {
            continue;
        };
        let inliers: Vec<usize> = (0..corrs.len())
            .filter(|&i| reprojection_px(&corrs[i], r, t, &v) <= thresh)
            .collect();
        if best.as_ref().is_none_or(|(_, b)| inliers.len() > b.len()) {
            best = Some((v, inliers));
        }
    }

    let (_, inliers) = best.context("RANSAC found no consistent plane at all")?;
    // Refit on the full consensus set: the minimal-sample model only ever
    // served to *select* inliers.
    let v = solve_v(corrs, r, t, &inliers).context("refit on the inlier set is degenerate")?;
    let mut resid: Vec<f64> = inliers
        .iter()
        .map(|&i| reprojection_px(&corrs[i], r, t, &v))
        .collect();
    resid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = resid[resid.len() / 2] as f32;
    Ok((v, inliers.len(), p50))
}

/// Least-squares `v` from the correspondences `idx` names.
///
/// Each correspondence contributes `[x1]× (R x0 + t (vᵀ x0)) = 0`, which is
/// **linear** in `v` — the whole reason the known `R`/`t` are worth using
/// instead of fitting a free homography and decomposing it. Accumulated as
/// normal equations in `f64` and solved by SVD, so a rank-deficient sample
/// (e.g. three collinear windows) reports failure instead of a wild answer.
fn solve_v(
    corrs: &[RayCorr],
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    idx: &[usize],
) -> Option<Vector3<f64>> {
    let mut ata = Matrix3::<f64>::zeros();
    let mut atb = Vector3::<f64>::zeros();
    for &i in idx {
        let c = &corrs[i];
        let cross = skew(&c.x1);
        let a = cross * (t * c.x0.transpose());
        let b = -(cross * (r * c.x0));
        ata += a.transpose() * a;
        atb += a.transpose() * b;
    }
    let v = ata.svd(true, true).solve(&atb, 1e-14).ok()?;
    v.iter().all(|x| x.is_finite()).then_some(v)
}

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Reprojection error of one correspondence under plane `v`, in camera1
/// pixels (scaled by camera1's focal length, `FOCAL1_PX`).
fn reprojection_px(c: &RayCorr, r: &Matrix3<f64>, t: &Vector3<f64>, v: &Vector3<f64>) -> f64 {
    let mapped = r * c.x0 + t * v.dot(&c.x0);
    if mapped.z.abs() < 1e-9 {
        return f64::INFINITY;
    }
    let dx = mapped.x / mapped.z - c.x1.x;
    let dy = mapped.y / mapped.z - c.x1.y;
    FOCAL1_PX * (dx * dx + dy * dy).sqrt()
}

/// Camera1's focal length in pixels, used only to express the fit's residual
/// in pixels. Read from `calibration.json` (`21343.4`) rather than threaded
/// through every call: this is a fixed-dataset example, and the number's only
/// job is to make a threshold and a printed residual legible.
const FOCAL1_PX: f64 = 21343.4;

// ── Grid and rectification ─────────────────────────────────────────────

/// A [`PlaneGrid`] covering camera0's own footprint on the plane (`pose`'s
/// `z = 0`), plus [`FOOTPRINT_MARGIN`].
///
/// The footprint is the quadrilateral camera0's four image corners cut out of
/// the plane, not a `standoff × cx/fx` half-width: once the plane is tilted
/// (and on this dataset it is, substantially), that rectangle-around-the-axis
/// shortcut is wrong on both extent and centre.
fn footprint_grid(
    camera: &CameraModel,
    pose: &Pose3,
    img_w: usize,
    img_h: usize,
    target_w: usize,
) -> Result<PlaneGrid> {
    let inv = pose.inverse(); // plane-from-camera
    let c = inv * Point3f::origin();
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    let corners = [
        (0.0, 0.0),
        (img_w as f32 - 1.0, 0.0),
        (0.0, img_h as f32 - 1.0),
        (img_w as f32 - 1.0, img_h as f32 - 1.0),
    ];
    for (px, py) in corners {
        let nrm = undistort_pixel(camera, Point2f::new(px, py));
        let dir = inv.rotation * Vec3f::new(nrm.x, nrm.y, 1.0);
        if dir.z.abs() < 1e-6 {
            bail!("camera corner ray is parallel to the estimated target plane");
        }
        let s = -c.z / dir.z;
        if s <= 0.0 {
            bail!("camera corner ray meets the estimated target plane behind the camera");
        }
        let p = c + dir * s;
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }

    let (cx, cy) = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y));
    let half_x = 0.5 * (max_x - min_x) * FOOTPRINT_MARGIN;
    let half_y = 0.5 * (max_y - min_y) * FOOTPRINT_MARGIN;
    let mm_per_px = (2.0 * half_x) / target_w as f32;
    if !mm_per_px.is_finite() || mm_per_px <= 0.0 {
        bail!("degenerate footprint on the estimated target plane");
    }
    let h = ((2.0 * half_y) / mm_per_px).round().max(1.0) as usize;
    // A plane so oblique that the footprint quadrilateral runs away would
    // produce a grid of absurd height; fail loudly rather than allocate it.
    if h > 4 * target_w {
        bail!(
            "footprint aspect {:.1} is implausible — the estimated plane is nearly edge-on to camera0",
            h as f32 / target_w as f32
        );
    }
    Ok(PlaneGrid {
        origin_mm: Point2f::new(cx - half_x, cy - half_y),
        mm_per_px,
        w: target_w,
        h,
    })
}

/// One rectified view per camera, each with its own validity mask (255 =
/// this camera covers that grid pixel).
type Rectified = (Vec<Image<u8>>, Vec<Vec<u8>>);

/// Rectifies every camera onto `grid`, returning the images and their
/// validity masks.
fn rectify_all(
    shifted: &[(CameraModel, Pose3)],
    raws: &[Image<u8>],
    grid: &PlaneGrid,
) -> Result<Rectified> {
    let mut rectified = Vec::with_capacity(shifted.len());
    let mut masks = Vec::with_capacity(shifted.len());
    for ((camera, pose), raw) in shifted.iter().zip(raws) {
        let map: Map = plane_grid_map(camera, pose, grid);
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
    Ok((rectified, masks))
}

// ── Seam metric: ZNCC over the jointly-valid overlap ────────────────────

/// Zero-mean normalized cross-correlation between two rectified views over
/// the pixels both cover — `None` when the overlap is empty or one side is
/// flat there (no contrast, so no correlation is defined).
///
/// This replaces an earlier `max − min` raw-intensity statistic that
/// saturated on this hard-edged target and needed a paragraph of excuses; see
/// this file's doc comment.
fn overlap_zncc(
    a: &Image<u8>,
    b: &Image<u8>,
    mask_a: &[u8],
    mask_b: &[u8],
) -> Option<(f32, usize)> {
    let idx: Vec<usize> = (0..mask_a.len())
        .filter(|&i| mask_a[i] == 255 && mask_b[i] == 255)
        .collect();
    if idx.len() < 2 {
        return None;
    }
    let n = idx.len() as f64;
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    for &i in &idx {
        sa += a.data()[i] as f64;
        sb += b.data()[i] as f64;
    }
    let (ma, mb) = (sa / n, sb / n);
    let (mut num, mut va, mut vb) = (0.0f64, 0.0f64, 0.0f64);
    for &i in &idx {
        let da = a.data()[i] as f64 - ma;
        let db = b.data()[i] as f64 - mb;
        num += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom <= 0.0 {
        return None;
    }
    Some(((num / denom) as f32, idx.len()))
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
