//! Real-data cross-check: `corr::displacement` on the `data/42781` glue rig.
//!
//! This dataset (`docs/backlog.md` / roadmap: no calibration, three cameras
//! glued vertically into one 320x291 frame) has no recorded ground truth for
//! inter-frame motion — `tools/glue-42781/motion.py` (phase correlation) only
//! ever wrote PNG plots, no numeric trajectory file to compare against (see
//! `data/42781/output/`). So this is **agreement, not a gate**: it asserts
//! the trajectory this crate's `displacement` produces is internally
//! consistent (bounded frame-to-frame delta, no NaN/degenerate steps) and
//! prints a summary for the wave report. Skips cleanly when the (gitignored,
//! locally-fetched) dataset is absent — nothing here should fail CI on a
//! fresh checkout.

use std::path::{Path, PathBuf};

use vision_metrology::corr::{DisplacementConfig, Refine, displacement};
use vision_metrology::{Image, Rect2f};

/// `../../data/42781` relative to this crate's manifest directory, so the
/// test finds the dataset regardless of the directory `cargo test` was
/// invoked from.
fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/42781")
}

fn frames_in(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("bmp" | "BMP")))
        .collect();
    out.sort();
    out
}

fn load_gray(path: &Path) -> Image<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("opening {}: {e}", path.display()))
        .to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Image::from_vec(w, h, img.into_raw()).expect("valid frame")
}

/// Strip `index` (rows `[index*h, (index+1)*h)`) of `frame`.
fn strip(frame: &Image<u8>, index: usize, h: usize) -> Image<u8> {
    let w = frame.width();
    let y0 = index * h;
    let mut data = vec![0u8; w * h];
    for row in 0..h {
        let src = &frame.data()[(y0 + row) * w..(y0 + row + 1) * w];
        data[row * w..(row + 1) * w].copy_from_slice(src);
    }
    Image::from_vec(w, h, data).expect("valid strip")
}

#[test]
fn glue_rig_trajectory_is_smooth() {
    let frames = frames_in(&data_dir());
    if frames.is_empty() {
        eprintln!(
            "glue_rig_trajectory_is_smooth: data/42781 not found locally, skipping \
             (gitignored dataset, not part of the checkout)"
        );
        return;
    }

    const STRIP_INDEX: usize = 1;
    const STRIP_H: usize = 97;
    // The glue nozzle's dome — the same teach ROI `examples/align_crops.rs`
    // uses, chosen for its real edges and frame-to-frame texture.
    let window = Rect2f {
        x: 110.0,
        y: 30.0,
        width: 100.0,
        height: 67.0,
    };
    let cfg = DisplacementConfig {
        window,
        search: (10, 10),
        refine: Refine::LucasKanade { iters: 3 },
        min_score: 0.2,
    };

    let n = frames.len().min(40);
    let strips: Vec<Image<u8>> = frames[..n]
        .iter()
        .map(|p| strip(&load_gray(p), STRIP_INDEX, STRIP_H))
        .collect();

    let mut cum = (0.0f32, 0.0f32);
    let mut max_step = 0.0f32;
    let mut sum_step = 0.0f32;
    let mut scores = Vec::with_capacity(strips.len() - 1);
    let mut steps = 0usize;

    for pair in strips.windows(2) {
        let d = displacement(&pair[0].as_view(), &pair[1].as_view(), &cfg)
            .expect("displacement succeeds on real frames");
        assert!(d.shift.x.is_finite() && d.shift.y.is_finite());

        let mag = d.shift.norm();
        max_step = max_step.max(mag);
        sum_step += mag;
        scores.push(d.score);
        cum.0 += d.shift.x;
        cum.1 += d.shift.y;
        steps += 1;

        // Consistency, not a ground-truth gate: a real rig between two
        // consecutive frames does not teleport. The bounded search itself
        // caps the *possible* answer at `search` px past the window edge —
        // this just confirms the accepted answer is well inside that, not
        // pinned to the search boundary every time (which would mean the
        // true target routinely left the search window).
        assert!(
            mag < 8.0,
            "frame-to-frame shift {mag:.3} px exceeds the sane bound (search radius {:?})",
            cfg.search
        );
    }

    let mean_step = sum_step / steps as f32;
    let mean_score = scores.iter().sum::<f32>() / scores.len() as f32;
    eprintln!(
        "glue_rig_trajectory_is_smooth: {steps} consecutive pairs, strip {STRIP_INDEX}, \
         window {window:?}\n  mean |shift| = {mean_step:.3} px, max |shift| = {max_step:.3} px, \
         mean score = {mean_score:.3}\n  cumulative displacement = ({:.2}, {:.2}) px",
        cum.0, cum.1
    );
}
