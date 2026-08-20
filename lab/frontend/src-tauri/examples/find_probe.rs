//! Where does `find` time actually go?
//!
//! Written to answer a specific complaint — "find takes several seconds" —
//! without guessing. The library publishes **3.5 ms** for a full 360° search of
//! a clean 1280×1024 frame (`docs/shape-matching.md`), and the canend frames
//! are exactly that size, so any large number had to come from either the shell
//! or the search *settings*, not from the matcher.
//!
//! ## Measured (2026-08-20, M4 Pro, release, canend set1/normal/bright)
//!
//! ```text
//! large ROI (60% of frame): 429 points, 5 levels
//!   min_score 0.4, 360deg, greed 0.9        47.8 ms  1 match
//!   min_score 0.7, 360deg, greed 0.9        22.8 ms  1 match
//!   min_score 0.7, +-15deg                  10.3 ms  1 match
//!   min_score 0.7, 360deg, greed 1.0         5.7 ms  0 match
//!
//! tight ROI (25% of frame): 250 points, 5 levels
//!   min_score 0.4, 360deg, greed 0.9         7.9 ms  1 match
//!   min_score 0.7, 360deg, greed 0.9         4.2 ms  1 match
//!   min_score 0.7, +-15deg                   2.7 ms  1 match
//!   min_score 0.7, 360deg, greed 1.0         3.9 ms  1 match
//! ```
//!
//! **The conclusion.** A model taught from a sensible ROI runs at **4.2 ms**,
//! in line with the published benchmark. The 48 ms row is a model whose ROI
//! covers 60% of the frame: its radius is ~400 px, and the per-level angle step
//! is `clamp(1/radius, …)`, so a full turn costs thousands of steps. Every row
//! here is a *setting*, and the old lab made the most expensive combination of
//! them the only one reachable — no angle range, no match cap, no greediness,
//! and no reported duration to notice it by. That, plus commands running on the
//! window's own thread and a PNG re-encode on every tier request, is what "the
//! library is slow" actually was.
//!
//! Note the last row of the first block: at `greediness = 1.0` the search is
//! fastest and finds *nothing*. That is the trade the knob names, and it is why
//! the UI exposes it with that warning rather than picking a value silently.
//!
//! ## Run
//! ```text
//! cargo run --release --example find_probe -- [image-folder]
//! ```

use std::time::Instant;

use vm_lab_desktop::commands;
use vm_lab_desktop::state::AppState;
use vm_lab_desktop::types::{FindRequest, ModelCreateRequest, SearchTuningIn};

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/privatedata/canend/set1/normal/bright",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    if !std::path::Path::new(&dir).is_dir() {
        eprintln!("no such folder: {dir}\nusage: find_probe [image-folder]");
        return;
    }

    let data = tempfile::tempdir().expect("tempdir");
    let cache = tempfile::tempdir().expect("tempdir");
    let state = AppState::with_cache_dir(data.path().into(), cache.path().into()).expect("state");

    let entries = commands::images::images_scan_dir(&dir, false).expect("scan");
    let opened = commands::images::images_open_paths(
        &state,
        entries.iter().take(2).map(|e| e.path.clone()).collect(),
    )
    .expect("open");
    let Some(first) = opened.first() else {
        eprintln!("no images in {dir}");
        return;
    };
    let id = first.id.clone();
    let (w, h) = (first.width as f32, first.height as f32);

    for (name, roi) in [
        (
            "large ROI (60% of frame)",
            [w * 0.2, h * 0.2, w * 0.6, h * 0.6],
        ),
        (
            "tight ROI (25% of frame)",
            [w * 0.375, h * 0.375, w * 0.25, h * 0.25],
        ),
    ] {
        let model = commands::models::models_create(
            &state,
            ModelCreateRequest {
                image_id: id.clone(),
                roi,
                min_contrast: 0.1,
                num_levels: None,
                keep_contours: None,
                origin: None,
                reference_angle: 0.0,
            },
        )
        .expect("teach");
        println!(
            "\n{name}: {} points, {} levels",
            model.point_counts[0], model.num_levels_built
        );

        let base = FindRequest {
            image_id: id.clone(),
            model_id: model.id.clone(),
            max_matches: Some(1),
            ..FindRequest::default()
        };
        for (label, req) in [
            (
                "min_score 0.4, 360deg, greed 0.9",
                FindRequest {
                    min_score: 0.4,
                    ..base.clone()
                },
            ),
            (
                "min_score 0.7, 360deg, greed 0.9",
                FindRequest {
                    min_score: 0.7,
                    ..base.clone()
                },
            ),
            (
                "min_score 0.7, +-15deg",
                FindRequest {
                    min_score: 0.7,
                    angle_range: Some((-0.26, 0.26)),
                    ..base.clone()
                },
            ),
            (
                "min_score 0.7, 360deg, greed 1.0",
                FindRequest {
                    min_score: 0.7,
                    tuning: Some(SearchTuningIn {
                        greediness: Some(1.0),
                        ..SearchTuningIn::default()
                    }),
                    ..base.clone()
                },
            ),
        ] {
            // Warmed first, so this times the search and not the frame's decode.
            let _ = commands::find::find(&state, req.clone());
            let started = Instant::now();
            let found = commands::find::find(&state, req).expect("find");
            println!(
                "  {label:36} {:7.1} ms  {} match",
                started.elapsed().as_secs_f64() * 1000.0,
                found.matches.len()
            );
        }
    }
}
