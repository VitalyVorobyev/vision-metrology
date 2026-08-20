//! The desktop flow, end to end, over a real folder of frames.
//!
//! `contract_parity.rs` proves the shared operations still agree with the
//! browser shell's golden numbers. This proves the things that have no browser
//! counterpart and were the point of the wave: open a folder without decoding
//! it, render tiers once and cache them as files, preview and curate contours,
//! build a model from the curated subset, read its geometry back, and run it
//! across the set.
//!
//! Gated on `~/privatedata/canend` because that is where the real frames live.
//! It reports what it skipped rather than passing silently — a test that
//! quietly does nothing is worse than no test.

use std::path::PathBuf;
use std::time::Instant;

use vm_lab_desktop::commands;
use vm_lab_desktop::state::AppState;
use vm_lab_desktop::types::{
    BatchFindRequest, BatchSearchIn, FindRequest, ModelCreateRequest, TeachPreviewRequest,
};

fn dataset() -> Option<PathBuf> {
    let dir =
        PathBuf::from(std::env::var("HOME").ok()?).join("privatedata/canend/set1/normal/bright");
    dir.is_dir().then_some(dir)
}

fn fresh_state() -> (AppState, tempfile::TempDir, tempfile::TempDir) {
    let data = tempfile::tempdir().expect("tempdir");
    let cache = tempfile::tempdir().expect("tempdir");
    let state = AppState::with_cache_dir(data.path().to_path_buf(), cache.path().to_path_buf())
        .expect("state");
    (state, data, cache)
}

#[test]
fn a_folder_opens_without_decoding_it_and_frames_register_by_path() {
    let Some(dir) = dataset() else {
        eprintln!("skipped: ~/privatedata/canend/set1/normal/bright is not present");
        return;
    };
    let (state, _data, cache) = fresh_state();

    // Scanning reads directory entries and image headers only.
    let started = Instant::now();
    let entries = commands::images::images_scan_dir(dir.to_str().unwrap(), false).expect("scan");
    let scan_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert!(!entries.is_empty(), "the folder should hold frames");
    assert!(
        entries.iter().all(|e| e.width > 0 && e.height > 0),
        "every entry should carry header dimensions"
    );
    println!("scanned {} frames in {scan_ms:.1} ms", entries.len());

    // Natural order, so frame 2 precedes frame 10.
    let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    if let (Some(i2), Some(i10)) = (
        names.iter().position(|n| *n == "2.bmp"),
        names.iter().position(|n| *n == "10.bmp"),
    ) {
        assert!(i2 < i10, "2.bmp should sort before 10.bmp, got {names:?}");
    }

    // Registering keeps the user's own files where they are.
    let paths: Vec<String> = entries.iter().take(4).map(|e| e.path.clone()).collect();
    let opened = commands::images::images_open_paths(&state, paths.clone()).expect("open");
    assert_eq!(opened.len(), 4);
    for (out, path) in opened.iter().zip(&paths) {
        assert_eq!(out.path.as_deref(), Some(path.as_str()));
        assert!(
            !out.sha256.is_empty(),
            "the content hash should be filled in"
        );
    }

    // Re-opening the same path is the same entry, not a duplicate.
    let again = commands::images::images_open_paths(&state, paths.clone()).expect("reopen");
    assert_eq!(again[0].id, opened[0].id);
    assert_eq!(commands::images::images_list(&state).len(), 4);

    // Tiers are rendered once, to files, and reused.
    let first = Instant::now();
    let tier = commands::images::image_tier_path(&state, &opened[0].id, "preview").expect("tier");
    let cold_ms = first.elapsed().as_secs_f64() * 1000.0;
    assert!(
        PathBuf::from(&tier).is_file(),
        "the tier should be a real file"
    );
    assert!(
        tier.starts_with(cache.path().to_str().unwrap()),
        "tiers belong in the cache dir, got {tier}"
    );

    let second = Instant::now();
    let again = commands::images::image_tier_path(&state, &opened[0].id, "preview").expect("tier");
    let warm_ms = second.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(tier, again);
    println!("preview tier: {cold_ms:.1} ms cold, {warm_ms:.3} ms warm");
    assert!(
        warm_ms < cold_ms,
        "a cached tier must not re-encode ({warm_ms:.3} ms vs {cold_ms:.1} ms)"
    );
}

#[test]
fn teaching_is_curated_and_the_model_can_be_drawn() {
    let Some(dir) = dataset() else {
        eprintln!("skipped: ~/privatedata/canend/set1/normal/bright is not present");
        return;
    };
    let (state, _data, _cache) = fresh_state();

    let entries = commands::images::images_scan_dir(dir.to_str().unwrap(), false).expect("scan");
    let opened = commands::images::images_open_paths(
        &state,
        entries.iter().take(6).map(|e| e.path.clone()).collect(),
    )
    .expect("open");
    let image_id = opened[0].id.clone();

    // A generous ROI over the middle of the frame — this is a smoke test of the
    // pipeline's shape, not of where the can end happens to sit.
    let (w, h) = (opened[0].width as f32, opened[0].height as f32);
    let roi = [w * 0.2, h * 0.2, w * 0.6, h * 0.6];

    let preview = commands::teach::teach_preview(
        &state,
        TeachPreviewRequest {
            image_id: image_id.clone(),
            roi,
            min_contrast: 0.1,
        },
    )
    .expect("preview");
    assert!(
        !preview.contours.is_empty(),
        "a real frame should offer candidate contours"
    );
    println!(
        "preview: {} contours, {} points",
        preview.contours.len(),
        preview.total_points
    );

    // Keep the longest few — the part, not the speckle around it.
    let mut by_length = preview.contours.clone();
    by_length.sort_by(|a, b| b.length.total_cmp(&a.length));
    let keep: Vec<usize> = by_length.iter().take(5).map(|c| c.id).collect();

    let curated = commands::models::models_create(
        &state,
        ModelCreateRequest {
            image_id: image_id.clone(),
            roi,
            min_contrast: 0.1,
            num_levels: None,
            keep_contours: Some(keep.clone()),
            origin: None,
            reference_angle: 0.5,
        },
    )
    .expect("curated teach");
    assert_eq!(curated.reference_angle, 0.5);

    let plain = commands::models::models_create(
        &state,
        ModelCreateRequest {
            image_id: image_id.clone(),
            roi,
            min_contrast: 0.1,
            num_levels: None,
            keep_contours: None,
            origin: None,
            reference_angle: 0.0,
        },
    )
    .expect("plain teach");

    println!(
        "curated {} points vs plain {} (kept {} of {} contours)",
        curated.point_counts[0],
        plain.point_counts[0],
        keep.len(),
        preview.contours.len()
    );

    // The geometry a UI draws: one point per model point, four floats each.
    let geom = commands::models::model_geometry(&state, &curated.id, 0, "reference")
        .expect("reference geometry");
    assert_eq!(geom.points.len(), curated.point_counts[0] * 4);
    assert_eq!(geom.frame, "reference");

    let model_frame =
        commands::models::model_geometry(&state, &curated.id, 0, "model").expect("model geometry");
    // A non-zero canonical orientation is exactly the case where the two frames
    // differ; if they did not, the whole feature would be a no-op.
    assert_ne!(
        geom.points, model_frame.points,
        "reference and model frames must differ once reference_angle is set"
    );

    // Every drawn point should land inside the ROI it was taught from.
    for [x, y, _, _] in geom.points.as_chunks::<4>().0 {
        assert!(*x >= roi[0] - 2.0 && *x <= roi[0] + roi[2] + 2.0);
        assert!(*y >= roi[1] - 2.0 && *y <= roi[1] + roi[3] + 2.0);
    }

    // And the model's own reference crop renders.
    let crop = commands::models::model_crop(
        &state,
        &vm_lab_desktop::types::ModelCropRequest {
            model_id: curated.id.clone(),
            rect: roi,
            px_per_unit: 0.5,
        },
    )
    .expect("model crop");
    assert!(crop.len() > 100, "the crop should be a real PNG");

    // Find, timed, then the same model across the whole set.
    let started = Instant::now();
    let found = commands::find::find(
        &state,
        FindRequest {
            image_id: image_id.clone(),
            model_id: plain.id.clone(),
            min_score: 0.4,
            max_matches: Some(1),
            ..FindRequest::default()
        },
    )
    .expect("find");
    println!(
        "find on the teach frame: {:.1} ms, {} match(es)",
        started.elapsed().as_secs_f64() * 1000.0,
        found.matches.len()
    );
    assert!(
        !found.matches.is_empty(),
        "a model must at least find itself in the frame it was taught from"
    );

    let mut seen = 0usize;
    let batch = commands::batch::batch_find(
        &state,
        &BatchFindRequest {
            model_id: plain.id.clone(),
            image_ids: opened.iter().map(|i| i.id.clone()).collect(),
            search: BatchSearchIn {
                min_score: 0.3,
                max_matches: Some(1),
                ..BatchSearchIn::default()
            },
        },
        |p| {
            seen += 1;
            assert_eq!(p.total, 6);
        },
    )
    .expect("batch");
    assert_eq!(batch.items.len(), 6);
    assert_eq!(seen, 6, "progress should fire once per frame");
    let total: f64 = batch.items.iter().map(|i| i.elapsed_ms).sum();
    println!("batch over 6 frames: {total:.0} ms total");
}
