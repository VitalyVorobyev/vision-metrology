//! Starting up must survive what a previous run left behind.
//!
//! `AppState::rehydrate` runs inside Tauri's `setup` hook, which runs *after* the
//! window exists. An error there used to propagate out of `.expect(...)` and kill
//! the process — leaving a window that had never been given anything to draw, i.e.
//! a black rectangle with the reason only in a terminal a bundled `.app` has none
//! of. One model file this build cannot read is not a reason to refuse to start.

use std::fs;
use std::path::Path;

use vm_lab_desktop::state::AppState;

fn write(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).expect("write fixture");
}

/// A real 4×4 grayscale PNG, so the image entry has pixels to point at.
fn tiny_png(path: &Path) {
    image::GrayImage::from_raw(4, 4, vec![128; 16])
        .expect("buffer")
        .save(path)
        .expect("save png");
}

#[test]
fn a_model_this_build_cannot_read_costs_the_model_not_the_session() {
    let data = tempfile::tempdir().expect("tempdir");
    let cache = tempfile::tempdir().expect("tempdir");
    let root = data.path();

    // The state's own constructor is what creates the registry directories, so
    // build it first and lay the previous session's files into them.
    let state = AppState::with_cache_dir(root.to_path_buf(), cache.path().to_path_buf())
        .expect("state should initialize");

    // A perfectly good frame from a previous session.
    tiny_png(&root.join("images/img-1.png"));
    write(
        &root.join("images/img-1.json"),
        br#"{"id":"img-1","filename":"8.bmp","width":4,"height":4,"sha256":"abc"}"#,
    );

    // A model whose sidecar is fine but whose payload this build cannot load —
    // a format from the future, or a truncated write.
    write(
        &root.join("models/model-1.json"),
        br#"{"id":"model-1","image_id":"img-1","roi":[0.0,0.0,4.0,4.0],"min_contrast":0.1,"num_levels":3}"#,
    );
    write(&root.join("models/model-1.bin"), b"not a shape model");

    // And a sidecar that is not even JSON.
    write(&root.join("models/model-2.json"), b"{ truncated");

    state
        .rehydrate()
        .expect("rehydrating must not fail over a file it cannot read");

    assert_eq!(
        state.images.lock().expect("images").len(),
        1,
        "the readable frame should still be there"
    );
    assert!(
        state.models.lock().expect("models").is_empty(),
        "an unreadable model is dropped, not fatal"
    );
}
