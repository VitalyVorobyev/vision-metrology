//! Tauri desktop shell over `vision-metrology` — commands and events, no HTTP.
//!
//! This is the lib half of the crate; `main.rs` is a one-line binary wrapper. Splitting
//! it out is what lets `tests/contract_parity.rs` call the command layer
//! (`commands::*`, plain functions over `&AppState`) without a running GUI: a Tauri
//! `#[tauri::command]` needs `tauri::State`/`tauri::AppHandle`, which only exist inside a
//! running `App` — the thin wrappers below are the only things that touch those types,
//! and every wrapper is a two-line adapter over a `commands::*` function that a test can
//! call directly.
//!
//! ## Every wrapper that touches pixels is `async`
//!
//! A synchronous `#[tauri::command]` runs on the main thread, which is the
//! thread that also draws the window. A `find` over a large frame, a PNG
//! encode, a folder scan — each of them froze the UI for its whole duration,
//! and a frozen window is indistinguishable from a slow library. The wrappers
//! below are `async` and hand the real work to `spawn_blocking`, so the window
//! keeps painting (and the progress events below keep arriving) while the work
//! runs.

pub mod commands;
pub mod error;
pub mod state;
pub mod types;

use std::sync::Arc;
use std::time::Instant;

use tauri::{Emitter, Manager};

use error::AppError;
use state::AppState;
use types::{
    BatchFindRequest, BatchFindResponse, CalibrationOut, DirEntryOut, DisplacementRequest,
    DisplacementResponse, FindRequest, FindResponse, ImageOut, MeasureRequest, MeasureResponse,
    ModelCreateRequest, ModelCropRequest, ModelGeometryOut, ModelOut, ProgressEvent,
    RectifyRequest, RectifyResponse, TeachPreviewRequest, TeachPreviewResponse, ThumbEvent,
};

fn state_err(e: AppError) -> String {
    e.0
}

/// Run `f` off the main thread, so the window keeps painting.
///
/// The state is an `Arc` clone rather than a borrow because the closure
/// outlives this function's stack frame.
async fn blocking<T, F>(app: &tauri::AppHandle, f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce(&AppState) -> Result<T, String> + Send + 'static,
{
    let state = app.state::<Arc<AppState>>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || f(&state))
        .await
        .map_err(|e| format!("task panicked: {e}"))?
}

/// Emit `lab://progress` around a named operation and report its duration.
///
/// The elapsed time was already being measured for `find` and thrown at an
/// event nothing listened to. Every heavy command reports it now, and the
/// frontend's status bar shows it — which is the difference between "the lab
/// feels slow" and a number that says where the time went.
fn timed<T>(app: &tauri::AppHandle, op: &'static str, f: impl FnOnce() -> T) -> T {
    let _ = app.emit(
        "lab://progress",
        ProgressEvent {
            op,
            stage: "started",
            elapsed_ms: None,
        },
    );
    let started = Instant::now();
    let out = f();
    let _ = app.emit(
        "lab://progress",
        ProgressEvent {
            op,
            stage: "finished",
            elapsed_ms: Some(started.elapsed().as_secs_f64() * 1000.0),
        },
    );
    out
}

// -- images ------------------------------------------------------------------

#[tauri::command]
async fn images_upload(
    app: tauri::AppHandle,
    filename: String,
    bytes: Vec<u8>,
) -> Result<ImageOut, String> {
    blocking(&app, move |state| {
        commands::images::images_upload(state, filename, &bytes).map_err(state_err)
    })
    .await
}

/// Register images already on disk, by path — the desktop's own "open".
#[tauri::command]
async fn images_open_paths(
    app: tauri::AppHandle,
    paths: Vec<String>,
) -> Result<Vec<ImageOut>, String> {
    let handle = app.clone();
    blocking(&app, move |state| {
        timed(&handle, "open", || {
            commands::images::images_open_paths(state, paths).map_err(state_err)
        })
    })
    .await
}

/// List a folder's image files without decoding any of them.
#[tauri::command]
async fn images_scan_dir(
    app: tauri::AppHandle,
    dir: String,
    recursive: bool,
) -> Result<Vec<DirEntryOut>, String> {
    let handle = app.clone();
    blocking(&app, move |_state| {
        timed(&handle, "scan", || {
            commands::images::images_scan_dir(&dir, recursive).map_err(state_err)
        })
    })
    .await
}

#[tauri::command]
async fn images_list(app: tauri::AppHandle) -> Result<Vec<ImageOut>, String> {
    blocking(&app, move |state| Ok(commands::images::images_list(state))).await
}

/// Absolute path to the cached PNG tier, which the frontend turns into an
/// `asset:` URL. See `commands::images` for why this is a path and not bytes.
#[tauri::command]
async fn image_tier_path(
    app: tauri::AppHandle,
    image_id: String,
    tier: String,
) -> Result<String, String> {
    blocking(&app, move |state| {
        commands::images::image_tier_path(state, &image_id, &tier).map_err(state_err)
    })
    .await
}

/// Render missing `thumb` tiers ahead of the grid scrolling onto them,
/// reporting each as `lab://thumb`.
#[tauri::command]
async fn prewarm_thumbnails(app: tauri::AppHandle, image_ids: Vec<String>) -> Result<(), String> {
    let handle = app.clone();
    blocking(&app, move |state| {
        commands::images::prewarm_thumbnails(state, &image_ids, |id, done, total| {
            let _ = handle.emit(
                "lab://thumb",
                ThumbEvent {
                    image_id: id.to_string(),
                    done,
                    total,
                },
            );
        });
        Ok(())
    })
    .await
}

#[tauri::command]
async fn image_data(
    app: tauri::AppHandle,
    image_id: String,
    tier: String,
) -> Result<tauri::ipc::Response, String> {
    let bytes = blocking(&app, move |state| {
        commands::images::image_data(state, &image_id, &tier).map_err(state_err)
    })
    .await?;
    Ok(tauri::ipc::Response::new(bytes))
}

// -- teach -------------------------------------------------------------------

/// The candidate contours to curate before building a model.
#[tauri::command]
async fn teach_preview(
    app: tauri::AppHandle,
    req: TeachPreviewRequest,
) -> Result<TeachPreviewResponse, String> {
    let handle = app.clone();
    blocking(&app, move |state| {
        timed(&handle, "teach_preview", || {
            commands::teach::teach_preview(state, req).map_err(state_err)
        })
    })
    .await
}

#[tauri::command]
async fn models_create(app: tauri::AppHandle, req: ModelCreateRequest) -> Result<ModelOut, String> {
    let handle = app.clone();
    blocking(&app, move |state| {
        timed(&handle, "teach", || {
            commands::models::models_create(state, req).map_err(state_err)
        })
    })
    .await
}

#[tauri::command]
async fn models_list(app: tauri::AppHandle) -> Result<Vec<ModelOut>, String> {
    blocking(&app, move |state| Ok(commands::models::models_list(state))).await
}

#[tauri::command]
async fn model_geometry(
    app: tauri::AppHandle,
    model_id: String,
    level: usize,
    frame: String,
) -> Result<ModelGeometryOut, String> {
    blocking(&app, move |state| {
        commands::models::model_geometry(state, &model_id, level, &frame).map_err(state_err)
    })
    .await
}

#[tauri::command]
async fn model_crop(
    app: tauri::AppHandle,
    req: ModelCropRequest,
) -> Result<tauri::ipc::Response, String> {
    let bytes = blocking(&app, move |state| {
        commands::models::model_crop(state, &req).map_err(state_err)
    })
    .await?;
    Ok(tauri::ipc::Response::new(bytes))
}

// -- calibration -------------------------------------------------------------

#[tauri::command]
async fn calibration_upload(
    app: tauri::AppHandle,
    filename: String,
    bytes: Vec<u8>,
) -> Result<CalibrationOut, String> {
    blocking(&app, move |state| {
        commands::calibration::calibration_upload(state, filename, bytes).map_err(state_err)
    })
    .await
}

#[tauri::command]
async fn calibration_list(app: tauri::AppHandle) -> Result<Vec<CalibrationOut>, String> {
    blocking(&app, move |state| {
        Ok(commands::calibration::calibration_list(state))
    })
    .await
}

// -- find / measure ----------------------------------------------------------

#[tauri::command]
async fn find(app: tauri::AppHandle, req: FindRequest) -> Result<FindResponse, String> {
    let handle = app.clone();
    blocking(&app, move |state| {
        timed(&handle, "find", || {
            commands::find::find(state, req).map_err(state_err)
        })
    })
    .await
}

/// One model over a whole set of frames, with `lab://batch` progress per image.
#[tauri::command]
async fn batch_find(
    app: tauri::AppHandle,
    req: BatchFindRequest,
) -> Result<BatchFindResponse, String> {
    let handle = app.clone();
    blocking(&app, move |state| {
        commands::batch::batch_find(state, &req, |p| {
            let _ = handle.emit("lab://batch", p);
        })
        .map_err(state_err)
    })
    .await
}

#[tauri::command]
async fn measure(app: tauri::AppHandle, req: MeasureRequest) -> Result<MeasureResponse, String> {
    let handle = app.clone();
    blocking(&app, move |state| {
        timed(&handle, "measure", || {
            commands::measure::measure(state, req).map_err(state_err)
        })
    })
    .await
}

#[tauri::command]
async fn rectify(app: tauri::AppHandle, req: RectifyRequest) -> Result<RectifyResponse, String> {
    let handle = app.clone();
    blocking(&app, move |state| {
        timed(&handle, "rectify", || {
            commands::rectify::rectify(state, req).map_err(state_err)
        })
    })
    .await
}

#[tauri::command]
async fn rectify_crop(
    image_id: String,
    model_id: String,
    index: usize,
) -> Result<tauri::ipc::Response, String> {
    let bytes = tauri::async_runtime::spawn_blocking(move || {
        commands::rectify::rectify_crop(&image_id, &model_id, index)
    })
    .await
    .map_err(|e| format!("task panicked: {e}"))?
    .map_err(state_err)?;
    Ok(tauri::ipc::Response::new(bytes))
}

#[tauri::command]
async fn displacement(
    app: tauri::AppHandle,
    req: DisplacementRequest,
) -> Result<DisplacementResponse, String> {
    let handle = app.clone();
    blocking(&app, move |state| {
        timed(&handle, "displacement", || {
            commands::displacement::displacement(state, req).map_err(state_err)
        })
    })
    .await
}

#[tauri::command]
fn health() -> &'static str {
    "ok"
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .expect("app data dir should be resolvable");
            // Derived tiers go to the OS cache directory, not app-data: every
            // byte there is reproducible from the source image, so it should
            // be free to evict and should not ride along in a backup.
            let cache_dir = app
                .path()
                .app_cache_dir()
                .expect("app cache dir should be resolvable");
            let state = AppState::with_cache_dir(data_dir, cache_dir)
                .and_then(|s| {
                    s.rehydrate()?;
                    Ok(s)
                })
                .expect("app state should initialize");
            app.manage(Arc::new(state));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            health,
            images_upload,
            images_open_paths,
            images_scan_dir,
            images_list,
            image_tier_path,
            prewarm_thumbnails,
            image_data,
            teach_preview,
            models_create,
            models_list,
            model_geometry,
            model_crop,
            calibration_upload,
            calibration_list,
            find,
            batch_find,
            measure,
            rectify,
            rectify_crop,
            displacement,
        ])
        .run(tauri::generate_context!())
        .expect("error while running vm-lab-desktop");
}
