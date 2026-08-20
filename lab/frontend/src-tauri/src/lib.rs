//! Tauri desktop shell over `vision-metrology` — commands and events, no HTTP.
//!
//! This is the lib half of the crate; `main.rs` is a one-line binary wrapper. Splitting
//! it out is what lets `tests/contract_parity.rs` call the command layer
//! (`commands::*`, plain functions over `&AppState`) without a running GUI: a Tauri
//! `#[tauri::command]` needs `tauri::State`/`tauri::AppHandle`, which only exist inside a
//! running `App` — the thin wrappers below are the only things that touch those types,
//! and every wrapper is a two-line adapter over a `commands::*` function that a test can
//! call directly.

pub mod commands;
pub mod error;
pub mod state;
pub mod types;

use std::time::Instant;

use tauri::{Emitter, Manager};

use error::AppError;
use state::AppState;
use types::{
    CalibrationOut, DisplacementRequest, DisplacementResponse, FindRequest, FindResponse, ImageOut,
    MeasureRequest, MeasureResponse, ModelCreateRequest, ModelOut, ProgressEvent, RectifyRequest,
    RectifyResponse,
};

fn state_err(e: AppError) -> String {
    e.0
}

#[tauri::command]
fn images_upload(
    state: tauri::State<'_, AppState>,
    filename: String,
    bytes: Vec<u8>,
) -> Result<ImageOut, String> {
    commands::images::images_upload(&state, filename, &bytes).map_err(state_err)
}

#[tauri::command]
fn images_list(state: tauri::State<'_, AppState>) -> Vec<ImageOut> {
    commands::images::images_list(&state)
}

#[tauri::command]
fn image_data(
    state: tauri::State<'_, AppState>,
    image_id: String,
    tier: String,
) -> Result<tauri::ipc::Response, String> {
    let bytes = commands::images::image_data(&state, &image_id, &tier).map_err(state_err)?;
    Ok(tauri::ipc::Response::new(bytes))
}

#[tauri::command]
fn models_create(
    state: tauri::State<'_, AppState>,
    req: ModelCreateRequest,
) -> Result<ModelOut, String> {
    commands::models::models_create(&state, req).map_err(state_err)
}

#[tauri::command]
fn models_list(state: tauri::State<'_, AppState>) -> Vec<ModelOut> {
    commands::models::models_list(&state)
}

#[tauri::command]
fn calibration_upload(
    state: tauri::State<'_, AppState>,
    filename: String,
    bytes: Vec<u8>,
) -> Result<CalibrationOut, String> {
    commands::calibration::calibration_upload(&state, filename, bytes).map_err(state_err)
}

#[tauri::command]
fn calibration_list(state: tauri::State<'_, AppState>) -> Vec<CalibrationOut> {
    commands::calibration::calibration_list(&state)
}

/// The one command wired with progress events (plan: "wire one real progress case") —
/// `find`'s cost scales with scene size in a way a user notices on a large frame or a
/// wide angle/scale sweep, unlike the effectively-instant `teach`/`images_*` calls.
#[tauri::command]
fn find(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    req: FindRequest,
) -> Result<FindResponse, String> {
    let _ = app.emit(
        "lab://progress",
        ProgressEvent {
            op: "find",
            stage: "started",
            elapsed_ms: None,
        },
    );
    let started = Instant::now();
    let result = commands::find::find(&state, req).map_err(state_err);
    let _ = app.emit(
        "lab://progress",
        ProgressEvent {
            op: "find",
            stage: "finished",
            elapsed_ms: Some(started.elapsed().as_secs_f64() * 1000.0),
        },
    );
    result
}

#[tauri::command]
fn measure(
    state: tauri::State<'_, AppState>,
    req: MeasureRequest,
) -> Result<MeasureResponse, String> {
    commands::measure::measure(&state, req).map_err(state_err)
}

#[tauri::command]
fn rectify(
    state: tauri::State<'_, AppState>,
    req: RectifyRequest,
) -> Result<RectifyResponse, String> {
    commands::rectify::rectify(&state, req).map_err(state_err)
}

#[tauri::command]
fn rectify_crop(
    image_id: String,
    model_id: String,
    index: usize,
) -> Result<tauri::ipc::Response, String> {
    let bytes = commands::rectify::rectify_crop(&image_id, &model_id, index).map_err(state_err)?;
    Ok(tauri::ipc::Response::new(bytes))
}

#[tauri::command]
fn displacement(
    state: tauri::State<'_, AppState>,
    req: DisplacementRequest,
) -> Result<DisplacementResponse, String> {
    commands::displacement::displacement(&state, req).map_err(state_err)
}

#[tauri::command]
fn health() -> &'static str {
    "ok"
}

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .expect("app data dir should be resolvable");
            let state = AppState::rehydrated(data_dir).expect("app state should initialize");
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            health,
            images_upload,
            images_list,
            image_data,
            models_create,
            models_list,
            calibration_upload,
            calibration_list,
            find,
            measure,
            rectify,
            rectify_crop,
            displacement,
        ])
        .run(tauri::generate_context!())
        .expect("error while running vm-lab-desktop");
}
