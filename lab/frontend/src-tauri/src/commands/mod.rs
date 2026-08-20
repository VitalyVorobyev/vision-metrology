//! Command handlers as plain functions over `&AppState` — the layer `main.rs`'s
//! `#[tauri::command]` wrappers call into, and what `tests/contract_parity.rs` calls
//! directly (no GUI, no `tauri::State` needed to exercise this layer).

pub mod calibration;
pub mod displacement;
pub mod find;
pub mod images;
pub mod measure;
pub mod models;
pub mod rectify;
