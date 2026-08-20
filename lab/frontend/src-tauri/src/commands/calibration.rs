//! `calibration_upload`, `calibration_list` — mirrors
//! `lab/backend/src/vm_lab/routers/calibration.py`.

use crate::error::AppResult;
use crate::state::AppState;
use crate::types::CalibrationOut;

pub fn calibration_upload(
    state: &AppState,
    filename: String,
    bytes: Vec<u8>,
) -> AppResult<CalibrationOut> {
    state.add_calibration(filename, bytes)
}

pub fn calibration_list(state: &AppState) -> Vec<CalibrationOut> {
    let calibrations = state
        .calibrations
        .lock()
        .expect("calibrations mutex poisoned");
    let mut out: Vec<CalibrationOut> = calibrations
        .values()
        .map(|e| CalibrationOut {
            id: e.id.clone(),
            filename: e.filename.clone(),
            format: e.format.clone(),
            n_cameras: e.cameras.len(),
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}
