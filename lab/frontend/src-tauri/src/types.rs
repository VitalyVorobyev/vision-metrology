//! Request/response DTOs for the Tauri commands.
//!
//! Field names and shapes deliberately mirror `lab/backend/src/vm_lab/schemas.py`'s
//! Pydantic models closely enough that `tauriBackend.ts` can reuse the *same* TypeScript
//! types `httpBackend` uses (`lab/frontend/src/api/generated.ts`, generated from
//! `lab/contract/openapi.json`) — this is a numeric-agreement contract between the two
//! backends (see `lab/contract/README.md`), not a byte-identical wire format, so a field
//! present-but-`null` here where the Python side would omit it entirely is fine; both
//! deserialize to the same TypeScript value.

use serde::{Deserialize, Serialize};

// -- images --------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ImageOut {
    pub id: String,
    pub filename: String,
    pub width: u32,
    pub height: u32,
    pub sha256: String,
}

// -- models (teach) --------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCreateRequest {
    pub image_id: String,
    pub roi: [f32; 4],
    #[serde(default = "default_min_contrast")]
    pub min_contrast: f32,
    #[serde(default)]
    pub num_levels: Option<usize>,
}

fn default_min_contrast() -> f32 {
    0.1
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelOut {
    pub id: String,
    pub image_id: String,
    pub roi: [f32; 4],
    pub min_contrast: f32,
    pub num_levels: Option<usize>,
    pub origin: [f32; 2],
    pub num_levels_built: usize,
    pub point_counts: Vec<usize>,
}

// -- find --------------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Default)]
pub struct FindRequest {
    pub image_id: String,
    pub model_id: String,
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    #[serde(default)]
    pub max_matches: Option<usize>,
    #[serde(default)]
    pub roi: Option<[f32; 4]>,
    #[serde(default)]
    pub angle_range: Option<(f32, f32)>,
}

fn default_min_score() -> f32 {
    0.7
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchOut {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub scale: f32,
    pub score: f32,
    pub support: usize,
    pub level: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FindResponse {
    pub matches: Vec<MatchOut>,
}

// -- measure -------------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasureShapeKind {
    Circle,
    Line,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MeasureConfigIn {
    pub sigma: Option<f32>,
    pub threshold: Option<f32>,
    pub polarity: Option<String>,
    pub max_obliquity_deg: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FitConfigIn {
    pub loss: Option<String>,
    pub inlier_tol: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MeasureObjectIn {
    pub kind: MeasureShapeKind,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub cx: Option<f32>,
    #[serde(default)]
    pub cy: Option<f32>,
    #[serde(default)]
    pub r: Option<f32>,
    #[serde(default)]
    pub arc: Option<(f32, f32)>,
    #[serde(default)]
    pub ax: Option<f32>,
    #[serde(default)]
    pub ay: Option<f32>,
    #[serde(default)]
    pub bx: Option<f32>,
    #[serde(default)]
    pub by: Option<f32>,
    #[serde(default = "default_n_calipers")]
    pub n_calipers: usize,
    #[serde(default = "default_caliper_len")]
    pub caliper_len: f32,
    #[serde(default = "default_caliper_width")]
    pub caliper_width: f32,
    #[serde(default)]
    pub measure: Option<MeasureConfigIn>,
    #[serde(default)]
    pub fit: Option<FitConfigIn>,
}

fn default_n_calipers() -> usize {
    12
}
fn default_caliper_len() -> f32 {
    20.0
}
fn default_caliper_width() -> f32 {
    5.0
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct FixtureIn {
    pub x: f32,
    pub y: f32,
    #[serde(default)]
    pub angle: f32,
    #[serde(default = "default_scale")]
    pub scale: f32,
}

fn default_scale() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlaneIn {
    #[serde(default)]
    pub nx: f32,
    #[serde(default)]
    pub ny: f32,
    #[serde(default = "default_nz")]
    pub nz: f32,
    #[serde(default)]
    pub d: f32,
}

fn default_nz() -> f32 {
    1.0
}

impl Default for PlaneIn {
    fn default() -> Self {
        Self {
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            d: 0.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MeasureRequest {
    pub image_id: String,
    pub model_id: String,
    #[serde(default)]
    pub fixture: Option<FixtureIn>,
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    pub objects: Vec<MeasureObjectIn>,
    #[serde(default)]
    pub calibration_id: Option<String>,
    #[serde(default)]
    pub camera_index: usize,
    #[serde(default)]
    pub plane: PlaneIn,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeMarkOut {
    pub pos_px: f32,
    pub polarity: String,
    pub x_mm: Option<f32>,
    pub y_mm: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaliperProfileOut {
    pub values: Vec<f32>,
    pub step_px: f32,
    pub edges: Vec<EdgeMarkOut>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaliperResultOut {
    pub index: usize,
    pub status: &'static str,
    pub reason: Option<String>,
    pub profile: CaliperProfileOut,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct OverlayPrimitiveOut {
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tone: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cx: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cy: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x1: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y1: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x2: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y2: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cross: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MeasureObjectResultOut {
    pub kind: &'static str,
    pub label: Option<String>,
    pub message: Option<String>,
    pub circle_cx: Option<f32>,
    pub circle_cy: Option<f32>,
    pub circle_r: Option<f32>,
    pub line_px: Option<f32>,
    pub line_py: Option<f32>,
    pub line_dx: Option<f32>,
    pub line_dy: Option<f32>,
    pub rms: Option<f32>,
    pub max_dev: Option<f32>,
    pub n_used: Option<usize>,
    pub circle_cx_mm: Option<f32>,
    pub circle_cy_mm: Option<f32>,
    pub circle_r_mm: Option<f32>,
    pub calipers: Vec<CaliperResultOut>,
    pub overlay: Vec<OverlayPrimitiveOut>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MeasureResponse {
    pub fixture: FixtureIn,
    pub fixture_source: &'static str,
    pub objects: Vec<MeasureObjectResultOut>,
}

// -- rectify -------------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct CropSpecIn {
    pub rect: [f32; 4],
    #[serde(default = "default_px_per_unit")]
    pub px_per_unit: f32,
    #[serde(default = "default_true")]
    pub normalize_scale: bool,
}

fn default_px_per_unit() -> f32 {
    1.0
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct RectifyRequest {
    pub image_id: String,
    pub model_id: String,
    pub crop: CropSpecIn,
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    #[serde(default)]
    pub max_matches: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RectifyMatchOut {
    pub index: usize,
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub scale: f32,
    pub score: f32,
    pub support: usize,
    pub level: usize,
    pub validity: f32,
    /// A key into `image_data`-style lookup: `"{image_id}/{model_id}/{index}"`.
    /// The desktop shell has no HTTP crop-cache URL — `tauriBackend.ts` builds an
    /// object URL from the `rectify_crop` command instead (see its own docs).
    pub crop_key: String,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RectifyResponse {
    pub width: usize,
    pub height: usize,
    pub matches: Vec<RectifyMatchOut>,
}

// -- displacement --------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct DisplacementRequest {
    pub image_ids: Vec<String>,
    pub window: [f32; 4],
    #[serde(default = "default_search")]
    pub search_x: i32,
    #[serde(default = "default_search")]
    pub search_y: i32,
    #[serde(default = "default_refine")]
    pub refine: String,
    #[serde(default = "default_lk_iters")]
    pub lk_iters: u32,
    #[serde(default = "default_disp_min_score")]
    pub min_score: f32,
}

fn default_search() -> i32 {
    12
}
fn default_refine() -> String {
    "lucas_kanade".to_string()
}
fn default_lk_iters() -> u32 {
    3
}
fn default_disp_min_score() -> f32 {
    0.5
}

#[derive(Debug, Clone, Serialize)]
pub struct DisplacementPairOut {
    pub from_image_id: String,
    pub to_image_id: String,
    pub dx: f32,
    pub dy: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DisplacementResponse {
    pub pairs: Vec<DisplacementPairOut>,
    pub cumulative_x: Vec<f32>,
    pub cumulative_y: Vec<f32>,
}

// -- calibration ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationOut {
    pub id: String,
    pub filename: String,
    pub format: String,
    pub n_cameras: usize,
}

// -- progress events -------------------------------------------------------------------------

/// Payload for the `"lab://progress"` event a long-running command emits at least twice
/// (`stage: "started"` then `stage: "finished"`), per the plan's "wire one real progress
/// case" — wired on `find`, the only command here whose cost scales with image size in a
/// way a user would notice.
#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    pub op: &'static str,
    pub stage: &'static str,
    pub elapsed_ms: Option<f64>,
}
