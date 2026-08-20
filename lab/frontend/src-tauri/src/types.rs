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
    /// The user's own file, for an image opened by path — `None` for one
    /// uploaded as bytes. Desktop-only; the browser shell has no such notion.
    pub path: Option<String>,
}

/// A thumbnail finished warming — `lab://thumb`.
#[derive(Debug, Clone, Serialize)]
pub struct ThumbEvent {
    pub image_id: String,
    pub done: usize,
    pub total: usize,
}

/// One image file found by `images_scan_dir`, described **without decoding
/// it**: name, size on disk, and the dimensions from its header.
#[derive(Debug, Clone, Serialize)]
pub struct DirEntryOut {
    pub path: String,
    pub name: String,
    pub bytes: u64,
    pub width: u32,
    pub height: u32,
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
    /// Contour ids from `teach_preview` to keep. `None` keeps everything the
    /// ROI holds — the old, rectangle-only behaviour. An empty list is a
    /// caller who deselected everything, and is an error rather than a model
    /// with no points.
    #[serde(default)]
    pub keep_contours: Option<Vec<usize>>,
    /// Reference point in image coordinates. `None` = the level-0 centroid.
    #[serde(default)]
    pub origin: Option<[f32; 2]>,
    /// The part's natural 0° direction in the reference image, radians.
    #[serde(default)]
    pub reference_angle: f32,
}

/// One candidate contour offered by `teach_preview`, for the caller to keep or
/// drop before the model is built.
#[derive(Debug, Clone, Serialize)]
pub struct ContourOut {
    pub id: usize,
    /// Polyline vertices in image coordinates, `[x, y]` pairs, flattened —
    /// a flat array rather than nested pairs because these go straight into
    /// an SVG path and there can be thousands of them.
    pub points: Vec<f32>,
    pub closed: bool,
    /// Contour length in pixels, and the mean gradient magnitude along it —
    /// what a UI sorts by to find the background structure worth dropping.
    pub length: f32,
    pub mean_strength: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCropRequest {
    pub model_id: String,
    /// Crop rectangle in model-frame coordinates, `[x, y, w, h]`.
    pub rect: [f32; 4],
    #[serde(default = "default_px_per_unit")]
    pub px_per_unit: f32,
}

fn default_px_per_unit() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct TeachPreviewRequest {
    pub image_id: String,
    pub roi: [f32; 4],
    #[serde(default = "default_min_contrast")]
    pub min_contrast: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TeachPreviewResponse {
    pub contours: Vec<ContourOut>,
    pub total_points: usize,
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
    /// The canonical orientation the model frame is rotated onto, radians.
    pub reference_angle: f32,
}

/// A model's own points, so a UI can draw what was actually learned rather
/// than print how many points there are.
#[derive(Debug, Clone, Serialize)]
pub struct ModelGeometryOut {
    pub model_id: String,
    pub level: usize,
    pub origin: [f32; 2],
    pub reference_angle: f32,
    /// `[x, y, dx, dy]` per point, flattened: position plus unit gradient
    /// direction. Flat because a model level is up to 512 points and this
    /// crosses the IPC boundary as JSON.
    pub points: Vec<f32>,
    /// Which frame `points` are in: `"reference"` (the teach image's own axes)
    /// or `"model"` (rotated onto the canonical orientation — the frame a
    /// match's pose consumes).
    pub frame: String,
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
    #[serde(default)]
    pub scale_range: Option<(f32, f32)>,
    /// `"none"`, `"interpolate"` (default) or `"least_squares"`.
    #[serde(default)]
    pub refinement: Option<String>,
    /// Scene gradient floor as a fraction of the image's dynamic range.
    #[serde(default)]
    pub min_contrast: Option<f32>,
    #[serde(default)]
    pub tuning: Option<SearchTuningIn>,
}

/// The search-effort knobs, mirroring `matching::ShapeSearchTuning`. Every
/// field trades run time against the chance of missing a match `min_score`
/// says should be reported; `None` keeps the library's own default.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct SearchTuningIn {
    pub greediness: Option<f32>,
    pub angle_step: Option<f32>,
    pub scale_step: Option<f32>,
    pub last_level: Option<usize>,
    pub max_candidates: Option<usize>,
    pub coarse_score_factor: Option<f32>,
}

/// One image's worth of `batch_find` result.
#[derive(Debug, Clone, Serialize)]
pub struct BatchFindItemOut {
    pub image_id: String,
    pub matches: Vec<MatchOut>,
    pub elapsed_ms: f64,
    /// Set when this image failed; `matches` is then empty. One unreadable
    /// frame must not abandon a run over three thousand of them.
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BatchFindRequest {
    pub model_id: String,
    pub image_ids: Vec<String>,
    #[serde(flatten)]
    pub search: BatchSearchIn,
}

/// The search half of a `BatchFindRequest`, i.e. a `FindRequest` without the
/// per-image fields.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct BatchSearchIn {
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    #[serde(default)]
    pub max_matches: Option<usize>,
    #[serde(default)]
    pub angle_range: Option<(f32, f32)>,
    #[serde(default)]
    pub scale_range: Option<(f32, f32)>,
    #[serde(default)]
    pub refinement: Option<String>,
    #[serde(default)]
    pub min_contrast: Option<f32>,
    #[serde(default)]
    pub tuning: Option<SearchTuningIn>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchFindResponse {
    pub items: Vec<BatchFindItemOut>,
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
    /// The rest of the search that produced the match list the caller is
    /// rectifying **by index**.
    ///
    /// `rectify` runs the search itself to place its crops, so it has to run
    /// the *same* one: a narrowed angle sweep on the caller's side and a full
    /// 360° here would return a different set, and index `2` would then name a
    /// different instance in the crop cache than it does in the caller's table.
    /// Every field defaults, so a request that does not care is unaffected —
    /// including the committed contract fixtures, which predate them.
    #[serde(default)]
    pub angle_range: Option<(f32, f32)>,
    #[serde(default)]
    pub scale_range: Option<(f32, f32)>,
    #[serde(default)]
    pub refinement: Option<String>,
    #[serde(default)]
    pub min_contrast: Option<f32>,
    #[serde(default)]
    pub tuning: Option<SearchTuningIn>,
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
