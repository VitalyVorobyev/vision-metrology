//! CaliperBench's JSONL protocol over [`Caliper`]: requests in, prediction rows out.
//!
//! A request is a strip (`start_xy`, `end_xy`, `width_px`, `samples`, `across`) plus an
//! ordered list of zero, one or two polarities; a prediction row is the edge distances from
//! `start`, or a failure reason. The methods reproduce CaliperBench's textbook baselines
//! operator for operator, so that the same request gives the same row:
//!
//! - the strip becomes a [`MeasureStrip`] with its exact endpoints and sample counts;
//! - `sigma` and `radius` are in samples there and in pixels here, converted with the
//!   strip's spacing;
//! - the derivative is a Gaussian then central differences (`np.gradient`);
//! - bounds are strict ([`OffImage::Reject`]), and there is no obliquity gate;
//! - `min_response` is the threshold.
//!
//! `gradient_gaussian` and `half_contrast` have no CaliperBench counterpart; they use the
//! same strip, smoothing and selection with a log-parabola refinement and a half-contrast
//! refinement.
//!
//! Shared by `examples/caliperbench_run.rs` (the command-line runner) and
//! `tests/caliperbench_protocol.rs` (the golden cross-check).

// `#[path]`-included from an example and a test, which each use a subset of it.
#![allow(dead_code)]

use std::collections::HashSet;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use image::DynamicImage;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use vision_metrology::measure::diagnostics::explain;
use vision_metrology::measure::{
    Caliper, Derivative, EdgeSelect, EdgeSequence, Locate, MeasureConfig, MeasureStrip, OffImage,
    PolaritySelect, ProfileConfig, RejectReason,
};
use vision_metrology::vm_primitives::LevelCrossing1D;
use vision_metrology::{BorderMode, EdgePolarity, Image, ImageView, Point2f, SubpixRefine};

/// The edge-location methods the runner offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Gradient peak, three-point parabola (CaliperBench's default baseline).
    GradientParabolic,
    /// Gradient peak at its integer sample.
    GradientInteger,
    /// Gradient peak, parabola through the logarithms.
    GradientGaussian,
    /// The crossing of the mean of the two end levels nearest the middle.
    MidpointCrossing,
    /// Gradient peaks moved to their local half-contrast crossings.
    HalfContrast,
}

impl Method {
    /// Every method by its protocol name.
    pub const ALL: [(&'static str, Method); 5] = [
        ("gradient_parabolic", Method::GradientParabolic),
        ("gradient_integer", Method::GradientInteger),
        ("gradient_gaussian", Method::GradientGaussian),
        ("midpoint_crossing", Method::MidpointCrossing),
        ("half_contrast", Method::HalfContrast),
    ];

    /// The method called `name`.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.iter().find(|(n, _)| *n == name).map(|&(_, m)| m)
    }

    /// The protocol name.
    pub fn name(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|&&(_, m)| m == self)
            .map_or("", |&(n, _)| n)
    }

    fn is_gradient(self) -> bool {
        !matches!(self, Method::MidpointCrossing)
    }
}

/// The derivative operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Operator {
    /// A Gaussian of half-width `radius`, then central differences — `np.gradient`.
    #[default]
    Central,
    /// The analytic derivative of a Gaussian (half-width `ceil(3σ)`); `radius` is unused.
    Dog,
}

impl Operator {
    /// The operator called `name`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "central" => Some(Self::Central),
            "dog" => Some(Self::Dog),
            _ => None,
        }
    }
}

/// CaliperBench's `BaselineParams`: the same keys, the same defaults, the same checks.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// Gaussian smoothing of the profile, in samples.
    pub sigma: f64,
    /// Smoothing kernel half-width, in samples.
    pub radius: usize,
    /// Gradient peaks at or below this are ignored.
    pub min_response: f64,
    /// Midpoint: samples at each end whose median gives that end's level.
    pub endpoint_samples: usize,
    /// Midpoint: required end-to-end contrast; half-contrast: required flank contrast.
    pub min_contrast: f64,
    /// Strip overrides; `None` keeps the request's value.
    pub width_px: Option<f64>,
    pub across: Option<usize>,
    pub samples: Option<usize>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            radius: 3,
            min_response: 0.01,
            endpoint_samples: 3,
            min_contrast: 0.05,
            width_px: None,
            across: None,
            samples: None,
        }
    }
}

impl Params {
    /// Parse a JSON object of overrides. Unknown keys and invalid values are errors.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(text).map_err(|e| format!("params: {e}"))?;
        let Value::Object(map) = value else {
            return Err("params: expected a JSON object".into());
        };
        let mut p = Params::default();
        for (key, v) in &map {
            let bad = || format!("params: invalid value for {key}: {v}");
            let number = || v.as_f64().filter(|x| x.is_finite()).ok_or_else(bad);
            let count = || {
                v.as_f64()
                    .filter(|x| x.fract() == 0.0 && *x >= 0.0 && *x <= 1e9)
                    .map(|x| x as usize)
                    .ok_or_else(bad)
            };
            match key.as_str() {
                "sigma" => p.sigma = number()?,
                "radius" => p.radius = count()?,
                "min_response" => p.min_response = number()?,
                "endpoint_samples" => p.endpoint_samples = count()?,
                "min_contrast" => p.min_contrast = number()?,
                "width_px" => p.width_px = if v.is_null() { None } else { Some(number()?) },
                "across" => p.across = if v.is_null() { None } else { Some(count()?) },
                "samples" => p.samples = if v.is_null() { None } else { Some(count()?) },
                _ => return Err(format!("params: unknown baseline parameter {key:?}")),
            }
        }
        if p.sigma <= 0.0 || p.radius < 1 || p.endpoint_samples < 1 {
            return Err(
                "params: sigma must be positive; radius and endpoint_samples at least 1".into(),
            );
        }
        if p.width_px.is_some_and(|w| w < 1.0)
            || p.across.is_some_and(|a| !(1..=1000).contains(&a))
            || p.samples.is_some_and(|s| !(5..=100_000).contains(&s))
        {
            return Err("params: width_px at least 1, across 1..=1000, samples 5..=100000".into());
        }
        Ok(p)
    }
}

/// A requested transition, along the scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Polarity {
    Rising,
    Falling,
    Either,
}

impl Polarity {
    fn select(self) -> PolaritySelect {
        match self {
            Polarity::Rising => PolaritySelect::Rising,
            Polarity::Falling => PolaritySelect::Falling,
            Polarity::Either => PolaritySelect::Any,
        }
    }
}

/// A scan strip, in image pixels with `(0, 0)` at the top-left pixel centre.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Strip {
    pub start_xy: [f64; 2],
    pub end_xy: [f64; 2],
    #[serde(default = "one")]
    pub width_px: f64,
    #[serde(default = "default_samples")]
    pub samples: usize,
    #[serde(default = "one_count")]
    pub across: usize,
}

fn one() -> f64 {
    1.0
}
fn one_count() -> usize {
    1
}
fn default_samples() -> usize {
    101
}

impl Strip {
    fn length(&self) -> f64 {
        (self.end_xy[0] - self.start_xy[0]).hypot(self.end_xy[1] - self.start_xy[1])
    }
}

/// One protocol request.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    #[serde(default = "one_count")]
    pub schema_version: usize,
    pub sample_id: String,
    pub image: String,
    pub image_sha256: String,
    pub strip: Strip,
    pub polarities: Vec<Polarity>,
}

impl Request {
    /// The checks CaliperBench's `Request` model makes.
    fn validate(&self) -> Result<(), String> {
        let s = &self.strip;
        let path = Path::new(&self.image);
        let relative = !self.image.is_empty()
            && !self.image.contains('\\')
            && path
                .components()
                .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
        let hex = self.image_sha256.len() == 64
            && self
                .image_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        let problem = if self.schema_version != 1 {
            "unsupported schema_version"
        } else if self.sample_id.is_empty() {
            "empty sample_id"
        } else if !relative {
            "image must be a relative POSIX path under the data root"
        } else if !hex {
            "image_sha256 must be 64 lowercase hex digits"
        } else if self.polarities.len() > 2 {
            "at most two polarities"
        } else if s.length() <= 0.0 {
            "strip endpoints must differ"
        } else if s.width_px < 1.0 || !(5..=100_000).contains(&s.samples) {
            "width_px at least 1, samples 5..=100000"
        } else if !(1..=1000).contains(&s.across) || (s.width_px > 1.0 && s.across < 2) {
            "across 1..=1000, and at least 2 for a wide strip"
        } else if s.samples * s.across > 2_000_000 {
            "strip exceeds the sampling budget"
        } else {
            return Ok(());
        };
        Err(format!("request {:?}: {problem}", self.sample_id))
    }
}

/// Read a requests file: one JSON object per non-empty line, every one valid, ids unique.
pub fn read_requests(path: &Path) -> Result<Vec<Request>, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut ids = HashSet::new();
    let mut out = Vec::new();
    for (n, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let req: Request =
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
        req.validate()?;
        if !ids.insert(req.sample_id.clone()) {
            return Err(format!("duplicate sample id {:?}", req.sample_id));
        }
        out.push(req);
    }
    Ok(out)
}

/// One prediction row: the edges, or why there are none.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    pub sample_id: String,
    /// Edge distances from `start`, in pixels, increasing; or the failure reason.
    pub outcome: Result<Vec<f64>, &'static str>,
    pub runtime_ms: f64,
}

impl Prediction {
    /// The protocol row: `schema_version`, `sample_id`, `status`, `edges_px`, `runtime_ms`,
    /// and `reason` when it failed. Nothing else, and nothing non-finite.
    pub fn to_json(&self) -> Value {
        let runtime_ms = if self.runtime_ms.is_finite() {
            self.runtime_ms.max(0.0)
        } else {
            0.0
        };
        let mut row = Map::new();
        row.insert("schema_version".into(), json!(1));
        row.insert("sample_id".into(), json!(self.sample_id));
        match &self.outcome {
            Ok(edges) => {
                row.insert("status".into(), json!("ok"));
                row.insert("edges_px".into(), json!(edges));
                row.insert("runtime_ms".into(), json!(runtime_ms));
            }
            Err(reason) => {
                row.insert("status".into(), json!("failed"));
                row.insert("edges_px".into(), json!([]));
                row.insert("runtime_ms".into(), json!(runtime_ms));
                row.insert("reason".into(), json!(reason));
            }
        }
        Value::Object(row)
    }
}

/// Flank windows of `half_contrast`, in pixels: CaliperBench's reference edge definition.
const FLANK_NEAR_PX: f32 = 3.0;
const FLANK_FAR_PX: f32 = 8.0;
const FLANK_TOL_PX: f32 = 0.01;
const FLANK_ITERATIONS: usize = 5;

/// A configured method with its reusable caliper.
#[derive(Debug, Clone)]
pub struct Runner {
    method: Method,
    operator: Operator,
    params: Params,
    cal: Caliper,
    levels: LevelCrossing1D,
}

impl Runner {
    pub fn new(method: Method, operator: Operator, params: Params) -> Self {
        let placeholder = MeasureStrip {
            start: Point2f::new(0.0, 0.0),
            end: Point2f::new(1.0, 0.0),
            half_width: 0.0,
            samples: None,
            across: None,
        };
        Self {
            method,
            operator,
            params,
            cal: Caliper::strip(placeholder, MeasureConfig::default()),
            levels: LevelCrossing1D::new(),
        }
    }

    /// Measure one request; with `trace`, also return CaliperBench's lab trace keys
    /// (`step`, `profile`, `smooth`, then `gradient` and `candidates` or `levels` and
    /// `threshold`). The trace is computed after the timed measurement.
    pub fn predict(
        &mut self,
        img: &ImageView<'_, f32>,
        req: &Request,
        trace: bool,
    ) -> (Prediction, Option<Map<String, Value>>) {
        let started = Instant::now();
        let strip = self.strip(&req.strip);
        self.cal.set_strip(strip);
        let spacing = self.cal.spacing();
        self.cal.set_config(self.config(spacing, &req.polarities));
        let measured = self
            .cal
            .measure(img)
            .map(|edges| edges.iter().map(|e| f64::from(e.t)).collect::<Vec<_>>());
        let outcome = self.outcome(measured, &req.polarities);
        let runtime_ms = started.elapsed().as_secs_f64() * 1e3;
        let prediction = Prediction {
            sample_id: req.sample_id.clone(),
            outcome,
            runtime_ms,
        };
        let trace = trace.then(|| self.trace(img, req));
        (prediction, trace)
    }

    /// The request's strip with the parameter overrides, as a [`MeasureStrip`].
    ///
    /// Transverse offsets are numpy's `linspace(−h, h, across)` with `h = (width − 1)/2`.
    /// One line of a wide strip is `linspace`'s first point, `−h`, so the strip moves by
    /// `−h` along the normal.
    fn strip(&self, s: &Strip) -> MeasureStrip {
        let width = self.params.width_px.unwrap_or(s.width_px);
        let across = self.params.across.unwrap_or(s.across);
        let samples = self.params.samples.unwrap_or(s.samples);
        let ([sx, sy], [ex, ey]) = (s.start_xy, s.end_xy);
        let length = s.length();
        let (ux, uy) = ((ex - sx) / length, (ey - sy) / length);
        let (nx, ny) = (-uy, ux);
        let h = (width - 1.0) / 2.0;
        let (shift, half_width) = if across == 1 && h > 0.0 {
            (-h, 0.0)
        } else {
            (0.0, h)
        };
        MeasureStrip {
            start: Point2f::new((sx + shift * nx) as f32, (sy + shift * ny) as f32),
            end: Point2f::new((ex + shift * nx) as f32, (ey + shift * ny) as f32),
            half_width: half_width as f32,
            samples: NonZeroUsize::new(samples),
            across: NonZeroUsize::new(across),
        }
    }

    /// The caliper config for a strip sampled every `spacing` pixels.
    fn config(&self, spacing: f32, polarities: &[Polarity]) -> MeasureConfig {
        let p = &self.params;
        let derivative = match self.operator {
            Operator::Central => Derivative::SmoothThenCentral {
                radius_px: p.radius as f32 * spacing,
            },
            Operator::Dog => Derivative::DerivativeOfGaussian,
        };
        let gradient = |refine| Locate::GradientPeak { refine };
        let locate = match self.method {
            Method::GradientParabolic => gradient(SubpixRefine::Parabolic3),
            Method::GradientInteger => gradient(SubpixRefine::None),
            Method::GradientGaussian => gradient(SubpixRefine::Gaussian3),
            Method::MidpointCrossing => Locate::MidpointCrossing {
                endpoint_samples: NonZeroUsize::new(p.endpoint_samples)
                    .unwrap_or(NonZeroUsize::MIN),
                min_contrast: p.min_contrast as f32,
            },
            Method::HalfContrast => Locate::HalfContrast {
                flank_near_px: FLANK_NEAR_PX,
                flank_far_px: FLANK_FAR_PX,
                tol_px: FLANK_TOL_PX,
                max_iter: NonZeroUsize::new(FLANK_ITERATIONS).unwrap_or(NonZeroUsize::MIN),
                min_contrast: p.min_contrast as f32,
            },
        };
        let select = match polarities {
            [] => EdgeSelect::Strongest,
            [first, rest @ ..] => EdgeSelect::StrongestInOrder(EdgeSequence {
                first: first.select(),
                second: rest.first().map(|p| p.select()),
            }),
        };
        MeasureConfig {
            threshold: p.min_response as f32,
            polarity: PolaritySelect::Any,
            select,
            locate,
            max_obliquity_deg: 180.0,
            profile: ProfileConfig {
                sigma: p.sigma as f32 * spacing,
                derivative,
                step: 1.0,
                border: BorderMode::Clamp,
                off_image: OffImage::Reject,
            },
        }
    }

    /// The row for a measurement, with CaliperBench's reasons in its order: bounds, the
    /// midpoint's one-edge rule, then the method's own checks.
    fn outcome(
        &self,
        measured: Result<Vec<f64>, RejectReason>,
        polarities: &[Polarity],
    ) -> Result<Vec<f64>, &'static str> {
        use RejectReason as R;
        let midpoint = self.method == Method::MidpointCrossing;
        let edges = match measured {
            Err(R::OffImage) => return Err("strip_out_of_bounds"),
            _ if midpoint && polarities.len() != 1 => {
                return Err("midpoint_crossing_requires_one_edge");
            }
            Ok(edges) => edges,
            // A negative task with nothing found is a correct "no edge".
            Err(R::NoEdge | R::LowContrast | R::NoCrossing) if polarities.is_empty() => Vec::new(),
            Err(R::LowContrast) if midpoint => return Err("insufficient_endpoint_contrast"),
            Err(R::LowContrast) => return Err("low_flank_contrast"),
            Err(R::WrongPolarity) if midpoint => return Err("wrong_polarity"),
            Err(R::NoCrossing) => return Err("missing_crossing"),
            // Half-contrast edges that converged on one crossing: the second has none.
            Err(R::IncompleteSequence) if !self.cal.levels().is_empty() => {
                return Err("missing_crossing");
            }
            Err(R::NoEdge | R::WrongPolarity | R::IncompleteSequence) => {
                return Err("missing_peak");
            }
            Err(R::ProfileTooShort) => return Err("profile_too_short"),
            Err(R::TooOblique) => return Err("too_oblique"),
        };
        if edges.iter().any(|e| !e.is_finite()) {
            return Err("non_finite_edge");
        }
        Ok(edges)
    }

    /// CaliperBench's lab trace for the last request, from [`explain`].
    fn trace(&mut self, img: &ImageView<'_, f32>, req: &Request) -> Map<String, Value> {
        let mut out = Map::new();
        out.insert("sample_id".into(), json!(req.sample_id));
        let t = explain(&mut self.cal, img);
        // CaliperBench rejects an out-of-bounds strip before it traces anything.
        if t.reject == Some(RejectReason::OffImage) {
            return out;
        }
        out.insert("step".into(), json!(f64::from(t.spacing)));
        out.insert("profile".into(), json!(t.profile));
        out.insert("smooth".into(), json!(t.smoothed));
        if self.method.is_gradient() {
            out.insert("gradient".into(), json!(t.response));
            // Rising candidates first, then falling, each along the scan.
            let mut cands = t.candidates;
            cands.sort_by_key(|c| c.polarity == EdgePolarity::Falling);
            let rows: Vec<Value> = cands
                .iter()
                .map(|c| {
                    let polarity = match c.polarity {
                        EdgePolarity::Rising => "rising",
                        EdgePolarity::Falling => "falling",
                    };
                    json!([f64::from(c.amplitude), f64::from(c.t), polarity])
                })
                .collect();
            out.insert("candidates".into(), Value::Array(rows));
        } else if req.polarities.len() == 1 && !t.smoothed.is_empty() {
            let n = NonZeroUsize::new(self.params.endpoint_samples).unwrap_or(NonZeroUsize::MIN);
            let (before, after) = self.levels.end_levels(&t.smoothed, n);
            let (before, after) = (f64::from(before), f64::from(after));
            out.insert("levels".into(), json!([before, after]));
            out.insert("threshold".into(), json!((before + after) / 2.0));
        }
        out
    }
}

/// Pillow's `convert("L")` luma of an 8-bit RGB pixel (ITU-R 601-2, fixed point).
pub fn pillow_luma(r: u8, g: u8, b: u8) -> u8 {
    ((19595 * u32::from(r) + 38470 * u32::from(g) + 7471 * u32::from(b) + 0x8000) >> 16) as u8
}

/// An 8-bit grayscale image as `f32` in `[0, 1]`, as `np.array(im) / 255` reads it.
pub fn gray_to_unit(width: usize, height: usize, pixels: &[u8]) -> Image<f32> {
    let data = pixels.iter().map(|&v| f32::from(v) / 255.0).collect();
    Image::from_vec(width, height, data).expect("pixel count matches the size")
}

/// Load an image the way CaliperBench's runner does: 8-bit grayscale (`L`) as is, 8-bit
/// RGB through Pillow's luma, anything else refused. Errors are row failure reasons.
pub fn load_gray(path: &Path) -> Result<Image<f32>, &'static str> {
    let bytes = fs::read(path).map_err(|_| "image_unreadable")?;
    // A palette PNG decodes to RGB here but is mode "P" to Pillow, which the reference
    // refuses.
    let png_palette = bytes.starts_with(b"\x89PNG\r\n\x1a\n") && bytes.get(25) == Some(&3);
    let img = image::load_from_memory(&bytes).map_err(|_| "image_unreadable")?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let gray: Vec<u8> = match img {
        DynamicImage::ImageLuma8(g) => g.into_raw(),
        DynamicImage::ImageRgb8(c) if !png_palette => c
            .as_raw()
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&[r, g, b]| pillow_luma(r, g, b))
            .collect(),
        _ => return Err("unsupported_image_mode"),
    };
    if w < 2 || h < 2 {
        return Err("invalid_grayscale_image");
    }
    Ok(gray_to_unit(w, h, &gray))
}

/// Everything one run needs.
#[derive(Debug, Clone)]
pub struct RunOptions {
    pub method: Method,
    pub operator: Operator,
    pub params: Params,
    pub requests: PathBuf,
    pub data_root: PathBuf,
    pub output: PathBuf,
    pub trace: Option<PathBuf>,
}

/// What a run wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunSummary {
    pub ok: usize,
    pub failed: usize,
}

/// Run every request and write one prediction row per request, in request order.
///
/// Each image is loaded once: requests are measured grouped by image. A request whose
/// image cannot be used fails on its own row. `Err` means nothing usable was written: the
/// requests file could not be read, or an output could not be written.
pub fn run(opts: &RunOptions) -> Result<RunSummary, String> {
    let requests = read_requests(&opts.requests)?;
    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_by(|&a, &b| requests[a].image.cmp(&requests[b].image));

    let mut runner = Runner::new(opts.method, opts.operator, opts.params.clone());
    let mut rows: Vec<Option<Value>> = vec![None; requests.len()];
    let mut traces: Vec<Option<Value>> = vec![None; requests.len()];
    let mut start = 0;
    while start < order.len() {
        let name = &requests[order[start]].image;
        let end = start + order[start..].partition_point(|&i| &requests[i].image == name);
        let img = load_gray(&opts.data_root.join(name));
        for &i in &order[start..end] {
            let req = &requests[i];
            let (prediction, trace) = match &img {
                Ok(img) => runner.predict(&img.as_view(), req, opts.trace.is_some()),
                Err(reason) => {
                    let mut trace = Map::new();
                    trace.insert("sample_id".into(), json!(req.sample_id));
                    let row = Prediction {
                        sample_id: req.sample_id.clone(),
                        outcome: Err(*reason),
                        runtime_ms: 0.0,
                    };
                    (row, Some(trace))
                }
            };
            rows[i] = Some(prediction.to_json());
            traces[i] = trace.map(Value::Object);
        }
        start = end;
    }

    let summary = RunSummary {
        ok: rows
            .iter()
            .flatten()
            .filter(|r| r["status"] == "ok")
            .count(),
        failed: rows
            .iter()
            .flatten()
            .filter(|r| r["status"] == "failed")
            .count(),
    };
    write_jsonl(&opts.output, rows.into_iter().flatten())?;
    if let Some(path) = &opts.trace {
        write_jsonl(path, traces.into_iter().flatten())?;
    }
    Ok(summary)
}

fn write_jsonl(path: &Path, rows: impl Iterator<Item = Value>) -> Result<(), String> {
    let mut text = String::new();
    for row in rows {
        text.push_str(&row.to_string());
        text.push('\n');
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    fs::write(path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
}
