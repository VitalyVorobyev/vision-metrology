//! Python-visible `corr` config mirrors: `CorrTemplateConfig`, `CorrConfig`,
//! `DisplacementConfig`, and their nested tuning/refine types.

use core::num::NonZeroUsize;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use vision_metrology::corr::{
    CorrConfig as NativeCorrConfig, CorrMetric as NativeCorrMetric,
    CorrSearchTuning as NativeCorrSearchTuning, CorrTemplateConfig as NativeCorrTemplateConfig,
    CorrTemplateTuning as NativeCorrTemplateTuning, DisplacementConfig as NativeDisplacementConfig,
    Refine as NativeRefine,
};
use vm_primitives::Rect2f;

fn to_rect(roi: (f32, f32, f32, f32)) -> Rect2f {
    Rect2f {
        x: roi.0,
        y: roi.1,
        width: roi.2,
        height: roi.3,
    }
}

fn parse_metric(s: &str) -> PyResult<NativeCorrMetric> {
    match s {
        "zncc" => Ok(NativeCorrMetric::Zncc),
        "ssd" => Ok(NativeCorrMetric::Ssd),
        other => Err(PyValueError::new_err(format!(
            "metric must be 'zncc' or 'ssd', got '{other}'"
        ))),
    }
}

fn metric_str(m: NativeCorrMetric) -> String {
    match m {
        NativeCorrMetric::Zncc => "zncc".to_string(),
        NativeCorrMetric::Ssd => "ssd".to_string(),
    }
}

/// Tagged [`DisplacementConfig`]`.refine` — construct with `Refine.none()`
/// or `Refine.lucas_kanade(iters=...)`, never a bare string, since the
/// second variant carries a payload the first does not.
#[pyclass(frozen, eq, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Refine {
    is_none: bool,
    iters: u32,
}

#[pymethods]
impl Refine {
    /// Report corrmatch's own quadratic-peak subpixel estimate directly.
    #[staticmethod]
    pub fn none() -> Self {
        Self {
            is_none: true,
            iters: 0,
        }
    }

    /// Follow with translation-only inverse-compositional Lucas-Kanade.
    #[staticmethod]
    #[pyo3(signature = (iters=3))]
    pub fn lucas_kanade(iters: u32) -> Self {
        Self {
            is_none: false,
            iters,
        }
    }

    fn __repr__(&self) -> String {
        if self.is_none {
            "Refine.none()".to_string()
        } else {
            format!("Refine.lucas_kanade(iters={})", self.iters)
        }
    }
}

impl Refine {
    pub fn to_native(self) -> NativeRefine {
        if self.is_none {
            NativeRefine::None
        } else {
            NativeRefine::LucasKanade { iters: self.iters }
        }
    }

    fn from_native(r: NativeRefine) -> Self {
        match r {
            NativeRefine::None => Self::none(),
            NativeRefine::LucasKanade { iters } => Self::lucas_kanade(iters),
        }
    }
}

impl Default for Refine {
    fn default() -> Self {
        Self::from_native(NativeRefine::default())
    }
}

/// Mirrors `vision_metrology::corr::CorrTemplateTuning`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct CorrTemplateTuning {
    pub coarse_angle_step_deg: f32,
    pub min_angle_step_deg: f32,
    pub fill_value: u8,
    pub precompute_coarsest: bool,
}

impl Default for CorrTemplateTuning {
    fn default() -> Self {
        let t = NativeCorrTemplateTuning::default();
        Self {
            coarse_angle_step_deg: t.coarse_angle_step_deg,
            min_angle_step_deg: t.min_angle_step_deg,
            fill_value: t.fill_value,
            precompute_coarsest: t.precompute_coarsest,
        }
    }
}

#[pymethods]
impl CorrTemplateTuning {
    #[new]
    #[pyo3(signature = (
        coarse_angle_step_deg=None,
        min_angle_step_deg=None,
        fill_value=None,
        precompute_coarsest=None,
    ))]
    pub fn new(
        coarse_angle_step_deg: Option<f32>,
        min_angle_step_deg: Option<f32>,
        fill_value: Option<u8>,
        precompute_coarsest: Option<bool>,
    ) -> Self {
        let d = Self::default();
        Self {
            coarse_angle_step_deg: coarse_angle_step_deg.unwrap_or(d.coarse_angle_step_deg),
            min_angle_step_deg: min_angle_step_deg.unwrap_or(d.min_angle_step_deg),
            fill_value: fill_value.unwrap_or(d.fill_value),
            precompute_coarsest: precompute_coarsest.unwrap_or(d.precompute_coarsest),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CorrTemplateTuning(coarse_angle_step_deg={:.3}, min_angle_step_deg={:.3})",
            self.coarse_angle_step_deg, self.min_angle_step_deg
        )
    }
}

impl CorrTemplateTuning {
    pub fn to_native(self) -> NativeCorrTemplateTuning {
        NativeCorrTemplateTuning {
            coarse_angle_step_deg: self.coarse_angle_step_deg,
            min_angle_step_deg: self.min_angle_step_deg,
            fill_value: self.fill_value,
            precompute_coarsest: self.precompute_coarsest,
        }
    }
}

/// Mirrors `vision_metrology::corr::CorrTemplateConfig`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct CorrTemplateConfig {
    pub rotation: bool,
    /// `None` uses corrmatch's own default (6).
    pub max_levels: Option<usize>,
    pub tuning: CorrTemplateTuning,
}

impl Default for CorrTemplateConfig {
    fn default() -> Self {
        let c = NativeCorrTemplateConfig::default();
        Self {
            rotation: c.rotation,
            max_levels: c.max_levels.map(NonZeroUsize::get),
            tuning: CorrTemplateTuning::default(),
        }
    }
}

#[pymethods]
impl CorrTemplateConfig {
    #[new]
    #[pyo3(signature = (rotation=None, max_levels=None, tuning=None))]
    pub fn new(
        rotation: Option<bool>,
        max_levels: Option<usize>,
        tuning: Option<CorrTemplateTuning>,
    ) -> Self {
        let d = Self::default();
        Self {
            rotation: rotation.unwrap_or(d.rotation),
            max_levels: max_levels.or(d.max_levels),
            tuning: tuning.unwrap_or(d.tuning),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CorrTemplateConfig(rotation={}, max_levels={:?})",
            self.rotation, self.max_levels
        )
    }
}

impl CorrTemplateConfig {
    pub fn to_native(&self) -> NativeCorrTemplateConfig {
        NativeCorrTemplateConfig {
            rotation: self.rotation,
            max_levels: self.max_levels.and_then(NonZeroUsize::new),
            tuning: self.tuning.to_native(),
        }
    }
}

/// Mirrors `vision_metrology::corr::CorrSearchTuning`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct CorrSearchTuning {
    pub parallel: bool,
    pub max_image_levels: Option<usize>,
    pub beam_width: usize,
    pub per_angle_topk: usize,
    pub nms_radius: usize,
    pub roi_radius: usize,
    pub angle_half_range_steps: usize,
    pub min_var_i: f32,
}

impl Default for CorrSearchTuning {
    fn default() -> Self {
        let t = NativeCorrSearchTuning::default();
        Self {
            parallel: t.parallel,
            max_image_levels: t.max_image_levels.map(NonZeroUsize::get),
            beam_width: t.beam_width,
            per_angle_topk: t.per_angle_topk,
            nms_radius: t.nms_radius,
            roi_radius: t.roi_radius,
            angle_half_range_steps: t.angle_half_range_steps,
            min_var_i: t.min_var_i,
        }
    }
}

#[pymethods]
impl CorrSearchTuning {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        parallel=None,
        max_image_levels=None,
        beam_width=None,
        per_angle_topk=None,
        nms_radius=None,
        roi_radius=None,
        angle_half_range_steps=None,
        min_var_i=None,
    ))]
    pub fn new(
        parallel: Option<bool>,
        max_image_levels: Option<usize>,
        beam_width: Option<usize>,
        per_angle_topk: Option<usize>,
        nms_radius: Option<usize>,
        roi_radius: Option<usize>,
        angle_half_range_steps: Option<usize>,
        min_var_i: Option<f32>,
    ) -> Self {
        let d = Self::default();
        Self {
            parallel: parallel.unwrap_or(d.parallel),
            max_image_levels: max_image_levels.or(d.max_image_levels),
            beam_width: beam_width.unwrap_or(d.beam_width),
            per_angle_topk: per_angle_topk.unwrap_or(d.per_angle_topk),
            nms_radius: nms_radius.unwrap_or(d.nms_radius),
            roi_radius: roi_radius.unwrap_or(d.roi_radius),
            angle_half_range_steps: angle_half_range_steps.unwrap_or(d.angle_half_range_steps),
            min_var_i: min_var_i.unwrap_or(d.min_var_i),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CorrSearchTuning(beam_width={}, nms_radius={})",
            self.beam_width, self.nms_radius
        )
    }
}

impl CorrSearchTuning {
    pub fn to_native(self) -> NativeCorrSearchTuning {
        NativeCorrSearchTuning {
            parallel: self.parallel,
            max_image_levels: self.max_image_levels.and_then(NonZeroUsize::new),
            beam_width: self.beam_width,
            per_angle_topk: self.per_angle_topk,
            nms_radius: self.nms_radius,
            roi_radius: self.roi_radius,
            angle_half_range_steps: self.angle_half_range_steps,
            min_var_i: self.min_var_i,
        }
    }
}

/// Mirrors `vision_metrology::corr::CorrConfig`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct CorrConfig {
    pub rotation: bool,
    /// `'zncc'` or `'ssd'`.
    pub metric: String,
    pub min_score: Option<f32>,
    pub tuning: CorrSearchTuning,
}

impl Default for CorrConfig {
    fn default() -> Self {
        let c = NativeCorrConfig::default();
        Self {
            rotation: c.rotation,
            metric: metric_str(c.metric),
            min_score: c.min_score,
            tuning: CorrSearchTuning::default(),
        }
    }
}

#[pymethods]
impl CorrConfig {
    #[new]
    #[pyo3(signature = (rotation=None, metric=None, min_score=None, tuning=None))]
    pub fn new(
        rotation: Option<bool>,
        metric: Option<String>,
        min_score: Option<f32>,
        tuning: Option<CorrSearchTuning>,
    ) -> Self {
        let d = Self::default();
        Self {
            rotation: rotation.unwrap_or(d.rotation),
            metric: metric.unwrap_or(d.metric),
            min_score: min_score.or(d.min_score),
            tuning: tuning.unwrap_or(d.tuning),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CorrConfig(rotation={}, metric='{}', min_score={:?})",
            self.rotation, self.metric, self.min_score
        )
    }
}

impl CorrConfig {
    pub fn to_native(&self) -> PyResult<NativeCorrConfig> {
        Ok(NativeCorrConfig {
            rotation: self.rotation,
            metric: parse_metric(&self.metric)?,
            min_score: self.min_score,
            tuning: self.tuning.to_native(),
        })
    }
}

/// Mirrors `vision_metrology::corr::DisplacementConfig`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct DisplacementConfig {
    /// `(x, y, width, height)` window in `prev`, level-0 pixel coordinates.
    pub window: (f32, f32, f32, f32),
    /// Search radius `(x, y)` in `curr` around the window's `prev` position.
    pub search: (i32, i32),
    pub refine: Refine,
    pub min_score: f32,
}

#[pymethods]
impl DisplacementConfig {
    #[new]
    #[pyo3(signature = (window, search=(12, 12), refine=None, min_score=0.5))]
    pub fn new(
        window: (f32, f32, f32, f32),
        search: (i32, i32),
        refine: Option<Refine>,
        min_score: f32,
    ) -> Self {
        Self {
            window,
            search,
            refine: refine.unwrap_or_default(),
            min_score,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DisplacementConfig(window={:?}, search={:?}, min_score={:.3})",
            self.window, self.search, self.min_score
        )
    }
}

impl DisplacementConfig {
    pub fn to_native(&self) -> NativeDisplacementConfig {
        NativeDisplacementConfig {
            window: to_rect(self.window),
            search: self.search,
            refine: self.refine.to_native(),
            min_score: self.min_score,
        }
    }
}
