//! Python-visible [`MeasureConfig`](vision_metrology::measure::MeasureConfig) mirror.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vision_metrology::measure::{
    EdgeSelect as NativeEdgeSelect, MeasureConfig as NativeMeasureConfig,
    PolaritySelect as NativePolaritySelect,
};
use vm_primitives::BorderMode;

/// Mirrors `vision_metrology::measure::MeasureConfig`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct MeasureConfig {
    /// Gaussian sigma of the 1-D derivative-of-Gaussian kernel, in pixels.
    pub sigma: f32,
    /// Minimum `|DoG response|` for an edge to be reported.
    pub threshold: f32,
    /// "any", "rising" or "falling".
    pub polarity: String,
    /// "all", "first", "last" or "strongest".
    pub select: String,
    /// Profile sampling step along the scan axis, in pixels.
    pub step: f32,
    /// Maximum angle, in degrees, between scan direction and image gradient.
    /// `180.0` disables the obliquity gate.
    pub max_obliquity_deg: f32,
    /// "clamp", "reflect101" or "constant".
    pub border_mode: String,
    pub border_constant: f32,
}

#[pymethods]
impl MeasureConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        sigma=None,
        threshold=None,
        polarity=None,
        select=None,
        step=None,
        max_obliquity_deg=None,
        border_mode=None,
        border_constant=None
    ))]
    pub fn new(
        sigma: Option<f32>,
        threshold: Option<f32>,
        polarity: Option<String>,
        select: Option<String>,
        step: Option<f32>,
        max_obliquity_deg: Option<f32>,
        border_mode: Option<String>,
        border_constant: Option<f32>,
    ) -> PyResult<Self> {
        let d = Self::default();
        for (name, value, allowed) in [
            (
                "polarity",
                polarity.as_deref(),
                &["any", "rising", "falling"][..],
            ),
            (
                "select",
                select.as_deref(),
                &["all", "first", "last", "strongest"][..],
            ),
            (
                "border_mode",
                border_mode.as_deref(),
                &["clamp", "reflect101", "constant"][..],
            ),
        ] {
            if let Some(v) = value
                && !allowed.contains(&v)
            {
                return Err(PyValueError::new_err(format!(
                    "{name} must be one of {allowed:?}, got '{v}'"
                )));
            }
        }
        Ok(Self {
            sigma: sigma.unwrap_or(d.sigma),
            threshold: threshold.unwrap_or(d.threshold),
            polarity: polarity.unwrap_or(d.polarity),
            select: select.unwrap_or(d.select),
            step: step.unwrap_or(d.step),
            max_obliquity_deg: max_obliquity_deg.unwrap_or(d.max_obliquity_deg),
            border_mode: border_mode.unwrap_or(d.border_mode),
            border_constant: border_constant.unwrap_or(d.border_constant),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MeasureConfig(sigma={:.3}, threshold={:.3}, polarity='{}', select='{}')",
            self.sigma, self.threshold, self.polarity, self.select
        )
    }
}

impl Default for MeasureConfig {
    fn default() -> Self {
        let n = NativeMeasureConfig::default();
        Self {
            sigma: n.sigma,
            threshold: n.threshold,
            polarity: "any".to_string(),
            select: "all".to_string(),
            step: n.step,
            max_obliquity_deg: n.max_obliquity_deg,
            border_mode: "clamp".to_string(),
            border_constant: 0.0,
        }
    }
}

impl MeasureConfig {
    pub fn to_native(&self) -> NativeMeasureConfig {
        NativeMeasureConfig {
            sigma: self.sigma,
            threshold: self.threshold,
            polarity: match self.polarity.as_str() {
                "rising" => NativePolaritySelect::Rising,
                "falling" => NativePolaritySelect::Falling,
                _ => NativePolaritySelect::Any,
            },
            select: match self.select.as_str() {
                "first" => NativeEdgeSelect::First,
                "last" => NativeEdgeSelect::Last,
                "strongest" => NativeEdgeSelect::Strongest,
                _ => NativeEdgeSelect::All,
            },
            step: self.step,
            max_obliquity_deg: self.max_obliquity_deg,
            border: match self.border_mode.as_str() {
                "reflect101" => BorderMode::Reflect101,
                "constant" => BorderMode::Constant(self.border_constant),
                _ => BorderMode::Clamp,
            },
        }
    }
}
