//! Python-visible [`MeasureConfig`](vision_metrology::measure::MeasureConfig) mirror.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vision_metrology::measure::{
    Derivative as NativeDerivative, EdgeSelect as NativeEdgeSelect, Locate as NativeLocate,
    MeasureConfig as NativeMeasureConfig, PolaritySelect as NativePolaritySelect,
    ProfileConfig as NativeProfileConfig,
};
use vm_primitives::{BorderMode, SubpixRefine};

/// How a caliper locates an edge on its profile — construct with
/// `Locate.gradient_peak(refine=...)`.
#[pyclass(frozen, get_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct Locate {
    /// "gradient_peak".
    pub kind: String,
    /// Subpixel refinement of a gradient peak: "none", "parabolic", "gaussian" or
    /// "centroid".
    pub refine: String,
    /// Half-width of the centroid window, in samples (only for `refine="centroid"`).
    pub centroid_radius: usize,
}

#[pymethods]
impl Locate {
    /// A local extremum of the derivative, refined to subpixel position.
    #[staticmethod]
    #[pyo3(signature = (refine="parabolic", centroid_radius=2))]
    pub fn gradient_peak(refine: &str, centroid_radius: usize) -> PyResult<Self> {
        let allowed = ["none", "parabolic", "gaussian", "centroid"];
        if !allowed.contains(&refine) {
            return Err(PyValueError::new_err(format!(
                "refine must be one of {allowed:?}, got '{refine}'"
            )));
        }
        Ok(Self {
            kind: "gradient_peak".into(),
            refine: refine.into(),
            centroid_radius,
        })
    }

    fn __repr__(&self) -> String {
        format!("Locate.gradient_peak(refine='{}')", self.refine)
    }
}

impl Locate {
    pub fn to_native(&self) -> NativeLocate {
        NativeLocate::GradientPeak {
            refine: match self.refine.as_str() {
                "none" => SubpixRefine::None,
                "gaussian" => SubpixRefine::Gaussian3,
                "centroid" => SubpixRefine::Centroid {
                    radius: self.centroid_radius,
                },
                _ => SubpixRefine::Parabolic3,
            },
        }
    }
}

impl Default for Locate {
    fn default() -> Self {
        Self {
            kind: "gradient_peak".into(),
            refine: "parabolic".into(),
            centroid_radius: 2,
        }
    }
}

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
    /// "dog" (derivative of Gaussian) or "smooth_central" (Gaussian, then central
    /// differences).
    pub derivative: String,
    /// Half-width of the smoothing kernel for `derivative="smooth_central"`, in pixels.
    pub kernel_radius_px: f32,
    /// How each edge is located on the profile.
    pub locate: Locate,
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
        border_constant=None,
        derivative=None,
        kernel_radius_px=None,
        locate=None
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
        derivative: Option<String>,
        kernel_radius_px: Option<f32>,
        locate: Option<Locate>,
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
            (
                "derivative",
                derivative.as_deref(),
                &["dog", "smooth_central"][..],
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
            derivative: derivative.unwrap_or(d.derivative),
            kernel_radius_px: kernel_radius_px.unwrap_or(d.kernel_radius_px),
            locate: locate.unwrap_or(d.locate),
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
            sigma: n.profile.sigma,
            threshold: n.threshold,
            polarity: "any".to_string(),
            select: "all".to_string(),
            step: n.profile.step,
            max_obliquity_deg: n.max_obliquity_deg,
            border_mode: "clamp".to_string(),
            border_constant: 0.0,
            derivative: "dog".to_string(),
            kernel_radius_px: 3.0,
            locate: Locate::default(),
        }
    }
}

impl MeasureConfig {
    pub fn to_native(&self) -> NativeMeasureConfig {
        NativeMeasureConfig {
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
            locate: self.locate.to_native(),
            max_obliquity_deg: self.max_obliquity_deg,
            profile: NativeProfileConfig {
                sigma: self.sigma,
                derivative: match self.derivative.as_str() {
                    "smooth_central" => NativeDerivative::SmoothThenCentral {
                        radius_px: self.kernel_radius_px,
                    },
                    _ => NativeDerivative::DerivativeOfGaussian,
                },
                step: self.step,
                border: match self.border_mode.as_str() {
                    "reflect101" => BorderMode::Reflect101,
                    "constant" => BorderMode::Constant(self.border_constant),
                    _ => BorderMode::Clamp,
                },
            },
        }
    }
}
