//! Python-visible [`Edge2DConfig`](vm_primitives::edge::Edge2DConfig) mirror.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vm_primitives::BorderMode;
use vm_primitives::edge::{Edge2DConfig, Hysteresis, SmoothKind, Subpix2D};

/// Mirrors `vm_primitives::edge::Edge2DConfig`.
///
/// `Hysteresis::{Auto, Manual}` is mapped onto the two-field sentinel pattern
/// familiar from `argparse`-style optional pairs rather than exposed as its
/// own enum type: `low_thresh=None, high_thresh=None` is `Auto`; setting
/// either selects `Manual`, and a threshold left at `None` defaults to `0.0`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct EdgeConfig {
    pub smooth_kind: String,
    /// `None` on either threshold selects the automatic pair.
    pub low_thresh: Option<f32>,
    pub high_thresh: Option<f32>,
    pub border_mode: String,
    pub border_constant: f32,
    pub subpix: String,
}

#[pymethods]
impl EdgeConfig {
    #[new]
    #[pyo3(signature = (
        smooth_kind=None,
        low_thresh=None,
        high_thresh=None,
        border_mode=None,
        border_constant=None,
        subpix=None
    ))]
    pub fn new(
        smooth_kind: Option<String>,
        low_thresh: Option<f32>,
        high_thresh: Option<f32>,
        border_mode: Option<String>,
        border_constant: Option<f32>,
        subpix: Option<String>,
    ) -> Self {
        let default = Self::default();
        Self {
            smooth_kind: smooth_kind.unwrap_or(default.smooth_kind),
            low_thresh: low_thresh.or(default.low_thresh),
            high_thresh: high_thresh.or(default.high_thresh),
            border_mode: border_mode.unwrap_or(default.border_mode),
            border_constant: border_constant.unwrap_or(default.border_constant),
            subpix: subpix.unwrap_or(default.subpix),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EdgeConfig(smooth_kind='{}', low_thresh={:?}, high_thresh={:?}, border_mode='{}', border_constant={:.3}, subpix='{}')",
            self.smooth_kind,
            self.low_thresh,
            self.high_thresh,
            self.border_mode,
            self.border_constant,
            self.subpix
        )
    }
}

impl Default for EdgeConfig {
    fn default() -> Self {
        let (low_thresh, high_thresh) = match Edge2DConfig::default().hysteresis {
            Hysteresis::Auto => (None, None),
            Hysteresis::Manual { low, high } => (Some(low), Some(high)),
        };
        Self {
            smooth_kind: "binomial3".to_string(),
            low_thresh,
            high_thresh,
            border_mode: "clamp".to_string(),
            border_constant: 0.0,
            subpix: "parabolic_along_normal".to_string(),
        }
    }
}

impl EdgeConfig {
    pub fn to_native(&self) -> PyResult<Edge2DConfig> {
        let smooth_kind = match self.smooth_kind.to_ascii_lowercase().as_str() {
            "none" => SmoothKind::None,
            "binomial3" => SmoothKind::Binomial3,
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid smooth_kind '{other}', expected one of: none, binomial3"
                )));
            }
        };

        let border = match self.border_mode.to_ascii_lowercase().as_str() {
            "clamp" => BorderMode::Clamp,
            "reflect101" => BorderMode::Reflect101,
            "constant" => BorderMode::Constant(self.border_constant),
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid border_mode '{other}', expected one of: clamp, reflect101, constant"
                )));
            }
        };

        let subpix = match self.subpix.to_ascii_lowercase().as_str() {
            "none" => Subpix2D::None,
            "parabolic_along_normal" | "parabolic" => Subpix2D::ParabolicAlongNormal,
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid subpix '{other}', expected one of: none, parabolic_along_normal"
                )));
            }
        };

        let hysteresis = match (self.low_thresh, self.high_thresh) {
            (None, None) => Hysteresis::Auto,
            (low, high) => Hysteresis::Manual {
                low: low.unwrap_or(0.0),
                high: high.unwrap_or(0.0),
            },
        };

        Ok(Edge2DConfig {
            smooth_kind,
            hysteresis,
            border,
            subpix,
        })
    }
}
