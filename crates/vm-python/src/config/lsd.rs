//! Python-visible [`LsdConfig`](vision_metrology::lsd::LsdConfig) mirror.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vision_metrology::lsd::LsdConfig as NativeLsdConfig;
use vm_primitives::PreSmooth as NativePreSmooth;

#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct LsdConfig {
    pub downscale_levels: u32,
    pub pre_smooth: String,
    pub ang_th: f32,
    pub log_eps: f32,
    pub density_th: f32,
    pub n_bins: usize,
    pub min_length: f32,
}

#[pymethods]
impl LsdConfig {
    #[new]
    #[pyo3(signature = (
        downscale_levels=None,
        pre_smooth=None,
        ang_th=None,
        log_eps=None,
        density_th=None,
        n_bins=None,
        min_length=None
    ))]
    pub fn new(
        downscale_levels: Option<u32>,
        pre_smooth: Option<String>,
        ang_th: Option<f32>,
        log_eps: Option<f32>,
        density_th: Option<f32>,
        n_bins: Option<usize>,
        min_length: Option<f32>,
    ) -> Self {
        let default = Self::default();
        Self {
            downscale_levels: downscale_levels.unwrap_or(default.downscale_levels),
            pre_smooth: pre_smooth.unwrap_or(default.pre_smooth),
            ang_th: ang_th.unwrap_or(default.ang_th),
            log_eps: log_eps.unwrap_or(default.log_eps),
            density_th: density_th.unwrap_or(default.density_th),
            n_bins: n_bins.unwrap_or(default.n_bins),
            min_length: min_length.unwrap_or(default.min_length),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LsdConfig(downscale_levels={}, pre_smooth={:?}, ang_th={:.3}, log_eps={:.3}, density_th={:.3}, n_bins={}, min_length={:.3})",
            self.downscale_levels,
            self.pre_smooth,
            self.ang_th,
            self.log_eps,
            self.density_th,
            self.n_bins,
            self.min_length
        )
    }
}

impl Default for LsdConfig {
    fn default() -> Self {
        let native = NativeLsdConfig::default();
        Self {
            downscale_levels: native.downscale_levels,
            pre_smooth: match native.pre_smooth {
                NativePreSmooth::None => "none".to_string(),
                NativePreSmooth::Binomial121 => "binomial121".to_string(),
            },
            ang_th: native.ang_th,
            log_eps: native.log_eps,
            density_th: native.density_th,
            n_bins: native.n_bins,
            min_length: native.min_length,
        }
    }
}

impl LsdConfig {
    pub fn to_native(&self) -> PyResult<NativeLsdConfig> {
        let pre_smooth = match self.pre_smooth.as_str() {
            "none" => NativePreSmooth::None,
            "binomial121" => NativePreSmooth::Binomial121,
            other => {
                return Err(PyValueError::new_err(format!(
                    "pre_smooth must be 'none' or 'binomial121', got {other:?}"
                )));
            }
        };
        Ok(NativeLsdConfig {
            downscale_levels: self.downscale_levels,
            pre_smooth,
            ang_th: self.ang_th,
            log_eps: self.log_eps,
            density_th: self.density_th,
            n_bins: self.n_bins,
            min_length: self.min_length,
        })
    }
}
