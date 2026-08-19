//! Python-visible [`FitConfig`](vision_metrology::fit::FitConfig) mirror.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vision_metrology::fit::{
    FitConfig as NativeFitConfig, RansacConfig as NativeRansacConfig,
    RobustLoss as NativeRobustLoss,
};

#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// `"none"`, `"huber"` or `"tukey"`.
    pub loss: String,
    /// Tuning constant in pixels for `huber` / `tukey`. Ignored for `none`.
    pub loss_scale: f32,
    /// `0` disables the RANSAC consensus stage.
    pub ransac_iters: usize,
    /// Inlier distance in pixels for RANSAC.
    pub inlier_tol: f32,
    /// Minimum consensus size.
    pub min_inliers: usize,
    /// LCG seed. The same seed always gives the same fit.
    pub seed: u64,
    /// Maximum refinement iterations.
    pub max_iters: usize,
    /// Convergence threshold in pixels.
    pub tol: f32,
}

#[pymethods]
impl FitConfig {
    #[new]
    #[pyo3(signature = (
        loss=None, loss_scale=None, ransac_iters=None, inlier_tol=None,
        min_inliers=None, seed=None, max_iters=None, tol=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: Option<String>,
        loss_scale: Option<f32>,
        ransac_iters: Option<usize>,
        inlier_tol: Option<f32>,
        min_inliers: Option<usize>,
        seed: Option<u64>,
        max_iters: Option<usize>,
        tol: Option<f32>,
    ) -> Self {
        let d = Self::default();
        Self {
            loss: loss.unwrap_or(d.loss),
            loss_scale: loss_scale.unwrap_or(d.loss_scale),
            ransac_iters: ransac_iters.unwrap_or(d.ransac_iters),
            inlier_tol: inlier_tol.unwrap_or(d.inlier_tol),
            min_inliers: min_inliers.unwrap_or(d.min_inliers),
            seed: seed.unwrap_or(d.seed),
            max_iters: max_iters.unwrap_or(d.max_iters),
            tol: tol.unwrap_or(d.tol),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FitConfig(loss={:?}, loss_scale={:.3}, ransac_iters={}, inlier_tol={:.3}, \
             min_inliers={}, seed={}, max_iters={}, tol={:.5})",
            self.loss,
            self.loss_scale,
            self.ransac_iters,
            self.inlier_tol,
            self.min_inliers,
            self.seed,
            self.max_iters,
            self.tol
        )
    }
}

impl Default for FitConfig {
    fn default() -> Self {
        let native = NativeFitConfig::default();
        let rc = NativeRansacConfig::default();
        Self {
            loss: "none".to_string(),
            loss_scale: 2.0,
            ransac_iters: 0,
            inlier_tol: rc.inlier_tol,
            min_inliers: rc.min_inliers,
            seed: rc.seed,
            max_iters: native.max_iters,
            tol: native.tol,
        }
    }
}

impl FitConfig {
    pub fn to_native(&self) -> PyResult<NativeFitConfig> {
        let loss = match self.loss.as_str() {
            "none" => NativeRobustLoss::None,
            "huber" => NativeRobustLoss::Huber { k: self.loss_scale },
            "tukey" => NativeRobustLoss::Tukey { c: self.loss_scale },
            other => {
                return Err(PyValueError::new_err(format!(
                    "loss must be 'none', 'huber' or 'tukey', got {other:?}"
                )));
            }
        };
        if self.loss != "none" && !(self.loss_scale.is_finite() && self.loss_scale > 0.0) {
            return Err(PyValueError::new_err("loss_scale must be positive"));
        }
        let ransac = (self.ransac_iters > 0).then_some(NativeRansacConfig {
            iters: self.ransac_iters,
            inlier_tol: self.inlier_tol,
            min_inliers: self.min_inliers,
            seed: self.seed,
        });
        Ok(NativeFitConfig {
            loss,
            ransac,
            max_iters: self.max_iters,
            tol: self.tol,
        })
    }
}
