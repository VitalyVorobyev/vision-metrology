//! Python-visible configuration objects and conversion to native Rust configs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vm_core::{BorderMode, Rect2f};
use vm_edge::edge2d::{Edge2DConfig, SmoothKind, Subpix2D};
use vm_match::RigidMatchConfig as NativeRigidMatchConfig;
use vm_multiscale::MultiScaleConfig as NativeMultiScaleConfig;
use vm_shape::{ConicFitConfig as NativeConicFitConfig, LsdConfig as NativeLsdConfig};

#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct EdgeConfig {
    pub pre_smooth: bool,
    pub smooth_kind: String,
    pub low_thresh: f32,
    pub high_thresh: f32,
    pub border_mode: String,
    pub border_constant: f32,
    pub subpix: String,
}

#[pymethods]
impl EdgeConfig {
    #[new]
    #[pyo3(signature = (
        pre_smooth=None,
        smooth_kind=None,
        low_thresh=None,
        high_thresh=None,
        border_mode=None,
        border_constant=None,
        subpix=None
    ))]
    pub fn new(
        pre_smooth: Option<bool>,
        smooth_kind: Option<String>,
        low_thresh: Option<f32>,
        high_thresh: Option<f32>,
        border_mode: Option<String>,
        border_constant: Option<f32>,
        subpix: Option<String>,
    ) -> Self {
        let default = Self::default();
        Self {
            pre_smooth: pre_smooth.unwrap_or(default.pre_smooth),
            smooth_kind: smooth_kind.unwrap_or(default.smooth_kind),
            low_thresh: low_thresh.unwrap_or(default.low_thresh),
            high_thresh: high_thresh.unwrap_or(default.high_thresh),
            border_mode: border_mode.unwrap_or(default.border_mode),
            border_constant: border_constant.unwrap_or(default.border_constant),
            subpix: subpix.unwrap_or(default.subpix),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EdgeConfig(pre_smooth={}, smooth_kind='{}', low_thresh={:.3}, high_thresh={:.3}, border_mode='{}', border_constant={:.3}, subpix='{}')",
            self.pre_smooth,
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
        let native = Edge2DConfig::default();
        Self {
            pre_smooth: native.pre_smooth,
            smooth_kind: "binomial3".to_string(),
            low_thresh: native.low_thresh,
            high_thresh: native.high_thresh,
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

        Ok(Edge2DConfig {
            pre_smooth: self.pre_smooth,
            smooth_kind,
            low_thresh: self.low_thresh,
            high_thresh: self.high_thresh,
            border,
            subpix,
        })
    }
}

#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct MultiScaleConfig {
    pub num_levels: usize,
    pub base_sigma: f32,
    pub merge_duplicates: bool,
    pub edge: EdgeConfig,
}

#[pymethods]
impl MultiScaleConfig {
    #[new]
    #[pyo3(signature = (num_levels=None, base_sigma=None, merge_duplicates=None, edge=None))]
    pub fn new(
        num_levels: Option<usize>,
        base_sigma: Option<f32>,
        merge_duplicates: Option<bool>,
        edge: Option<EdgeConfig>,
    ) -> Self {
        let default = Self::default();
        Self {
            num_levels: num_levels.unwrap_or(default.num_levels),
            base_sigma: base_sigma.unwrap_or(default.base_sigma),
            merge_duplicates: merge_duplicates.unwrap_or(default.merge_duplicates),
            edge: edge.unwrap_or(default.edge),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiScaleConfig(num_levels={}, base_sigma={:.3}, merge_duplicates={})",
            self.num_levels, self.base_sigma, self.merge_duplicates
        )
    }
}

impl Default for MultiScaleConfig {
    fn default() -> Self {
        let native = NativeMultiScaleConfig::default();
        Self {
            num_levels: native.num_levels,
            base_sigma: native.base_sigma,
            merge_duplicates: native.merge_duplicates,
            edge: EdgeConfig::default(),
        }
    }
}

impl MultiScaleConfig {
    pub fn to_native(&self) -> PyResult<NativeMultiScaleConfig> {
        if self.num_levels == 0 {
            return Err(PyValueError::new_err("num_levels must be >= 1"));
        }
        Ok(NativeMultiScaleConfig {
            num_levels: self.num_levels,
            base_sigma: self.base_sigma,
            edge: self.edge.to_native()?,
            merge_duplicates: self.merge_duplicates,
        })
    }
}

#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct LsdConfig {
    pub scale: f32,
    pub sigma_scale: f32,
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
        scale=None,
        sigma_scale=None,
        ang_th=None,
        log_eps=None,
        density_th=None,
        n_bins=None,
        min_length=None
    ))]
    pub fn new(
        scale: Option<f32>,
        sigma_scale: Option<f32>,
        ang_th: Option<f32>,
        log_eps: Option<f32>,
        density_th: Option<f32>,
        n_bins: Option<usize>,
        min_length: Option<f32>,
    ) -> Self {
        let default = Self::default();
        Self {
            scale: scale.unwrap_or(default.scale),
            sigma_scale: sigma_scale.unwrap_or(default.sigma_scale),
            ang_th: ang_th.unwrap_or(default.ang_th),
            log_eps: log_eps.unwrap_or(default.log_eps),
            density_th: density_th.unwrap_or(default.density_th),
            n_bins: n_bins.unwrap_or(default.n_bins),
            min_length: min_length.unwrap_or(default.min_length),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LsdConfig(scale={:.3}, sigma_scale={:.3}, ang_th={:.3}, log_eps={:.3}, density_th={:.3}, n_bins={}, min_length={:.3})",
            self.scale,
            self.sigma_scale,
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
            scale: native.scale,
            sigma_scale: native.sigma_scale,
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
        if !(self.scale > 0.0 && self.scale <= 1.0) {
            return Err(PyValueError::new_err("scale must be in (0.0, 1.0]"));
        }
        Ok(NativeLsdConfig {
            scale: self.scale,
            sigma_scale: self.sigma_scale,
            ang_th: self.ang_th,
            log_eps: self.log_eps,
            density_th: self.density_th,
            n_bins: self.n_bins,
            min_length: self.min_length,
        })
    }
}

#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct ConicFitConfig {
    pub use_bookstein: bool,
    pub ransac_iters: usize,
    pub inlier_tol: f32,
    pub min_inliers: usize,
    pub rng_seed: u64,
}

#[pymethods]
impl ConicFitConfig {
    #[new]
    #[pyo3(signature = (use_bookstein=None, ransac_iters=None, inlier_tol=None, min_inliers=None, rng_seed=None))]
    pub fn new(
        use_bookstein: Option<bool>,
        ransac_iters: Option<usize>,
        inlier_tol: Option<f32>,
        min_inliers: Option<usize>,
        rng_seed: Option<u64>,
    ) -> Self {
        let default = Self::default();
        Self {
            use_bookstein: use_bookstein.unwrap_or(default.use_bookstein),
            ransac_iters: ransac_iters.unwrap_or(default.ransac_iters),
            inlier_tol: inlier_tol.unwrap_or(default.inlier_tol),
            min_inliers: min_inliers.unwrap_or(default.min_inliers),
            rng_seed: rng_seed.unwrap_or(default.rng_seed),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ConicFitConfig(use_bookstein={}, ransac_iters={}, inlier_tol={:.3}, min_inliers={}, rng_seed={})",
            self.use_bookstein, self.ransac_iters, self.inlier_tol, self.min_inliers, self.rng_seed
        )
    }
}

impl Default for ConicFitConfig {
    fn default() -> Self {
        let native = NativeConicFitConfig::default();
        Self {
            use_bookstein: native.use_bookstein,
            ransac_iters: native.ransac_iters,
            inlier_tol: native.inlier_tol,
            min_inliers: native.min_inliers,
            rng_seed: native.rng_seed,
        }
    }
}

impl ConicFitConfig {
    pub fn to_native(&self) -> PyResult<NativeConicFitConfig> {
        if self.inlier_tol <= 0.0 {
            return Err(PyValueError::new_err("inlier_tol must be > 0"));
        }
        if self.min_inliers < 5 {
            return Err(PyValueError::new_err("min_inliers must be >= 5"));
        }
        Ok(NativeConicFitConfig {
            use_bookstein: self.use_bookstein,
            ransac_iters: self.ransac_iters,
            inlier_tol: self.inlier_tol,
            min_inliers: self.min_inliers,
            rng_seed: self.rng_seed,
        })
    }
}

#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct RigidMatchConfig {
    pub angle_min: f32,
    pub angle_max: f32,
    pub angle_step: f32,
    pub scale_min: f32,
    pub scale_max: f32,
    pub scale_step: f32,
    pub position_x: f32,
    pub position_y: f32,
    pub position_width: f32,
    pub position_height: f32,
    pub chamfer_threshold: f32,
    pub min_score: f32,
    pub refine_icp: bool,
    pub top_k: usize,
    pub resolution_factor: f32,
}

#[pymethods]
impl RigidMatchConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        angle_min=None,
        angle_max=None,
        angle_step=None,
        scale_min=None,
        scale_max=None,
        scale_step=None,
        position_x=None,
        position_y=None,
        position_width=None,
        position_height=None,
        chamfer_threshold=None,
        min_score=None,
        refine_icp=None,
        top_k=None,
        resolution_factor=None
    ))]
    pub fn new(
        angle_min: Option<f32>,
        angle_max: Option<f32>,
        angle_step: Option<f32>,
        scale_min: Option<f32>,
        scale_max: Option<f32>,
        scale_step: Option<f32>,
        position_x: Option<f32>,
        position_y: Option<f32>,
        position_width: Option<f32>,
        position_height: Option<f32>,
        chamfer_threshold: Option<f32>,
        min_score: Option<f32>,
        refine_icp: Option<bool>,
        top_k: Option<usize>,
        resolution_factor: Option<f32>,
    ) -> Self {
        let default = Self::default();
        Self {
            angle_min: angle_min.unwrap_or(default.angle_min),
            angle_max: angle_max.unwrap_or(default.angle_max),
            angle_step: angle_step.unwrap_or(default.angle_step),
            scale_min: scale_min.unwrap_or(default.scale_min),
            scale_max: scale_max.unwrap_or(default.scale_max),
            scale_step: scale_step.unwrap_or(default.scale_step),
            position_x: position_x.unwrap_or(default.position_x),
            position_y: position_y.unwrap_or(default.position_y),
            position_width: position_width.unwrap_or(default.position_width),
            position_height: position_height.unwrap_or(default.position_height),
            chamfer_threshold: chamfer_threshold.unwrap_or(default.chamfer_threshold),
            min_score: min_score.unwrap_or(default.min_score),
            refine_icp: refine_icp.unwrap_or(default.refine_icp),
            top_k: top_k.unwrap_or(default.top_k),
            resolution_factor: resolution_factor.unwrap_or(default.resolution_factor),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RigidMatchConfig(angle=({:.3},{:.3}), angle_step={:.3}, scale=({:.3},{:.3}), scale_step={:.3}, pos=({:.1},{:.1},{:.1},{:.1}), chamfer_threshold={:.3}, min_score={:.3}, refine_icp={}, top_k={}, resolution_factor={:.3})",
            self.angle_min,
            self.angle_max,
            self.angle_step,
            self.scale_min,
            self.scale_max,
            self.scale_step,
            self.position_x,
            self.position_y,
            self.position_width,
            self.position_height,
            self.chamfer_threshold,
            self.min_score,
            self.refine_icp,
            self.top_k,
            self.resolution_factor
        )
    }
}

impl Default for RigidMatchConfig {
    fn default() -> Self {
        let native = NativeRigidMatchConfig::default();
        Self {
            angle_min: native.angle_range.0,
            angle_max: native.angle_range.1,
            angle_step: native.angle_step,
            scale_min: native.scale_range.0,
            scale_max: native.scale_range.1,
            scale_step: native.scale_step,
            position_x: native.position_search.x,
            position_y: native.position_search.y,
            position_width: native.position_search.width,
            position_height: native.position_search.height,
            chamfer_threshold: native.chamfer_threshold,
            min_score: native.min_score,
            refine_icp: native.refine_icp,
            top_k: native.top_k,
            resolution_factor: native.resolution_factor,
        }
    }
}

impl RigidMatchConfig {
    pub fn to_native(&self) -> PyResult<NativeRigidMatchConfig> {
        if self.angle_step <= 0.0 {
            return Err(PyValueError::new_err("angle_step must be > 0"));
        }
        if self.scale_min <= 0.0 || self.scale_max <= 0.0 {
            return Err(PyValueError::new_err("scale_min and scale_max must be > 0"));
        }
        if self.scale_step < 0.0 {
            return Err(PyValueError::new_err("scale_step must be >= 0"));
        }
        if self.position_width <= 0.0 || self.position_height <= 0.0 {
            return Err(PyValueError::new_err(
                "position_width and position_height must be > 0",
            ));
        }
        if self.chamfer_threshold <= 0.0 {
            return Err(PyValueError::new_err("chamfer_threshold must be > 0"));
        }
        if !(0.0..=1.0).contains(&self.min_score) {
            return Err(PyValueError::new_err("min_score must be in [0, 1]"));
        }
        if self.top_k == 0 {
            return Err(PyValueError::new_err("top_k must be >= 1"));
        }

        Ok(NativeRigidMatchConfig {
            angle_range: (self.angle_min, self.angle_max),
            angle_step: self.angle_step,
            scale_range: (self.scale_min, self.scale_max),
            scale_step: self.scale_step,
            position_search: Rect2f {
                x: self.position_x,
                y: self.position_y,
                width: self.position_width,
                height: self.position_height,
            },
            chamfer_threshold: self.chamfer_threshold,
            min_score: self.min_score,
            refine_icp: self.refine_icp,
            top_k: self.top_k,
            resolution_factor: self.resolution_factor,
        })
    }
}
