//! Python-visible configuration objects and conversion to native Rust configs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vision_metrology::fit::{
    FitConfig as NativeFitConfig, RansacConfig as NativeRansacConfig,
    RobustLoss as NativeRobustLoss,
};
use vision_metrology::matching::Polarity as NativePolarity;
use vision_metrology::matching::Refinement as NativeRefinement;
use vision_metrology::matching::ShapeModelConfig as NativeShapeModelConfig;
use vision_metrology::matching::ShapeSearchConfig as NativeShapeSearchConfig;
use vision_metrology::shape::LsdConfig as NativeLsdConfig;
use vm_primitives::BorderMode;
use vm_primitives::PreSmooth as NativePreSmooth;
use vm_primitives::edge::{Edge2DConfig, SmoothKind, Subpix2D};

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

#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct ShapeModelConfig {
    pub num_levels: usize,
    pub min_contrast: f32,
    pub max_points: usize,
    pub angle_min: f32,
    pub angle_max: f32,
    pub scale_min: f32,
    pub scale_max: f32,
    /// "match", "ignore_global" or "ignore_local".
    pub polarity: String,
    pub min_points_per_level: usize,
}

impl Default for ShapeModelConfig {
    fn default() -> Self {
        let n = NativeShapeModelConfig::default();
        Self {
            num_levels: n.num_levels,
            min_contrast: n.min_contrast,
            max_points: n.max_points,
            angle_min: n.angle_range.0,
            angle_max: n.angle_range.1,
            scale_min: n.scale_range.0,
            scale_max: n.scale_range.1,
            polarity: "match".to_string(),
            min_points_per_level: n.min_points_per_level,
        }
    }
}

#[pymethods]
impl ShapeModelConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        num_levels=None,
        min_contrast=None,
        max_points=None,
        angle_min=None,
        angle_max=None,
        scale_min=None,
        scale_max=None,
        polarity=None,
        min_points_per_level=None
    ))]
    pub fn new(
        num_levels: Option<usize>,
        min_contrast: Option<f32>,
        max_points: Option<usize>,
        angle_min: Option<f32>,
        angle_max: Option<f32>,
        scale_min: Option<f32>,
        scale_max: Option<f32>,
        polarity: Option<String>,
        min_points_per_level: Option<usize>,
    ) -> Self {
        let d = Self::default();
        Self {
            num_levels: num_levels.unwrap_or(d.num_levels),
            min_contrast: min_contrast.unwrap_or(d.min_contrast),
            max_points: max_points.unwrap_or(d.max_points),
            angle_min: angle_min.unwrap_or(d.angle_min),
            angle_max: angle_max.unwrap_or(d.angle_max),
            scale_min: scale_min.unwrap_or(d.scale_min),
            scale_max: scale_max.unwrap_or(d.scale_max),
            polarity: polarity.unwrap_or(d.polarity),
            min_points_per_level: min_points_per_level.unwrap_or(d.min_points_per_level),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapeModelConfig(num_levels={}, max_points={}, angle=({:.3},{:.3}), scale=({:.3},{:.3}), polarity='{}')",
            self.num_levels,
            self.max_points,
            self.angle_min,
            self.angle_max,
            self.scale_min,
            self.scale_max,
            self.polarity
        )
    }
}

fn parse_polarity(s: &str) -> PyResult<NativePolarity> {
    match s {
        "match" => Ok(NativePolarity::Match),
        "ignore_global" => Ok(NativePolarity::IgnoreGlobal),
        "ignore_local" => Ok(NativePolarity::IgnoreLocal),
        other => Err(PyValueError::new_err(format!(
            "polarity must be 'match', 'ignore_global' or 'ignore_local', got '{other}'"
        ))),
    }
}

impl ShapeModelConfig {
    pub fn to_native(&self) -> PyResult<NativeShapeModelConfig> {
        if self.scale_min <= 0.0 || self.scale_max < self.scale_min {
            return Err(PyValueError::new_err(
                "scale_min must be > 0 and scale_max >= scale_min",
            ));
        }
        if self.angle_max < self.angle_min {
            return Err(PyValueError::new_err("angle_max must be >= angle_min"));
        }
        Ok(NativeShapeModelConfig {
            num_levels: self.num_levels,
            min_contrast: self.min_contrast,
            max_points: self.max_points,
            angle_range: (self.angle_min, self.angle_max),
            scale_range: (self.scale_min, self.scale_max),
            polarity: parse_polarity(&self.polarity)?,
            min_points_per_level: self.min_points_per_level,
            ..NativeShapeModelConfig::default()
        })
    }
}

/// Python-visible [`ShapeSearchConfig`].
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct ShapeSearchConfig {
    pub min_score: f32,
    pub greediness: f32,
    pub max_matches: usize,
    pub max_overlap: f32,
    pub min_contrast: f32,
    /// "none", "interpolate" or "least_squares".
    pub refinement: String,
    pub last_level: usize,
    pub max_candidates: usize,
    pub coarse_score_factor: f32,
}

impl Default for ShapeSearchConfig {
    fn default() -> Self {
        let n = NativeShapeSearchConfig::default();
        Self {
            min_score: n.min_score,
            greediness: n.greediness,
            max_matches: n.max_matches,
            max_overlap: n.max_overlap,
            min_contrast: n.min_contrast,
            refinement: "interpolate".to_string(),
            last_level: n.last_level,
            max_candidates: n.max_candidates,
            coarse_score_factor: n.coarse_score_factor,
        }
    }
}

#[pymethods]
impl ShapeSearchConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        min_score=None,
        greediness=None,
        max_matches=None,
        max_overlap=None,
        min_contrast=None,
        refinement=None,
        last_level=None,
        max_candidates=None,
        coarse_score_factor=None
    ))]
    pub fn new(
        min_score: Option<f32>,
        greediness: Option<f32>,
        max_matches: Option<usize>,
        max_overlap: Option<f32>,
        min_contrast: Option<f32>,
        refinement: Option<String>,
        last_level: Option<usize>,
        max_candidates: Option<usize>,
        coarse_score_factor: Option<f32>,
    ) -> Self {
        let d = Self::default();
        Self {
            min_score: min_score.unwrap_or(d.min_score),
            greediness: greediness.unwrap_or(d.greediness),
            max_matches: max_matches.unwrap_or(d.max_matches),
            max_overlap: max_overlap.unwrap_or(d.max_overlap),
            min_contrast: min_contrast.unwrap_or(d.min_contrast),
            refinement: refinement.unwrap_or(d.refinement),
            last_level: last_level.unwrap_or(d.last_level),
            max_candidates: max_candidates.unwrap_or(d.max_candidates),
            coarse_score_factor: coarse_score_factor.unwrap_or(d.coarse_score_factor),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapeSearchConfig(min_score={:.3}, greediness={:.2}, max_matches={}, refinement='{}')",
            self.min_score, self.greediness, self.max_matches, self.refinement
        )
    }
}

impl ShapeSearchConfig {
    pub fn to_native(&self) -> PyResult<NativeShapeSearchConfig> {
        if !(0.0..=1.0).contains(&self.min_score) {
            return Err(PyValueError::new_err("min_score must be in [0, 1]"));
        }
        if !(0.0..=1.0).contains(&self.greediness) {
            return Err(PyValueError::new_err("greediness must be in [0, 1]"));
        }
        let refinement = match self.refinement.as_str() {
            "none" => NativeRefinement::None,
            "interpolate" => NativeRefinement::Interpolate,
            "least_squares" => NativeRefinement::LeastSquares,
            other => {
                return Err(PyValueError::new_err(format!(
                    "refinement must be 'none', 'interpolate' or 'least_squares', got '{other}'"
                )));
            }
        };
        Ok(NativeShapeSearchConfig {
            min_score: self.min_score,
            greediness: self.greediness,
            max_matches: self.max_matches,
            max_overlap: self.max_overlap,
            min_contrast: self.min_contrast,
            refinement,
            last_level: self.last_level,
            max_candidates: self.max_candidates,
            coarse_score_factor: self.coarse_score_factor,
            ..NativeShapeSearchConfig::default()
        })
    }
}
