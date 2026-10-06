//! Python-visible [`BeadConfig`](vision_metrology::measure::BeadConfig) mirror: the
//! config, its two stage calipers and its tuning.
//!
//! The string fields are checked when the config is used (`BeadTracker(config)` or
//! assigning `BeadTracker.config`), which raises `ValueError` naming the allowed values;
//! so do the numeric ranges the native tracker validates.

use std::num::NonZeroUsize;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vision_metrology::fit::RobustLoss as NativeRobustLoss;
use vision_metrology::measure::{
    BeadCaliper as NativeBeadCaliper, BeadConfig as NativeBeadConfig,
    BeadPolarity as NativeBeadPolarity, BeadTuning as NativeBeadTuning,
};

use super::measure::{
    BORDER_MODES, DERIVATIVES, Locate, MeasureConfig, OFF_IMAGES, Profile, check_name, not_one_of,
    profile_names,
};

const POLARITIES: &[&str] = &["light", "dark"];
const LOSSES: &[&str] = &["none", "huber", "tukey"];

/// The caliper one stage of a `BeadTracker` measures with — mirrors
/// `vision_metrology::measure::BeadCaliper`.
///
/// The default is the tracking stage's caliper; `BeadConfig()` measures with a stricter
/// one. Every strip keeps all edges of both polarities, because the tracker chooses the
/// pair itself.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct BeadCaliper {
    /// Largest distance, in px, between a pair's midpoint and its station along the
    /// normal: the stage's search range. With `BeadConfig.max_width` it sets the strip
    /// length.
    pub max_offset: f32,
    /// Half-width of each strip along the curve, in px.
    pub half_width: f32,
    /// Minimum `|derivative response|` for an edge, on the input pixel scale.
    pub threshold: f32,
    /// How each edge is located; `Locate.midpoint_crossing` is rejected.
    pub locate: Locate,
    /// Maximum angle, in degrees, between the strip and the image gradient at an edge.
    pub max_obliquity_deg: f32,
    /// Gaussian sigma of the profile smoothing, in px.
    pub sigma: f32,
    /// Profile sampling step along the strip, in px.
    pub step: f32,
    /// "clamp", "reflect101" or "constant".
    pub border_mode: String,
    pub border_constant: f32,
    /// "dog" or "smooth_central".
    pub derivative: String,
    /// Half-width of the smoothing kernel for `derivative="smooth_central"`, in px.
    pub kernel_radius_px: f32,
    /// "fill" or "reject".
    pub off_image: String,
}

#[pymethods]
impl BeadCaliper {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        max_offset=None, half_width=None, threshold=None, locate=None, max_obliquity_deg=None,
        sigma=None, step=None, border_mode=None, border_constant=None, derivative=None,
        kernel_radius_px=None, off_image=None
    ))]
    pub fn new(
        max_offset: Option<f32>,
        half_width: Option<f32>,
        threshold: Option<f32>,
        locate: Option<Locate>,
        max_obliquity_deg: Option<f32>,
        sigma: Option<f32>,
        step: Option<f32>,
        border_mode: Option<String>,
        border_constant: Option<f32>,
        derivative: Option<String>,
        kernel_radius_px: Option<f32>,
        off_image: Option<String>,
    ) -> PyResult<Self> {
        for (name, value, allowed) in [
            ("border_mode", border_mode.as_deref(), BORDER_MODES),
            ("derivative", derivative.as_deref(), DERIVATIVES),
            ("off_image", off_image.as_deref(), OFF_IMAGES),
        ] {
            if let Some(v) = value {
                check_name(name, v, allowed)?;
            }
        }
        let d = Self::default();
        Ok(Self {
            max_offset: max_offset.unwrap_or(d.max_offset),
            half_width: half_width.unwrap_or(d.half_width),
            threshold: threshold.unwrap_or(d.threshold),
            locate: locate.unwrap_or(d.locate),
            max_obliquity_deg: max_obliquity_deg.unwrap_or(d.max_obliquity_deg),
            sigma: sigma.unwrap_or(d.sigma),
            step: step.unwrap_or(d.step),
            border_mode: border_mode.unwrap_or(d.border_mode),
            border_constant: border_constant.unwrap_or(d.border_constant),
            derivative: derivative.unwrap_or(d.derivative),
            kernel_radius_px: kernel_radius_px.unwrap_or(d.kernel_radius_px),
            off_image: off_image.unwrap_or(d.off_image),
        })
    }

    /// The `MeasureConfig` each strip of this stage measures with: every edge
    /// (`select="all"`) of either polarity (`polarity="any"`). Raises `ValueError` when a
    /// string field holds a name it does not accept.
    pub fn to_measure_config(&self) -> PyResult<MeasureConfig> {
        self.to_native()?;
        Ok(MeasureConfig {
            sigma: self.sigma,
            threshold: self.threshold,
            polarity: "any".into(),
            select: "all".into(),
            sequence: Vec::new(),
            step: self.step,
            max_obliquity_deg: self.max_obliquity_deg,
            border_mode: self.border_mode.clone(),
            border_constant: self.border_constant,
            derivative: self.derivative.clone(),
            kernel_radius_px: self.kernel_radius_px,
            locate: self.locate.clone(),
            off_image: self.off_image.clone(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "BeadCaliper(max_offset={}, half_width={}, threshold={}, sigma={}, step={})",
            self.max_offset, self.half_width, self.threshold, self.sigma, self.step
        )
    }
}

impl Default for BeadCaliper {
    fn default() -> Self {
        Self::from_native(&NativeBeadCaliper::default())
    }
}

impl BeadCaliper {
    fn from_native(c: &NativeBeadCaliper) -> Self {
        let (border_mode, border_constant, derivative, kernel_radius_px, off_image) =
            profile_names(&c.profile);
        Self {
            max_offset: c.max_offset,
            half_width: c.half_width,
            threshold: c.threshold,
            locate: Locate::from_native(c.locate),
            max_obliquity_deg: c.max_obliquity_deg,
            sigma: c.profile.sigma,
            step: c.profile.step,
            border_mode: border_mode.into(),
            border_constant,
            derivative: derivative.into(),
            kernel_radius_px,
            off_image: off_image.into(),
        }
    }

    fn to_native(&self) -> PyResult<NativeBeadCaliper> {
        Ok(NativeBeadCaliper {
            max_offset: self.max_offset,
            half_width: self.half_width,
            threshold: self.threshold,
            locate: self.locate.to_native()?,
            max_obliquity_deg: self.max_obliquity_deg,
            profile: Profile {
                sigma: self.sigma,
                step: self.step,
                border_mode: &self.border_mode,
                border_constant: self.border_constant,
                derivative: &self.derivative,
                kernel_radius_px: self.kernel_radius_px,
                off_image: &self.off_image,
            }
            .to_native()?,
        })
    }
}

/// How hard a `BeadTracker` works, and how stiff its corrections are — mirrors
/// `vision_metrology::measure::BeadTuning`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct BeadTuning {
    /// The most tracking passes (at least 1).
    pub passes: i64,
    /// Stop ("converged") once a pass's solved correction is below this at every station,
    /// in px, and was applied in full.
    pub tol: f32,
    /// Trust in the prior: a penalty on each correction's size; dimensionless.
    pub damping: f32,
    /// Tension length, px: the penalty on a correction's slope.
    pub tension_px: f32,
    /// Bending length, px: corrections shorter than about `2π·bending_px` are suppressed.
    pub bending_px: f32,
    /// "none", "huber" or "tukey".
    pub loss: String,
    /// The loss constant in px, for "huber" and "tukey".
    pub loss_scale: f32,
    /// The most reweighted solves per pass after the first, least-squares one (at least 1).
    pub irls_iters: i64,
    /// Half-length of each tangent's chord, in px of arc length.
    pub tangent_window_px: f32,
    /// The fraction of stations, in [0, 1], that must find the bead for a pass to move the
    /// curve.
    pub min_support: f32,
}

#[pymethods]
impl BeadTuning {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        passes=None, tol=None, damping=None, tension_px=None, bending_px=None, loss=None,
        loss_scale=None, irls_iters=None, tangent_window_px=None, min_support=None
    ))]
    pub fn new(
        passes: Option<i64>,
        tol: Option<f32>,
        damping: Option<f32>,
        tension_px: Option<f32>,
        bending_px: Option<f32>,
        loss: Option<String>,
        loss_scale: Option<f32>,
        irls_iters: Option<i64>,
        tangent_window_px: Option<f32>,
        min_support: Option<f32>,
    ) -> PyResult<Self> {
        if let Some(l) = loss.as_deref() {
            check_name("loss", l, LOSSES)?;
        }
        let d = Self::default();
        Ok(Self {
            passes: passes.unwrap_or(d.passes),
            tol: tol.unwrap_or(d.tol),
            damping: damping.unwrap_or(d.damping),
            tension_px: tension_px.unwrap_or(d.tension_px),
            bending_px: bending_px.unwrap_or(d.bending_px),
            loss: loss.unwrap_or(d.loss),
            loss_scale: loss_scale.unwrap_or(d.loss_scale),
            irls_iters: irls_iters.unwrap_or(d.irls_iters),
            tangent_window_px: tangent_window_px.unwrap_or(d.tangent_window_px),
            min_support: min_support.unwrap_or(d.min_support),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "BeadTuning(passes={}, tol={}, tension_px={}, bending_px={}, loss='{}', \
             loss_scale={})",
            self.passes, self.tol, self.tension_px, self.bending_px, self.loss, self.loss_scale
        )
    }
}

impl Default for BeadTuning {
    fn default() -> Self {
        Self::from_native(&NativeBeadTuning::default())
    }
}

impl BeadTuning {
    fn from_native(n: &NativeBeadTuning) -> Self {
        let (loss, loss_scale) = match n.loss {
            NativeRobustLoss::None => ("none", 1.0),
            NativeRobustLoss::Huber { k } => ("huber", k),
            NativeRobustLoss::Tukey { c } => ("tukey", c),
        };
        Self {
            passes: n.passes.get() as i64,
            tol: n.tol,
            damping: n.damping,
            tension_px: n.tension_px,
            bending_px: n.bending_px,
            loss: loss.into(),
            loss_scale,
            irls_iters: n.irls_iters.get() as i64,
            tangent_window_px: n.tangent_window_px,
            min_support: n.min_support,
        }
    }

    fn to_native(&self) -> PyResult<NativeBeadTuning> {
        let count = |name: &str, v: i64| {
            usize::try_from(v)
                .ok()
                .and_then(NonZeroUsize::new)
                .ok_or_else(|| PyValueError::new_err(format!("{name} must be at least 1, got {v}")))
        };
        Ok(NativeBeadTuning {
            passes: count("passes", self.passes)?,
            tol: self.tol,
            damping: self.damping,
            tension_px: self.tension_px,
            bending_px: self.bending_px,
            loss: match self.loss.as_str() {
                "none" => NativeRobustLoss::None,
                "huber" => NativeRobustLoss::Huber { k: self.loss_scale },
                "tukey" => NativeRobustLoss::Tukey { c: self.loss_scale },
                other => return Err(not_one_of("loss", other, LOSSES)),
            },
            irls_iters: count("irls_iters", self.irls_iters)?,
            tangent_window_px: self.tangent_window_px,
            min_support: self.min_support,
        })
    }
}

/// What a `BeadTracker` tracks and measures — mirrors
/// `vision_metrology::measure::BeadConfig`.
///
/// `track`, `measure` and `tuning` are copies: change one by assigning a whole new value
/// (`cfg.track = vm.BeadCaliper(max_offset=20.0)`), not by setting its fields in place.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct BeadConfig {
    /// "light" (brighter than the background) or "dark".
    pub polarity: String,
    /// The narrowest pair, in px, inclusive.
    pub min_width: f32,
    /// The widest pair, in px, inclusive.
    pub max_width: f32,
    /// Arc length between stations, in px.
    pub spacing: f32,
    /// When set, reject a pair with another edge within this many px outside it.
    pub clearance: Option<f32>,
    /// When set, reject a pair whose score margin over the next is below this, in (0, 1).
    pub min_margin: Option<f32>,
    /// The tracking stage's caliper.
    pub track: BeadCaliper,
    /// The final stage's caliper.
    pub measure: BeadCaliper,
    pub tuning: BeadTuning,
}

#[pymethods]
impl BeadConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        polarity=None, min_width=None, max_width=None, spacing=None, clearance=None,
        min_margin=None, track=None, measure=None, tuning=None
    ))]
    pub fn new(
        polarity: Option<String>,
        min_width: Option<f32>,
        max_width: Option<f32>,
        spacing: Option<f32>,
        clearance: Option<f32>,
        min_margin: Option<f32>,
        track: Option<BeadCaliper>,
        measure: Option<BeadCaliper>,
        tuning: Option<BeadTuning>,
    ) -> PyResult<Self> {
        if let Some(p) = polarity.as_deref() {
            check_name("polarity", p, POLARITIES)?;
        }
        let d = Self::default();
        Ok(Self {
            polarity: polarity.unwrap_or(d.polarity),
            min_width: min_width.unwrap_or(d.min_width),
            max_width: max_width.unwrap_or(d.max_width),
            spacing: spacing.unwrap_or(d.spacing),
            clearance,
            min_margin,
            track: track.unwrap_or(d.track),
            measure: measure.unwrap_or(d.measure),
            tuning: tuning.unwrap_or(d.tuning),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "BeadConfig(polarity='{}', min_width={}, max_width={}, spacing={}, clearance={:?}, \
             min_margin={:?})",
            self.polarity,
            self.min_width,
            self.max_width,
            self.spacing,
            self.clearance,
            self.min_margin
        )
    }
}

impl Default for BeadConfig {
    fn default() -> Self {
        let n = NativeBeadConfig::default();
        Self {
            polarity: "light".into(),
            min_width: n.min_width,
            max_width: n.max_width,
            spacing: n.spacing,
            clearance: n.clearance,
            min_margin: n.min_margin,
            track: BeadCaliper::from_native(&n.track),
            measure: BeadCaliper::from_native(&n.measure),
            tuning: BeadTuning::from_native(&n.tuning),
        }
    }
}

impl BeadConfig {
    /// The native config. String fields are checked here; the numeric ranges are checked
    /// by the native tracker.
    pub fn to_native(&self) -> PyResult<NativeBeadConfig> {
        let polarity = match self.polarity.as_str() {
            "light" => NativeBeadPolarity::Light,
            "dark" => NativeBeadPolarity::Dark,
            other => return Err(not_one_of("polarity", other, POLARITIES)),
        };
        Ok(NativeBeadConfig {
            polarity,
            min_width: self.min_width,
            max_width: self.max_width,
            spacing: self.spacing,
            clearance: self.clearance,
            min_margin: self.min_margin,
            track: self.track.to_native()?,
            measure: self.measure.to_native()?,
            tuning: self.tuning.to_native()?,
        })
    }
}
