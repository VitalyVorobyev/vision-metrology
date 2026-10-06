//! What a bead tracker looks for, how far each stage searches, and how hard it works.

use std::num::NonZeroUsize;

use vm_primitives::Error;

use crate::fit::RobustLoss;
use crate::measure::{
    Derivative, EdgeSelect, Locate, MeasureConfig, PolaritySelect, ProfileConfig,
};

/// Which way a bead differs from its background.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BeadPolarity {
    /// Brighter than the background: along each scan (towards +n), a rising edge, then a
    /// falling one.
    #[default]
    Light,
    /// Darker than the background: a falling edge, then a rising one.
    Dark,
}

/// The caliper one stage of a [`BeadTracker`](super::BeadTracker) measures with.
///
/// The subset of [`MeasureConfig`] a bead stage can usefully set. Every strip keeps all
/// edges of both polarities (`select` [`EdgeSelect::All`], `polarity`
/// [`PolaritySelect::Any`]), because the tracker chooses the pair itself.
///
/// `Default` is the tracking stage's caliper: a wide, permissive search.
/// [`BeadConfig::default`] measures with a stricter one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeadCaliper {
    /// The largest distance, in pixels, between a pair's midpoint and its station along
    /// the normal. It is the search range of the stage, and with
    /// [`BeadConfig::max_width`] it sets the strip's length.
    pub max_offset: f32,
    /// Half-width of each strip along the curve, in pixels: how much of the bead each
    /// profile averages. Keep it small on a tight bend, where a straight strip averages
    /// across the curve.
    pub half_width: f32,
    /// Minimum `|derivative response|` for an edge, on the input pixel scale: re-tune for
    /// `u16` and `f32` images.
    pub threshold: f32,
    /// How each edge is located on the profile. [`Locate::MidpointCrossing`] finds one
    /// edge per profile and is rejected.
    pub locate: Locate,
    /// Maximum angle, in degrees, between the strip and the image gradient at an edge.
    /// `180.0` disables the check.
    pub max_obliquity_deg: f32,
    /// How each profile is sampled and smoothed.
    pub profile: ProfileConfig,
}

impl Default for BeadCaliper {
    fn default() -> Self {
        let m = MeasureConfig::default();
        Self {
            max_offset: 15.0,
            half_width: 2.0,
            threshold: m.threshold,
            locate: m.locate,
            max_obliquity_deg: 180.0,
            profile: m.profile,
        }
    }
}

impl BeadCaliper {
    /// The caliper config each strip of this stage measures with: every edge
    /// ([`EdgeSelect::All`]) of either polarity ([`PolaritySelect::Any`]).
    pub fn to_measure_config(&self) -> MeasureConfig {
        MeasureConfig {
            threshold: self.threshold,
            polarity: PolaritySelect::Any,
            select: EdgeSelect::All,
            locate: self.locate,
            max_obliquity_deg: self.max_obliquity_deg,
            profile: self.profile,
        }
    }
}

/// How hard the tracking loop works, and how stiff the curve's corrections are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeadTuning {
    /// The most tracking passes: measure, solve, move, resample.
    pub passes: NonZeroUsize,
    /// The loop stops once a pass moves no station by more than this, in pixels.
    pub tol: f32,
    /// Trust in the prior, `λ0`: a penalty on the size of each correction, relative to a
    /// station's data weight of 1. Dimensionless; `0.0` lets data move the curve freely.
    pub damping: f32,
    /// Tension length `ℓ1`, in pixels: the penalty on a correction's slope along the curve.
    pub tension_px: f32,
    /// Bending length `ℓ2`, in pixels: the penalty on a correction's curvature. A
    /// correction whose wavelength is shorter than about `2π·ℓ2` is suppressed in one
    /// pass; a longer one passes.
    pub bending_px: f32,
    /// How each station's observation is weighted against the curve, with its constant in
    /// pixels. A robust loss keeps a wrong pair from pulling the curve.
    pub loss: RobustLoss,
    /// The most reweighted solves per pass.
    pub irls_iters: NonZeroUsize,
    /// Half-length, in pixels of arc length, of the chord each tangent is taken over.
    /// Longer chords smooth a coarse prior's corners.
    pub tangent_window_px: f32,
    /// The fraction of stations, in `[0, 1]`, that must find the bead for a pass to move
    /// the curve.
    pub min_support: f32,
}

impl Default for BeadTuning {
    fn default() -> Self {
        Self {
            passes: NonZeroUsize::new(3).expect("non-zero"),
            tol: 0.05,
            damping: 0.0,
            tension_px: 2.0,
            bending_px: 8.0,
            loss: RobustLoss::Huber { k: 1.0 },
            irls_iters: NonZeroUsize::new(5).expect("non-zero"),
            tangent_window_px: 10.0,
            min_support: 0.2,
        }
    }
}

/// What a [`BeadTracker`](super::BeadTracker) tracks and measures.
///
/// Widths, offsets, spacing and clearance are in pixels. The defaults describe a wide
/// light bead, 30 to 80 px across, whose prior is within 15 px.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeadConfig {
    /// Whether the bead is lighter or darker than its background.
    pub polarity: BeadPolarity,
    /// The narrowest pair, in pixels, inclusive.
    pub min_width: f32,
    /// The widest pair, in pixels, inclusive.
    pub max_width: f32,
    /// Arc length between stations, in pixels. The station count is fixed per call:
    /// `round(L / spacing) + 1`, at least 2, for a prior of length `L`.
    pub spacing: f32,
    /// When set, a pair is rejected if any other edge lies within this many pixels
    /// outside either of its edges. Edges between the pair (a highlight) are allowed.
    pub clearance: Option<f32>,
    /// When set, a pair is rejected as ambiguous unless its margin over the best other
    /// pair, `1 − s₂/s₁` in scores, reaches this value, in `(0, 1)`.
    pub min_margin: Option<f32>,
    /// The tracking stage: evidence for where the bead is.
    pub track: BeadCaliper,
    /// The final stage: the reported positions and widths.
    pub measure: BeadCaliper,
    /// The loop's effort and the correction's stiffness.
    pub tuning: BeadTuning,
}

impl Default for BeadConfig {
    fn default() -> Self {
        let track = BeadCaliper::default();
        Self {
            polarity: BeadPolarity::Light,
            min_width: 30.0,
            max_width: 80.0,
            spacing: 4.0,
            clearance: None,
            min_margin: None,
            track,
            measure: BeadCaliper {
                max_offset: 3.0,
                half_width: 1.0,
                max_obliquity_deg: 30.0,
                profile: ProfileConfig {
                    step: 0.5,
                    ..track.profile
                },
                ..track
            },
            tuning: BeadTuning::default(),
        }
    }
}

/// `Ok` when `v` is finite and `v > 0`.
fn positive(v: f32, what: &'static str) -> Result<(), Error> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidConfig(what))
    }
}

/// `Ok` when `v` is finite and `v ≥ 0`.
fn non_negative(v: f32, what: &'static str) -> Result<(), Error> {
    if v.is_finite() && v >= 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidConfig(what))
    }
}

/// Check every setting of `cfg`.
pub(super) fn validate(cfg: &BeadConfig) -> Result<(), Error> {
    non_negative(
        cfg.min_width,
        "bead min_width must be finite and non-negative",
    )?;
    positive(cfg.max_width, "bead max_width must be finite and positive")?;
    if cfg.min_width > cfg.max_width {
        return Err(Error::InvalidConfig("bead min_width exceeds max_width"));
    }
    positive(cfg.spacing, "bead spacing must be finite and positive")?;
    if let Some(c) = cfg.clearance {
        positive(c, "bead clearance must be finite and positive")?;
    }
    if let Some(m) = cfg.min_margin
        && !(m > 0.0 && m < 1.0)
    {
        return Err(Error::InvalidConfig("bead min_margin must be in (0, 1)"));
    }
    validate_caliper(&cfg.track, cfg.min_width)?;
    validate_caliper(&cfg.measure, cfg.min_width)?;
    validate_tuning(&cfg.tuning)
}

fn validate_caliper(c: &BeadCaliper, min_width: f32) -> Result<(), Error> {
    positive(c.max_offset, "bead max_offset must be finite and positive")?;
    non_negative(
        c.half_width,
        "bead half_width must be finite and non-negative",
    )?;
    non_negative(
        c.threshold,
        "bead threshold must be finite and non-negative",
    )?;
    if !(c.max_obliquity_deg > 0.0 && c.max_obliquity_deg <= 180.0) {
        return Err(Error::InvalidConfig(
            "bead max_obliquity_deg must be in (0, 180]",
        ));
    }
    positive(
        c.profile.sigma,
        "bead profile sigma must be finite and positive",
    )?;
    positive(
        c.profile.step,
        "bead profile step must be finite and positive",
    )?;
    if let Derivative::SmoothThenCentral { radius_px } = c.profile.derivative {
        positive(
            radius_px,
            "bead smoothing radius must be finite and positive",
        )?;
    }
    match c.locate {
        Locate::GradientPeak { .. } => Ok(()),
        Locate::MidpointCrossing { .. } => Err(Error::InvalidConfig(
            "bead calipers need every edge; MidpointCrossing finds one per profile",
        )),
        Locate::HalfContrast {
            flank_near_px,
            flank_far_px,
            tol_px,
            min_contrast,
            ..
        } => {
            non_negative(flank_near_px, "bead half-contrast flanks must be finite")?;
            positive(flank_far_px, "bead half-contrast flanks must be finite")?;
            positive(tol_px, "bead half-contrast tolerance must be positive")?;
            non_negative(
                min_contrast,
                "bead half-contrast min_contrast must be finite",
            )?;
            if flank_near_px > flank_far_px {
                return Err(Error::InvalidConfig(
                    "bead half-contrast flank_near_px exceeds flank_far_px",
                ));
            }
            // Each edge's inner flank must stay on the bead, short of the opposite edge.
            if flank_far_px >= min_width {
                return Err(Error::InvalidConfig(
                    "bead half-contrast flanks reach the opposite edge (flank_far_px >= min_width)",
                ));
            }
            Ok(())
        }
    }
}

fn validate_tuning(t: &BeadTuning) -> Result<(), Error> {
    positive(t.tol, "bead tol must be finite and positive")?;
    non_negative(t.damping, "bead damping must be finite and non-negative")?;
    non_negative(
        t.tension_px,
        "bead tension_px must be finite and non-negative",
    )?;
    non_negative(
        t.bending_px,
        "bead bending_px must be finite and non-negative",
    )?;
    positive(
        t.tangent_window_px,
        "bead tangent_window_px must be finite and positive",
    )?;
    if !(t.min_support >= 0.0 && t.min_support <= 1.0) {
        return Err(Error::InvalidConfig("bead min_support must be in [0, 1]"));
    }
    match t.loss {
        RobustLoss::None => Ok(()),
        RobustLoss::Huber { k: v } | RobustLoss::Tukey { c: v } => {
            positive(v, "bead loss constant must be finite and positive")
        }
    }
}
