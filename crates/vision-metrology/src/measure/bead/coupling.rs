//! How a bead tracker's stages choose the walls: per station, or along the whole curve at
//! once (ADR-0019).

use vm_primitives::Error;

use super::config::{BeadConfig, non_negative, positive};

/// How a [`BeadTracker`](super::BeadTracker) stage chooses the bead's walls.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BeadPairing {
    /// Each station picks its own edge pair from its strip's edges, through the gates of
    /// [`BeadConfig`] (ADR-0018).
    #[default]
    PerStation,
    /// One exact path chooses both walls at every station together, under continuity
    /// along the curve, with states for one visible wall and for absence (ADR-0019).
    /// [`BeadConfig::min_margin`] is a per-station gate and must be `None`;
    /// [`BeadConfig::clearance`] becomes a cost: a wall pays the evidence of the strongest
    /// other edge within `clearance` outside it, beyond a pixel.
    Coupled(BeadCoupling),
}

/// The costs of [`BeadPairing::Coupled`]'s path.
///
/// At each station the path's state is the bead's centre `c` and width `w` on a grid of
/// [`bin_px`](Self::bin_px), and which walls it measures: both, one, or none. A wall's
/// evidence is its stage's caliper response `g` at the wall, per pixel, signed so that the
/// bead's polarity is positive and softly saturated, `e(g) = (g/s)/(1 + |g|/s)` for the
/// [`saturation`](Self::saturation) `s`, so that one strong edge cannot carry a pair. Per
/// pixel of curve a state costs:
///
/// - both walls: `−e(g_l) − e(g_r) + λ0·(c/reach)²`, plus each wall's clearance cost
///   when [`BeadConfig::clearance`] is set;
/// - one wall, only where the other wall's position has no valid response (off the
///   image): `−e(g) − e(absent_level) + λ0·(c/reach)²`;
/// - absent: `−2·e(absent_level)`.
///
/// Between stations `h` px apart the walls `l` and `r` pay `κ·(Δl² + Δr²)/h + κ_w·Δw²/h`,
/// with `|Δc|` and `|Δw|` bounded by the slopes times `h` (and at least one bin), and a
/// change between a present and the absent state pays [`switch_cost`](Self::switch_cost).
/// Data per pixel and smoothness per unit slope make the result independent of the
/// station spacing. The absent state keeps a centre and width, so a gap is bridged
/// smoothly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeadCoupling {
    /// `s`, the response at which a wall's evidence reaches half its ceiling, in input
    /// units per pixel: re-tune for `u16` and `f32` images.
    pub saturation: f32,
    /// The response, in input units per pixel, above which the path prefers measuring a
    /// wall to calling it absent.
    pub absent_level: f32,
    /// `κ`, the stiffness of each wall's position along the curve; dimensionless.
    pub wall_stiffness: f32,
    /// `κ_w`, the extra stiffness of the width; dimensionless.
    pub width_stiffness: f32,
    /// The largest change of the centre per pixel of curve.
    pub max_wall_slope: f32,
    /// The largest change of the width per pixel of curve.
    pub max_width_slope: f32,
    /// The cost of each change between a present state and the absent one, and between
    /// the two one-wall states. A change between both walls and one wall is free: where
    /// a wall leaves the image decides it.
    pub switch_cost: f32,
    /// `λ0`, trust in the prior: the penalty on the centre's offset from the station,
    /// relative to the stage's reach; dimensionless.
    pub offset_prior: f32,
    /// The centre's grid, in pixels, rounded to a whole number of the stage's profile
    /// samples (at least one). The width's grid is twice it, so both walls fall on the
    /// centre's grid. Each chosen wall then moves to its caliper's sub-pixel edge.
    pub bin_px: f32,
    /// A different answer at a station moves a wall further than this, in pixels. The
    /// station's margin is what the best such answer costs more than the chosen one.
    pub margin_tol_px: f32,
    /// The quantile, in `(0, 1]`, of the station margins that scores a whole stage:
    /// [`BeadPathScore::margin`](super::BeadPathScore::margin).
    pub margin_quantile: f32,
}

impl Default for BeadCoupling {
    fn default() -> Self {
        Self {
            saturation: 10.0,
            absent_level: 4.0,
            wall_stiffness: 0.2,
            width_stiffness: 0.5,
            max_wall_slope: 1.0,
            max_width_slope: 0.5,
            switch_cost: 2.0,
            offset_prior: 0.1,
            bin_px: 0.5,
            margin_tol_px: 5.0,
            margin_quantile: 0.25,
        }
    }
}

/// The most (centre, width) cells per station a coupled path may have.
const MAX_PATH_CELLS: f64 = (1u64 << 16) as f64;

/// Check `cfg.pairing` against the rest of `cfg`.
pub(super) fn validate_pairing(cfg: &BeadConfig) -> Result<(), Error> {
    let BeadPairing::Coupled(c) = &cfg.pairing else {
        return Ok(());
    };
    if cfg.min_margin.is_some() {
        return Err(Error::InvalidConfig(
            "bead min_margin is a per-station gate: coupled pairing needs it None",
        ));
    }
    positive(
        c.saturation,
        "bead coupling saturation must be finite and positive",
    )?;
    for (v, what) in [
        (
            c.absent_level,
            "bead coupling absent_level must be finite and non-negative",
        ),
        (
            c.wall_stiffness,
            "bead coupling wall_stiffness must be finite and non-negative",
        ),
        (
            c.width_stiffness,
            "bead coupling width_stiffness must be finite and non-negative",
        ),
        (
            c.switch_cost,
            "bead coupling switch_cost must be finite and non-negative",
        ),
        (
            c.offset_prior,
            "bead coupling offset_prior must be finite and non-negative",
        ),
    ] {
        non_negative(v, what)?;
    }
    for (v, what) in [
        (
            c.max_wall_slope,
            "bead coupling max_wall_slope must be finite and positive",
        ),
        (
            c.max_width_slope,
            "bead coupling max_width_slope must be finite and positive",
        ),
        (c.bin_px, "bead coupling bin_px must be finite and positive"),
        (
            c.margin_tol_px,
            "bead coupling margin_tol_px must be finite and positive",
        ),
    ] {
        positive(v, what)?;
    }
    if !(c.margin_quantile > 0.0 && c.margin_quantile <= 1.0) {
        return Err(Error::InvalidConfig(
            "bead coupling margin_quantile must be in (0, 1]",
        ));
    }
    for stage in [&cfg.track, &cfg.measure] {
        let bin = f64::from(c.bin_px).max(f64::from(stage.profile.step));
        let centres = 2.0 * f64::from(stage.max_offset) / bin + 1.0;
        let widths = f64::from(cfg.max_width - cfg.min_width) / (2.0 * bin) + 1.0;
        if centres * widths > MAX_PATH_CELLS {
            return Err(Error::InvalidConfig(
                "bead coupling grid exceeds 2^16 cells per station: raise bin_px, or lower \
                 max_offset or the width range",
            ));
        }
    }
    Ok(())
}
