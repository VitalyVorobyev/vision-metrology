//! Python-visible [`BeadCoupling`](vision_metrology::measure::BeadCoupling) mirror: the
//! costs of a bead tracker's coupled pairing. `BeadConfig.pairing` holds one, or `None`
//! for per-station pairing.

use pyo3::prelude::*;
use vision_metrology::measure::BeadCoupling as NativeBeadCoupling;

/// The costs of a `BeadTracker`'s coupled pairing — mirrors
/// `vision_metrology::measure::BeadCoupling`.
///
/// Setting `BeadConfig.pairing` to one chooses both walls at every station together, as
/// one exact path along the curve, with states for one visible wall and for absence.
/// `saturation` and `absent_level` are derivative responses in input units per px; the
/// stiffnesses, `switch_cost` and `offset_prior` are dimensionless; the slopes are px per
/// px of curve; `bin_px` and `margin_tol_px` are px; `margin_quantile` is in (0, 1].
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct BeadCoupling {
    pub saturation: f32,
    pub absent_level: f32,
    pub wall_stiffness: f32,
    pub width_stiffness: f32,
    pub max_wall_slope: f32,
    pub max_width_slope: f32,
    pub switch_cost: f32,
    pub offset_prior: f32,
    pub bin_px: f32,
    pub margin_tol_px: f32,
    pub margin_quantile: f32,
}

#[pymethods]
impl BeadCoupling {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        saturation=None, absent_level=None, wall_stiffness=None, width_stiffness=None,
        max_wall_slope=None, max_width_slope=None, switch_cost=None, offset_prior=None,
        bin_px=None, margin_tol_px=None, margin_quantile=None
    ))]
    pub fn new(
        saturation: Option<f32>,
        absent_level: Option<f32>,
        wall_stiffness: Option<f32>,
        width_stiffness: Option<f32>,
        max_wall_slope: Option<f32>,
        max_width_slope: Option<f32>,
        switch_cost: Option<f32>,
        offset_prior: Option<f32>,
        bin_px: Option<f32>,
        margin_tol_px: Option<f32>,
        margin_quantile: Option<f32>,
    ) -> Self {
        let d = NativeBeadCoupling::default();
        Self {
            saturation: saturation.unwrap_or(d.saturation),
            absent_level: absent_level.unwrap_or(d.absent_level),
            wall_stiffness: wall_stiffness.unwrap_or(d.wall_stiffness),
            width_stiffness: width_stiffness.unwrap_or(d.width_stiffness),
            max_wall_slope: max_wall_slope.unwrap_or(d.max_wall_slope),
            max_width_slope: max_width_slope.unwrap_or(d.max_width_slope),
            switch_cost: switch_cost.unwrap_or(d.switch_cost),
            offset_prior: offset_prior.unwrap_or(d.offset_prior),
            bin_px: bin_px.unwrap_or(d.bin_px),
            margin_tol_px: margin_tol_px.unwrap_or(d.margin_tol_px),
            margin_quantile: margin_quantile.unwrap_or(d.margin_quantile),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BeadCoupling(saturation={}, absent_level={}, wall_stiffness={}, \
             width_stiffness={}, switch_cost={}, bin_px={})",
            self.saturation,
            self.absent_level,
            self.wall_stiffness,
            self.width_stiffness,
            self.switch_cost,
            self.bin_px
        )
    }
}

impl BeadCoupling {
    pub(super) fn from_native(n: &NativeBeadCoupling) -> Self {
        Self {
            saturation: n.saturation,
            absent_level: n.absent_level,
            wall_stiffness: n.wall_stiffness,
            width_stiffness: n.width_stiffness,
            max_wall_slope: n.max_wall_slope,
            max_width_slope: n.max_width_slope,
            switch_cost: n.switch_cost,
            offset_prior: n.offset_prior,
            bin_px: n.bin_px,
            margin_tol_px: n.margin_tol_px,
            margin_quantile: n.margin_quantile,
        }
    }

    pub(super) fn to_native(&self) -> NativeBeadCoupling {
        NativeBeadCoupling {
            saturation: self.saturation,
            absent_level: self.absent_level,
            wall_stiffness: self.wall_stiffness,
            width_stiffness: self.width_stiffness,
            max_wall_slope: self.max_wall_slope,
            max_width_slope: self.max_width_slope,
            switch_cost: self.switch_cost,
            offset_prior: self.offset_prior,
            bin_px: self.bin_px,
            margin_tol_px: self.margin_tol_px,
            margin_quantile: self.margin_quantile,
        }
    }
}
