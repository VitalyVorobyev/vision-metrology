//! Python configuration for the planar rig pose estimator (`metric::rig`).

use pyo3::{exceptions::PyValueError, prelude::*};
use std::num::NonZeroUsize;
use vision_metrology::metric::rig::{
    MotionNoise as NativeNoise, RigPoseConfig as NativeConfig, RigPoseTuning as NativeTuning,
    RigSmootherConfig as NativeSmoother, RotationHold,
};
use vm_primitives::{Isometry2f, Vec2f};

/// Per-step 1-σ change of the increment, at the rig's lever centre.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Clone)]
pub struct MotionNoise {
    pub translation_mm: f32,
    pub rotation_rad: f32,
}

#[pymethods]
impl MotionNoise {
    #[new]
    #[pyo3(signature=(*,translation_mm=0.1,rotation_rad=1e-3))]
    fn new(translation_mm: f32, rotation_rad: f32) -> Self {
        Self {
            translation_mm,
            rotation_rad,
        }
    }
}

impl Default for MotionNoise {
    fn default() -> Self {
        let n = NativeNoise::default();
        Self {
            translation_mm: n.translation_mm,
            rotation_rad: n.rotation_rad,
        }
    }
}

impl MotionNoise {
    fn to_native(&self) -> NativeNoise {
        NativeNoise {
            translation_mm: self.translation_mm,
            rotation_rad: self.rotation_rad,
        }
    }
}

/// Numerical effort of the per-step solve.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Clone)]
pub struct RigPoseTuning {
    pub iterations: usize,
    pub tolerance_mm: f32,
    pub shift_sigma_floor_px: f32,
}

#[pymethods]
impl RigPoseTuning {
    #[new]
    #[pyo3(signature=(*,iterations=20,tolerance_mm=1e-7,shift_sigma_floor_px=0.05))]
    fn new(iterations: usize, tolerance_mm: f32, shift_sigma_floor_px: f32) -> Self {
        Self {
            iterations,
            tolerance_mm,
            shift_sigma_floor_px,
        }
    }
}

impl Default for RigPoseTuning {
    fn default() -> Self {
        let n = NativeTuning::default();
        Self {
            iterations: n.iterations.get(),
            tolerance_mm: n.tolerance_mm,
            shift_sigma_floor_px: n.shift_sigma_floor_px,
        }
    }
}

/// Planar rig pose estimation. `anchor` is the frame-0 substrate→plane pose
/// `(tx, ty, theta)`; `rotation_hold` is `'previous'` or `'zero'`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Clone)]
pub struct RigPoseConfig {
    pub anchor: (f32, f32, f32),
    pub huber_k: f32,
    pub tukey_c: Option<f32>,
    pub max_closure_px: Option<f32>,
    pub rotation_hold: String,
    pub rotation_prior_sigma_rad: f32,
    pub motion_noise: MotionNoise,
    pub tuning: RigPoseTuning,
}

#[pymethods]
impl RigPoseConfig {
    #[new]
    #[pyo3(signature=(*,anchor=(0.0,0.0,0.0),huber_k=2.0,tukey_c=Some(6.0),max_closure_px=None,
        rotation_hold="previous".to_string(),rotation_prior_sigma_rad=1e-3,motion_noise=None,tuning=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        anchor: (f32, f32, f32),
        huber_k: f32,
        tukey_c: Option<f32>,
        max_closure_px: Option<f32>,
        rotation_hold: String,
        rotation_prior_sigma_rad: f32,
        motion_noise: Option<MotionNoise>,
        tuning: Option<RigPoseTuning>,
    ) -> Self {
        Self {
            anchor,
            huber_k,
            tukey_c,
            max_closure_px,
            rotation_hold,
            rotation_prior_sigma_rad,
            motion_noise: motion_noise.unwrap_or_default(),
            tuning: tuning.unwrap_or_default(),
        }
    }
}

impl Default for RigPoseConfig {
    fn default() -> Self {
        let n = NativeConfig::default();
        Self {
            anchor: (
                n.anchor.translation.x,
                n.anchor.translation.y,
                n.anchor.rotation.angle(),
            ),
            huber_k: n.huber_k,
            tukey_c: n.tukey_c,
            max_closure_px: n.max_closure_px,
            rotation_hold: "previous".to_string(),
            rotation_prior_sigma_rad: n.rotation_prior_sigma_rad,
            motion_noise: MotionNoise::default(),
            tuning: RigPoseTuning::default(),
        }
    }
}

impl RigPoseConfig {
    pub(crate) fn to_native(&self) -> PyResult<NativeConfig> {
        let rotation_hold = match self.rotation_hold.as_str() {
            "previous" => RotationHold::Previous,
            "zero" => RotationHold::Zero,
            other => {
                return Err(PyValueError::new_err(format!(
                    "rotation_hold must be 'previous' or 'zero', got {other:?}"
                )));
            }
        };
        let (tx, ty, theta) = self.anchor;
        Ok(NativeConfig {
            anchor: Isometry2f::new(Vec2f::new(tx, ty), theta),
            huber_k: self.huber_k,
            tukey_c: self.tukey_c,
            max_closure_px: self.max_closure_px,
            rotation_hold,
            rotation_prior_sigma_rad: self.rotation_prior_sigma_rad,
            motion_noise: self.motion_noise.to_native(),
            tuning: NativeTuning {
                iterations: NonZeroUsize::new(self.tuning.iterations)
                    .ok_or_else(|| PyValueError::new_err("iterations must be positive"))?,
                tolerance_mm: self.tuning.tolerance_mm,
                shift_sigma_floor_px: self.tuning.shift_sigma_floor_px,
            },
        })
    }
}

/// The fixed-interval smoother over the increments.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Clone)]
pub struct RigSmootherConfig {
    pub motion_noise: MotionNoise,
    pub gate_sigma: Option<f32>,
}

#[pymethods]
impl RigSmootherConfig {
    #[new]
    #[pyo3(signature=(*,motion_noise=None,gate_sigma=Some(5.0)))]
    fn new(motion_noise: Option<MotionNoise>, gate_sigma: Option<f32>) -> Self {
        Self {
            motion_noise: motion_noise.unwrap_or_default(),
            gate_sigma,
        }
    }
}

impl Default for RigSmootherConfig {
    fn default() -> Self {
        let n = NativeSmoother::default();
        Self {
            motion_noise: MotionNoise::default(),
            gate_sigma: n.gate_sigma,
        }
    }
}

impl RigSmootherConfig {
    pub(crate) fn to_native(&self) -> NativeSmoother {
        NativeSmoother {
            motion_noise: self.motion_noise.to_native(),
            gate_sigma: self.gate_sigma,
        }
    }
}
