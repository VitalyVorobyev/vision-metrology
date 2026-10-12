//! Python configuration for native masked translation.

use pyo3::{exceptions::PyValueError, prelude::*};
use std::num::NonZeroUsize;
use vision_metrology::corr::{
    TranslationConfig as NativeConfig, TranslationTuning as NativeTuning,
};

fn positive(value: usize) -> PyResult<NonZeroUsize> {
    NonZeroUsize::new(value).ok_or_else(|| PyValueError::new_err("count must be positive"))
}

/// Numerical effort for masked translation.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Clone)]
pub struct TranslationTuning {
    pub max_levels: usize,
    pub iterations: usize,
    pub tolerance: f32,
    pub huber: f32,
    pub min_texture: f32,
}

#[pymethods]
impl TranslationTuning {
    #[new]
    #[pyo3(signature=(*,max_levels=3,iterations=12,tolerance=0.005,huber=1.5,min_texture=1e-5))]
    fn new(
        max_levels: usize,
        iterations: usize,
        tolerance: f32,
        huber: f32,
        min_texture: f32,
    ) -> Self {
        Self {
            max_levels,
            iterations,
            tolerance,
            huber,
            min_texture,
        }
    }
}

impl Default for TranslationTuning {
    fn default() -> Self {
        let n = NativeTuning::default();
        Self {
            max_levels: n.max_levels.get(),
            iterations: n.iterations.get(),
            tolerance: n.tolerance,
            huber: n.huber,
            min_texture: n.min_texture,
        }
    }
}

impl TranslationTuning {
    pub(crate) fn to_native(&self) -> PyResult<NativeTuning> {
        Ok(NativeTuning {
            max_levels: positive(self.max_levels)?,
            iterations: positive(self.iterations)?,
            tolerance: self.tolerance,
            huber: self.huber,
            min_texture: self.min_texture,
        })
    }
}

/// Bounded translation search and acceptance gates. Masks are uint8, nonzero valid.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Clone)]
pub struct TranslationConfig {
    pub search_radius: usize,
    pub min_support: usize,
    pub min_score: f32,
    pub min_margin: f32,
    pub max_region_error: Option<f32>,
    pub tuning: TranslationTuning,
}

#[pymethods]
impl TranslationConfig {
    #[new]
    #[pyo3(signature=(*,search_radius=12,min_support=400,min_score=0.5,min_margin=0.02,max_region_error=Some(0.5),tuning=None))]
    fn new(
        search_radius: usize,
        min_support: usize,
        min_score: f32,
        min_margin: f32,
        max_region_error: Option<f32>,
        tuning: Option<TranslationTuning>,
    ) -> Self {
        Self {
            search_radius,
            min_support,
            min_score,
            min_margin,
            max_region_error,
            tuning: tuning.unwrap_or_default(),
        }
    }
}

impl Default for TranslationConfig {
    fn default() -> Self {
        let n = NativeConfig::default();
        Self {
            search_radius: n.search_radius.get(),
            min_support: n.min_support.get(),
            min_score: n.min_score,
            min_margin: n.min_margin,
            max_region_error: n.max_region_error,
            tuning: TranslationTuning::default(),
        }
    }
}

impl TranslationConfig {
    pub(crate) fn to_native(&self) -> PyResult<NativeConfig> {
        Ok(NativeConfig {
            search_radius: positive(self.search_radius)?,
            min_support: positive(self.min_support)?,
            min_score: self.min_score,
            min_margin: self.min_margin,
            max_region_error: self.max_region_error,
            tuning: self.tuning.to_native()?,
        })
    }
}
