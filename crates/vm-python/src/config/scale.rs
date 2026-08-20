//! Python-visible `scale` config mirrors: `MomentScaleConfig`,
//! `LogPolarScaleConfig`, `ScaleInvariantConfig`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use vision_metrology::scale::{
    BlobPolarity as NativeBlobPolarity, LogPolarScaleConfig as NativeLogPolarScaleConfig,
    MomentScaleConfig as NativeMomentScaleConfig,
    ScaleInvariantConfig as NativeScaleInvariantConfig,
};

use super::EdgeConfig;
use super::ShapeSearchConfig;

fn parse_polarity(s: &str) -> PyResult<NativeBlobPolarity> {
    match s {
        "dark_on_bright" => Ok(NativeBlobPolarity::DarkOnBright),
        "bright_on_dark" => Ok(NativeBlobPolarity::BrightOnDark),
        other => Err(PyValueError::new_err(format!(
            "polarity must be 'dark_on_bright' or 'bright_on_dark', got '{other}'"
        ))),
    }
}

fn polarity_str(p: NativeBlobPolarity) -> String {
    match p {
        NativeBlobPolarity::DarkOnBright => "dark_on_bright".to_string(),
        NativeBlobPolarity::BrightOnDark => "bright_on_dark".to_string(),
    }
}

/// Mirrors `vision_metrology::scale::MomentScaleConfig`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct MomentScaleConfig {
    /// `'dark_on_bright'` or `'bright_on_dark'`.
    pub polarity: String,
    pub min_area: u32,
}

impl Default for MomentScaleConfig {
    fn default() -> Self {
        let c = NativeMomentScaleConfig::default();
        Self {
            polarity: polarity_str(c.polarity),
            min_area: c.min_area,
        }
    }
}

#[pymethods]
impl MomentScaleConfig {
    #[new]
    #[pyo3(signature = (polarity=None, min_area=None))]
    pub fn new(polarity: Option<String>, min_area: Option<u32>) -> Self {
        let d = Self::default();
        Self {
            polarity: polarity.unwrap_or(d.polarity),
            min_area: min_area.unwrap_or(d.min_area),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MomentScaleConfig(polarity='{}', min_area={})",
            self.polarity, self.min_area
        )
    }
}

impl MomentScaleConfig {
    pub fn to_native(&self) -> PyResult<NativeMomentScaleConfig> {
        Ok(NativeMomentScaleConfig {
            polarity: parse_polarity(&self.polarity)?,
            min_area: self.min_area,
        })
    }
}

/// Mirrors `vision_metrology::scale::LogPolarScaleConfig`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct LogPolarScaleConfig {
    pub scale_search: (f32, f32),
    /// `None` searches scale only; `Some(margin)` also searches rotation
    /// within `+/- margin` radians.
    pub angle_margin: Option<f32>,
    pub edge: EdgeConfig,
}

impl Default for LogPolarScaleConfig {
    fn default() -> Self {
        let c = NativeLogPolarScaleConfig::default();
        Self {
            scale_search: c.scale_search,
            angle_margin: c.angle_margin,
            edge: EdgeConfig::default(),
        }
    }
}

#[pymethods]
impl LogPolarScaleConfig {
    #[new]
    #[pyo3(signature = (scale_search=None, angle_margin=None, edge=None))]
    pub fn new(
        scale_search: Option<(f32, f32)>,
        angle_margin: Option<f32>,
        edge: Option<EdgeConfig>,
    ) -> Self {
        let d = Self::default();
        Self {
            scale_search: scale_search.unwrap_or(d.scale_search),
            angle_margin: angle_margin.or(d.angle_margin),
            edge: edge.unwrap_or(d.edge),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LogPolarScaleConfig(scale_search={:?}, angle_margin={:?})",
            self.scale_search, self.angle_margin
        )
    }
}

impl LogPolarScaleConfig {
    pub fn to_native(&self) -> PyResult<NativeLogPolarScaleConfig> {
        Ok(NativeLogPolarScaleConfig {
            scale_search: self.scale_search,
            angle_margin: self.angle_margin,
            edge: self.edge.to_native()?,
        })
    }
}

/// Mirrors `vision_metrology::scale::ScaleInvariantConfig`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone, Default)]
pub struct ScaleInvariantConfig {
    pub moments: MomentScaleConfig,
    pub logpolar: LogPolarScaleConfig,
    pub search: ShapeSearchConfig,
}

#[pymethods]
impl ScaleInvariantConfig {
    #[new]
    #[pyo3(signature = (moments=None, logpolar=None, search=None))]
    pub fn new(
        moments: Option<MomentScaleConfig>,
        logpolar: Option<LogPolarScaleConfig>,
        search: Option<ShapeSearchConfig>,
    ) -> Self {
        let d = Self::default();
        Self {
            moments: moments.unwrap_or(d.moments),
            logpolar: logpolar.unwrap_or(d.logpolar),
            search: search.unwrap_or(d.search),
        }
    }

    fn __repr__(&self) -> String {
        "ScaleInvariantConfig(...)".to_string()
    }
}

impl ScaleInvariantConfig {
    pub fn to_native(&self) -> PyResult<NativeScaleInvariantConfig> {
        Ok(NativeScaleInvariantConfig {
            moments: self.moments.to_native()?,
            logpolar: self.logpolar.to_native()?,
            search: self.search.to_native()?,
        })
    }
}
