//! PyO3 Python extension module for the `vision-metrology` workspace.
//!
//! ## Module: `vision_metrology`
//!
//! This module exposes both object-oriented detector/matcher classes and
//! declarative free functions.

mod config_py;
mod convert;
mod detector;
mod functions;
mod match_py;
mod segment;
mod shape;
mod types;

use pyo3::prelude::*;

use config_py::{EdgeConfig, FitConfig, LsdConfig, ShapeModelConfig, ShapeSearchConfig};
use detector::EdgeDetector;
use match_py::{ShapeMatcher, ShapeModel};
use segment::Segmenter;
use shape::{Fitter, LsdDetector};
use types::{Circle, ComponentStats, Edgel, Ellipse, LineSegment, ShapeMatch};

/// `vision_metrology` — Python bindings for the vision-metrology workspace.
#[pymodule]
fn vision_metrology(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Config types
    m.add_class::<EdgeConfig>()?;
    m.add_class::<LsdConfig>()?;
    m.add_class::<FitConfig>()?;
    m.add_class::<ShapeModelConfig>()?;
    m.add_class::<ShapeSearchConfig>()?;

    // Stateful classes
    m.add_class::<EdgeDetector>()?;
    m.add_class::<LsdDetector>()?;
    m.add_class::<Fitter>()?;
    m.add_class::<ShapeModel>()?;
    m.add_class::<ShapeMatcher>()?;
    m.add_class::<Segmenter>()?;

    // Result types
    m.add_class::<Edgel>()?;
    m.add_class::<LineSegment>()?;
    m.add_class::<Circle>()?;
    m.add_class::<Ellipse>()?;
    m.add_class::<ShapeMatch>()?;
    m.add_class::<ComponentStats>()?;

    // Declarative free functions
    functions::register(m)?;
    Ok(())
}
