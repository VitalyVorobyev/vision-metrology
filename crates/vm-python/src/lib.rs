//! PyO3 Python extension module for the `vision-metrology` workspace.
//!
//! ## Module: `vision_metrology`
//!
//! This module exposes both object-oriented detector/matcher classes and
//! declarative free functions.

mod config;
mod contour_py;
mod convert;
mod detector;
mod functions;
mod match_py;
mod measure_py;
mod morph_py;
mod segment;
mod shape;
mod types;
mod warp_py;

use pyo3::prelude::*;

use config::{
    Contrast, EdgeConfig, FitConfig, LsdConfig, MeasureConfig, ShapeModelConfig, ShapeSearchConfig,
    ShapeSearchTuning,
};
use detector::EdgeDetector;
use match_py::{CropSpec, ShapeMatcher, ShapeModel};
use measure_py::{
    Caliper, MeasureRejected, MetrologyError, MetrologyModel, MetrologyObject, MetrologyResult,
    MetrologyShape,
};
use segment::Segmenter;
use shape::{Fitter, LsdDetector};
use types::{
    Circle, ComponentStats, Edgel, Ellipse, Line, LineSegment, MeasureEdge, MeasurePair, ShapeMatch,
};
use warp_py::Map;

/// `vision_metrology` — Python bindings for the vision-metrology workspace.
#[pymodule]
fn vision_metrology(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Config types
    m.add_class::<EdgeConfig>()?;
    m.add_class::<LsdConfig>()?;
    m.add_class::<FitConfig>()?;
    m.add_class::<Contrast>()?;
    m.add_class::<ShapeModelConfig>()?;
    m.add_class::<ShapeSearchTuning>()?;
    m.add_class::<ShapeSearchConfig>()?;
    m.add_class::<MeasureConfig>()?;
    m.add_class::<CropSpec>()?;

    // Stateful classes
    m.add_class::<EdgeDetector>()?;
    m.add_class::<LsdDetector>()?;
    m.add_class::<Fitter>()?;
    m.add_class::<ShapeModel>()?;
    m.add_class::<ShapeMatcher>()?;
    m.add_class::<Segmenter>()?;
    m.add_class::<Caliper>()?;
    m.add_class::<MetrologyModel>()?;
    m.add_class::<Map>()?;

    // Result types
    m.add_class::<Edgel>()?;
    m.add_class::<LineSegment>()?;
    m.add_class::<Circle>()?;
    m.add_class::<Ellipse>()?;
    m.add_class::<Line>()?;
    m.add_class::<ShapeMatch>()?;
    m.add_class::<ComponentStats>()?;
    m.add_class::<MeasureEdge>()?;
    m.add_class::<MeasurePair>()?;
    m.add_class::<MetrologyShape>()?;
    m.add_class::<MetrologyObject>()?;
    m.add_class::<MetrologyResult>()?;
    m.add_class::<MetrologyError>()?;

    // Exceptions
    m.add("MeasureRejected", m.py().get_type::<MeasureRejected>())?;

    // Declarative free functions
    functions::register(m)?;
    Ok(())
}
