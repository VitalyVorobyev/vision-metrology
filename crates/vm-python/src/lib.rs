//! PyO3 Python extension module for the `vision-metrology` workspace.
//!
//! ## Module: `vision_metrology`
//!
//! This module exposes both object-oriented detector/matcher classes and
//! declarative free functions.

mod config;
mod contour_py;
mod convert;
mod corr_py;
mod detector;
mod functions;
mod match_py;
mod measure_py;
mod metric_py;
mod morph_py;
mod segment;
mod shape;
mod types;
mod warp_py;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use config::{
    Contrast, CorrConfig, CorrSearchTuning, CorrTemplateConfig, CorrTemplateTuning,
    DisplacementConfig, EdgeConfig, FitConfig, LsdConfig, MeasureConfig, Refine, ShapeModelConfig,
    ShapeSearchConfig, ShapeSearchTuning,
};
use corr_py::{CorrMatch, CorrTemplate, Displacement, displacement, find, find_topk};
use detector::EdgeDetector;
use match_py::{CropSpec, ShapeMatcher, ShapeModel};
use measure_py::{
    Caliper, MeasureRejected, MetrologyError, MetrologyModel, MetrologyObject, MetrologyResult,
    MetrologyShape,
};
use metric_py::{
    BrownConrady5, CameraModel, PinholeIntrinsics, Plane3, PlaneGrid, load_rig_extrinsics,
    load_table_calibration, pixel_to_plane, plane_grid_map, project_plane_points, undistort_map,
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
    m.add_class::<CorrTemplateTuning>()?;
    m.add_class::<CorrTemplateConfig>()?;
    m.add_class::<CorrSearchTuning>()?;
    m.add_class::<CorrConfig>()?;
    m.add_class::<Refine>()?;
    m.add_class::<DisplacementConfig>()?;

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
    m.add_class::<CorrTemplate>()?;

    // metric: the calibration bridge
    m.add_class::<PinholeIntrinsics>()?;
    m.add_class::<BrownConrady5>()?;
    m.add_class::<CameraModel>()?;
    m.add_class::<Plane3>()?;
    m.add_class::<PlaneGrid>()?;

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
    m.add_class::<CorrMatch>()?;
    m.add_class::<Displacement>()?;

    // Exceptions
    m.add("MeasureRejected", m.py().get_type::<MeasureRejected>())?;

    // metric: free functions
    m.add_function(wrap_pyfunction!(pixel_to_plane, m)?)?;
    m.add_function(wrap_pyfunction!(plane_grid_map, m)?)?;
    m.add_function(wrap_pyfunction!(undistort_map, m)?)?;
    m.add_function(wrap_pyfunction!(project_plane_points, m)?)?;
    m.add_function(wrap_pyfunction!(load_rig_extrinsics, m)?)?;
    m.add_function(wrap_pyfunction!(load_table_calibration, m)?)?;

    // corr: cross-correlation matching + displacement
    m.add_function(wrap_pyfunction!(find, m)?)?;
    m.add_function(wrap_pyfunction!(find_topk, m)?)?;
    m.add_function(wrap_pyfunction!(displacement, m)?)?;

    // Declarative free functions
    functions::register(m)?;
    Ok(())
}
