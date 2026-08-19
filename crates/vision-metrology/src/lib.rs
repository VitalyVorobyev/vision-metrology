//! Industrial machine-vision metrology library.
//!
//! `vision-metrology` provides a complete pipeline for high-precision image
//! analysis in industrial settings. It builds on [`vm_primitives`], which it
//! re-exports in full, so this is the only dependency you need.
//!
//! ## Modules
//!
//! | Module        | Content |
//! |---------------|---------|
//! | [`contour`]   | Junction-aware contour graph extraction from 2D edgels |
//! | [`laser`]     | Laser stripe extraction using opposite-polarity edge pairs |
//! | [`matching`]  | Shape-based object detection: gradient-orientation model matching |
//! | [`segment`]   | Otsu / adaptive threshold, CCL, watershed, region growing |
//! | [`shape`]     | LSD line detection, Bookstein / Fitzgibbon conic / ellipse fitting |
//!
//! All of `vm_primitives` (image, geometry, edge, pyramid, morphology) is
//! also re-exported here for one-stop access.

pub mod contour;
pub mod laser;
pub mod matching;
pub mod segment;
pub mod shape;

// Re-export all primitives so users only need one dependency.
pub use vm_primitives::*;

// ---------------------------------------------------------------------------
// Flat domain re-exports
// ---------------------------------------------------------------------------

pub use contour::{
    Connectivity, ContourBuildConfig, ContourGraph, EdgeId, GraphEdge, MAX_KERNEL_PTS, Node,
    NodeId, NodeKind, build_graph_from_detector_output, build_graph_from_edgels, smooth_polyline,
};
pub use laser::{
    CoarseMethod, ColAccess, LaserExtractConfig, LaserExtractor, LaserLine, LaserSample, ScanAxis,
    best_pair_with_prior, coarse_center_f32, coarse_center_u8, coarse_center_u16,
};
pub use matching::{
    ContourOrientation, ModelPoint, Polarity, Refinement, ShapeMatch, ShapeMatcher, ShapeModel,
    ShapeModelBuilder, ShapeModelConfig, ShapeModelLevel, ShapeSearchConfig,
    create_shape_model_f32, create_shape_model_u8, create_shape_model_u16, match_point_scores,
};
pub use segment::{
    AdaptiveThreshConfig, CcLabel, ComponentStats, RegionGrowConfig, adaptive_threshold_u8,
    component_stats, grow_regions, label_connected_components_u8, otsu_threshold_u8, watershed,
};
pub use shape::{
    Conic2f, ConicFitConfig, ConicFitter, Ellipse2f, LineSegment2f, LsdConfig, LsdDetector,
};
