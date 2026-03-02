//! Industrial machine-vision metrology library.
//!
//! `vision-metrology` provides a complete pipeline for high-precision image
//! analysis in industrial settings. It bundles all workspace crates into a
//! single dependency and re-exports their APIs flat at crate root.
//!
//! ## Modules
//!
//! | Module        | Content |
//! |---------------|---------|
//! | [`contour`]   | Junction-aware contour graph extraction from 2D edgels |
//! | [`laser`]     | Laser stripe extraction using opposite-polarity edge pairs |
//! | [`matching`]  | Rigid / similarity edge-model matching with chamfer + ICP |
//! | [`multiscale`]| Multi-scale 2D edge detection across pyramid levels |
//! | [`segment`]   | Otsu / adaptive threshold, CCL, watershed, region growing |
//! | [`shape`]     | LSD line detection, Bookstein / Fitzgibbon conic / ellipse fitting |
//!
//! All of `vm_primitives` (image, geometry, edge, pyramid, morphology) is
//! also re-exported here for one-stop access.

pub mod contour;
pub mod laser;
pub mod matching;
pub mod multiscale;
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
    EdgeModel, MatchConfig, MatchResult, RigidEdgeMatcher, RigidMatchConfig, RigidMatchResult,
    build_scene_chamfer, chamfer_score, icp_refine, normal_score, transform_points,
};
pub use multiscale::{MultiScaleConfig, MultiScaleEdgeDetector, ScaleAnnotatedEdgel};
pub use segment::{
    AdaptiveThreshConfig, CcLabel, ComponentStats, RegionGrowConfig, adaptive_threshold_u8,
    component_stats, grow_regions, label_connected_components_u8, otsu_threshold_u8, watershed,
};
pub use shape::{
    Conic2f, ConicFitConfig, ConicFitter, Ellipse2f, LineSegment2f, LsdConfig, LsdDetector,
};
