//! Junction-aware contour graph extraction from 2D edgels.
//!
//! This crate builds contour topology from integer-grid adjacency while keeping
//! subpixel polyline geometry from `vm-edge` edgels:
//! - Node pixels are defined by degree `!= 2` in the chosen connectivity.
//! - Degree-2 pixels are traced as chain points between nodes.
//! - Pure loop components (all degree-2) are represented by a loop edge and a
//!   `LoopAnchor` node.
//!
//! Connectivity options:
//! - [`Connectivity::C4`]: axis-aligned neighbors only.
//! - [`Connectivity::C8`]: includes diagonals (recommended for diagonal edges).
//!
//! This is topology + polyline extraction. Smoothing and curve fitting are
//! intentionally left for later stages.

mod build;
mod graph;

pub use build::{
    Connectivity, ContourBuildConfig, build_graph_from_detector_output, build_graph_from_edgels,
};
pub use graph::{ContourGraph, EdgeId, GraphEdge, Node, NodeId, NodeKind};
