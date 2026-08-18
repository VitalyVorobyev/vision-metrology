//! Junction-aware contour graph extraction from 2D edgels.
//!
//! This module builds contour topology from integer-grid adjacency while keeping
//! subpixel polyline geometry from `vm_primitives::edge` edgels:
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
mod smooth;

pub use build::{
    Connectivity, ContourBuildConfig, build_graph_from_detector_output, build_graph_from_edgels,
};
pub use graph::{ContourGraph, EdgeId, GraphEdge, Node, NodeId, NodeKind};
pub use smooth::{MAX_KERNEL_PTS, smooth_polyline};

#[cfg(test)]
mod geom_tests {
    use vm_primitives::{Point2f, Vec2f};

    use crate::{Connectivity, ContourBuildConfig, build_graph_from_edgels};

    fn e(x: usize, y: usize) -> vm_primitives::edge::edge2d::Edgel {
        vm_primitives::edge::edge2d::Edgel {
            p: Point2f {
                x: x as f32,
                y: y as f32,
            },
            n: Vec2f { x: 1.0, y: 0.0 },
            strength: 1.0,
            idx: (x, y),
        }
    }

    #[test]
    fn compute_geometry_horizontal_chain() {
        // Horizontal chain: (0,0), (1,0), (2,0), (3,0), (4,0)
        // Expected tangent at every point: (1, 0).
        // Expected curvature: 0 everywhere (straight line).
        let edgels: Vec<_> = (0..5).map(|x| e(x, 2)).collect();
        let cfg = ContourBuildConfig {
            connectivity: Connectivity::C4,
            min_component_size: 2,
            record_strengths: false,
            record_geometry: true,
            ..Default::default()
        };
        let g = build_graph_from_edgels(10, 5, &edgels, &cfg);
        assert_eq!(g.edges.len(), 1);

        let edge = &g.edges[0];
        let tangents = edge.tangents.as_ref().expect("geometry was requested");
        let curvatures = edge.curvatures.as_ref().expect("geometry was requested");
        let arc_params = edge.arc_params.as_ref().expect("geometry was requested");

        // Arc params: 0 at first point, 1 at last.
        assert!((arc_params[0]).abs() < 1e-6);
        assert!((arc_params[arc_params.len() - 1] - 1.0).abs() < 1e-6);

        // All tangents should point along +x.
        for t in tangents {
            assert!(
                (t.x - 1.0).abs() < 1e-5,
                "tangent.x expected ~1, got {}",
                t.x
            );
            assert!((t.y).abs() < 1e-5, "tangent.y expected ~0, got {}", t.y);
        }

        // Curvature of a straight line is 0.
        for &k in curvatures {
            assert!(k.abs() < 1e-4, "curvature expected ~0, got {k}");
        }
    }

    #[test]
    fn compute_geometry_no_record() {
        // record_geometry = false: all optional fields stay None.
        let edgels: Vec<_> = (0..4).map(|x| e(x, 0)).collect();
        let cfg = ContourBuildConfig::default(); // record_geometry: false
        let g = build_graph_from_edgels(6, 3, &edgels, &cfg);
        for edge in &g.edges {
            assert!(edge.tangents.is_none());
            assert!(edge.curvatures.is_none());
            assert!(edge.arc_params.is_none());
        }
    }

    #[test]
    fn filter_edges_min_length() {
        // Long chain (7 px) + short chain (2 px) in separate components.
        let mut edgels: Vec<_> = (0..7).map(|x| e(x, 0)).collect();
        edgels.extend((0..2).map(|x| e(x, 4)));
        let cfg = ContourBuildConfig {
            connectivity: Connectivity::C4,
            min_component_size: 2,
            record_geometry: false,
            record_strengths: false,
            ..Default::default()
        };
        let g = build_graph_from_edgels(10, 6, &edgels, &cfg);
        let long_edges: Vec<_> = g.filter_edges_min_length(5.0).collect();
        assert_eq!(long_edges.len(), 1);
        assert!(long_edges[0].length >= 5.0);
    }

    #[test]
    fn iter_edges_by_length_order() {
        // Build a graph with two separate horizontal chains of different lengths.
        let mut edgels: Vec<_> = (0..6).map(|x| e(x, 0)).collect();
        edgels.extend((0..3).map(|x| e(x, 4)));
        let cfg = ContourBuildConfig {
            connectivity: Connectivity::C4,
            min_component_size: 2,
            record_geometry: false,
            record_strengths: false,
            ..Default::default()
        };
        let g = build_graph_from_edgels(10, 6, &edgels, &cfg);
        let sorted: Vec<_> = g.iter_edges_by_length().collect();
        for pair in sorted.windows(2) {
            assert!(pair[0].length >= pair[1].length);
        }
    }
}
