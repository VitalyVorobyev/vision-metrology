use vm_primitives::{Point2f, Vec2f};

/// Index of a node in [`ContourGraph::nodes`].
pub type NodeId = usize;
/// Index of an edge in [`ContourGraph::edges`].
pub type EdgeId = usize;

/// Topological role of a contour graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// Degree-1 node: free endpoint of an open contour chain.
    End,
    /// Degree ≥ 3 node: branching point where multiple chains meet (T/Y/X junction).
    Junction,
    /// Degree-0 node: isolated pixel not connected to any chain.
    Isolated,
    /// Anchor node for a loop component (closed curve with no end-points).
    LoopAnchor,
}

/// A topological node (endpoint or junction) in a [`ContourGraph`].
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier; equals the index of this node in [`ContourGraph::nodes`].
    pub id: NodeId,
    /// Topological role.
    pub kind: NodeKind,
    /// Subpixel position (from the originating edgel).
    pub p: Point2f,
    /// Integer pixel grid cell containing this node.
    pub idx: (usize, usize),
    /// Number of incident arcs. Equals `incident_edges.len()`.
    pub degree: usize,
    /// Indices of all arcs incident to this node in [`ContourGraph::edges`].
    pub incident_edges: Vec<EdgeId>,
}

/// A polyline arc connecting two [`Node`]s in a [`ContourGraph`].
///
/// Geometry fields (`tangents`, `curvatures`, `arc_params`) are populated
/// lazily by calling [`GraphEdge::compute_geometry`] or by setting
/// `ContourBuildConfig::record_geometry` to `true` at build time.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Unique identifier; equals the index of this edge in [`ContourGraph::edges`].
    pub id: EdgeId,
    /// [`NodeId`] of the first endpoint.
    pub a: NodeId,
    /// [`NodeId`] of the second endpoint (equals `a` for loop edges).
    pub b: NodeId,
    /// Ordered subpixel points along the arc from `a` to `b`.
    pub points: Vec<Point2f>,
    /// Edgel gradient magnitudes at each point, if recorded.
    pub strengths: Option<Vec<f32>>,
    /// Euclidean arc length of the polyline in pixels.
    pub length: f32,
    /// Mean edgel strength along the arc.
    pub score: f32,
    /// `true` when this edge forms a closed loop (its two endpoints are the same node).
    pub is_loop: bool,
    /// Unit tangent direction at each point, computed by [`GraphEdge::compute_geometry`].
    /// `tangents[i]` is the symmetric finite-difference tangent at `points[i]`.
    pub tangents: Option<Vec<Vec2f>>,
    /// Signed curvature κ = (t′ × t) / |t|³ at each interior point (radians / pixel).
    /// Endpoints use one-sided differences.
    pub curvatures: Option<Vec<f32>>,
    /// Cumulative arc-length parameter in `[0, 1]` at each point.
    /// `arc_params[0] == 0.0`, `arc_params[last] == 1.0`.
    pub arc_params: Option<Vec<f32>>,
}

impl GraphEdge {
    /// Compute and store tangents, curvatures, and arc-length parameters.
    ///
    /// Safe to call multiple times; re-computes on every call.
    /// Has no effect if `points.len() < 2`.
    pub fn compute_geometry(&mut self) {
        let n = self.points.len();
        if n < 2 {
            self.tangents = None;
            self.curvatures = None;
            self.arc_params = None;
            return;
        }

        // --- cumulative arc lengths ---
        let mut cum = vec![0.0_f32; n];
        for i in 1..n {
            let dx = self.points[i].x - self.points[i - 1].x;
            let dy = self.points[i].y - self.points[i - 1].y;
            cum[i] = cum[i - 1] + (dx * dx + dy * dy).sqrt();
        }
        let total = cum[n - 1];
        let arc_params: Vec<f32> = if total > 0.0 {
            cum.iter().map(|&s| s / total).collect()
        } else {
            vec![0.0; n]
        };

        // --- tangents via symmetric finite differences ---
        let mut tangents = Vec::with_capacity(n);
        for i in 0..n {
            let prev = if i == 0 { 0 } else { i - 1 };
            let next = if i + 1 < n { i + 1 } else { n - 1 };
            let dx = self.points[next].x - self.points[prev].x;
            let dy = self.points[next].y - self.points[prev].y;
            let len = (dx * dx + dy * dy).sqrt();
            tangents.push(if len > 0.0 {
                Vec2f::new(dx / len, dy / len)
            } else {
                Vec2f::new(0.0, 0.0)
            });
        }

        // --- signed curvature: κ = (t × dt/ds) where × is the 2-D cross product ---
        // Use discrete approximation: κ[i] ≈ cross(tangent[i-1], tangent[i+1]) / (2 * ds)
        // At endpoints fall back to one-sided difference.
        let curvatures: Vec<f32> = (0..n)
            .map(|i| {
                let prev = if i == 0 { 0 } else { i - 1 };
                let next = if i + 1 < n { i + 1 } else { n - 1 };
                if prev == next {
                    return 0.0;
                }
                let t0 = tangents[prev];
                let t1 = tangents[next];
                // 2-D cross product t0 × t1 = t0.x*t1.y - t0.y*t1.x
                let cross = t0.x * t1.y - t0.y * t1.x;
                // `ds` must be a distance in pixels. Using `arc_params` here
                // instead would divide by a fraction of the total length, which
                // makes the result depend on how long the contour happens to be
                // -- the same circle sampled as a longer arc would report a
                // different curvature, and the documented radians/pixel unit
                // would not hold.
                let ds = cum[next] - cum[prev];
                if ds > 0.0 { cross / ds } else { 0.0 }
            })
            .collect();

        self.tangents = Some(tangents);
        self.curvatures = Some(curvatures);
        self.arc_params = Some(arc_params);
    }
}

/// Topological graph of contour chains extracted from 2-D edgels.
///
/// Nodes represent endpoints and junctions; edges are polyline arcs
/// connecting them. Build via [`build_graph_from_edgels`].
///
/// [`build_graph_from_edgels`]: crate::contour::build_graph_from_edgels
#[derive(Debug, Clone, Default)]
pub struct ContourGraph {
    /// Width of the source image (used as grid map extent).
    pub width: usize,
    /// Height of the source image.
    pub height: usize,
    /// All nodes (endpoints, junctions, isolated points, loop anchors).
    pub nodes: Vec<Node>,
    /// All arc edges.
    pub edges: Vec<GraphEdge>,
}

impl ContourGraph {
    /// Number of junction nodes (degree ≥ 3).
    pub fn num_junctions(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Junction)
            .count()
    }

    /// Number of end nodes (degree 1).
    pub fn num_ends(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.kind == NodeKind::End)
            .count()
    }

    /// Iterate over all junction nodes.
    pub fn iter_junctions(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(|n| n.kind == NodeKind::Junction)
    }

    /// Iterate over all arc edges.
    pub fn iter_edges(&self) -> impl Iterator<Item = &GraphEdge> {
        self.edges.iter()
    }

    /// Return the polyline points for the arc with the given `id`.
    ///
    /// # Panics
    /// Panics if `id >= self.edges.len()`.
    pub fn edge_polyline(&self, id: EdgeId) -> &[Point2f] {
        &self.edges[id].points
    }

    /// Iterate edges sorted by arc length, longest first.
    pub fn iter_edges_by_length(&self) -> impl Iterator<Item = &GraphEdge> {
        let mut indices: Vec<usize> = (0..self.edges.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            self.edges[b]
                .length
                .partial_cmp(&self.edges[a].length)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        // Return owned iterator over cloned indices.
        // SAFETY: indices are all valid edge ids.
        indices.into_iter().map(|i| &self.edges[i])
    }

    /// Return edges whose arc length is ≥ `min_length` pixels.
    pub fn filter_edges_min_length(&self, min_length: f32) -> impl Iterator<Item = &GraphEdge> {
        self.edges.iter().filter(move |e| e.length >= min_length)
    }

    /// Compute geometry (tangents, curvatures, arc params) for all edges in place.
    pub fn compute_all_geometry(&mut self) {
        for edge in &mut self.edges {
            edge.compute_geometry();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: NodeId, kind: NodeKind) -> Node {
        Node {
            id,
            kind,
            p: Point2f::new(id as f32, 0.0),
            idx: (id, 0),
            degree: match kind {
                NodeKind::End => 1,
                NodeKind::Junction => 3,
                NodeKind::Isolated => 0,
                NodeKind::LoopAnchor => 2,
            },
            incident_edges: Vec::new(),
        }
    }

    fn edge(id: EdgeId, points: Vec<Point2f>) -> GraphEdge {
        let length = points
            .windows(2)
            .map(|w| ((w[1].x - w[0].x).powi(2) + (w[1].y - w[0].y).powi(2)).sqrt())
            .sum();
        GraphEdge {
            id,
            a: 0,
            b: 1,
            points,
            strengths: None,
            length,
            score: 0.0,
            is_loop: false,
            tangents: None,
            curvatures: None,
            arc_params: None,
        }
    }

    fn line(id: EdgeId, n: usize) -> GraphEdge {
        edge(id, (0..n).map(|i| Point2f::new(i as f32, 0.0)).collect())
    }

    /// Samples a circle of radius `r` at `n` evenly spaced angles.
    fn arc(id: EdgeId, r: f32, n: usize) -> GraphEdge {
        edge(
            id,
            (0..n)
                .map(|i| {
                    let t = i as f32 / n as f32 * core::f32::consts::TAU;
                    Point2f::new(r * t.cos(), r * t.sin())
                })
                .collect(),
        )
    }

    fn graph(nodes: Vec<Node>, edges: Vec<GraphEdge>) -> ContourGraph {
        ContourGraph {
            width: 64,
            height: 64,
            nodes,
            edges,
        }
    }

    #[test]
    fn node_counts_select_by_kind() {
        let g = graph(
            vec![
                node(0, NodeKind::End),
                node(1, NodeKind::Junction),
                node(2, NodeKind::Junction),
                node(3, NodeKind::Isolated),
                node(4, NodeKind::LoopAnchor),
                node(5, NodeKind::End),
            ],
            vec![],
        );
        assert_eq!(g.num_junctions(), 2);
        assert_eq!(g.num_ends(), 2);
        assert_eq!(g.iter_junctions().count(), 2);
        assert!(
            g.iter_junctions().all(|n| n.kind == NodeKind::Junction),
            "iter_junctions must not leak other kinds"
        );
        // Isolated and LoopAnchor are counted by neither accessor.
        assert_eq!(g.nodes.len(), 6);
    }

    #[test]
    fn edges_iterate_longest_first() {
        // Lengths 2, 5, 9 built out of order, so a stable pass-through would fail.
        let g = graph(vec![], vec![line(0, 3), line(1, 10), line(2, 6)]);
        let by_len: Vec<EdgeId> = g.iter_edges_by_length().map(|e| e.id).collect();
        assert_eq!(by_len, vec![1, 2, 0]);

        let lengths: Vec<f32> = g.iter_edges_by_length().map(|e| e.length).collect();
        assert!(
            lengths.windows(2).all(|w| w[0] >= w[1]),
            "lengths must be non-increasing, got {lengths:?}"
        );
        assert_eq!(g.iter_edges().count(), 3);
    }

    #[test]
    fn equal_lengths_do_not_lose_edges() {
        // `partial_cmp` ties must still yield every edge exactly once.
        let g = graph(vec![], vec![line(0, 5), line(1, 5), line(2, 5)]);
        let mut ids: Vec<EdgeId> = g.iter_edges_by_length().map(|e| e.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn min_length_filter_is_inclusive() {
        let g = graph(vec![], vec![line(0, 3), line(1, 5), line(2, 9)]);
        // Lengths are 2, 4 and 8.
        let kept: Vec<EdgeId> = g.filter_edges_min_length(4.0).map(|e| e.id).collect();
        assert_eq!(kept, vec![1, 2], "the threshold itself must be kept");
        assert_eq!(g.filter_edges_min_length(0.0).count(), 3);
        assert_eq!(g.filter_edges_min_length(100.0).count(), 0);
    }

    #[test]
    fn edge_polyline_returns_that_edges_points() {
        let g = graph(vec![], vec![line(0, 3), line(1, 5)]);
        assert_eq!(g.edge_polyline(0).len(), 3);
        assert_eq!(g.edge_polyline(1).len(), 5);
        assert_eq!(g.edge_polyline(1)[4].x, 4.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn edge_polyline_panics_on_unknown_id() {
        // Documented behaviour: the id is an index, not a lookup key.
        let g = graph(vec![], vec![line(0, 3)]);
        let _ = g.edge_polyline(7);
    }

    #[test]
    fn geometry_of_a_straight_line() {
        let mut e = line(0, 6);
        e.compute_geometry();

        let tangents = e.tangents.as_ref().expect("tangents computed");
        let curvatures = e.curvatures.as_ref().expect("curvatures computed");
        let arc = e.arc_params.as_ref().expect("arc params computed");
        assert_eq!(tangents.len(), 6);

        for (i, t) in tangents.iter().enumerate() {
            assert!(
                (t.x - 1.0).abs() < 1e-5 && t.y.abs() < 1e-5,
                "tangent {i} of a +x line must be (1, 0), got ({}, {})",
                t.x,
                t.y
            );
        }
        for (i, k) in curvatures.iter().enumerate() {
            assert!(k.abs() < 1e-4, "curvature {i} of a line must be 0, got {k}");
        }
        // Arc parameter spans [0, 1] and, for evenly spaced points, is uniform.
        assert!((arc[0] - 0.0).abs() < 1e-6);
        assert!((arc[5] - 1.0).abs() < 1e-6);
        assert!(
            arc.windows(2).all(|w| w[1] > w[0]),
            "arc parameter must increase strictly"
        );
        assert!((arc[1] - 0.2).abs() < 1e-5, "expected uniform spacing");
    }

    #[test]
    fn geometry_of_a_circle_has_constant_curvature() {
        // A circle of radius r has curvature 1/r everywhere.
        let r = 20.0;
        let mut e = arc(0, r, 64);
        e.compute_geometry();
        let curvatures = e.curvatures.as_ref().expect("curvatures computed");

        // Skip two points at each end. The endpoint tangents are one-sided, so
        // they are wrong at index 0 and n-1, and that feeds the symmetric
        // difference at index 1 and n-2 as well.
        //
        // The remaining points agree with 1/r to 6e-4, which is the exact
        // discrete-chord error for this sampling: sin(2*tau/n) / (2 * chord)
        // with chord = 2*r*sin(pi/n) gives 0.049699 against 1/r = 0.05.
        for (i, k) in curvatures.iter().enumerate().take(62).skip(2) {
            assert!(
                (k.abs() - 1.0 / r).abs() < 1e-3,
                "curvature {i} should be ~{:.4}, got {k:.4}",
                1.0 / r
            );
        }

        // Curvature must be a property of the shape, not of how long the
        // contour is. Half the circle, same radius, same curvature. This is
        // what fails if `ds` is taken from the normalised `arc_params`.
        let mut half = arc(1, r, 64);
        half.points.truncate(32);
        half.compute_geometry();
        let half_k = half.curvatures.as_ref().expect("curvatures");
        for (i, k) in half_k.iter().enumerate().take(30).skip(2) {
            assert!(
                (k.abs() - 1.0 / r).abs() < 1e-3,
                "curvature {i} of the half arc should still be ~{:.4}, got {k:.4}",
                1.0 / r
            );
        }
        // Tangents stay unit length.
        for t in e.tangents.as_ref().expect("tangents").iter() {
            let n = (t.x * t.x + t.y * t.y).sqrt();
            assert!((n - 1.0).abs() < 1e-4, "tangent must be unit, got {n}");
        }
    }

    #[test]
    fn geometry_is_a_no_op_below_two_points() {
        for n in [0usize, 1] {
            let mut e = line(0, n);
            e.compute_geometry();
            assert!(e.tangents.is_none(), "n={n} must leave tangents unset");
            assert!(e.curvatures.is_none());
            assert!(e.arc_params.is_none());
        }
    }

    #[test]
    fn geometry_is_idempotent() {
        let mut e = arc(0, 10.0, 32);
        e.compute_geometry();
        let first = e.curvatures.clone().expect("curvatures");
        e.compute_geometry();
        let second = e.curvatures.clone().expect("curvatures");
        assert_eq!(first, second, "recomputing must be stable");
    }

    #[test]
    fn compute_all_geometry_covers_every_edge() {
        let mut g = graph(vec![], vec![line(0, 4), arc(1, 8.0, 16), line(2, 1)]);
        g.compute_all_geometry();
        assert!(g.edges[0].tangents.is_some());
        assert!(g.edges[1].tangents.is_some());
        // Single-point edge stays untouched rather than panicking.
        assert!(g.edges[2].tangents.is_none());
    }
}
