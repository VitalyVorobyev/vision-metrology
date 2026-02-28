use vm_core::{Point2f, Vec2f};

pub type NodeId = usize;
pub type EdgeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    End,
    Junction,
    Isolated,
    LoopAnchor,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub p: Point2f,
    pub idx: (usize, usize),
    pub degree: usize,
    pub incident_edges: Vec<EdgeId>,
}

/// A polyline arc connecting two [`Node`]s in a [`ContourGraph`].
///
/// Geometry fields (`tangents`, `curvatures`, `arc_params`) are populated
/// lazily by calling [`GraphEdge::compute_geometry`] or by setting
/// [`ContourBuildConfig::record_geometry`] to `true` at build time.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub id: EdgeId,
    pub a: NodeId,
    pub b: NodeId,
    pub points: Vec<Point2f>,
    pub strengths: Option<Vec<f32>>,
    /// Euclidean arc length of the polyline in pixels.
    pub length: f32,
    /// Mean edgel strength along the arc.
    pub score: f32,
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
                Vec2f {
                    x: dx / len,
                    y: dy / len,
                }
            } else {
                Vec2f { x: 0.0, y: 0.0 }
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
                let ds = arc_params[next] - arc_params[prev];
                if ds > 0.0 { cross / ds } else { 0.0 }
            })
            .collect();

        self.tangents = Some(tangents);
        self.curvatures = Some(curvatures);
        self.arc_params = Some(arc_params);
    }
}

#[derive(Debug, Clone, Default)]
pub struct ContourGraph {
    pub width: usize,
    pub height: usize,
    pub nodes: Vec<Node>,
    pub edges: Vec<GraphEdge>,
}

impl ContourGraph {
    pub fn num_junctions(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Junction)
            .count()
    }

    pub fn num_ends(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.kind == NodeKind::End)
            .count()
    }

    pub fn iter_junctions(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(|n| n.kind == NodeKind::Junction)
    }

    pub fn iter_edges(&self) -> impl Iterator<Item = &GraphEdge> {
        self.edges.iter()
    }

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
