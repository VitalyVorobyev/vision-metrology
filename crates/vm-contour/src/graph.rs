use vm_core::Point2f;

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

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub id: EdgeId,
    pub a: NodeId,
    pub b: NodeId,
    pub points: Vec<Point2f>,
    pub strengths: Option<Vec<f32>>,
    pub length: f32,
    pub score: f32,
    pub is_loop: bool,
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
}
