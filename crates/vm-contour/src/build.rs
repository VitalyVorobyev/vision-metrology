use vm_core::{ImageView, Point2f};
use vm_edge::edge2d::{Edge2DConfig, Edge2DDetector, Edgel};

use crate::graph::{ContourGraph, EdgeId, GraphEdge, Node, NodeId, NodeKind};

const DX: [isize; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
const DY: [isize; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
const DIRS_C4: [u8; 4] = [0, 2, 4, 6];
const DIRS_C8: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    C4,
    C8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContourBuildConfig {
    pub connectivity: Connectivity,
    pub min_component_size: usize,
    pub record_strengths: bool,
}

impl Default for ContourBuildConfig {
    fn default() -> Self {
        Self {
            connectivity: Connectivity::C8,
            min_component_size: 2,
            record_strengths: false,
        }
    }
}

pub fn build_graph_from_detector_output(
    img: &ImageView<'_, u8>,
    detector: &mut Edge2DDetector,
    edge_cfg: &Edge2DConfig,
    contour_cfg: &ContourBuildConfig,
) -> ContourGraph {
    let edgels = detector.detect_u8(img, edge_cfg);
    build_graph_from_edgels(img.width(), img.height(), &edgels, contour_cfg)
}

pub fn build_graph_from_edgels(
    width: usize,
    height: usize,
    edgels: &[Edgel],
    cfg: &ContourBuildConfig,
) -> ContourGraph {
    let n = match width.checked_mul(height) {
        Some(v) => v,
        None => {
            return ContourGraph {
                width,
                height,
                nodes: Vec::new(),
                edges: Vec::new(),
            };
        }
    };

    if width == 0 || height == 0 || edgels.is_empty() {
        return ContourGraph {
            width,
            height,
            nodes: Vec::new(),
            edges: Vec::new(),
        };
    }

    let dirs = dirs_for(cfg.connectivity);

    let mut edgel_at = vec![-1_i32; n];
    for (i, e) in edgels.iter().enumerate() {
        let Some(i32_idx) = i32::try_from(i).ok() else {
            break;
        };

        let (x, y) = e.idx;
        if x >= width || y >= height {
            continue;
        }

        let p = y * width + x;
        let prev = edgel_at[p];
        if prev < 0 {
            edgel_at[p] = i32_idx;
            continue;
        }

        let prev_idx = prev as usize;
        if e.strength > edgels[prev_idx].strength {
            edgel_at[p] = i32_idx;
        }
    }

    let mut raw_edge = vec![0_u8; n];
    for i in 0..n {
        if edgel_at[i] >= 0 {
            raw_edge[i] = 1;
        }
    }

    let active = build_active_mask(
        &raw_edge,
        width,
        height,
        cfg.connectivity,
        cfg.min_component_size,
    );

    let mut deg = vec![0_u8; n];
    for p in 0..n {
        if active[p] == 0 {
            continue;
        }

        let mut d = 0_u8;
        for &dir in dirs {
            if connected_neighbor(p, dir, &active, width, height, cfg.connectivity).is_some() {
                d = d.saturating_add(1);
            }
        }
        deg[p] = d;
    }

    let mut nodes = Vec::new();
    let mut node_at = vec![-1_i32; n];
    for p in 0..n {
        if active[p] == 0 || deg[p] == 2 {
            continue;
        }

        let eidx = edgel_at[p] as usize;
        let (x, y) = (p % width, p / width);
        let node_id = nodes.len();
        node_at[p] = node_id as i32;

        nodes.push(Node {
            id: node_id,
            kind: kind_from_degree(deg[p]),
            p: edgels[eidx].p,
            idx: (x, y),
            degree: deg[p] as usize,
            incident_edges: Vec::new(),
        });
    }

    let mut edges = Vec::new();
    let mut used_link = vec![0_u8; n];

    let mut start_node = 0_usize;
    while start_node < nodes.len() {
        let start_idx = linear_idx(nodes[start_node].idx, width);

        for &dir in dirs {
            let Some(first) =
                connected_neighbor(start_idx, dir, &active, width, height, cfg.connectivity)
            else {
                continue;
            };
            if is_link_used(&used_link, start_idx, dir) {
                continue;
            }

            let (points, strengths, end_idx, closed, score) = trace_chain(
                start_idx,
                first,
                dir,
                &active,
                &edgel_at,
                edgels,
                &node_at,
                &mut used_link,
                width,
                height,
                cfg.connectivity,
                cfg.record_strengths,
            );

            let end_node = if let Some(id) = end_idx {
                id
            } else {
                ensure_terminal_node(
                    end_pixel_from_points(&points, &edgel_at, edgels, width, height),
                    &mut node_at,
                    &mut nodes,
                    &deg,
                    &edgel_at,
                    edgels,
                    width,
                    NodeKind::End,
                )
            };

            let edge_id = edges.len();
            edges.push(GraphEdge {
                id: edge_id,
                a: start_node,
                b: end_node,
                length: arc_length(&points, closed),
                score,
                is_loop: closed && start_node == end_node,
                points,
                strengths,
            });
        }

        start_node += 1;
    }

    // Handle loop components with no degree!=2 nodes.
    for p in 0..n {
        if active[p] == 0 {
            continue;
        }

        for &dir in dirs {
            let Some(next) = connected_neighbor(p, dir, &active, width, height, cfg.connectivity)
            else {
                continue;
            };
            if is_link_used(&used_link, p, dir) {
                continue;
            }

            let anchor = ensure_terminal_node(
                p,
                &mut node_at,
                &mut nodes,
                &deg,
                &edgel_at,
                edgels,
                width,
                NodeKind::LoopAnchor,
            );

            let (points, strengths, _end_idx, closed, score) = trace_chain(
                p,
                next,
                dir,
                &active,
                &edgel_at,
                edgels,
                &node_at,
                &mut used_link,
                width,
                height,
                cfg.connectivity,
                cfg.record_strengths,
            );

            let (a, b, is_loop) = if closed {
                (anchor, anchor, true)
            } else {
                let end_node = ensure_terminal_node(
                    end_pixel_from_points(&points, &edgel_at, edgels, width, height),
                    &mut node_at,
                    &mut nodes,
                    &deg,
                    &edgel_at,
                    edgels,
                    width,
                    NodeKind::End,
                );
                (anchor, end_node, anchor == end_node)
            };

            let edge_id = edges.len();
            edges.push(GraphEdge {
                id: edge_id,
                a,
                b,
                length: arc_length(&points, is_loop),
                score,
                is_loop,
                points,
                strengths,
            });
        }
    }

    for edge in &edges {
        let a = edge.a;
        let b = edge.b;
        let id: EdgeId = edge.id;
        nodes[a].incident_edges.push(id);
        if a != b {
            nodes[b].incident_edges.push(id);
        }
    }

    ContourGraph {
        width,
        height,
        nodes,
        edges,
    }
}

#[allow(clippy::too_many_arguments)]
fn trace_chain(
    start: usize,
    first: usize,
    start_dir: u8,
    active: &[u8],
    edgel_at: &[i32],
    edgels: &[Edgel],
    node_at: &[i32],
    used_link: &mut [u8],
    width: usize,
    height: usize,
    connectivity: Connectivity,
    record_strengths: bool,
) -> (Vec<Point2f>, Option<Vec<f32>>, Option<NodeId>, bool, f32) {
    let dirs = dirs_for(connectivity);

    let mut points = Vec::new();
    let mut strengths = record_strengths.then(Vec::new);

    let start_edgel = edgel_at[start] as usize;
    points.push(edgels[start_edgel].p);

    let mut sum_strength = edgels[start_edgel].strength;
    let mut num_strength = 1_usize;
    if let Some(s) = &mut strengths {
        s.push(edgels[start_edgel].strength);
    }

    let mut prev = start;
    let mut cur = first;
    let mut dir = start_dir;
    let mut end_node = None;
    let mut is_closed = false;

    let max_steps = active.len().max(1);
    for _ in 0..max_steps {
        mark_link_both(used_link, prev, dir, cur);

        let cur_edgel = edgel_at[cur] as usize;
        points.push(edgels[cur_edgel].p);
        sum_strength += edgels[cur_edgel].strength;
        num_strength += 1;
        if let Some(s) = &mut strengths {
            s.push(edgels[cur_edgel].strength);
        }

        if cur == start {
            is_closed = true;
            end_node = node_at[cur].try_into().ok();
            break;
        }

        if node_at[cur] >= 0 {
            end_node = node_at[cur].try_into().ok();
            break;
        }

        let Some((next_dir, next)) = find_next_neighbor(
            cur,
            prev,
            active,
            used_link,
            width,
            height,
            connectivity,
            dirs,
        ) else {
            break;
        };

        if next == start {
            mark_link_both(used_link, cur, next_dir, start);
            is_closed = true;
            end_node = node_at[start].try_into().ok();
            break;
        }

        prev = cur;
        cur = next;
        dir = next_dir;
    }

    let score = if num_strength == 0 {
        0.0
    } else {
        sum_strength / num_strength as f32
    };

    (points, strengths, end_node, is_closed, score)
}

#[allow(clippy::too_many_arguments)]
fn find_next_neighbor(
    cur: usize,
    prev: usize,
    active: &[u8],
    used_link: &[u8],
    width: usize,
    height: usize,
    connectivity: Connectivity,
    dirs: &[u8],
) -> Option<(u8, usize)> {
    let mut fallback = None;
    for &dir in dirs {
        let Some(nb) = connected_neighbor(cur, dir, active, width, height, connectivity) else {
            continue;
        };
        if nb == prev {
            continue;
        }

        if !is_link_used(used_link, cur, dir) {
            return Some((dir, nb));
        }

        if fallback.is_none() {
            fallback = Some((dir, nb));
        }
    }

    fallback
}

fn build_active_mask(
    raw_edge: &[u8],
    width: usize,
    height: usize,
    connectivity: Connectivity,
    min_component_size: usize,
) -> Vec<u8> {
    let n = raw_edge.len();
    let mut active = vec![0_u8; n];

    let min_size = min_component_size.max(1);
    if min_size <= 1 {
        for i in 0..n {
            if raw_edge[i] != 0 {
                active[i] = 1;
            }
        }
        return active;
    }

    let dirs = dirs_for(connectivity);
    let mut seen = vec![0_u8; n];
    let mut stack = Vec::new();
    let mut component = Vec::new();

    for i in 0..n {
        if raw_edge[i] == 0 || seen[i] != 0 {
            continue;
        }

        stack.clear();
        component.clear();
        seen[i] = 1;
        stack.push(i);

        while let Some(p) = stack.pop() {
            component.push(p);
            for &dir in dirs {
                let Some(nb) = connected_neighbor(p, dir, raw_edge, width, height, connectivity)
                else {
                    continue;
                };
                if seen[nb] == 0 {
                    seen[nb] = 1;
                    stack.push(nb);
                }
            }
        }

        if component.len() >= min_size {
            for &p in &component {
                active[p] = 1;
            }
        }
    }

    active
}

#[allow(clippy::too_many_arguments)]
fn ensure_terminal_node(
    pixel: usize,
    node_at: &mut [i32],
    nodes: &mut Vec<Node>,
    deg: &[u8],
    edgel_at: &[i32],
    edgels: &[Edgel],
    width: usize,
    fallback_kind: NodeKind,
) -> NodeId {
    if node_at[pixel] >= 0 {
        return node_at[pixel] as usize;
    }

    let eidx = edgel_at[pixel] as usize;
    let (x, y) = (pixel % width, pixel / width);

    let degree = deg[pixel] as usize;
    let kind = match degree {
        0 => NodeKind::Isolated,
        1 => NodeKind::End,
        2 => fallback_kind,
        _ => NodeKind::Junction,
    };

    let node_id = nodes.len();
    node_at[pixel] = node_id as i32;

    nodes.push(Node {
        id: node_id,
        kind,
        p: edgels[eidx].p,
        idx: (x, y),
        degree,
        incident_edges: Vec::new(),
    });

    node_id
}

fn end_pixel_from_points(
    points: &[Point2f],
    edgel_at: &[i32],
    edgels: &[Edgel],
    width: usize,
    height: usize,
) -> usize {
    // Fallback for malformed chains: map last point back to integer idx if exact,
    // otherwise scan for the nearest available edgel pixel.
    let last = points[points.len().saturating_sub(1)];
    let xr = last.x.round();
    let yr = last.y.round();
    if xr >= 0.0 && yr >= 0.0 {
        let x = xr as usize;
        let y = yr as usize;
        if x < width && y < height {
            let idx = y * width + x;
            if edgel_at[idx] >= 0 {
                return idx;
            }
        }
    }

    let mut best = 0usize;
    let mut best_d2 = f32::INFINITY;
    for (i, e) in edgels.iter().enumerate() {
        let (x, y) = e.idx;
        if x >= width || y >= height {
            continue;
        }
        let dx = e.p.x - last.x;
        let dy = e.p.y - last.y;
        let d2 = dx * dx + dy * dy;
        if d2 < best_d2 {
            best_d2 = d2;
            best = i;
        }
    }

    let (bx, by) = edgels[best].idx;
    by * width + bx
}

fn kind_from_degree(d: u8) -> NodeKind {
    match d {
        0 => NodeKind::Isolated,
        1 => NodeKind::End,
        _ => NodeKind::Junction,
    }
}

fn arc_length(points: &[Point2f], closed: bool) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }

    let mut len = 0.0_f32;
    for i in 1..points.len() {
        let dx = points[i].x - points[i - 1].x;
        let dy = points[i].y - points[i - 1].y;
        len += (dx * dx + dy * dy).sqrt();
    }

    if closed {
        let first = points[0];
        let last = points[points.len() - 1];
        let dx = first.x - last.x;
        let dy = first.y - last.y;
        len += (dx * dx + dy * dy).sqrt();
    }

    len
}

#[inline]
fn linear_idx(idx: (usize, usize), width: usize) -> usize {
    idx.1 * width + idx.0
}

#[inline]
fn dirs_for(connectivity: Connectivity) -> &'static [u8] {
    match connectivity {
        Connectivity::C4 => &DIRS_C4,
        Connectivity::C8 => &DIRS_C8,
    }
}

#[inline]
fn opposite_dir(dir: u8) -> u8 {
    (dir + 4) & 7
}

#[inline]
fn is_link_used(used_link: &[u8], p: usize, dir: u8) -> bool {
    let bit = 1_u8 << dir;
    (used_link[p] & bit) != 0
}

#[inline]
fn mark_link_both(used_link: &mut [u8], a: usize, dir_ab: u8, b: usize) {
    used_link[a] |= 1_u8 << dir_ab;
    used_link[b] |= 1_u8 << opposite_dir(dir_ab);
}

#[inline]
fn neighbor_index(p: usize, dir: u8, width: usize, height: usize) -> Option<usize> {
    if width == 0 || height == 0 {
        return None;
    }

    let x = p % width;
    let y = p / width;
    let nx = x as isize + DX[dir as usize];
    let ny = y as isize + DY[dir as usize];
    if nx < 0 || ny < 0 {
        return None;
    }

    let (nxu, nyu) = (nx as usize, ny as usize);
    if nxu >= width || nyu >= height {
        return None;
    }

    Some(nyu * width + nxu)
}

#[inline]
fn connected_neighbor(
    p: usize,
    dir: u8,
    occupancy: &[u8],
    width: usize,
    height: usize,
    connectivity: Connectivity,
) -> Option<usize> {
    let nb = neighbor_index(p, dir, width, height)?;
    if occupancy[nb] == 0 {
        return None;
    }

    if connectivity == Connectivity::C8 && is_diagonal_dir(dir) {
        let x = p % width;
        let y = p / width;
        let dx = DX[dir as usize];
        let dy = DY[dir as usize];

        let side_a = index_if_in_bounds(x as isize + dx, y as isize, width, height);
        let side_b = index_if_in_bounds(x as isize, y as isize + dy, width, height);
        if side_a.is_some_and(|i| occupancy[i] != 0) || side_b.is_some_and(|i| occupancy[i] != 0) {
            return None;
        }
    }

    Some(nb)
}

#[inline]
fn is_diagonal_dir(dir: u8) -> bool {
    DX[dir as usize] != 0 && DY[dir as usize] != 0
}

#[inline]
fn index_if_in_bounds(x: isize, y: isize, width: usize, height: usize) -> Option<usize> {
    if x < 0 || y < 0 {
        return None;
    }

    let (xu, yu) = (x as usize, y as usize);
    if xu >= width || yu >= height {
        return None;
    }

    Some(yu * width + xu)
}

#[cfg(test)]
mod tests {
    use vm_core::{Image, Point2f, Vec2f};
    use vm_edge::edge2d::{Edge2DConfig, Edge2DDetector, Edgel};

    use crate::{
        Connectivity, ContourBuildConfig, NodeKind, build_graph_from_detector_output,
        build_graph_from_edgels,
    };

    fn e(x: usize, y: usize) -> Edgel {
        Edgel {
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
    fn t_junction_graph() {
        let mut edgels = Vec::new();
        for y in 1..=7 {
            edgels.push(e(4, y));
        }
        for x in 5..=7 {
            edgels.push(e(x, 4));
        }

        let cfg = ContourBuildConfig {
            connectivity: Connectivity::C8,
            min_component_size: 2,
            record_strengths: false,
        };
        let g = build_graph_from_edgels(9, 9, &edgels, &cfg);

        assert_eq!(g.num_junctions(), 1);
        assert_eq!(g.num_ends(), 3);
        assert_eq!(g.edges.len(), 3);

        let j = g
            .iter_junctions()
            .next()
            .expect("junction should be present");
        assert_eq!(j.idx, (4, 4));
        assert_eq!(j.degree, 3);

        for edge in g.iter_edges() {
            assert!(edge.points.len() >= 2);
        }
    }

    #[test]
    fn y_junction_graph() {
        let mut edgels = Vec::new();
        for y in 1..=4 {
            edgels.push(e(4, y));
        }
        edgels.push(e(3, 5));
        edgels.push(e(2, 6));
        edgels.push(e(5, 5));
        edgels.push(e(6, 6));

        let cfg = ContourBuildConfig {
            connectivity: Connectivity::C8,
            min_component_size: 2,
            record_strengths: false,
        };
        let g = build_graph_from_edgels(9, 9, &edgels, &cfg);

        assert_eq!(g.num_junctions(), 1);
        assert_eq!(g.num_ends(), 3);
        assert_eq!(g.edges.len(), 3);
        for edge in g.iter_edges() {
            assert!(edge.points.len() >= 2);
        }
    }

    #[test]
    fn loop_component_creates_loop_edge() {
        let mut edgels = Vec::new();
        for x in 2..=5 {
            edgels.push(e(x, 2));
            edgels.push(e(x, 5));
        }
        for y in 3..=4 {
            edgels.push(e(2, y));
            edgels.push(e(5, y));
        }

        let cfg = ContourBuildConfig {
            connectivity: Connectivity::C4,
            min_component_size: 2,
            record_strengths: false,
        };
        let g = build_graph_from_edgels(9, 9, &edgels, &cfg);

        assert_eq!(g.num_junctions(), 0);
        assert_eq!(g.num_ends(), 0);
        assert_eq!(g.edges.len(), 1);
        assert!(g.edges[0].is_loop);
        assert_eq!(g.edges[0].a, g.edges[0].b);

        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].kind, NodeKind::LoopAnchor);
    }

    #[test]
    fn detector_output_builder_smoke() {
        let w = 64;
        let h = 48;
        let mut data = vec![0_u8; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = if x >= 32 { 255 } else { 0 };
            }
        }

        let img = Image::from_vec(w, h, data).expect("valid image");
        let mut detector = Edge2DDetector::new();
        let edge_cfg = Edge2DConfig::default();
        let contour_cfg = ContourBuildConfig::default();

        let g = build_graph_from_detector_output(
            &img.as_view(),
            &mut detector,
            &edge_cfg,
            &contour_cfg,
        );

        assert!(!g.edges.is_empty());
    }
}
