//! Edges and pairs found by a caliper, and how the survivors are narrowed down.

use vm_primitives::{EdgePolarity, Point2f};

use super::config::EdgeSelect;

/// One edge found by a caliper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureEdge {
    /// Subpixel position in **image** coordinates.
    pub p: Point2f,
    /// Position along the scan, in pixels: the signed distance from the centre for
    /// rect and radial placements, the arc length from `angle_start` for an arc.
    pub t: f32,
    /// `|DoG response|` at the edge — the local contrast.
    pub amplitude: f32,
    /// Direction of the intensity transition along the scan axis.
    pub polarity: EdgePolarity,
}

/// A pair of opposite-polarity edges, i.e. one bar or gap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurePair {
    /// The earlier edge along the scan axis.
    pub first: MeasureEdge,
    /// The later edge.
    pub second: MeasureEdge,
    /// Midpoint in image coordinates.
    pub center: Point2f,
    /// Distance between the two edges along the scan axis, in pixels.
    pub width: f32,
}

/// Append the edges `select` keeps from `cands` (in profile order) to `out`.
pub(crate) fn select_edges(cands: &[MeasureEdge], select: EdgeSelect, out: &mut Vec<MeasureEdge>) {
    match select {
        EdgeSelect::All => out.extend_from_slice(cands),
        EdgeSelect::First => out.extend(cands.first().copied()),
        EdgeSelect::Last => out.extend(cands.last().copied()),
        // Equal amplitudes resolve to the later edge.
        EdgeSelect::Strongest => out.extend(
            cands
                .iter()
                .max_by(|a, b| a.amplitude.total_cmp(&b.amplitude))
                .copied(),
        ),
    }
}

/// Pair `edges` greedily in scan order: each edge with the next one of the opposite
/// polarity, both then consumed.
pub(crate) fn pair_edges(edges: &[MeasureEdge], out: &mut Vec<MeasurePair>) {
    out.clear();
    let mut i = 0;
    while i + 1 < edges.len() {
        let a = edges[i];
        let Some(off) = edges[i + 1..].iter().position(|e| e.polarity != a.polarity) else {
            break;
        };
        let b = edges[i + 1 + off];
        out.push(MeasurePair {
            first: a,
            second: b,
            center: Point2f::new(0.5 * (a.p.x + b.p.x), 0.5 * (a.p.y + b.p.y)),
            width: b.t - a.t,
        });
        i += off + 2;
    }
}
