//! Choosing the bead's edge pair on one strip, gate by gate.

use std::cmp::Ordering;

use vm_primitives::{EdgePolarity, Point2f};

use super::config::BeadPolarity;
use super::result::{BeadHit, BeadReject};
use crate::measure::{MeasureEdge, MeasurePair};

/// The gates a stage's pairs pass. Lengths are in pixels along the scan.
#[derive(Debug, Clone, Copy)]
pub(super) struct Gates {
    pub polarity: BeadPolarity,
    pub min_width: f64,
    pub max_width: f64,
    /// The stage's offset reach: the score's scale, before any curvature clip.
    pub max_offset: f64,
    pub clearance: Option<f64>,
    pub min_margin: Option<f64>,
}

/// How far a candidate pair got through the gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Reached {
    Nothing,
    Ordered,
    Width,
    Offset,
    All,
}

/// Sort `edges` by `t`, ties by polarity then amplitude, so pairing never depends on the
/// order a locate mode returned them in.
pub(super) fn sort_edges(edges: &mut [MeasureEdge]) {
    let rank = |p: EdgePolarity| match p {
        EdgePolarity::Rising => 0u8,
        EdgePolarity::Falling => 1u8,
    };
    edges.sort_unstable_by(|a, b| {
        a.t.total_cmp(&b.t)
            .then(rank(a.polarity).cmp(&rank(b.polarity)))
            .then(a.amplitude.total_cmp(&b.amplitude))
    });
}

/// A pair that passed every gate, with its sort key.
#[derive(Debug, Clone, Copy)]
struct Scored {
    i: usize,
    j: usize,
    score: f64,
    offset: f64,
}

impl Scored {
    /// `(−s, |o|, t₁, t₂)`: the higher score, then the smaller offset, then the earlier
    /// edges.
    fn cmp(&self, other: &Self, edges: &[MeasureEdge]) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.offset.abs().total_cmp(&other.offset.abs()))
            .then(edges[self.i].t.total_cmp(&edges[other.i].t))
            .then(edges[self.j].t.total_cmp(&edges[other.j].t))
    }
}

/// Whether another edge lies within `c` px outside the pair `(i, j)` of the sorted `edges`.
fn crowded(edges: &[MeasureEdge], i: usize, j: usize, c: f64) -> bool {
    let (t1, t2) = (f64::from(edges[i].t), f64::from(edges[j].t));
    let before = edges[..i]
        .iter()
        .rev()
        .map(|e| f64::from(e.t))
        .take_while(|&t| t >= t1 - c)
        .any(|t| t < t1);
    let after = edges[j + 1..]
        .iter()
        .map(|e| f64::from(e.t))
        .take_while(|&t| t <= t2 + c)
        .any(|t| t > t2);
    before || after
}

/// The bead's pair among the sorted `edges` of one strip.
///
/// `center_t` is the station's position along the strip, and `[lo, hi]` the window the
/// pair's midpoint offset from it must fall in. The gates run in a fixed order; when no
/// pair survives, the reason is the gate that removed the last one.
pub(super) fn choose(
    edges: &[MeasureEdge],
    center_t: f64,
    (lo, hi): (f64, f64),
    g: &Gates,
) -> Result<BeadHit, BeadReject> {
    let (lead, trail) = match g.polarity {
        BeadPolarity::Light => (EdgePolarity::Rising, EdgePolarity::Falling),
        BeadPolarity::Dark => (EdgePolarity::Falling, EdgePolarity::Rising),
    };
    let usable = |e: &MeasureEdge| e.t.is_finite() && e.amplitude.is_finite();
    let mut reached = Reached::Nothing;
    let mut best: Option<Scored> = None;
    // The best score among the survivors other than `best`.
    let mut runner_up: Option<f64> = None;
    for (i, a) in edges.iter().enumerate() {
        if a.polarity != lead || !usable(a) {
            continue;
        }
        for (j, b) in edges.iter().enumerate().skip(i + 1) {
            if b.polarity != trail || !usable(b) || b.t <= a.t {
                continue;
            }
            reached = reached.max(Reached::Ordered);
            let (t1, t2) = (f64::from(a.t), f64::from(b.t));
            let width = t2 - t1;
            if width < g.min_width || width > g.max_width {
                continue;
            }
            reached = reached.max(Reached::Width);
            let offset = 0.5 * (t1 + t2) - center_t;
            if offset < lo || offset > hi {
                continue;
            }
            reached = reached.max(Reached::Offset);
            if g.clearance.is_some_and(|c| crowded(edges, i, j, c)) {
                continue;
            }
            reached = Reached::All;
            let r = offset / g.max_offset;
            let score = f64::from(a.amplitude.min(b.amplitude)) / (1.0 + r * r);
            let cand = Scored {
                i,
                j,
                score,
                offset,
            };
            match best {
                Some(prev) if cand.cmp(&prev, edges) != Ordering::Less => {
                    runner_up = Some(runner_up.map_or(score, |s| s.max(score)));
                }
                _ => {
                    if let Some(prev) = best {
                        runner_up = Some(runner_up.map_or(prev.score, |s| s.max(prev.score)));
                    }
                    best = Some(cand);
                }
            }
        }
    }
    let Some(best) = best else {
        return Err(match reached {
            Reached::Nothing => BeadReject::NoPair,
            Reached::Ordered => BeadReject::Width,
            Reached::Width => BeadReject::Offset,
            Reached::Offset | Reached::All => BeadReject::Clearance,
        });
    };
    let margin = match runner_up {
        None => 1.0,
        Some(s2) if best.score > 0.0 => (1.0 - s2 / best.score).clamp(0.0, 1.0),
        Some(_) => 0.0,
    };
    if g.min_margin.is_some_and(|m| margin < m) {
        return Err(BeadReject::Ambiguous);
    }
    let (first, second) = (edges[best.i], edges[best.j]);
    let (a_lo, a_hi) = (
        first.amplitude.min(second.amplitude),
        first.amplitude.max(second.amplitude),
    );
    let balance = if a_hi > 0.0 {
        f64::from(a_lo / a_hi)
    } else {
        0.0
    };
    Ok(BeadHit {
        pair: MeasurePair {
            first,
            second,
            center: Point2f::new(
                0.5 * (first.p.x + second.p.x),
                0.5 * (first.p.y + second.p.y),
            ),
            width: second.t - first.t,
        },
        offset: best.offset as f32,
        confidence: (margin * balance) as f32,
    })
}
