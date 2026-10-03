//! Edges and pairs found by a caliper, and how the survivors are narrowed down.

use std::cmp::Ordering;

use vm_primitives::{EdgePolarity, Point2f};

use super::config::{EdgeSelect, RejectReason};

/// One edge found by a caliper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureEdge {
    /// Subpixel position in **image** coordinates.
    pub p: Point2f,
    /// Position along the scan, in pixels: the distance from `start` for a strip,
    /// the signed distance from the centre for rect and radial placements, the arc
    /// length from `angle_start` for an arc.
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

/// An edge that passed threshold and polarity, with its subpixel position on the profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Candidate {
    /// Subpixel profile index, in samples.
    pub x: f32,
    pub edge: MeasureEdge,
}

/// Append the edges `select` keeps from `cands` (in profile order) to `out`.
///
/// Only [`EdgeSelect::StrongestInOrder`] can come up empty on a non-empty `cands`; it
/// then leaves `out` as it was and names the entry that failed.
pub(crate) fn select_edges(
    cands: &[Candidate],
    select: EdgeSelect,
    out: &mut Vec<MeasureEdge>,
) -> Result<(), RejectReason> {
    match select {
        EdgeSelect::All => out.extend(cands.iter().map(|c| c.edge)),
        EdgeSelect::First => out.extend(cands.first().map(|c| c.edge)),
        EdgeSelect::Last => out.extend(cands.last().map(|c| c.edge)),
        // Equal amplitudes resolve to the later edge.
        EdgeSelect::Strongest => out.extend(
            cands
                .iter()
                .max_by(|a, b| a.edge.amplitude.total_cmp(&b.edge.amplitude))
                .map(|c| c.edge),
        ),
        EdgeSelect::StrongestInOrder(seq) => {
            let start = out.len();
            let mut after: Option<f32> = None;
            for (k, want) in [Some(seq.first), seq.second]
                .into_iter()
                .flatten()
                .enumerate()
            {
                let best = cands
                    .iter()
                    .filter(|c| want.admits(c.edge.polarity) && after.is_none_or(|a| c.x > a))
                    .reduce(|best, c| if stronger(c, best) { c } else { best });
                let Some(best) = best else {
                    out.truncate(start);
                    return Err(match k {
                        0 if cands.is_empty() => RejectReason::NoEdge,
                        0 => RejectReason::WrongPolarity,
                        _ => RejectReason::IncompleteSequence,
                    });
                };
                out.push(best.edge);
                after = Some(best.x);
            }
        }
    }
    Ok(())
}

/// `c` beats `best`: a larger amplitude, or an equal one earlier on the profile.
fn stronger(c: &Candidate, best: &Candidate) -> bool {
    match c.edge.amplitude.total_cmp(&best.edge.amplitude) {
        Ordering::Greater => true,
        Ordering::Equal => c.x < best.x,
        Ordering::Less => false,
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

#[cfg(test)]
mod tests {
    use super::{Candidate, MeasureEdge, select_edges};
    use crate::measure::{EdgeSelect, EdgeSequence, PolaritySelect, RejectReason};
    use vm_primitives::{EdgePolarity, Point2f};

    const R: EdgePolarity = EdgePolarity::Rising;
    const F: EdgePolarity = EdgePolarity::Falling;

    /// A candidate at profile position `x`; `t` is `x` so the chosen edges can be read back.
    fn cand(x: f32, amplitude: f32, polarity: EdgePolarity) -> Candidate {
        Candidate {
            x,
            edge: MeasureEdge {
                p: Point2f::new(x, 0.0),
                t: x,
                amplitude,
                polarity,
            },
        }
    }

    fn in_order(first: PolaritySelect, second: Option<PolaritySelect>) -> EdgeSelect {
        EdgeSelect::StrongestInOrder(EdgeSequence { first, second })
    }

    fn select(cands: &[Candidate], sel: EdgeSelect) -> Result<Vec<f32>, RejectReason> {
        let mut out = Vec::new();
        select_edges(cands, sel, &mut out).map(|()| out.iter().map(|e| e.t).collect())
    }

    #[test]
    fn equal_strength_goes_to_the_earlier_edge() {
        let cands = [cand(10.0, 3.0, R), cand(20.0, 3.0, R), cand(30.0, 1.0, F)];
        let any = PolaritySelect::Any;
        assert_eq!(select(&cands, in_order(any, None)), Ok(vec![10.0]));
        // The second entry starts after 10: 20 and 30 remain, and 20 is stronger.
        assert_eq!(
            select(&cands, in_order(any, Some(any))),
            Ok(vec![10.0, 20.0])
        );
        // `Strongest` keeps its own rule: equal amplitudes go to the later edge.
        assert_eq!(select(&cands, EdgeSelect::Strongest), Ok(vec![20.0]));
    }

    /// Each entry looks strictly after the previous choice: a stronger falling edge
    /// before the chosen rising one does not count, and no edge is chosen twice.
    #[test]
    fn later_entries_look_strictly_after_the_previous_choice() {
        let (rising, falling) = (PolaritySelect::Rising, PolaritySelect::Falling);
        let cands = [
            cand(5.0, 9.0, F),
            cand(12.5, 4.0, R),
            cand(20.0, 2.0, F),
            cand(30.0, 3.0, F),
        ];
        assert_eq!(
            select(&cands, in_order(rising, Some(falling))),
            Ok(vec![12.5, 30.0])
        );

        // One edge cannot fill two entries, even at an equal position.
        let one = [cand(12.5, 4.0, R)];
        let any = PolaritySelect::Any;
        assert_eq!(
            select(&one, in_order(any, Some(any))),
            Err(RejectReason::IncompleteSequence)
        );
        let tied = [cand(12.5, 4.0, R), cand(12.5, 4.0, F)];
        assert_eq!(
            select(&tied, in_order(rising, Some(falling))),
            Err(RejectReason::IncompleteSequence)
        );
    }

    #[test]
    fn a_missing_entry_names_which_one() {
        let (rising, falling) = (PolaritySelect::Rising, PolaritySelect::Falling);
        let cands = [cand(10.0, 3.0, R), cand(20.0, 2.0, R)];
        assert_eq!(
            select(&cands, in_order(rising, Some(falling))),
            Err(RejectReason::IncompleteSequence)
        );
        assert_eq!(
            select(&cands, in_order(falling, None)),
            Err(RejectReason::WrongPolarity)
        );
        assert_eq!(
            select(&[], in_order(rising, None)),
            Err(RejectReason::NoEdge)
        );

        // A failed selection leaves the output untouched.
        let mut out = vec![cands[0].edge];
        let r = select_edges(&cands, in_order(rising, Some(falling)), &mut out);
        assert_eq!(r, Err(RejectReason::IncompleteSequence));
        assert_eq!(out, vec![cands[0].edge]);
    }

    /// CaliperBench's greedy selection, line for line: for each requested polarity the
    /// eligible candidates are those of that polarity strictly after the last chosen
    /// position, and the winner is `max(eligible, key=(strength, -position))`. Its
    /// candidate list holds every rising peak, then every falling one.
    fn reference(cands: &[Candidate], sequence: &[PolaritySelect]) -> Option<Vec<f32>> {
        let listed: Vec<&Candidate> = cands
            .iter()
            .filter(|c| c.edge.polarity == R)
            .chain(cands.iter().filter(|c| c.edge.polarity == F))
            .collect();
        let mut edges: Vec<f32> = Vec::new();
        for want in sequence {
            let mut best: Option<&Candidate> = None;
            for c in listed.iter().copied() {
                let eligible =
                    want.admits(c.edge.polarity) && edges.last().is_none_or(|&l| c.x > l);
                // Python's `max` keeps the first of equal keys.
                let key = |c: &Candidate| (c.edge.amplitude, -c.x);
                if eligible && best.is_none_or(|b| key(c) > key(b)) {
                    best = Some(c);
                }
            }
            edges.push(best?.x);
        }
        Some(edges)
    }

    /// The selection reproduces CaliperBench's on seeded candidate lists with many equal
    /// amplitudes. Positions are distinct, as they are on a profile: two candidates at
    /// one subpixel position with one amplitude is the only tie the two orders break
    /// differently.
    #[test]
    fn matches_the_caliperbench_selection() {
        let pols = [
            PolaritySelect::Any,
            PolaritySelect::Rising,
            PolaritySelect::Falling,
        ];
        let mut state = 0x2545_f491_u32;
        let mut next = |m: u32| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) % m
        };
        let mut compared = 0;
        for _ in 0..2000 {
            let n = next(7) as usize;
            let mut x = 0.0f32;
            let cands: Vec<Candidate> = (0..n)
                .map(|_| {
                    x += 0.25 * (1 + next(8)) as f32;
                    let amplitude = (1 + next(3)) as f32;
                    cand(x, amplitude, if next(2) == 0 { R } else { F })
                })
                .collect();
            for &a in &pols {
                for second in [
                    None,
                    Some(PolaritySelect::Any),
                    Some(PolaritySelect::Rising),
                    Some(PolaritySelect::Falling),
                ] {
                    let sequence: Vec<PolaritySelect> = std::iter::once(a).chain(second).collect();
                    let ours = select(&cands, in_order(a, second)).ok();
                    assert_eq!(ours, reference(&cands, &sequence), "{cands:?} {sequence:?}");
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, 2000 * 12);
    }
}
