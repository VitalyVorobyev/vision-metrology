//! Multi-instance suppression by model-point overlap.
//!
//! Bounding-box IoU is the usual choice and is wrong for this problem in two
//! ways. The axis-aligned box of an elongated part at 45° is up to twice the
//! part's area, so two genuinely disjoint instances can report an IoU above 0.5
//! and one is suppressed for no reason; and a box is shape-blind, so a ring and
//! a disc occupying the same box read as fully overlapping.
//!
//! Instead each accepted instance stamps its own transformed model points into
//! a reusable mask, and a candidate is suppressed when too large a fraction of
//! *its* points land on an already-stamped cell. That is rotation-aware,
//! shape-faithful, and linear in the model size.

use super::matcher::ShapeMatch;
use super::model::ModelPoint;

/// Reusable stamp buffer for overlap tests.
///
/// Allocated lazily: a single-instance search never touches it.
#[derive(Debug, Default)]
pub(crate) struct InstanceMask {
    data: Vec<u32>,
    width: usize,
    height: usize,
    stamp: u32,
}

impl InstanceMask {
    /// Prepare for a new suppression pass over a `width × height` image.
    fn begin(&mut self, width: usize, height: usize) {
        let n = width.saturating_mul(height);
        if self.width != width || self.height != height || self.data.len() != n {
            self.width = width;
            self.height = height;
            self.data = vec![0; n];
            self.stamp = 0;
        }
        // A generation counter avoids clearing the buffer between calls; only
        // the wrap needs a real clear.
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.data.fill(0);
            self.stamp = 1;
        }
    }

    #[inline]
    fn hit(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return false;
        }
        self.data[y as usize * self.width + x as usize] == self.stamp
    }

    /// Stamp a 3×3 footprint so that a neighbouring instance one pixel off
    /// still registers as overlapping.
    fn mark(&mut self, x: i32, y: i32) {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx as usize >= self.width || ny as usize >= self.height {
                    continue;
                }
                self.data[ny as usize * self.width + nx as usize] = self.stamp;
            }
        }
    }
}

/// Sort by score and greedily suppress overlapping instances.
///
/// The sort key is `(−score, x, y)`. The positional tiebreak is not decoration:
/// two identical instances can score bit-identically, and an unstable sort with
/// no tiebreak would then order them differently between runs, making the
/// output of a deterministic algorithm non-deterministic.
pub(crate) fn suppress(
    mask: &mut InstanceMask,
    (width, height): (usize, usize),
    points: &[ModelPoint],
    mut matches: Vec<ShapeMatch>,
    max_overlap: f32,
    max_matches: usize,
    scratch: &mut Vec<(i32, i32)>,
) -> Vec<ShapeMatch> {
    matches.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .expect("scores are finite")
            .then_with(|| {
                a.position
                    .x
                    .partial_cmp(&b.position.x)
                    .expect("positions are finite")
            })
            .then_with(|| {
                a.position
                    .y
                    .partial_cmp(&b.position.y)
                    .expect("positions are finite")
            })
    });

    // A single-instance search needs no mask: after sorting, the answer is the
    // first element. This is the common case, and it is why the mask is lazily
    // allocated rather than sized up front.
    if max_matches == 1 || matches.len() < 2 {
        matches.truncate(if max_matches == 0 {
            matches.len()
        } else {
            max_matches
        });
        return matches;
    }

    mask.begin(width, height);
    let mut kept: Vec<ShapeMatch> = Vec::new();
    let n = points.len().max(1);

    for m in matches {
        if max_matches != 0 && kept.len() >= max_matches {
            break;
        }
        let (sn, cs) = m.angle().sin_cos();
        let s = m.scale();
        scratch.clear();
        scratch.extend(points.iter().map(|p| {
            (
                (m.position.x + s * (cs * p.d.x - sn * p.d.y)).round() as i32,
                (m.position.y + s * (sn * p.d.x + cs * p.d.y)).round() as i32,
            )
        }));

        let hits = scratch.iter().filter(|&&(x, y)| mask.hit(x, y)).count();
        if hits as f32 / n as f32 > max_overlap {
            continue;
        }
        for &(x, y) in scratch.iter() {
            mask.mark(x, y);
        }
        kept.push(m);
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::{InstanceMask, suppress};
    use crate::matching::matcher::ShapeMatch;
    use crate::matching::model::ModelPoint;
    use vm_primitives::{Point2f, Vec2f, similarity_from_parts};

    /// A 40-point ring of radius 20 centred on the origin.
    fn ring() -> Vec<ModelPoint> {
        (0..40)
            .map(|i| {
                let a = i as f32 * core::f32::consts::TAU / 40.0;
                ModelPoint {
                    d: Vec2f::new(20.0 * a.cos(), 20.0 * a.sin()),
                    t: Vec2f::new(a.cos(), a.sin()),
                }
            })
            .collect()
    }

    fn at(x: f32, y: f32, score: f32) -> ShapeMatch {
        ShapeMatch {
            pose: similarity_from_parts(Vec2f::new(x, y), 0.0, 1.0),
            position: Point2f::new(x, y),
            score,
            support: 40,
            level: 0,
        }
    }

    #[test]
    fn coincident_instances_collapse_to_the_best_one() {
        let mut mask = InstanceMask::default();
        let mut scratch = Vec::new();
        let out = suppress(
            &mut mask,
            (200, 200),
            &ring(),
            vec![at(100.0, 100.0, 0.8), at(101.0, 100.0, 0.9)],
            0.5,
            0,
            &mut scratch,
        );
        assert_eq!(out.len(), 1);
        assert!((out[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn disjoint_instances_both_survive() {
        let mut mask = InstanceMask::default();
        let mut scratch = Vec::new();
        let out = suppress(
            &mut mask,
            (300, 300),
            &ring(),
            vec![at(60.0, 60.0, 0.8), at(220.0, 220.0, 0.9)],
            0.5,
            0,
            &mut scratch,
        );
        assert_eq!(out.len(), 2);
        // Highest score first.
        assert!(out[0].score > out[1].score);
    }

    #[test]
    fn concentric_rings_of_different_radius_do_not_suppress_each_other() {
        // The decisive case for point overlap over bounding-box IoU: a small
        // ring sits entirely inside the large ring's bounding box, yet shares
        // no edge pixels with it.
        let big = ring();
        let small: Vec<ModelPoint> = big
            .iter()
            .map(|p| ModelPoint {
                d: p.d * 0.35,
                t: p.t,
            })
            .collect();
        let mut mask = InstanceMask::default();
        let mut scratch = Vec::new();

        // Stamp the big ring first, then test the small one against it.
        let out = suppress(
            &mut mask,
            (200, 200),
            &big,
            vec![at(100.0, 100.0, 0.9)],
            0.5,
            0,
            &mut scratch,
        );
        assert_eq!(out.len(), 1);

        let out = suppress(
            &mut mask,
            (200, 200),
            &small,
            vec![at(100.0, 100.0, 0.9), at(100.0, 100.0, 0.8)],
            0.5,
            0,
            &mut scratch,
        );
        assert_eq!(out.len(), 1, "the two identical small rings must collapse");
    }

    #[test]
    fn max_matches_caps_the_output() {
        let mut mask = InstanceMask::default();
        let mut scratch = Vec::new();
        let out = suppress(
            &mut mask,
            (400, 400),
            &ring(),
            vec![
                at(60.0, 60.0, 0.7),
                at(200.0, 60.0, 0.9),
                at(60.0, 200.0, 0.8),
            ],
            0.5,
            2,
            &mut scratch,
        );
        assert_eq!(out.len(), 2);
        assert!((out[0].score - 0.9).abs() < 1e-6);
        assert!((out[1].score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn output_order_is_deterministic_for_equal_scores() {
        let mut mask = InstanceMask::default();
        let mut scratch = Vec::new();
        let input = vec![
            at(200.0, 60.0, 0.75),
            at(60.0, 200.0, 0.75),
            at(60.0, 60.0, 0.75),
        ];
        let a = suppress(
            &mut mask,
            (400, 400),
            &ring(),
            input.clone(),
            0.5,
            0,
            &mut scratch,
        );
        let b = suppress(&mut mask, (400, 400), &ring(), input, 0.5, 0, &mut scratch);
        let key = |v: &Vec<ShapeMatch>| {
            v.iter()
                .map(|m| (m.position.x, m.position.y))
                .collect::<Vec<_>>()
        };
        assert_eq!(key(&a), key(&b));
        // Equal scores are broken by position, smallest x first.
        assert_eq!(key(&a)[0], (60.0, 60.0));
    }
}
