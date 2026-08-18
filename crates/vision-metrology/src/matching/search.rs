//! Pose sweep grids and local-maximum extraction.

use vm_primitives::Rect2f;

/// A pose found by the search, in one pyramid level's coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Candidate {
    pub x: i32,
    pub y: i32,
    pub angle: f32,
    pub scale: f32,
    pub score: f32,
}

/// A one-dimensional sweep over angle or scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Grid {
    start: f32,
    step: f32,
    count: usize,
    /// True when the last sample is adjacent to the first, i.e. a full circle.
    pub cyclic: bool,
}

impl Grid {
    /// Angle sweep covering `[range.0, range.1]` with a step of at most `step`.
    ///
    /// The range is used verbatim, never wrapped: a range straddling ±π is
    /// given as e.g. `(3.0, 3.4)` and swept as such. Only the reported angle of
    /// a finished match is wrapped into `(-π, π]`.
    pub(crate) fn angles(range: (f32, f32), step: f32) -> Self {
        let two_pi = core::f32::consts::TAU;
        let span = (range.1 - range.0).max(0.0);
        let step = step.max(1e-4);
        if span >= two_pi - step {
            let count = ((two_pi / step).round() as usize).max(1);
            return Self {
                start: range.0,
                step: two_pi / count as f32,
                count,
                cyclic: true,
            };
        }
        let count = ((span / step).round() as usize) + 1;
        Self {
            start: range.0,
            step: if count > 1 {
                span / (count - 1) as f32
            } else {
                0.0
            },
            count,
            cyclic: false,
        }
    }

    /// Geometric scale sweep over `[range.0, range.1]`.
    ///
    /// Geometric, not linear: the quantity that matters is the *relative*
    /// change, so a linear step of 0.05 over `[0.5, 2.0]` would be a 10 % step
    /// at the bottom of the range and 2.5 % at the top.
    pub(crate) fn scales(range: (f32, f32), rel_step: f32) -> Self {
        let (lo, hi) = (range.0.max(1e-6), range.1.max(range.0.max(1e-6)));
        let ratio = hi / lo;
        let rel = rel_step.max(1e-4);
        if ratio <= 1.0 + 1e-6 {
            return Self {
                start: lo,
                step: 0.0,
                count: 1,
                cyclic: false,
            };
        }
        let count = ((ratio.ln() / (1.0 + rel).ln()).round() as usize).max(1) + 1;
        Self {
            start: lo,
            // Stored as the per-index growth factor minus one.
            step: ratio.powf(1.0 / (count - 1) as f32) - 1.0,
            count,
            cyclic: false,
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub(crate) fn angle_at(&self, i: usize) -> f32 {
        self.start + self.step * i as f32
    }

    #[inline]
    pub(crate) fn scale_at(&self, i: usize) -> f32 {
        self.start * (1.0 + self.step).powi(i as i32)
    }
}

/// Inclusive integer bounds of the translation sweep at one level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Span {
    pub x0: i32,
    pub x1: i32,
    pub y0: i32,
    pub y1: i32,
}

impl Span {
    #[inline]
    pub(crate) fn width(self) -> usize {
        (self.x1 - self.x0 + 1).max(0) as usize
    }

    #[inline]
    pub(crate) fn height(self) -> usize {
        (self.y1 - self.y0 + 1).max(0) as usize
    }

    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        self.x1 < self.x0 || self.y1 < self.y0
    }

    #[inline]
    pub(crate) fn contains(self, x: i32, y: i32) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }

    /// The positions of level `level` that a level-0 ROI maps onto.
    pub(crate) fn for_level(roi: Option<Rect2f>, level: usize, w: usize, h: usize) -> Self {
        let s = (1usize << level) as f32;
        let shift = 0.5 * (s - 1.0);
        let (mut x0, mut y0, mut x1, mut y1) = (0i32, 0i32, w as i32 - 1, h as i32 - 1);
        if let Some(r) = roi {
            x0 = x0.max((((r.x - shift) / s).floor()) as i32);
            y0 = y0.max((((r.y - shift) / s).floor()) as i32);
            x1 = x1.min(((((r.x + r.width) - shift) / s).ceil()) as i32);
            y1 = y1.min(((((r.y + r.height) - shift) / s).ceil()) as i32);
        }
        Self { x0, x1, y0, y1 }
    }
}

/// Emit every strict 3-D local maximum of `cur` over `(x, y, angle)`.
///
/// The angle dimension is not optional. Without it a single object produces one
/// candidate per angle step — up to 40 at a coarse level — and the candidate
/// budget is spent on one instance.
///
/// Plateaus are broken deterministically: a cell must be strictly greater than
/// every neighbour that precedes it in `(Δangle, Δy, Δx)` order and at least
/// equal to every neighbour that follows, so exactly one cell of a connected
/// plateau survives.
#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_local_maxima(
    prev: &[f32],
    cur: &[f32],
    next: &[f32],
    span: Span,
    angle: f32,
    scale: f32,
    threshold: f32,
    out: &mut Vec<Candidate>,
) {
    let (mw, mh) = (span.width(), span.height());
    if mw == 0 || mh == 0 {
        return;
    }
    let planes = [prev, cur, next];

    for iy in 0..mh {
        for ix in 0..mw {
            // Aborted positions were written as -inf, so this rejects them too.
            let v = cur[iy * mw + ix];
            if v < threshold {
                continue;
            }
            let mut is_max = true;
            'nb: for (pi, plane) in planes.iter().enumerate() {
                let da = pi as i32 - 1;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if da == 0 && dy == 0 && dx == 0 {
                            continue;
                        }
                        let (nx, ny) = (ix as i32 + dx, iy as i32 + dy);
                        let nv = if nx < 0 || ny < 0 || nx >= mw as i32 || ny >= mh as i32 {
                            f32::NEG_INFINITY
                        } else {
                            plane[ny as usize * mw + nx as usize]
                        };
                        let earlier = (da, dy, dx) < (0, 0, 0);
                        let ok = if earlier { v > nv } else { v >= nv };
                        if !ok {
                            is_max = false;
                            break 'nb;
                        }
                    }
                }
            }
            if is_max {
                out.push(Candidate {
                    x: span.x0 + ix as i32,
                    y: span.y0 + iy as i32,
                    angle,
                    scale,
                    score: v,
                });
            }
        }
    }
}

/// Keep the `max` highest-scoring candidates.
///
/// Returns `true` when candidates were dropped, so the caller can tell a user
/// "not found" apart from "gave up". Uses a linear-time selection rather than
/// the full sort a top-K insertion would repeat on every insert.
pub(crate) fn cap_candidates(cands: &mut Vec<Candidate>, max: usize) -> bool {
    if max == 0 || cands.len() <= max {
        return false;
    }
    cands.select_nth_unstable_by(max, |a, b| {
        b.score
            .partial_cmp(&a.score)
            .expect("scores are finite")
            .then_with(|| a.x.cmp(&b.x))
            .then_with(|| a.y.cmp(&b.y))
    });
    cands.truncate(max);
    true
}

#[cfg(test)]
mod tests {
    use super::{Candidate, Grid, Span, cap_candidates, collect_local_maxima};

    #[test]
    fn a_full_circle_angle_grid_is_cyclic_and_evenly_spaced() {
        let g = Grid::angles((-core::f32::consts::PI, core::f32::consts::PI), 0.1);
        assert!(g.cyclic);
        assert_eq!(g.len(), 63);
        let sweep = g.step * g.len() as f32;
        assert!((sweep - core::f32::consts::TAU).abs() < 1e-4);
    }

    #[test]
    fn a_narrow_range_hits_both_endpoints_without_wrapping() {
        // 170 deg to 190 deg: expressed unwrapped, so it straddles +-pi without
        // the grid ever having to think about it.
        let (a, b) = (170f32.to_radians(), 190f32.to_radians());
        let g = Grid::angles((a, b), 0.02);
        assert!(!g.cyclic);
        assert!((g.angle_at(0) - a).abs() < 1e-6);
        assert!((g.angle_at(g.len() - 1) - b).abs() < 1e-5);
        assert!(g.angle_at(g.len() - 1) > core::f32::consts::PI);
    }

    #[test]
    fn scale_grid_is_geometric() {
        let g = Grid::scales((0.5, 2.0), 0.1);
        assert!((g.scale_at(0) - 0.5).abs() < 1e-6);
        assert!((g.scale_at(g.len() - 1) - 2.0).abs() < 1e-4);
        // Constant ratio between successive samples is what "geometric" means.
        for i in 1..g.len() {
            let r = g.scale_at(i) / g.scale_at(i - 1);
            assert!((r - (1.0 + g.step)).abs() < 1e-5, "ratio {r} at {i}");
        }
    }

    #[test]
    fn a_degenerate_scale_range_yields_one_sample() {
        let g = Grid::scales((1.0, 1.0), 0.1);
        assert_eq!(g.len(), 1);
        assert!((g.scale_at(0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_plateau_yields_exactly_one_maximum() {
        let span = Span {
            x0: 0,
            x1: 4,
            y0: 0,
            y1: 4,
        };
        let mut cur = vec![0.0f32; 25];
        // 2x2 plateau of equal values.
        for &(x, y) in &[(1usize, 1usize), (2, 1), (1, 2), (2, 2)] {
            cur[y * 5 + x] = 0.9;
        }
        let neg = vec![f32::NEG_INFINITY; 25];
        let mut out = Vec::new();
        collect_local_maxima(&neg, &cur, &neg, span, 0.0, 1.0, 0.5, &mut out);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!((out[0].x, out[0].y), (1, 1));
    }

    #[test]
    fn a_maximum_in_the_angle_dimension_is_required_too() {
        let span = Span {
            x0: 0,
            x1: 2,
            y0: 0,
            y1: 2,
        };
        let mut cur = vec![0.0f32; 9];
        cur[4] = 0.8;
        let mut better = vec![0.0f32; 9];
        better[4] = 0.9;
        let neg = vec![f32::NEG_INFINITY; 9];

        let mut out = Vec::new();
        collect_local_maxima(&neg, &cur, &neg, span, 0.0, 1.0, 0.5, &mut out);
        assert_eq!(out.len(), 1, "isolated peak is a maximum");

        out.clear();
        collect_local_maxima(&neg, &cur, &better, span, 0.0, 1.0, 0.5, &mut out);
        assert!(out.is_empty(), "the neighbouring angle scores higher");
    }

    #[test]
    fn capping_keeps_the_best_and_reports_the_truncation() {
        let mut c: Vec<Candidate> = (0..10)
            .map(|i| Candidate {
                x: i,
                y: 0,
                angle: 0.0,
                scale: 1.0,
                score: i as f32 / 10.0,
            })
            .collect();
        assert!(cap_candidates(&mut c, 3));
        assert_eq!(c.len(), 3);
        let mut scores: Vec<f32> = c.iter().map(|k| k.score).collect();
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(scores, vec![0.9, 0.8, 0.7]);

        let mut small = c.clone();
        assert!(!cap_candidates(&mut small, 100));
    }
}
