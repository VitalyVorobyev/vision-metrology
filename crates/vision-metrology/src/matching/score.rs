//! The similarity measure and its greedy early-termination bound.
//!
//! For a model point with unit direction `t_i` placed at scene position `p_i`
//! under a pose, the term is `c_i = (R·t_i) · ĝ(p_i)`, where `ĝ` is the scene's
//! unit gradient direction — zero wherever the gradient is weaker than the
//! configured contrast floor.
//!
//! Two properties are load-bearing:
//!
//! * **Isotropic scale does not rotate a gradient.** For `I'(x) = I(R⁻¹(x−q)/s)`
//!   the chain rule gives `∇I'(p_i) = (1/s)·R·∇I(d_i)`, and the `1/s` cancels on
//!   normalisation. One rotated direction set therefore serves every scale; only
//!   the offsets `d_i` are scaled.
//! * **The sum is divided by `n`, the full model point count** — never by the
//!   number of points that happened to contribute. That is what turns the score
//!   into a direct occlusion measure: half the contour hidden gives ~0.5.

use vm_primitives::DirectionField;

use super::config::Polarity;
use super::model::ModelPoint;

/// A model point rotated and scaled for one pose, ready for integer lookup.
///
/// The offsets are pre-rounded because the translation sweep visits integer
/// positions only, which removes a round-and-convert from the inner loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RotPoint {
    pub ox: i32,
    pub oy: i32,
    pub tx: f32,
    pub ty: f32,
}

pub(crate) const POL_MATCH: u8 = 0;
pub(crate) const POL_GLOBAL: u8 = 1;
pub(crate) const POL_LOCAL: u8 = 2;

const fn pol_code(p: Polarity) -> u8 {
    match p {
        Polarity::Match => POL_MATCH,
        Polarity::IgnoreGlobal => POL_GLOBAL,
        Polarity::IgnoreLocal => POL_LOCAL,
    }
}

/// Rotate and scale a level's points into the caller's scratch buffer.
pub(crate) fn rotate_into(points: &[ModelPoint], angle: f32, scale: f32, out: &mut Vec<RotPoint>) {
    let (sn, cs) = angle.sin_cos();
    out.clear();
    out.reserve(points.len());
    for p in points {
        out.push(RotPoint {
            ox: (scale * (cs * p.d.x - sn * p.d.y)).round() as i32,
            oy: (scale * (sn * p.d.x + cs * p.d.y)).round() as i32,
            tx: cs * p.t.x - sn * p.t.y,
            ty: sn * p.t.x + cs * p.t.y,
        });
    }
}

/// Greedy early-termination bound, factored so the check is one FMA and one
/// compare.
///
/// With partial sum `S_j` after `j` points, abort when `S_j < a + b·j`.
///
/// * `greediness = 0` gives `a = n(min_score − 1)`, `b = 1`, i.e. abort exactly
///   when even a perfect score on every remaining point could not reach
///   `min_score`. That bound is **safe**: it never rejects a pose that would
///   have qualified.
/// * `greediness = 1` gives `a = 0`, `b = min_score`: the running mean must stay
///   above the threshold at every step. Fastest, and it does reject qualifying
///   poses whose early points are the occluded ones.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Bound {
    pub a: f32,
    pub b: f32,
}

impl Bound {
    pub(crate) fn new(n: usize, greediness: f32, min_score: f32) -> Self {
        let g = greediness.clamp(0.0, 1.0);
        Self {
            a: n as f32 * (1.0 - g) * (min_score - 1.0),
            b: (1.0 - g) + g * min_score,
        }
    }

    /// A bound that never aborts, for exhaustive reference scoring.
    pub(crate) fn never() -> Self {
        Self {
            a: f32::NEG_INFINITY,
            b: 0.0,
        }
    }
}

/// Points evaluated between two checks of the greedy bound.
///
/// `a + b·j` is monotone increasing in `j`, so checking less often can only
/// cost a few extra evaluations — it can never abort a pose the per-point check
/// would have kept. In exchange the inner loop stays an 8-wide block.
const CHUNK: usize = 8;

#[inline]
fn accumulate<const POL: u8>(
    dir: &[f32],
    w: usize,
    h: usize,
    qx: i32,
    qy: i32,
    p: RotPoint,
) -> f32 {
    let px = (qx + p.ox) as usize;
    let py = (qy + p.oy) as usize;
    // A negative coordinate wraps to a huge `usize`, so one unsigned compare
    // per axis rejects both sides of the image at once. Out-of-image points
    // contribute nothing, which is what lets a partially out-of-frame object
    // still be found with a correspondingly reduced score.
    if px >= w || py >= h {
        return 0.0;
    }
    let k = 2 * (py * w + px);
    let c = p.tx * dir[k] + p.ty * dir[k + 1];
    if POL == POL_LOCAL { c.abs() } else { c }
}

#[inline]
fn score_impl<const POL: u8>(
    field: &DirectionField,
    rot: &[RotPoint],
    qx: i32,
    qy: i32,
    bound: Bound,
) -> Option<f32> {
    let n = rot.len();
    if n == 0 {
        return None;
    }
    let (w, h) = (field.width(), field.height());
    let dir = field.dir();

    let mut sum = 0.0f32;
    let mut j = 0usize;
    while j < n {
        let end = (j + CHUNK).min(n);
        for p in &rot[j..end] {
            sum += accumulate::<POL>(dir, w, h, qx, qy, *p);
        }
        j = end;
        let partial = if POL == POL_GLOBAL { sum.abs() } else { sum };
        if partial < bound.a + bound.b * j as f32 {
            return None;
        }
    }

    let total = if POL == POL_GLOBAL { sum.abs() } else { sum };
    Some(total / n as f32)
}

/// Score one pose at one integer position, or `None` if the greedy bound
/// aborted it.
pub(crate) fn score_at(
    field: &DirectionField,
    rot: &[RotPoint],
    qx: i32,
    qy: i32,
    polarity: Polarity,
    bound: Bound,
) -> Option<f32> {
    match pol_code(polarity) {
        POL_MATCH => score_impl::<POL_MATCH>(field, rot, qx, qy, bound),
        POL_GLOBAL => score_impl::<POL_GLOBAL>(field, rot, qx, qy, bound),
        _ => score_impl::<POL_LOCAL>(field, rot, qx, qy, bound),
    }
}

/// Score a horizontal run of positions `x0..x0 + out.len()` at row `qy`.
///
/// Dispatching on polarity once per row instead of once per position keeps the
/// scoring loop monomorphic. Aborted positions are written as
/// [`f32::NEG_INFINITY`], which the local-maximum pass treats as "no candidate".
pub(crate) fn score_span(
    field: &DirectionField,
    rot: &[RotPoint],
    qy: i32,
    x0: i32,
    polarity: Polarity,
    bound: Bound,
    out: &mut [f32],
) {
    match pol_code(polarity) {
        POL_MATCH => span_impl::<POL_MATCH>(field, rot, qy, x0, bound, out),
        POL_GLOBAL => span_impl::<POL_GLOBAL>(field, rot, qy, x0, bound, out),
        _ => span_impl::<POL_LOCAL>(field, rot, qy, x0, bound, out),
    }
}

/// Positions per block in the blocked span scorer.
///
/// Inside a block the loop runs point-major: one model point is applied to all
/// `BLOCK` consecutive positions before the next point. For an in-bounds run
/// the reads are `dir[2k], dir[2k+1], dir[2k+2], …` — a contiguous
/// interleaved pattern the autovectorizer turns into deinterleaving loads and
/// FMAs. Per-lane sums accumulate in the identical order as the scalar
/// `score_impl` (point 0, 1, 2, …), and the greedy bound is checked at the
/// same 8-point boundaries, so the output is **bit-identical** to the scalar
/// path — the blocking changes only how much work aborted positions cost.
const BLOCK: usize = 32;

fn span_impl<const POL: u8>(
    field: &DirectionField,
    rot: &[RotPoint],
    qy: i32,
    x0: i32,
    bound: Bound,
    out: &mut [f32],
) {
    let n = rot.len();
    if n == 0 {
        out.fill(f32::NEG_INFINITY);
        return;
    }
    let (w, h) = (field.width(), field.height());
    let dir = field.dir();

    let mut i = 0usize;
    while i < out.len() {
        let bw = (out.len() - i).min(BLOCK);
        span_block::<POL>(
            dir,
            w,
            h,
            rot,
            x0 + i as i32,
            qy,
            bound,
            &mut out[i..i + bw],
        );
        i += bw;
    }
}

#[allow(clippy::too_many_arguments)]
fn span_block<const POL: u8>(
    dir: &[f32],
    w: usize,
    h: usize,
    rot: &[RotPoint],
    bx: i32,
    qy: i32,
    bound: Bound,
    out: &mut [f32],
) {
    let bw = out.len();
    let n = rot.len();
    let mut sums = [0.0f32; BLOCK];
    let mut dead = [false; BLOCK];
    let mut alive = bw;

    let mut j = 0usize;
    while j < n {
        let end = (j + CHUNK).min(n);
        for p in &rot[j..end] {
            let py = qy + p.oy;
            let xs = bx + p.ox;
            // Fast path: the whole block row is inside the image.
            if (py as usize) < h && xs >= 0 && xs as usize + bw <= w {
                let base = 2 * (py as usize * w + xs as usize);
                let row = &dir[base..base + 2 * bw];
                for (l, s) in sums[..bw].iter_mut().enumerate() {
                    let c = p.tx * row[2 * l] + p.ty * row[2 * l + 1];
                    *s += if POL == POL_LOCAL { c.abs() } else { c };
                }
            } else {
                for (l, s) in sums[..bw].iter_mut().enumerate() {
                    *s += accumulate::<POL>(dir, w, h, bx + l as i32, qy, *p);
                }
            }
        }
        j = end;

        let thr = bound.a + bound.b * j as f32;
        for l in 0..bw {
            if !dead[l] {
                let partial = if POL == POL_GLOBAL {
                    sums[l].abs()
                } else {
                    sums[l]
                };
                if partial < thr {
                    dead[l] = true;
                    alive -= 1;
                }
            }
        }
        if alive == 0 {
            break;
        }
        // Once only a few positions survive, the block-wide vector work costs
        // more than it saves: one deep survivor would drag all lanes to the
        // end. Finish the survivors lane by lane instead — same chunk
        // boundaries, same order, same results.
        if alive <= bw / 2 && j < n {
            for l in 0..bw {
                if !dead[l] {
                    dead[l] = !finish_lane::<POL>(
                        dir,
                        w,
                        h,
                        bx + l as i32,
                        qy,
                        rot,
                        j,
                        bound,
                        &mut sums[l],
                    );
                }
            }
            break;
        }
    }

    for (l, slot) in out.iter_mut().enumerate() {
        *slot = if dead[l] {
            f32::NEG_INFINITY
        } else {
            let total = if POL == POL_GLOBAL {
                sums[l].abs()
            } else {
                sums[l]
            };
            total / n as f32
        };
    }
}

/// Continue one lane from point `j` to the end with the standard chunked
/// bound checks. Returns `false` if the bound fired.
#[allow(clippy::too_many_arguments)]
fn finish_lane<const POL: u8>(
    dir: &[f32],
    w: usize,
    h: usize,
    qx: i32,
    qy: i32,
    rot: &[RotPoint],
    mut j: usize,
    bound: Bound,
    sum: &mut f32,
) -> bool {
    let n = rot.len();
    while j < n {
        let end = (j + CHUNK).min(n);
        for p in &rot[j..end] {
            *sum += accumulate::<POL>(dir, w, h, qx, qy, *p);
        }
        j = end;
        let partial = if POL == POL_GLOBAL { sum.abs() } else { *sum };
        if partial < bound.a + bound.b * j as f32 {
            return false;
        }
    }
    true
}

/// Score a subpixel pose exhaustively, also reporting how many model points
/// found any gradient at all.
///
/// Used once per reported match so that the score and the support count belong
/// to the pose actually returned, rather than to the integer grid cell the
/// search stopped at.
pub(crate) fn score_pose(
    field: &DirectionField,
    points: &[ModelPoint],
    qx: f32,
    qy: f32,
    angle: f32,
    scale: f32,
    polarity: Polarity,
) -> (f32, usize) {
    let n = points.len();
    if n == 0 {
        return (0.0, 0);
    }
    let (sn, cs) = angle.sin_cos();
    let (w, h) = (field.width(), field.height());
    let dir = field.dir();

    let mut sum = 0.0f32;
    let mut support = 0usize;
    for p in points {
        let px = (qx + scale * (cs * p.d.x - sn * p.d.y)).round();
        let py = (qy + scale * (sn * p.d.x + cs * p.d.y)).round();
        if px < 0.0 || py < 0.0 {
            continue;
        }
        let (ux, uy) = (px as usize, py as usize);
        if ux >= w || uy >= h {
            continue;
        }
        let k = 2 * (uy * w + ux);
        let tx = cs * p.t.x - sn * p.t.y;
        let ty = sn * p.t.x + cs * p.t.y;
        let c = tx * dir[k] + ty * dir[k + 1];
        if c != 0.0 {
            support += 1;
        }
        sum += if polarity == Polarity::IgnoreLocal {
            c.abs()
        } else {
            c
        };
    }

    let total = if polarity == Polarity::IgnoreGlobal {
        sum.abs()
    } else {
        sum
    };
    (total / n as f32, support)
}

#[cfg(test)]
mod tests {
    use super::{Bound, RotPoint, rotate_into, score_at};
    use crate::matching::config::Polarity;
    use crate::matching::model::ModelPoint;
    use vm_primitives::{DirectionField, Image, SmoothKind, Vec2f};

    fn unit(x: f32, y: f32) -> Vec2f {
        Vec2f { x, y }.normalize()
    }

    #[test]
    fn safe_bound_reduces_to_the_reachability_test() {
        // greediness 0: abort when even a perfect remainder cannot reach the
        // threshold, i.e. S_j < n*min - (n - j).
        let b = Bound::new(100, 0.0, 0.5);
        assert!((b.a - (100.0 * (0.5 - 1.0))).abs() < 1e-5);
        assert!((b.b - 1.0).abs() < 1e-5);

        // greediness 1: the running mean must already reach the threshold.
        let b = Bound::new(100, 1.0, 0.5);
        assert!(b.a.abs() < 1e-5);
        assert!((b.b - 0.5).abs() < 1e-5);
    }

    #[test]
    fn rotation_moves_offsets_and_directions_together() {
        let pts = [ModelPoint {
            d: Vec2f { x: 10.0, y: 0.0 },
            t: unit(1.0, 0.0),
        }];
        let mut out = Vec::new();
        rotate_into(&pts, core::f32::consts::FRAC_PI_2, 1.0, &mut out);
        assert_eq!(out[0].ox, 0);
        assert_eq!(out[0].oy, 10);
        assert!(out[0].tx.abs() < 1e-6 && (out[0].ty - 1.0).abs() < 1e-6);

        // Scale moves the offset but leaves the direction alone: an isotropic
        // scale does not rotate a gradient.
        rotate_into(&pts, 0.0, 2.5, &mut out);
        assert_eq!((out[0].ox, out[0].oy), (25, 0));
        assert!((out[0].tx - 1.0).abs() < 1e-6);
    }

    /// Vertical step edge at x = 16, dark left / bright right.
    fn step_field(min_mag: f32) -> DirectionField {
        let mut data = vec![0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 16..32 {
                data[y * 32 + x] = 255.0;
            }
        }
        let img = Image::from_vec(32, 32, data).unwrap();
        let mut f = DirectionField::new();
        f.build_f32(&img.as_view(), SmoothKind::None, min_mag);
        f
    }

    #[test]
    fn polarity_modes_differ_exactly_where_documented() {
        let field = step_field(10.0);
        // Four points on the step, all pointing +x (dark to bright).
        let rot: Vec<RotPoint> = (0..4)
            .map(|i| RotPoint {
                ox: 0,
                oy: i - 2,
                tx: 1.0,
                ty: 0.0,
            })
            .collect();
        let flipped: Vec<RotPoint> = rot.iter().map(|p| RotPoint { tx: -1.0, ..*p }).collect();
        let half: Vec<RotPoint> = rot
            .iter()
            .enumerate()
            .map(|(i, p)| RotPoint {
                tx: if i % 2 == 0 { 1.0 } else { -1.0 },
                ..*p
            })
            .collect();

        let b = Bound::never();
        let at = |r: &[RotPoint], pol| score_at(&field, r, 15, 16, pol, b).unwrap();

        assert!(at(&rot, Polarity::Match) > 0.99);
        assert!(at(&flipped, Polarity::Match) < -0.99);
        assert!(at(&flipped, Polarity::IgnoreGlobal) > 0.99);
        // Half inverted: the signed mean cancels, the per-point mean does not.
        assert!(at(&half, Polarity::Match).abs() < 0.01);
        assert!(at(&half, Polarity::IgnoreGlobal).abs() < 0.01);
        assert!(at(&half, Polarity::IgnoreLocal) > 0.99);
    }

    #[test]
    fn out_of_image_points_score_zero_rather_than_wrapping() {
        let field = step_field(10.0);
        // Two points on the edge, two far outside the image on both sides.
        let rot = vec![
            RotPoint {
                ox: 0,
                oy: 0,
                tx: 1.0,
                ty: 0.0,
            },
            RotPoint {
                ox: 0,
                oy: 1,
                tx: 1.0,
                ty: 0.0,
            },
            RotPoint {
                ox: -900,
                oy: 0,
                tx: 1.0,
                ty: 0.0,
            },
            RotPoint {
                ox: 900,
                oy: 0,
                tx: 1.0,
                ty: 0.0,
            },
        ];
        let s = score_at(&field, &rot, 15, 16, Polarity::Match, Bound::never()).unwrap();
        // Two of four contributed 1.0 each, divided by the full count.
        assert!((s - 0.5).abs() < 0.01, "got {s}");
    }

    #[test]
    fn the_safe_bound_never_rejects_a_qualifying_pose() {
        let field = step_field(10.0);
        let rot: Vec<RotPoint> = (0..40)
            .map(|i| RotPoint {
                ox: 0,
                oy: i - 20,
                tx: 1.0,
                ty: 0.0,
            })
            .collect();

        for qx in 0..32i32 {
            for qy in 8..24i32 {
                let exact = score_at(&field, &rot, qx, qy, Polarity::Match, Bound::never())
                    .expect("never aborts");
                let safe = score_at(
                    &field,
                    &rot,
                    qx,
                    qy,
                    Polarity::Match,
                    Bound::new(rot.len(), 0.0, 0.5),
                );
                if exact >= 0.5 {
                    assert_eq!(
                        safe,
                        Some(exact),
                        "safe bound dropped a qualifying pose at ({qx},{qy})"
                    );
                }
            }
        }
    }

    /// T13 — the greedy bound at `greediness = 0` is provably safe.
    ///
    /// Sweep every pose of a textured 64x64 scene. Wherever the exhaustive
    /// reference (a bound that never fires) scores at or above `min_score`,
    /// the guarded evaluation must return the **bit-identical** score; where
    /// the guarded evaluation aborts, the reference must be strictly below
    /// `min_score`. This is the single easiest property to get subtly wrong
    /// in the whole matcher, so it is pinned exactly, for all three
    /// polarities and several angles.
    #[test]
    fn t13_safe_bound_never_rejects_a_qualifying_pose() {
        // Deterministic texture: LCG noise plus low-frequency structure, so
        // the field has both coherent edges and clutter.
        let (w, h) = (64usize, 64usize);
        let mut data = vec![0.0f32; w * h];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for y in 0..h {
            for x in 0..w {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = ((state >> 33) & 0xff) as f32;
                let wave = 96.0 * (0.35 * x as f32).sin() * (0.22 * y as f32).cos();
                data[y * w + x] = (96.0 + 0.35 * noise + wave).clamp(0.0, 255.0);
            }
        }
        let img = Image::from_vec(w, h, data).unwrap();
        let mut field = DirectionField::new();
        field.build_f32(&img.as_view(), SmoothKind::Binomial3, 10.0);

        // A 24-point model with deterministic pseudo-random offsets and unit
        // directions -- realistic enough that partial sums dip and recover.
        let mut pts = Vec::new();
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..24 {
            let mut next = || {
                st = st
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((st >> 33) & 0xffff) as f32 / 65_535.0
            };
            let d = Vec2f {
                x: (next() - 0.5) * 16.0,
                y: (next() - 0.5) * 16.0,
            };
            let a = next() * core::f32::consts::TAU;
            pts.push(ModelPoint {
                d,
                t: Vec2f {
                    x: a.cos(),
                    y: a.sin(),
                },
            });
        }

        let min_score = 0.5f32;
        let mut rot = Vec::new();
        for polarity in [
            Polarity::Match,
            Polarity::IgnoreGlobal,
            Polarity::IgnoreLocal,
        ] {
            for k in 0..8 {
                let angle = k as f32 * core::f32::consts::TAU / 8.0;
                rotate_into(&pts, angle, 1.0, &mut rot);
                let safe = Bound::new(rot.len(), 0.0, min_score);

                let mut qualifying = 0usize;
                for qy in 0..h as i32 {
                    for qx in 0..w as i32 {
                        let reference = score_at(&field, &rot, qx, qy, polarity, Bound::never())
                            .expect("never-bound always yields a score");
                        let guarded = score_at(&field, &rot, qx, qy, polarity, safe);
                        if reference >= min_score {
                            qualifying += 1;
                            assert_eq!(
                                guarded,
                                Some(reference),
                                "safe bound rejected or altered a qualifying pose at                                  ({qx},{qy}) angle {k} polarity {polarity:?}"
                            );
                        } else if let Some(g) = guarded {
                            // Surviving a non-qualifying pose is allowed (the
                            // bound is necessary, not sufficient) -- but the
                            // score must still be identical arithmetic.
                            assert_eq!(g, reference);
                        }
                    }
                }
                // The fixture must actually exercise the property: IgnoreLocal
                // scores are generous, so at least that polarity has to
                // produce qualifying poses.
                if polarity == Polarity::IgnoreLocal {
                    assert!(qualifying > 0, "fixture produced no qualifying poses");
                }
            }
        }
    }
}
