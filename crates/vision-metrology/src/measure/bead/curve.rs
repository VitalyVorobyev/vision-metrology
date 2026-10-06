//! Stations along a curve: arc-length resampling, chord tangents, curvature, and the
//! guards that keep a correction from folding the curve.
//!
//! Everything here is `f64` (invariant 20). A station polyline is uniform: station `i`
//! sits at arc length `i·h`, and a point between stations is linear between them.

use vm_primitives::{Error, Point2f};

/// A point or a vector, `[x, y]`, in pixels.
pub(super) type V2 = [f64; 2];

/// The share of a station's room a correction may use before the curve folds. Evidence on
/// the concave side stops at `FOLD / |κ|`; a step moves no station more than `FOLD` of the
/// way to its centre of curvature, and keeps every segment at least `1 − FOLD` of its
/// length along its old direction.
pub(super) const FOLD: f64 = 0.9;

/// The most stations one call may place.
const MAX_STATIONS: usize = 1 << 22;

#[inline]
pub(super) fn sub(a: V2, b: V2) -> V2 {
    [a[0] - b[0], a[1] - b[1]]
}

#[inline]
pub(super) fn dot(a: V2, b: V2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// `a × b = a.x·b.y − a.y·b.x`: positive when `b` turns from `a` towards `a`'s normal.
#[inline]
pub(super) fn cross(a: V2, b: V2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

/// The normal `t.perp() = (−t_y, t_x)`.
#[inline]
pub(super) fn perp(t: V2) -> V2 {
    [-t[1], t[0]]
}

#[inline]
fn lerp(a: V2, b: V2, f: f64) -> V2 {
    [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]
}

/// Copy `prior` to `out` in `f64`, after checking it has two finite points or more.
pub(super) fn load_prior(prior: &[Point2f], out: &mut Vec<V2>) -> Result<(), Error> {
    if prior.len() < 2 {
        return Err(Error::InsufficientData {
            need: 2,
            got: prior.len(),
        });
    }
    if prior.iter().any(|p| !(p.x.is_finite() && p.y.is_finite())) {
        return Err(Error::Degenerate("bead prior has a non-finite point"));
    }
    out.clear();
    out.extend(prior.iter().map(|p| [f64::from(p.x), f64::from(p.y)]));
    Ok(())
}

/// Fill `cum` with the cumulative chord length of `pts` (`cum[0] = 0`) and return the
/// total.
pub(super) fn arc_lengths(pts: &[V2], cum: &mut Vec<f64>) -> f64 {
    cum.clear();
    let mut acc = 0.0;
    cum.push(acc);
    for w in pts.windows(2) {
        let d = sub(w[1], w[0]);
        acc += d[0].hypot(d[1]);
        cum.push(acc);
    }
    acc
}

/// The station count for a curve of `length` px at `spacing` px: `round(L/spacing) + 1`,
/// at least 2.
pub(super) fn station_count(length: f64, spacing: f64) -> Result<usize, Error> {
    let n = (length / spacing).round() + 1.0;
    if n > MAX_STATIONS as f64 {
        return Err(Error::InvalidConfig(
            "bead prior needs more than 2^22 stations at this spacing",
        ));
    }
    Ok((n as usize).max(2))
}

/// Resample the polyline `src`, whose cumulative lengths are `cum`, to `n ≥ 2` stations
/// uniform in arc length. The first and last stations are `src`'s ends exactly;
/// zero-length segments are skipped.
pub(super) fn resample(src: &[V2], cum: &[f64], n: usize, out: &mut Vec<V2>) {
    debug_assert!(n >= 2 && src.len() >= 2 && cum.len() == src.len());
    out.clear();
    let last = src.len() - 1;
    let total = cum[last];
    let mut seg = 0;
    out.push(src[0]);
    for i in 1..n - 1 {
        let s = i as f64 * (total / (n - 1) as f64);
        // The segment whose end reaches `s`.
        while seg + 1 < last && cum[seg + 1] < s {
            seg += 1;
        }
        let span = cum[seg + 1] - cum[seg];
        let f = if span > 0.0 {
            ((s - cum[seg]) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        out.push(lerp(src[seg], src[seg + 1], f));
    }
    out.push(src[last]);
}

/// The point at arc length `s` on the uniform stations `pts`, `h` apart.
fn at(pts: &[V2], h: f64, s: f64) -> V2 {
    let last = pts.len() - 1;
    let u = (s / h).clamp(0.0, last as f64);
    let k = (u.floor() as usize).min(last - 1);
    lerp(pts[k], pts[k + 1], u - k as f64)
}

/// Fill `tan` with each station's unit tangent: the chord from arc length `s − r` to
/// `s + r`, where `r = min(window, s, L − s)` shrinks near the ends so the chord stays
/// symmetric, which makes it exact on a circle. The two end stations take the one-sided
/// chord to their neighbour. A clamped full-window chord would tilt an end normal by about
/// `κ·window/2`, and a width measured along it by `1/cos` of that.
///
/// A station whose chord has no length takes its nearest neighbour's tangent; `(1, 0)`
/// when none has one.
pub(super) fn chord_tangents(pts: &[V2], h: f64, window: f64, tan: &mut Vec<V2>) {
    let n = pts.len();
    let length = h * (n - 1) as f64;
    tan.clear();
    for i in 0..n {
        let s = i as f64 * h;
        let r = window.min(s).min(length - s);
        let (a, b) = if i == 0 {
            (pts[0], pts[1])
        } else if i == n - 1 {
            (pts[n - 2], pts[n - 1])
        } else {
            (at(pts, h, s - r), at(pts, h, s + r))
        };
        let d = sub(b, a);
        let norm = d[0].hypot(d[1]);
        tan.push(if norm > 1e-12 {
            [d[0] / norm, d[1] / norm]
        } else {
            [f64::NAN, f64::NAN]
        });
    }
    // Degenerate chords: the previous good tangent, then the next one for a leading run.
    let mut prev: Option<V2> = None;
    for t in tan.iter_mut() {
        if t[0].is_nan() {
            if let Some(p) = prev {
                *t = p;
            }
        } else {
            prev = Some(*t);
        }
    }
    let first = tan.iter().copied().find(|t| !t[0].is_nan());
    for t in tan.iter_mut() {
        if t[0].is_nan() {
            *t = first.unwrap_or([1.0, 0.0]);
        }
    }
}

/// Fill `kappa` with each station's signed curvature, in 1/px, from the turn between its
/// neighbours' tangents (one-sided at the ends). Positive turns towards `+n`, the side the
/// centre of curvature is on.
pub(super) fn curvature(tan: &[V2], h: f64, kappa: &mut Vec<f64>) {
    let n = tan.len();
    kappa.clear();
    for i in 0..n {
        let (a, b) = (i.saturating_sub(1), (i + 1).min(n - 1));
        let (ta, tb) = (tan[a], tan[b]);
        let turn = cross(ta, tb).atan2(dot(ta, tb));
        kappa.push(turn / ((b - a) as f64 * h));
    }
}

/// The window `[lo, hi]` a pair's centre offset may fall in at curvature `kappa`: `±reach`,
/// clipped on the concave side to `FOLD / |κ|`, so no observation asks a station to cross
/// its centre of curvature. It gates evidence only; [`step_scale`] keeps a step from
/// folding the curve.
pub(super) fn offset_window(kappa: f64, reach: f64) -> (f64, f64) {
    let fold = if kappa != 0.0 {
        FOLD / kappa.abs()
    } else {
        f64::INFINITY
    };
    if kappa > 0.0 {
        (-reach, reach.min(fold))
    } else {
        (-reach.min(fold), reach)
    }
}

/// The largest `α ≤ 1` for which moving each station by `α·d[i]` along `normals[i]`
/// cannot fold the curve:
/// - no station moves more than `FOLD` of the way to its centre of curvature,
///   `1 − κᵢ·α·dᵢ ≥ 1 − FOLD`, with `κ` from the chord tangents;
/// - every segment keeps at least `1 − FOLD` of its length along its old direction.
pub(super) fn step_scale(pts: &[V2], normals: &[V2], kappa: &[f64], d: &[f64]) -> f64 {
    let mut alpha: f64 = 1.0;
    for (&k, &di) in kappa.iter().zip(d) {
        if k * di > FOLD {
            alpha = alpha.min(FOLD / (k * di));
        }
    }
    for i in 0..pts.len().saturating_sub(1) {
        let seg = sub(pts[i + 1], pts[i]);
        let (a, b) = (normals[i], normals[i + 1]);
        let delta = [d[i + 1] * b[0] - d[i] * a[0], d[i + 1] * b[1] - d[i] * a[1]];
        let along = dot(seg, delta);
        if along < 0.0 {
            alpha = alpha.min(FOLD * dot(seg, seg) / -along);
        }
    }
    alpha
}
