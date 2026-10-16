use nalgebra::{Matrix4, Vector4};

use super::{TranslationTuning, plane::Plane, result::TranslationStatus};

#[derive(Clone, Copy)]
pub(super) struct Refined {
    pub p: [f64; 2],
    pub gain: f64,
    pub offset: f64,
    pub n: usize,
    pub rms: f64,
    pub max: f64,
    pub covariance: [f32; 3],
}

#[derive(Default)]
pub(super) struct Workspace {
    pub residuals: Vec<f64>,
    /// Frozen previous-image pixel indices of the current refinement.
    pub observations: Vec<u32>,
}

/// Freeze the observations: valid previous pixels whose current-image taps stay valid for
/// every shift within ±1 px of `centre`. The set then stays fixed while the shift moves, so
/// taps entering or leaving validity cannot make the objective chatter.
fn freeze(a: &Plane, b: &Plane, centre: [f64; 2], region: [usize; 4], out: &mut Vec<u32>) {
    out.clear();
    for y in region[2]..region[3] {
        for x in region[0]..region[1] {
            let i = y * a.w + x;
            if a.valid[i] && b.window_valid(x as f64 + centre[0], y as f64 + centre[1]) {
                out.push(i as u32);
            }
        }
    }
}

fn observations(
    a: &Plane,
    b: &Plane,
    p: [f64; 2],
    frozen: &[u32],
    mut f: impl FnMut(f64, f64, f64, f64),
) {
    for &i in frozen {
        let i = i as usize;
        let (x, y) = ((i % a.w) as f64, (i / a.w) as f64);
        // Within the freeze window every tap is valid; `None` cannot occur there.
        if let Some((v, gx, gy)) = b.sample(x + p[0], y + p[1]) {
            f(a.data[i] as f64, v, gx, gy);
        }
    }
}

/// Robust scale of the photometric residuals: 1.4826 × median absolute residual.
fn robust_scale(
    a: &Plane,
    b: &Plane,
    p: [f64; 2],
    photometry: [f64; 2],
    scratch: &mut Workspace,
) -> f64 {
    let Workspace {
        residuals,
        observations: frozen,
    } = scratch;
    residuals.clear();
    observations(a, b, p, frozen, |u, v, _, _| {
        residuals.push((photometry[0] * v + photometry[1] - u).abs())
    });
    let n = residuals.len();
    if n == 0 {
        return 1.0;
    }
    let (_, median, _) = residuals.select_nth_unstable_by(n / 2, f64::total_cmp);
    (*median * 1.4826).max(1e-4)
}

/// Robust Gauss–Newton for `(dx, dy, gain, offset)` over a frozen observation set.
///
/// The set is frozen about the seed, and re-frozen only if the shift leaves that ±1 px
/// window. The Huber scale is estimated at the seed and once more after the first step,
/// then fixed. Convergence tests the translation step alone.
pub(super) fn refine(
    a: &Plane,
    b: &Plane,
    seed: [f64; 2],
    region: [usize; 4],
    min_support: usize,
    tuning: &TranslationTuning,
    scratch: &mut Workspace,
) -> Result<Refined, TranslationStatus> {
    let (mut p, mut gain, mut offset) = (seed, 1.0, 0.0);
    let mut centre = seed;
    freeze(a, b, centre, region, &mut scratch.observations);
    let mut cutoff = 0.0;
    let mut converged = false;
    for iteration in 0..tuning.iterations.get() {
        if (p[0] - centre[0]).abs() >= 1.0 || (p[1] - centre[1]).abs() >= 1.0 {
            centre = p;
            freeze(a, b, centre, region, &mut scratch.observations);
        }
        if scratch.observations.len() < min_support {
            return Err(TranslationStatus::InsufficientSupport);
        }
        if iteration < 2 {
            cutoff = tuning.huber as f64 * robust_scale(a, b, p, [gain, offset], scratch);
        }
        let (h, rhs, _, _, n) = system(a, b, p, gain, offset, cutoff, &scratch.observations);
        if n < min_support {
            return Err(TranslationStatus::InsufficientSupport);
        }
        let Some(chol) = h.cholesky() else {
            return Err(TranslationStatus::LowTexture);
        };
        let inv = chol.inverse();
        // The top-left block of the full inverse is the inverse Schur complement,
        // so gain/offset degeneracy cannot masquerade as translation texture.
        let worst = 0.5 * (inv[(0, 0)] + inv[(1, 1)])
            + (0.25 * (inv[(0, 0)] - inv[(1, 1)]).powi(2) + inv[(0, 1)].powi(2)).sqrt();
        if !worst.is_finite()
            || worst <= 0.0
            || 1.0 / (worst * n as f64) < tuning.min_texture as f64
        {
            return Err(TranslationStatus::LowTexture);
        }
        let mut step = chol.solve(&rhs);
        // A bounded step avoids crossing several interpolation cells at once.
        let movement = step[0].hypot(step[1]);
        if movement > 1.0 {
            step /= movement;
        }
        p[0] -= step[0];
        p[1] -= step[1];
        gain -= step[2];
        offset -= step[3];
        if !p.iter().all(|v| v.is_finite())
            || !gain.is_finite()
            || gain <= 0.0
            || !offset.is_finite()
        {
            return Err(TranslationStatus::RefinementFailed);
        }
        if step[0].hypot(step[1]) < tuning.tolerance as f64 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(TranslationStatus::RefinementFailed);
    }
    if (p[0] - centre[0]).abs() >= 1.0 || (p[1] - centre[1]).abs() >= 1.0 {
        freeze(a, b, p, region, &mut scratch.observations);
    }
    let (h, _, sum, max, n) = system(a, b, p, gain, offset, cutoff, &scratch.observations);
    if n < min_support {
        return Err(TranslationStatus::InsufficientSupport);
    }
    let inv = h.cholesky().ok_or(TranslationStatus::LowTexture)?.inverse();
    let variance = sum / (n.saturating_sub(4).max(1)) as f64;
    let covariance = [inv[(0, 0)], inv[(0, 1)], inv[(1, 1)]].map(|v| (v * variance) as f32);
    if covariance.iter().any(|v| !v.is_finite()) {
        return Err(TranslationStatus::RefinementFailed);
    }
    Ok(Refined {
        p,
        gain,
        offset,
        n,
        rms: (sum / n as f64).sqrt(),
        max,
        covariance,
    })
}

fn system(
    a: &Plane,
    b: &Plane,
    p: [f64; 2],
    gain: f64,
    offset: f64,
    cutoff: f64,
    frozen: &[u32],
) -> (Matrix4<f64>, Vector4<f64>, f64, f64, usize) {
    let (mut h, mut rhs, mut sum, mut max, mut n) =
        (Matrix4::zeros(), Vector4::zeros(), 0.0, 0.0f64, 0usize);
    observations(a, b, p, frozen, |u, v, gx, gy| {
        let r = gain * v + offset - u;
        let w = (cutoff / r.abs().max(cutoff)).min(1.0);
        let j = Vector4::new(gain * gx, gain * gy, v, 1.0);
        h += j * j.transpose() * w;
        rhs += j * (w * r);
        sum += r * r;
        max = max.max(r.abs());
        n += 1;
    });
    (h, rhs, sum, max, n)
}
