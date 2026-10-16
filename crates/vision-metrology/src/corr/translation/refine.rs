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
}

fn observations(
    a: &Plane,
    b: &Plane,
    p: [f64; 2],
    region: [usize; 4],
    mut f: impl FnMut(f64, f64, f64, f64),
) {
    for y in region[2]..region[3] {
        for x in region[0]..region[1] {
            let i = y * a.w + x;
            if !a.valid[i] {
                continue;
            }
            if let Some((v, gx, gy)) = b.sample(x as f64 + p[0], y as f64 + p[1]) {
                f(a.data[i] as f64, v, gx, gy);
            }
        }
    }
}

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
    let mut converged = false;
    for _ in 0..tuning.iterations.get() {
        let (h, rhs, _, _, n) = system(a, b, p, gain, offset, region, tuning, scratch);
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
        if step[0].hypot(step[1]) < tuning.tolerance as f64
            && step[2].abs() < tuning.tolerance as f64
            && step[3].abs() < tuning.tolerance as f64
        {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(TranslationStatus::RefinementFailed);
    }
    let (h, _, sum, max, n) = system(a, b, p, gain, offset, region, tuning, scratch);
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

#[allow(clippy::too_many_arguments)]
fn system(
    a: &Plane,
    b: &Plane,
    p: [f64; 2],
    gain: f64,
    offset: f64,
    region: [usize; 4],
    tuning: &TranslationTuning,
    scratch: &mut Workspace,
) -> (Matrix4<f64>, Vector4<f64>, f64, f64, usize) {
    scratch.residuals.clear();
    observations(a, b, p, region, |u, v, _, _| {
        scratch.residuals.push((gain * v + offset - u).abs())
    });
    let n = scratch.residuals.len();
    let scale = if n == 0 {
        1.0
    } else {
        let (_, median, _) = scratch
            .residuals
            .select_nth_unstable_by(n / 2, f64::total_cmp);
        (*median * 1.4826).max(1e-4)
    };
    let cutoff = tuning.huber as f64 * scale;
    let (mut h, mut rhs, mut sum, mut max) = (Matrix4::zeros(), Vector4::zeros(), 0.0, 0.0f64);
    observations(a, b, p, region, |u, v, gx, gy| {
        let r = gain * v + offset - u;
        let w = (cutoff / r.abs().max(cutoff)).min(1.0);
        let j = Vector4::new(gain * gx, gain * gy, v, 1.0);
        h += j * j.transpose() * w;
        rhs += j * (w * r);
        sum += r * r;
        max = max.max(r.abs());
    });
    (h, rhs, sum, max, n)
}
