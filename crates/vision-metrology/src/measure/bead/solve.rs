//! The regularised normal-offset solve.
//!
//! Station `i` observed the bead's centre at offset `d̂ᵢ` along its normal (or not at all,
//! weight 0). The corrections `d` minimise
//!
//! ```text
//! Σ wᵢ ρ(dᵢ − d̂ᵢ) + λ0 Σ dᵢ² + (ℓ1/h)² Σ (dᵢ₊₁ − dᵢ)² + (ℓ2/h)⁴ Σ (dᵢ₋₁ − 2dᵢ + dᵢ₊₁)²
//! ```
//!
//! the discretisation, at station spacing `h`, of `∫ w ρ + λ0 d² + ℓ1² d′² + ℓ2⁴ d″² ds`, so
//! the answer does not depend on `h`. On uniform data the correction passes a component of
//! angular frequency `ω` by `1 / (1 + λ0 + ℓ1²ω² + ℓ2⁴ω⁴)`.
//!
//! `ρ` is applied by iteratively reweighted least squares. Each weighted problem is a
//! symmetric positive definite pentadiagonal system, factored by a banded `LDLᵀ` in `O(N)`:
//! the shared smoother of `measure::numeric::whittaker` (ADR-0020), with 0/1 base weights.

// Invariant 20: every sum is f64.

use crate::fit::RobustLoss;
pub(super) use crate::measure::numeric::whittaker::Penalty;
use crate::measure::numeric::whittaker::Smoother;

/// IRLS stops once no correction moves by more than this between solves, in pixels.
const IRLS_TOL: f64 = 1e-4;

/// Reusable buffers for [`solve_offsets`].
#[derive(Debug, Clone, Default)]
pub(super) struct SolveScratch {
    /// The smoother; its `weights` are the last solve's, 0 at an invalid station, and its
    /// `u` the solved corrections.
    pub smooth: Smoother,
    /// The base weights: 1 at a valid station, 0 elsewhere.
    base: Vec<f64>,
}

impl SolveScratch {
    /// The solved corrections.
    pub(super) fn d(&self) -> &[f64] {
        &self.smooth.u
    }

    /// The last solve's weights.
    pub(super) fn weights(&self) -> &[f64] {
        &self.smooth.weights
    }
}

/// What one pass's solve did.
#[derive(Debug, Clone, Copy)]
pub(super) struct Solved {
    /// Reweighted solves run after the first, least-squares one.
    pub irls_iters: usize,
    /// RMS and largest `|d̂ − d|` over the valid stations.
    pub residual_rms: f64,
    pub residual_max: f64,
}

/// Solve for the corrections `s.d()` from the observations `obs`, read where `valid`.
///
/// A least-squares solve, then up to `irls_iters` solves reweighted by `loss` on the last
/// residuals, as the fitters do: Tukey anneals from the largest least-squares residual
/// down to its constant, and the loop stops early once the corrections settle at the
/// final constant. `RobustLoss::None` stops after the first solve. `None` when a
/// factorisation loses positive definiteness.
pub(super) fn solve_offsets(
    obs: &[f64],
    valid: &[bool],
    p: Penalty,
    loss: RobustLoss,
    irls_iters: usize,
    s: &mut SolveScratch,
) -> Option<Solved> {
    s.base.clear();
    s.base
        .extend(valid.iter().map(|&v| if v { 1.0 } else { 0.0 }));
    let reweighted = if loss == RobustLoss::None {
        0
    } else {
        irls_iters
    };
    let iters = s
        .smooth
        .solve_robust(obs, &s.base, p, reweighted, IRLS_TOL, |k, r0| {
            let step = loss.annealed(k, r0 as f32);
            (move |r: f64| step.weight(r as f32), step == loss)
        })?;
    let (residual_rms, residual_max) = residuals(obs, valid, s.d());
    Some(Solved {
        irls_iters: iters,
        residual_rms,
        residual_max,
    })
}

/// RMS and largest `|obs − d|` over the valid stations; zeros without any.
fn residuals(obs: &[f64], valid: &[bool], d: &[f64]) -> (f64, f64) {
    let (mut sum, mut max, mut n) = (0.0f64, 0.0f64, 0usize);
    for ((&o, &x), &v) in obs.iter().zip(d).zip(valid) {
        if v {
            let r = o - x;
            sum += r * r;
            max = max.max(r.abs());
            n += 1;
        }
    }
    if n == 0 {
        (0.0, 0.0)
    } else {
        ((sum / n as f64).sqrt(), max)
    }
}
