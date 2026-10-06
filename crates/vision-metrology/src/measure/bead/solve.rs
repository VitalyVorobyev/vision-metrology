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
//! angular frequency `ω` by `1 / (1 + ℓ1²ω² + ℓ2⁴ω⁴)`.
//!
//! `ρ` is applied by iteratively reweighted least squares. Each weighted problem is a
//! symmetric positive definite pentadiagonal system, factored by a banded `LDLᵀ` in `O(N)`.

// Invariant 13: nalgebra has no banded solver; this one lives beside its only caller, in
// f64, and is tested against nalgebra's dense Cholesky. Invariant 20: every sum is f64.

use crate::fit::RobustLoss;

/// A diagonal guard, relative to the largest diagonal entry, that keeps a system with no
/// data and no damping positive definite.
const GUARD: f64 = 1e-9;

/// IRLS stops once no correction moves by more than this between solves, in pixels.
const IRLS_TOL: f64 = 1e-4;

/// The penalty coefficients for one pass: `λ0`, `(ℓ1/h)²` and `(ℓ2/h)⁴`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Penalty {
    pub damping: f64,
    pub tension: f64,
    pub bending: f64,
}

impl Penalty {
    /// The coefficients for tension and bending lengths `l1`, `l2` px at spacing `h` px.
    pub(super) fn new(damping: f64, l1: f64, l2: f64, h: f64) -> Self {
        Self {
            damping,
            tension: (l1 / h).powi(2),
            bending: (l2 / h).powi(4),
        }
    }
}

/// A symmetric pentadiagonal matrix and its `LDLᵀ` factors, in reusable buffers.
#[derive(Debug, Clone, Default)]
pub(super) struct Band {
    /// `A[i][i]`.
    pub diag: Vec<f64>,
    /// `A[i][i+1]`, `n − 1` entries.
    pub off1: Vec<f64>,
    /// `A[i][i+2]`, `n − 2` entries.
    pub off2: Vec<f64>,
    /// `L[i][i−1]`, at index `i`.
    l1: Vec<f64>,
    /// `L[i][i−2]`, at index `i`.
    l2: Vec<f64>,
    /// The pivots `D[i]`.
    piv: Vec<f64>,
}

impl Band {
    /// Resize to `n × n`, all zero.
    pub(super) fn zero(&mut self, n: usize) {
        for (v, len) in [
            (&mut self.diag, n),
            (&mut self.off1, n.saturating_sub(1)),
            (&mut self.off2, n.saturating_sub(2)),
        ] {
            v.clear();
            v.resize(len, 0.0);
        }
    }

    /// `diag(w) + λ0 I + (ℓ1/h)² D1ᵀD1 + (ℓ2/h)⁴ D2ᵀD2`, plus the guard.
    pub(super) fn assemble(&mut self, w: &[f64], p: Penalty) {
        let n = w.len();
        self.zero(n);
        for (a, &wi) in self.diag.iter_mut().zip(w) {
            *a = wi + p.damping;
        }
        // D1: rows e[r+1] − e[r].
        for r in 0..n.saturating_sub(1) {
            self.diag[r] += p.tension;
            self.diag[r + 1] += p.tension;
            self.off1[r] -= p.tension;
        }
        // D2: rows e[r] − 2e[r+1] + e[r+2]; the outer products of (1, −2, 1).
        for r in 0..n.saturating_sub(2) {
            self.diag[r] += p.bending;
            self.diag[r + 1] += 4.0 * p.bending;
            self.diag[r + 2] += p.bending;
            self.off1[r] -= 2.0 * p.bending;
            self.off1[r + 1] -= 2.0 * p.bending;
            self.off2[r] += p.bending;
        }
        let guard = GUARD * self.diag.iter().copied().fold(1.0, f64::max);
        for a in &mut self.diag {
            *a += guard;
        }
    }

    /// Factor `A = L D Lᵀ` in place of the factor buffers. `false` when a pivot is not
    /// positive and finite: the matrix is not positive definite.
    pub(super) fn factor(&mut self) -> bool {
        let n = self.diag.len();
        for v in [&mut self.l1, &mut self.l2, &mut self.piv] {
            v.clear();
            v.resize(n, 0.0);
        }
        for i in 0..n {
            let mut d = self.diag[i];
            if i >= 2 {
                self.l2[i] = self.off2[i - 2] / self.piv[i - 2];
                d -= self.l2[i] * self.l2[i] * self.piv[i - 2];
            }
            if i >= 1 {
                let mut a = self.off1[i - 1];
                if i >= 2 {
                    a -= self.l2[i] * self.piv[i - 2] * self.l1[i - 1];
                }
                self.l1[i] = a / self.piv[i - 1];
                d -= self.l1[i] * self.l1[i] * self.piv[i - 1];
            }
            if !(d.is_finite() && d > 0.0) {
                return false;
            }
            self.piv[i] = d;
        }
        true
    }

    /// Solve `A x = x` in place with the factors of the last [`factor`](Self::factor).
    pub(super) fn solve_in_place(&self, x: &mut [f64]) {
        let n = x.len();
        for i in 1..n {
            let mut v = x[i] - self.l1[i] * x[i - 1];
            if i >= 2 {
                v -= self.l2[i] * x[i - 2];
            }
            x[i] = v;
        }
        for (v, &d) in x.iter_mut().zip(&self.piv) {
            *v /= d;
        }
        for i in (0..n.saturating_sub(1)).rev() {
            let mut v = x[i] - self.l1[i + 1] * x[i + 1];
            if i + 2 < n {
                v -= self.l2[i + 2] * x[i + 2];
            }
            x[i] = v;
        }
    }
}

/// Reusable buffers for [`solve_offsets`].
#[derive(Debug, Clone, Default)]
pub(super) struct SolveScratch {
    pub band: Band,
    /// The last solve's weights, 0 at an invalid station.
    pub weights: Vec<f64>,
    /// The solved corrections.
    pub d: Vec<f64>,
    prev: Vec<f64>,
}

/// What one pass's solve did.
#[derive(Debug, Clone, Copy)]
pub(super) struct Solved {
    pub irls_iters: usize,
    /// RMS and largest `|d̂ − d|` over the valid stations.
    pub residual_rms: f64,
    pub residual_max: f64,
}

/// Solve for the corrections `s.d` from the observations `obs`, read where `valid`.
///
/// Up to `irls_iters` solves: least squares, then each reweighted by `loss` on the last
/// residuals (Tukey annealed from the largest residual, as the fitters do). `None` when a
/// factorisation loses positive definiteness.
pub(super) fn solve_offsets(
    obs: &[f64],
    valid: &[bool],
    p: Penalty,
    loss: RobustLoss,
    irls_iters: usize,
    s: &mut SolveScratch,
) -> Option<Solved> {
    s.weights.clear();
    s.weights
        .extend(valid.iter().map(|&v| if v { 1.0 } else { 0.0 }));
    solve_weighted(obs, p, s)?;
    let mut iters = 1;
    let max_residual = residuals(obs, valid, &s.d).1;
    if loss != RobustLoss::None {
        for k in 0..irls_iters.saturating_sub(1) {
            let step = loss.annealed(k, max_residual as f32);
            for ((w, (&o, &d)), &v) in s.weights.iter_mut().zip(obs.iter().zip(&s.d)).zip(valid) {
                *w = if v { step.weight((o - d) as f32) } else { 0.0 };
            }
            s.prev.clone_from(&s.d);
            solve_weighted(obs, p, s)?;
            iters += 1;
            let moved =
                s.d.iter()
                    .zip(&s.prev)
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            if moved < IRLS_TOL && step == loss {
                break;
            }
        }
    }
    let (residual_rms, residual_max) = residuals(obs, valid, &s.d);
    Some(Solved {
        irls_iters: iters,
        residual_rms,
        residual_max,
    })
}

/// One weighted solve with `s.weights`, into `s.d`.
fn solve_weighted(obs: &[f64], p: Penalty, s: &mut SolveScratch) -> Option<()> {
    s.band.assemble(&s.weights, p);
    if !s.band.factor() {
        return None;
    }
    s.d.clear();
    // A zero-weight station's observation is never read, whatever it holds.
    s.d.extend(
        s.weights
            .iter()
            .zip(obs)
            .map(|(&w, &o)| if w > 0.0 { w * o } else { 0.0 }),
    );
    s.band.solve_in_place(&mut s.d);
    Some(())
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
