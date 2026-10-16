//! A robust Whittaker smoother: weighted, regularised least squares on a uniformly sampled
//! sequence, solved by a banded `LDLᵀ` in `O(n)`.
//!
//! The sequence `u` minimises
//!
//! ```text
//! Σ wᵢ ρ(uᵢ − ûᵢ) + λ0 Σ uᵢ² + (ℓ1/h)² Σ (uᵢ₊₁ − uᵢ)² + (ℓ2/h)⁴ Σ (uᵢ₋₁ − 2uᵢ + uᵢ₊₁)²
//! ```
//!
//! the discretisation, at spacing `h`, of `∫ w ρ + λ0 u² + ℓ1² u′² + ℓ2⁴ u″² ds`, so the
//! answer does not depend on `h`. With `λ0 = ℓ1 = 0` it is the Whittaker smoother
//! `Σ wᵢ ρ(uᵢ − ûᵢ) + (ℓ/h)⁴ Σ (Δ²u)²`. On uniform data it passes a component of angular
//! frequency `ω` by `1 / (1 + λ0 + ℓ1²ω² + ℓ2⁴ω⁴)`.
//!
//! A weight of 0 is a gap: that sample's observation is never read, whatever it holds. A
//! heavy weight is a pin: the solution passes through that observation. `ρ` is applied by
//! iteratively reweighted least squares; each weighted problem is a symmetric positive
//! definite pentadiagonal system.
//!
//! The module depends on nothing else in the crate, so the bench and the allocation gate
//! compile it on its own (`benches/measure_numeric.rs`, `tests/measure_numeric_alloc.rs`).

// Invariant 13: nalgebra has no banded solver; this one lives beside its callers in
// `measure`, in f64, and is tested against nalgebra's dense Cholesky. Invariant 20: every
// sum is f64. Invariant 6: the buffers are reused.

/// A diagonal guard, relative to the largest diagonal entry, that keeps a system with no
/// data and no damping positive definite.
const GUARD: f64 = 1e-9;

/// The penalty coefficients: `λ0`, `(ℓ1/h)²` and `(ℓ2/h)⁴`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Penalty {
    pub damping: f64,
    pub tension: f64,
    pub bending: f64,
}

impl Penalty {
    /// The coefficients for damping `λ0`, tension and bending lengths `l1`, `l2` and
    /// spacing `h`, all lengths in the same unit.
    pub(crate) fn new(damping: f64, l1: f64, l2: f64, h: f64) -> Self {
        Self {
            damping,
            tension: (l1 / h).powi(2),
            bending: (l2 / h).powi(4),
        }
    }
}

/// A symmetric pentadiagonal matrix and its `LDLᵀ` factors, in reusable buffers.
#[derive(Debug, Clone, Default)]
pub(crate) struct Band {
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
    pub(crate) fn zero(&mut self, n: usize) {
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
    pub(crate) fn assemble(&mut self, w: &[f64], p: Penalty) {
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
    pub(crate) fn factor(&mut self) -> bool {
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
    pub(crate) fn solve_in_place(&self, x: &mut [f64]) {
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

/// The smoother: its system and buffers, reused from call to call.
#[derive(Debug, Clone, Default)]
pub(crate) struct Smoother {
    pub band: Band,
    /// The last solve's weights: the base weights times the robust ones, 0 at a gap.
    pub weights: Vec<f64>,
    /// The solution `u`.
    pub u: Vec<f64>,
    prev: Vec<f64>,
}

impl Smoother {
    /// One weighted least-squares solve with weights `w`, into [`u`](Self::u). `false`
    /// when the factorisation loses positive definiteness.
    pub(crate) fn solve(&mut self, obs: &[f64], w: &[f64], p: Penalty) -> bool {
        self.weights.clear();
        self.weights.extend_from_slice(w);
        self.solve_weighted(obs, p)
    }

    /// The robust solve: a least-squares solve with the base weights `base`, then up to
    /// `max_iters` solves reweighted by `schedule`. Returns the reweighted solves run, or
    /// `None` when a factorisation loses positive definiteness.
    ///
    /// `schedule(k, r0)` gives the robust weight function for reweighted solve `k` (from
    /// 0), with `r0` the largest least-squares residual `|ûᵢ − uᵢ|` over the weighted
    /// samples, and whether it is the final loss. Sample `i` is then weighted
    /// `baseᵢ · ψ(ûᵢ − uᵢ)` on the last solve's residual. A loss that anneals, such as a
    /// graduated Tukey, returns a wider function first. The loop stops early once no
    /// sample moves by more than `tol` between solves under the final loss.
    pub(crate) fn solve_robust<F, W>(
        &mut self,
        obs: &[f64],
        base: &[f64],
        p: Penalty,
        max_iters: usize,
        tol: f64,
        mut schedule: F,
    ) -> Option<usize>
    where
        F: FnMut(usize, f64) -> (W, bool),
        W: Fn(f64) -> f64,
    {
        debug_assert_eq!(obs.len(), base.len());
        if !self.solve(obs, base, p) {
            return None;
        }
        let r0 = max_residual(obs, base, &self.u);
        let mut iters = 0;
        for k in 0..max_iters {
            let (psi, last) = schedule(k, r0);
            for ((w, &b), (&o, &u)) in self
                .weights
                .iter_mut()
                .zip(base)
                .zip(obs.iter().zip(&self.u))
            {
                *w = if b > 0.0 { b * psi(o - u) } else { 0.0 };
            }
            self.prev.clone_from(&self.u);
            if !self.solve_weighted(obs, p) {
                return None;
            }
            iters += 1;
            let moved = self
                .u
                .iter()
                .zip(&self.prev)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            if moved < tol && last {
                break;
            }
        }
        Some(iters)
    }

    /// One weighted solve with [`weights`](Self::weights), into [`u`](Self::u).
    fn solve_weighted(&mut self, obs: &[f64], p: Penalty) -> bool {
        self.band.assemble(&self.weights, p);
        if !self.band.factor() {
            return false;
        }
        self.u.clear();
        // A zero-weight sample's observation is never read, whatever it holds.
        self.u.extend(
            self.weights
                .iter()
                .zip(obs)
                .map(|(&w, &o)| if w > 0.0 { w * o } else { 0.0 }),
        );
        self.band.solve_in_place(&mut self.u);
        true
    }
}

/// The largest `|obs − u|` over the samples with a positive weight; 0 without any.
fn max_residual(obs: &[f64], w: &[f64], u: &[f64]) -> f64 {
    obs.iter()
        .zip(w)
        .zip(u)
        .filter(|((_, w), _)| **w > 0.0)
        .fold(0.0f64, |m, ((o, _), x)| m.max((o - x).abs()))
}
