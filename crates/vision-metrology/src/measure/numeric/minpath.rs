//! An exact min-path over layered states, with max-marginals.
//!
//! A path picks one state in each of `L` layers. Every layer's states are the same
//! [`Grid`]: `planes` copies of a row-major grid of one or two dimensions. A path costs
//!
//! ```text
//! Σₖ u_k(s_k) + Σₖ T(s_k, s_k+1),   T(s, s′) = S[p][p′] + Σ_d c_d (i′_d − i_d)²
//! ```
//!
//! with the unary costs `u` that the caller fills, a switch cost `S` between planes `p` and
//! `p′`, and a quadratic step cost on each grid dimension `d`, bounded by
//! `|i′_d − i_d| ≤ B_d`. The step cost is separable over the dimensions, so the minimum
//! over a layer's predecessors is one banded pass per dimension, not a pass over every
//! pair of states: `O(L · planes · cells · (planes + B_0 + B_1))` in all.
//!
//! - **Uses.** The coupled two-boundary extraction runs over (centre, width) with a
//!   present and an absent plane: walls `l = c − w/2`, `r = c + w/2` give
//!   `Δl² + Δr² = 2Δc² + Δw²/2`, which is separable. The bead-end rim runs over one
//!   dimension, the radius, along rays, with the first and last rays pinned.
//! - **A layer with fewer states** is the same grid with `+∞` unaries where a state does
//!   not exist; a layer with none feasible makes [`solve`](LayeredPath::solve) return
//!   `None`.
//! - **Soft pins** are strong finite unaries ([`pin`](LayeredPath::pin)). A hard pin
//!   (`+∞`) can make a step-bounded path infeasible.
//! - **Max-marginals.** `M_k(s)`, the cost of the best path through state `s` at layer
//!   `k`, is `α_k(s) + β_k(s)`: the forward cost of reaching `s`, its unary included, plus
//!   the backward cost of finishing from it. `min_s M_k(s)` is the best cost at every `k`.
//! - **Ties** go to the lower state index (plane, then row-major cell): at the last layer,
//!   and for each predecessor while backtracking.
//! - **Precision.** Costs are stored in `f32`, each layer relative to its own minimum, and
//!   the minima accumulate in `f64`. A stored value is then small near the optimum
//!   whatever the path's length, and a max-marginal is exact to `f32` rounding of a few
//!   layers' costs, not of the whole path's.
//!
//! The module depends on nothing else in the crate, so the bench and the allocation gate
//! compile it on its own (`benches/measure_numeric.rs`, `tests/measure_numeric_alloc.rs`).

// Invariants 6 (scratch reused), 12 (explicit tie-breaks) and 20 (the path's cost
// accumulates in f64).

/// The states of one layer: `planes` copies of a `dims[0] × dims[1]` row-major grid. A 1-D
/// grid is `[n, 1]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Grid {
    pub planes: usize,
    pub dims: [usize; 2],
}

impl Grid {
    /// The cells of one plane.
    pub(crate) fn cells(&self) -> usize {
        self.dims[0] * self.dims[1]
    }

    /// The states of one layer.
    pub(crate) fn states(&self) -> usize {
        self.planes * self.cells()
    }

    /// The state index of cell `(i0, i1)` on `plane`.
    pub(crate) fn index(&self, plane: usize, i0: usize, i1: usize) -> usize {
        debug_assert!(plane < self.planes && i0 < self.dims[0] && i1 < self.dims[1]);
        (plane * self.dims[0] + i0) * self.dims[1] + i1
    }

    /// The plane and cell of a state index.
    pub(crate) fn coords(&self, state: usize) -> (usize, usize, usize) {
        let i1 = state % self.dims[1];
        let rest = state / self.dims[1];
        (rest / self.dims[0], rest % self.dims[0], i1)
    }
}

/// What a step between adjacent layers costs.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Steps {
    /// `c_d` of the step cost `Σ_d c_d Δ_d²`, per grid dimension; finite, non-negative.
    pub coef: [f32; 2],
    /// The largest `|Δ_d|`, per grid dimension.
    pub band: [usize; 2],
    /// `switch[p · planes + p′]`: the cost of a step from plane `p` to plane `p′`,
    /// `planes²` entries, finite and non-negative. The diagonal is normally 0.
    pub switch: Vec<f32>,
}

/// The solver and every buffer it uses, reused from call to call.
#[derive(Debug, Clone, Default)]
pub(crate) struct LayeredPath {
    grid: Grid,
    layers: usize,
    steps: Steps,
    /// `pen[d][k] = c_d · k²`, for `k ≤ B_d`.
    pen: [Vec<f32>; 2],
    /// `layers × states`.
    unary: Vec<f32>,
    /// The forward costs, each layer less `alpha_off`; its minimum is 0.
    alpha: Vec<f32>,
    alpha_off: Vec<f64>,
    /// The backward costs, each layer less `beta_off`.
    beta: Vec<f32>,
    beta_off: Vec<f64>,
    /// One layer: the backward pass's `β + u`.
    acc: Vec<f32>,
    /// One plane: a band pass's source.
    tmp: Vec<f32>,
    path: Vec<usize>,
    solved: bool,
    /// Whether the last solve ran the backward pass.
    marginals: bool,
    /// `layers × planes`: whether a plane has a finite unary in a layer.
    live: Vec<bool>,
}

impl LayeredPath {
    /// Size for `layers` layers on `grid` with `steps`, every unary 0. Allocates only when
    /// the problem outgrows the buffers.
    ///
    /// # Panics
    /// On an empty problem (no layer, plane or cell), or a switch table that is not
    /// `planes²` long.
    pub(crate) fn reset(&mut self, layers: usize, grid: Grid, steps: &Steps) {
        assert!(layers > 0 && grid.states() > 0, "an empty path problem");
        assert_eq!(steps.switch.len(), grid.planes * grid.planes);
        debug_assert!(
            steps
                .coef
                .iter()
                .chain(&steps.switch)
                .all(|&c| c >= 0.0 && c.is_finite()),
            "step costs are finite and non-negative"
        );
        self.grid = grid;
        self.layers = layers;
        // Field by field: a derived `clone_from` would reallocate the switch table.
        self.steps.coef = steps.coef;
        self.steps.switch.clone_from(&steps.switch);
        for d in 0..2 {
            // A step longer than the grid never happens.
            self.steps.band[d] = steps.band[d].min(grid.dims[d] - 1);
            let c = steps.coef[d];
            self.pen[d].clear();
            self.pen[d].extend((0..=self.steps.band[d]).map(|k| c * (k * k) as f32));
        }
        let n = layers * grid.states();
        for v in [&mut self.unary, &mut self.alpha, &mut self.beta] {
            v.clear();
            v.resize(n, 0.0);
        }
        for v in [&mut self.alpha_off, &mut self.beta_off] {
            v.clear();
            v.resize(layers, 0.0);
        }
        self.acc.clear();
        self.acc.resize(grid.states(), 0.0);
        self.tmp.clear();
        self.tmp.resize(grid.cells(), 0.0);
        self.path.clear();
        self.path.resize(layers, 0);
        self.solved = false;
    }

    /// The grid of the last [`reset`](Self::reset).
    pub(crate) fn grid(&self) -> Grid {
        self.grid
    }

    /// The layer count of the last [`reset`](Self::reset).
    pub(crate) fn layers(&self) -> usize {
        self.layers
    }

    /// Layer `k`'s unary costs, indexed by [`Grid::index`], to fill. Never NaN; `+∞` marks
    /// a state that does not exist.
    pub(crate) fn unary_mut(&mut self, k: usize) -> &mut [f32] {
        self.solved = false;
        let s = self.grid.states();
        &mut self.unary[k * s..(k + 1) * s]
    }

    /// Layer `k`'s unary costs.
    pub(crate) fn unary(&self, k: usize) -> &[f32] {
        let s = self.grid.states();
        &self.unary[k * s..(k + 1) * s]
    }

    /// A soft pin: add `cost` to every state of layer `k` that is not on `plane` within
    /// the inclusive cell box `lo..=hi`.
    pub(crate) fn pin(
        &mut self,
        k: usize,
        plane: usize,
        lo: [usize; 2],
        hi: [usize; 2],
        cost: f32,
    ) {
        let g = self.grid;
        let u = self.unary_mut(k);
        for (s, v) in u.iter_mut().enumerate() {
            let (p, i0, i1) = g.coords(s);
            let inside =
                p == plane && (lo[0]..=hi[0]).contains(&i0) && (lo[1]..=hi[1]).contains(&i1);
            if !inside {
                *v += cost;
            }
        }
    }

    /// Solve: the forward and backward passes, then the best path by backtracking.
    /// Returns the best path's cost, or `None` when no path has a finite cost.
    pub(crate) fn solve(&mut self) -> Option<f64> {
        self.solve_with(true)
    }

    /// [`solve`](Self::solve), with the backward pass only when `marginals`: the path and
    /// its cost need the forward pass alone, the max-marginals both. Without it, the
    /// max-marginals are not available until the next solve that computes them.
    pub(crate) fn solve_with(&mut self, marginals: bool) -> Option<f64> {
        debug_assert!(!self.unary.iter().any(|v| v.is_nan()), "a NaN unary");
        self.solved = false;
        self.marginals = false;
        let (g, s, n) = (self.grid, self.grid.states(), self.layers);
        let (cells, planes) = (g.cells(), g.planes);
        self.live.clear();
        self.live.extend(
            self.unary
                .chunks_exact(cells)
                .map(|plane| plane.iter().any(|u| u.is_finite())),
        );
        let live = |k: usize| &self.live[k * planes..(k + 1) * planes];

        // Forward: α_0 = u_0, α_k = min over predecessors (α_{k−1} + T) + u_k.
        self.alpha[..s].copy_from_slice(&self.unary[..s]);
        let mut off = normalise(&mut self.alpha[..s])?;
        self.alpha_off[0] = off;
        for k in 1..n {
            let (done, rest) = self.alpha.split_at_mut(k * s);
            let dst = &mut rest[..s];
            transfer(
                &done[(k - 1) * s..],
                dst,
                g,
                &self.steps,
                &self.pen,
                &mut self.tmp,
                false,
                (live(k - 1), live(k)),
            );
            for (a, &u) in dst.iter_mut().zip(&self.unary[k * s..(k + 1) * s]) {
                *a += u;
            }
            off += normalise(dst)?;
            self.alpha_off[k] = off;
        }
        let best = off;

        // Backward: β_{L−1} = 0, β_k = min over successors (T + u_{k+1} + β_{k+1}).
        self.beta[(n - 1) * s..].fill(0.0);
        self.beta_off[n - 1] = 0.0;
        let mut off = 0.0;
        for k in (0..n - 1).rev().filter(|_| marginals) {
            for ((a, &b), &u) in self
                .acc
                .iter_mut()
                .zip(&self.beta[(k + 1) * s..(k + 2) * s])
                .zip(&self.unary[(k + 1) * s..(k + 2) * s])
            {
                *a = b + u;
            }
            let dst = &mut self.beta[k * s..(k + 1) * s];
            transfer(
                &self.acc,
                dst,
                g,
                &self.steps,
                &self.pen,
                &mut self.tmp,
                true,
                (live(k + 1), live(k)),
            );
            off += normalise(dst)?;
            self.beta_off[k] = off;
        }

        // Backtrack from the last layer's cheapest state.
        self.path[n - 1] = argmin(&self.alpha[(n - 1) * s..]);
        for k in (0..n - 1).rev() {
            self.path[k] = self.predecessor(k, self.path[k + 1]);
        }
        self.solved = true;
        self.marginals = marginals;
        Some(best)
    }

    /// The cheapest predecessor in layer `k` of state `next` in layer `k + 1`, with the
    /// forward pass's arithmetic, so it is the state that pass minimised over.
    fn predecessor(&self, k: usize, next: usize) -> usize {
        let g = self.grid;
        let (q, a, b) = g.coords(next);
        let alpha = &self.alpha[k * g.states()..(k + 1) * g.states()];
        let [b0, b1] = self.steps.band;
        let (a_lo, a_hi) = (a.saturating_sub(b0), (a + b0).min(g.dims[0] - 1));
        let (b_lo, b_hi) = (b.saturating_sub(b1), (b + b1).min(g.dims[1] - 1));
        let (mut best, mut arg) = (f32::INFINITY, usize::MAX);
        for p in 0..g.planes {
            let sw = self.steps.switch[p * g.planes + q];
            for i0 in a_lo..=a_hi {
                let p0 = self.pen[0][i0.abs_diff(a)];
                for i1 in b_lo..=b_hi {
                    let st = g.index(p, i0, i1);
                    let c = ((alpha[st] + sw) + self.pen[1][i1.abs_diff(b)]) + p0;
                    if c < best || arg == usize::MAX {
                        best = c;
                        arg = st;
                    }
                }
            }
        }
        arg
    }

    /// The best path's state per layer, from the last [`solve`](Self::solve) that
    /// returned a cost.
    pub(crate) fn path(&self) -> &[usize] {
        debug_assert!(self.solved, "no solved path");
        &self.path
    }

    /// Layer `k`'s max-marginals in parts: `M_k(s) = off + (α[s] + β[s])`, for a caller
    /// that scans a layer. The parts are relative to the layer's own minima, so their `f32`
    /// sum is exact to the rounding of a few layers' costs.
    pub(crate) fn marginal_parts(&self, k: usize) -> (f64, &[f32], &[f32]) {
        debug_assert!(self.solved && self.marginals, "no max-marginals");
        let s = self.grid.states();
        (
            self.alpha_off[k] + self.beta_off[k],
            &self.alpha[k * s..(k + 1) * s],
            &self.beta[k * s..(k + 1) * s],
        )
    }

    /// `M_k(s)`: the cost of the best path through state `s` at layer `k`; `+∞` when no
    /// path through it is finite.
    pub(crate) fn max_marginal(&self, k: usize, state: usize) -> f64 {
        debug_assert!(self.solved && self.marginals, "no max-marginals");
        let i = k * self.grid.states() + state;
        self.alpha_off[k] + self.beta_off[k] + f64::from(self.alpha[i]) + f64::from(self.beta[i])
    }
}

/// Subtract the slice's minimum from it, and return that minimum; `None` when no entry is
/// finite.
fn normalise(v: &mut [f32]) -> Option<f64> {
    let m = v.iter().copied().fold(f32::INFINITY, f32::min);
    if !m.is_finite() {
        return None;
    }
    for x in v.iter_mut() {
        *x -= m;
    }
    Some(f64::from(m))
}

/// The index of the slice's minimum, the first on a tie.
fn argmin(v: &[f32]) -> usize {
    let mut arg = 0;
    for (i, &x) in v.iter().enumerate() {
        if x < v[arg] {
            arg = i;
        }
    }
    arg
}

/// `dst(q, ·) = bandmin(min_p src(p, ·) + S[p][q])`, or with `Sᵀ` when `transpose` (the
/// backward pass). The step cost is symmetric in `Δ`, so one band pass serves both ways.
///
/// A plane is *live* in a layer when one of its unaries there is finite. A source plane
/// that is not live is `+∞` throughout and is skipped; a destination plane that is not
/// live is set to `+∞`, which is exact wherever it is read: its forward cost adds its own
/// `+∞` unaries, and its backward cost meets them in every max-marginal and earlier layer.
#[allow(clippy::too_many_arguments)]
fn transfer(
    src: &[f32],
    dst: &mut [f32],
    g: Grid,
    steps: &Steps,
    pen: &[Vec<f32>; 2],
    tmp: &mut [f32],
    transpose: bool,
    (src_live, dst_live): (&[bool], &[bool]),
) {
    let cells = g.cells();
    let sw = |p: usize, q: usize| {
        if transpose {
            steps.switch[q * g.planes + p]
        } else {
            steps.switch[p * g.planes + q]
        }
    };
    for q in 0..g.planes {
        let out = &mut dst[q * cells..(q + 1) * cells];
        out.fill(f32::INFINITY);
        if !dst_live[q] {
            continue;
        }
        for p in (0..g.planes).filter(|&p| src_live[p]) {
            let sp = sw(p, q);
            for (o, &v) in out.iter_mut().zip(&src[p * cells..(p + 1) * cells]) {
                *o = o.min(v + sp);
            }
        }
        band_min(out, tmp, g.dims, pen);
    }
}

/// `f(i0, i1) ← min over |Δ_d| ≤ B_d of f(i0 + Δ0, i1 + Δ1) + pen0[|Δ0|] + pen1[|Δ1|]`,
/// as a pass along dimension 1 and then one along dimension 0. On a tie the value is the
/// same whichever candidate gives it, so `f32::min` is exact here.
fn band_min(f: &mut [f32], tmp: &mut [f32], dims: [usize; 2], pen: &[Vec<f32>; 2]) {
    let n1 = dims[1];
    tmp.copy_from_slice(f);
    if pen[1].len() > 1 {
        for (row, src) in f.chunks_exact_mut(n1).zip(tmp.chunks_exact(n1)) {
            for (d, &c) in pen[1].iter().enumerate().skip(1) {
                min_shifted(&mut row[d..], &src[..n1 - d], c);
                min_shifted(&mut row[..n1 - d], &src[d..], c);
            }
        }
        tmp.copy_from_slice(f);
    }
    // Along dimension 0 a shift by `d` rows is a shift by `d · n1` in the flat plane.
    let len = f.len();
    for (d, &c) in pen[0].iter().enumerate().skip(1) {
        let k = d * n1;
        min_shifted(&mut f[k..], &tmp[..len - k], c);
        min_shifted(&mut f[..len - k], &tmp[k..], c);
    }
}

/// `dst ← min(dst, src + c)`, elementwise.
#[inline]
fn min_shifted(dst: &mut [f32], src: &[f32], c: f32) {
    for (o, &v) in dst.iter_mut().zip(src) {
        *o = o.min(v + c);
    }
}
