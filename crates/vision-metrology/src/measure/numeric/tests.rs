//! The min-path against brute force, and the smoother against dense solves.

use nalgebra::{DMatrix, DVector};

use super::minpath::{Grid, LayeredPath, Steps};
use super::whittaker::{Band, Penalty, Smoother};

/// A seeded LCG in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.next() * n as f64) as usize).min(n - 1)
    }
}

// ── the min-path ───────────────────────────────────────────────────────────

/// One step's cost in f64, `+∞` past the band.
fn step_cost(g: Grid, st: &Steps, from: usize, to: usize) -> f64 {
    let (p, a0, a1) = g.coords(from);
    let (q, b0, b1) = g.coords(to);
    let (d0, d1) = (a0.abs_diff(b0), a1.abs_diff(b1));
    if d0 > st.band[0] || d1 > st.band[1] {
        return f64::INFINITY;
    }
    f64::from(st.switch[p * g.planes + q])
        + f64::from(st.coef[0]) * (d0 * d0) as f64
        + f64::from(st.coef[1]) * (d1 * d1) as f64
}

/// Every path's cost, by enumeration: the best cost and path, the runner-up's cost, and
/// the max-marginals `layers × states`.
struct Brute {
    best: f64,
    second: f64,
    path: Vec<usize>,
    marginals: Vec<f64>,
}

fn brute(dp: &LayeredPath, st: &Steps) -> Brute {
    let (g, n) = (dp.grid(), dp.layers());
    let s = g.states();
    let mut out = Brute {
        best: f64::INFINITY,
        second: f64::INFINITY,
        path: vec![0; n],
        marginals: vec![f64::INFINITY; n * s],
    };
    let mut path = vec![0usize; n];
    loop {
        let mut cost = f64::from(dp.unary(0)[path[0]]);
        for k in 1..n {
            cost += step_cost(g, st, path[k - 1], path[k]) + f64::from(dp.unary(k)[path[k]]);
        }
        if cost.is_finite() {
            if cost < out.best {
                out.second = out.best;
                out.best = cost;
                out.path.clone_from(&path);
            } else if cost < out.second {
                out.second = cost;
            }
            for (k, &p) in path.iter().enumerate() {
                let m = &mut out.marginals[k * s + p];
                *m = m.min(cost);
            }
        }
        // The next path, as an odometer over the layers.
        let mut k = 0;
        loop {
            if k == n {
                return out;
            }
            path[k] += 1;
            if path[k] < s {
                break;
            }
            path[k] = 0;
            k += 1;
        }
    }
}

/// A random instance: grid, steps and unaries, some states missing and some pinned.
fn random_instance(rng: &mut Lcg, dp: &mut LayeredPath) -> Steps {
    let planes = 1 + rng.below(2);
    let two_d = rng.next() < 0.5;
    let dims = if two_d {
        [2 + rng.below(2), 2 + rng.below(2)]
    } else {
        [3 + rng.below(4), 1]
    };
    let layers = 2 + rng.below(if planes * dims[0] * dims[1] > 8 { 2 } else { 3 });
    let grid = Grid { planes, dims };
    let mut switch = vec![0.0f32; planes * planes];
    for (i, v) in switch.iter_mut().enumerate() {
        if i % (planes + 1) != 0 {
            *v = (3.0 * rng.next()) as f32;
        }
    }
    let steps = Steps {
        coef: [(2.0 * rng.next()) as f32, (2.0 * rng.next()) as f32],
        band: [1 + rng.below(3), 1 + rng.below(2)],
        switch,
    };
    dp.reset(layers, grid, &steps);
    // Now and then most states are missing, so some instances have no path.
    let missing = if rng.next() < 0.15 { 0.6 } else { 0.1 };
    for k in 0..layers {
        for u in dp.unary_mut(k) {
            *u = if rng.next() < missing {
                f32::INFINITY
            } else {
                (10.0 * rng.next() - 5.0) as f32
            };
        }
    }
    // Now and then a whole plane is missing from a layer, which the solver skips.
    if planes > 1 && rng.next() < 0.4 {
        let (k, p) = (rng.below(layers), rng.below(planes));
        let cells = dims[0] * dims[1];
        dp.unary_mut(k)[p * cells..(p + 1) * cells].fill(f32::INFINITY);
    }
    if rng.next() < 0.5 {
        let k = rng.below(layers);
        let (i0, i1) = (rng.below(dims[0]), rng.below(dims[1]));
        dp.pin(k, rng.below(planes), [i0, i1], [i0, i1], 50.0);
    }
    steps
}

#[test]
fn the_path_and_max_marginals_equal_brute_force() {
    let mut rng = Lcg(83);
    let mut dp = LayeredPath::default();
    let (mut checked, mut paths, mut infeasible) = (0, 0, 0);
    let mut worst = 0.0f64;
    for _ in 0..400 {
        let steps = random_instance(&mut rng, &mut dp);
        let want = brute(&dp, &steps);
        let got = dp.solve();
        let Some(best) = got else {
            assert!(
                want.best.is_infinite(),
                "the DP found no path, brute force {}",
                want.best
            );
            infeasible += 1;
            continue;
        };
        let tol = |x: f64| 1e-4 * (1.0 + x.abs());
        assert!(
            (best - want.best).abs() < tol(want.best),
            "best {best} vs {}",
            want.best
        );
        // The path is the brute-force one wherever the optimum is unique.
        if want.second - want.best > 1e-3 {
            assert_eq!(dp.path(), &want.path[..]);
            paths += 1;
        }
        let s = dp.grid().states();
        for k in 0..dp.layers() {
            for st in 0..s {
                let (m, w) = (dp.max_marginal(k, st), want.marginals[k * s + st]);
                if w.is_infinite() {
                    assert!(m.is_infinite(), "layer {k} state {st}: {m}, brute force ∞");
                } else {
                    worst = worst.max((m - w).abs() / (1.0 + w.abs()));
                    assert!((m - w).abs() < tol(w), "layer {k} state {st}: {m} vs {w}");
                }
            }
        }
        checked += 1;
    }
    eprintln!(
        "{checked} instances, {paths} unique paths, {infeasible} infeasible; worst {worst:e}"
    );
    assert!(checked > 300 && paths > 250 && infeasible > 5);
}

#[test]
fn the_forward_pass_alone_gives_the_same_path() {
    let mut rng = Lcg(19);
    let mut dp = LayeredPath::default();
    for _ in 0..200 {
        random_instance(&mut rng, &mut dp);
        let full = dp.solve().map(|c| (c, dp.path().to_vec()));
        let fwd = dp.solve_with(false).map(|c| (c, dp.path().to_vec()));
        assert_eq!(full, fwd);
    }
}

#[test]
fn ties_go_to_the_lower_index_and_repeat_bit_for_bit() {
    let mut dp = LayeredPath::default();
    let steps = Steps {
        coef: [1.0, 1.0],
        band: [1, 1],
        switch: vec![0.0, 1.0, 1.0, 0.0],
    };
    dp.reset(
        5,
        Grid {
            planes: 2,
            dims: [3, 2],
        },
        &steps,
    );
    assert_eq!(dp.solve(), Some(0.0));
    assert_eq!(dp.path(), &[0; 5]);

    // A random instance solved twice gives the same bits.
    let mut rng = Lcg(5);
    let steps = random_instance(&mut rng, &mut dp);
    let first = dp.solve();
    let marg: Vec<f64> = (0..dp.grid().states())
        .map(|s| dp.max_marginal(0, s))
        .collect();
    let path = dp.path().to_vec();
    let mut again = LayeredPath::default();
    again.reset(dp.layers(), dp.grid(), &steps);
    for k in 0..dp.layers() {
        again.unary_mut(k).copy_from_slice(dp.unary(k));
    }
    assert_eq!(again.solve().map(f64::to_bits), first.map(f64::to_bits));
    assert_eq!(again.path(), &path[..]);
    for (s, m) in marg.iter().enumerate() {
        assert_eq!(again.max_marginal(0, s).to_bits(), m.to_bits());
    }
}

#[test]
fn a_hard_constraint_can_make_the_path_infeasible() {
    // Layer 0 only at cell 0, layer 1 only at cell 5, steps of at most 2 cells.
    let mut dp = LayeredPath::default();
    let steps = Steps {
        coef: [0.1, 0.0],
        band: [2, 0],
        switch: vec![0.0],
    };
    dp.reset(
        2,
        Grid {
            planes: 1,
            dims: [6, 1],
        },
        &steps,
    );
    dp.unary_mut(0)[1..].fill(f32::INFINITY);
    dp.unary_mut(1)[..5].fill(f32::INFINITY);
    assert_eq!(dp.solve(), None);
    // As soft pins it has a path, which breaks the weaker pin.
    dp.reset(
        2,
        Grid {
            planes: 1,
            dims: [6, 1],
        },
        &steps,
    );
    dp.pin(0, 0, [0, 0], [0, 0], 100.0);
    dp.pin(1, 0, [5, 0], [5, 0], 50.0);
    assert_eq!(dp.solve(), Some(50.0));
    assert_eq!(dp.path(), &[0, 0]);
}

/// The bead-end rim: a 1-D radius state over rays across a half disc, first and last rays
/// pinned to the chord's half length.
#[test]
fn a_pinned_polar_rim_follows_a_half_ellipse() {
    let (rays, step) = (61usize, 0.25f64);
    let (half_chord, depth) = (10.0f64, 14.0f64);
    let radii = (2.0 * depth / step) as usize + 1;
    let mut dp = LayeredPath::default();
    let steps = Steps {
        coef: [0.05, 0.0],
        band: [8, 0],
        switch: vec![0.0],
    };
    dp.reset(
        rays,
        Grid {
            planes: 1,
            dims: [radii, 1],
        },
        &steps,
    );
    // The rim of a half ellipse with semi-axes half_chord along the chord and depth out.
    let truth = |k: usize| {
        let a = std::f64::consts::PI * k as f64 / (rays - 1) as f64;
        let (c, s) = (a.cos(), a.sin());
        1.0 / ((c / half_chord).powi(2) + (s / depth).powi(2)).sqrt()
    };
    for k in 0..rays {
        let r = truth(k);
        for (i, u) in dp.unary_mut(k).iter_mut().enumerate() {
            // A soft-saturated edge response, blurred over a pixel, and a weaker distractor
            // ring 4 px further out.
            let x = i as f64 * step;
            let e = (-((x - r) / 1.0).powi(2)).exp() + 0.6 * (-((x - r - 4.0) / 1.0).powi(2)).exp();
            *u = -e as f32;
        }
    }
    let pinned = (half_chord / step).round() as usize;
    for k in [0, rays - 1] {
        dp.pin(k, 0, [pinned - 1, 0], [pinned + 1, 0], 1e3);
    }
    dp.solve().expect("a rim");
    let mut worst = 0.0f64;
    for (k, &s) in dp.path().iter().enumerate() {
        let got = dp.grid().coords(s).1 as f64 * step;
        worst = worst.max((got - truth(k)).abs());
    }
    let ends = [dp.path()[0], dp.path()[rays - 1]];
    assert!(
        ends.iter().all(|&s| s.abs_diff(pinned) <= 1),
        "ends {ends:?}"
    );
    assert!(worst <= step, "worst {worst} px");
}

// ── the banded solve ───────────────────────────────────────────────────────

fn dense(b: &Band) -> DMatrix<f64> {
    let n = b.diag.len();
    DMatrix::from_fn(n, n, |r, c| match r.abs_diff(c) {
        0 => b.diag[r],
        1 => b.off1[r.min(c)],
        2 => b.off2[r.min(c)],
        _ => 0.0,
    })
}

/// `x` from the band solver and from nalgebra's dense Cholesky, `‖Δ‖ / ‖x‖`.
fn band_vs_dense(band: &mut Band, rhs: &[f64]) -> f64 {
    let want = dense(band)
        .cholesky()
        .expect("SPD")
        .solve(&DVector::from_column_slice(rhs));
    assert!(band.factor());
    let mut got = rhs.to_vec();
    band.solve_in_place(&mut got);
    rel_diff(&got, want.as_slice())
}

fn rel_diff(got: &[f64], want: &[f64]) -> f64 {
    let diff: f64 = got.iter().zip(want).map(|(a, b)| (a - b).powi(2)).sum();
    let norm: f64 = want.iter().map(|b| b * b).sum();
    (diff / norm).sqrt()
}

#[test]
fn the_band_solver_matches_a_dense_cholesky() {
    let mut rng = Lcg(7);
    let mut band = Band::default();
    let mut worst = 0.0f64;
    for &n in &[1usize, 2, 3, 4, 5, 17, 64, 300] {
        for _ in 0..5 {
            // Diagonally dominant: symmetric positive definite.
            band.zero(n);
            for v in band.off1.iter_mut().chain(band.off2.iter_mut()) {
                *v = 2.0 * rng.next() - 1.0;
            }
            for d in &mut band.diag {
                *d = 4.5 + 4.0 * rng.next();
            }
            let rhs: Vec<f64> = (0..n).map(|_| 20.0 * rng.next() - 10.0).collect();
            worst = worst.max(band_vs_dense(&mut band, &rhs));

            // Data weights, damping, tension and bending.
            let w: Vec<f64> = (0..n).map(|_| 0.5 + 0.5 * rng.next()).collect();
            let p = Penalty::new(0.1 * rng.next(), 2.0, 8.0, 2.0 + 4.0 * rng.next());
            band.assemble(&w, p);
            worst = worst.max(band_vs_dense(&mut band, &rhs));
        }
    }
    assert!(worst < 1e-12, "band vs dense: relative error {worst:e}");
}

#[test]
fn a_non_positive_pivot_is_reported() {
    let mut band = Band::default();
    band.zero(3);
    band.diag.copy_from_slice(&[1.0, 1.0, 1.0]);
    band.off1.copy_from_slice(&[2.0, 0.0]);
    assert!(!band.factor());
}

// ── the smoother ───────────────────────────────────────────────────────────

/// The normal equations of the weighted problem, built densely from the difference
/// matrices, and solved by nalgebra.
fn dense_smooth(obs: &[f64], w: &[f64], p: Penalty) -> Vec<f64> {
    let n = obs.len();
    let d1 = DMatrix::from_fn(n.saturating_sub(1), n, |r, c| match c.wrapping_sub(r) {
        0 => -1.0,
        1 => 1.0,
        _ => 0.0,
    });
    let d2 = DMatrix::from_fn(n.saturating_sub(2), n, |r, c| match c.wrapping_sub(r) {
        0 | 2 => 1.0,
        1 => -2.0,
        _ => 0.0,
    });
    let mut a = DMatrix::from_diagonal(&DVector::from_column_slice(w))
        + DMatrix::identity(n, n) * p.damping
        + d1.transpose() * &d1 * p.tension
        + d2.transpose() * &d2 * p.bending;
    let guard = 1e-9 * (0..n).map(|i| a[(i, i)]).fold(1.0, f64::max);
    for i in 0..n {
        a[(i, i)] += guard;
    }
    let rhs = DVector::from_iterator(
        n,
        w.iter()
            .zip(obs)
            .map(|(&w, &o)| if w > 0.0 { w * o } else { 0.0 }),
    );
    a.cholesky().expect("SPD").solve(&rhs).as_slice().to_vec()
}

/// A Huber weight of scale `k`.
fn huber(k: f64) -> impl Fn(f64) -> f64 {
    move |r: f64| if r.abs() <= k { 1.0 } else { k / r.abs() }
}

#[test]
fn the_smoother_matches_a_dense_solve_with_gaps_and_pins() {
    let mut rng = Lcg(11);
    let mut sm = Smoother::default();
    let mut worst = 0.0f64;
    for &n in &[3usize, 4, 9, 40, 257] {
        for trial in 0..4 {
            let obs: Vec<f64> = (0..n)
                .map(|i| (0.07 * i as f64).sin() * 5.0 + rng.next() - 0.5)
                .collect();
            let mut w: Vec<f64> = (0..n).map(|_| 0.2 + rng.next()).collect();
            let mut garbage = obs.clone();
            for i in 0..n {
                if rng.next() < 0.2 {
                    // A gap: its observation must never be read.
                    w[i] = 0.0;
                    garbage[i] = f64::NAN;
                }
            }
            // Whittaker only, or every term.
            let p = if trial % 2 == 0 {
                Penalty::new(0.0, 0.0, 6.0, 1.0 + rng.next())
            } else {
                Penalty::new(0.05, 2.0, 6.0, 1.0 + rng.next())
            };
            assert!(sm.solve(&garbage, &w, p));
            worst = worst.max(rel_diff(&sm.u, &dense_smooth(&obs, &w, p)));
            assert!(sm.u.iter().all(|u| u.is_finite()));
        }
    }
    assert!(worst < 1e-10, "smoother vs dense: relative error {worst:e}");

    // A heavy weight pins the solution to its observation.
    let n = 50;
    let obs: Vec<f64> = (0..n).map(|i| if i == 20 { 7.0 } else { 0.0 }).collect();
    let mut w = vec![1.0; n];
    w[20] = 1e8;
    assert!(sm.solve(&obs, &w, Penalty::new(0.0, 0.0, 4.0, 1.0)));
    assert!((sm.u[20] - 7.0).abs() < 1e-3, "pinned at {}", sm.u[20]);
}

#[test]
fn the_robust_smoother_matches_a_dense_irls() {
    let n = 120;
    let mut rng = Lcg(3);
    let mut obs: Vec<f64> = (0..n)
        .map(|i| (0.05 * i as f64).cos() + 0.1 * rng.next())
        .collect();
    for i in [17, 60, 61, 95] {
        obs[i] += 8.0;
    }
    let mut base = vec![1.0; n];
    base[40] = 0.0;
    base[41] = 0.5;
    let p = Penalty::new(0.0, 0.0, 5.0, 1.0);
    let mut sm = Smoother::default();
    let iters = sm
        .solve_robust(&obs, &base, p, 6, 0.0, |_, _| (huber(0.3), true))
        .expect("solved");
    assert_eq!(iters, 6);

    // The same iterations, densely.
    let mut u = dense_smooth(&obs, &base, p);
    for _ in 0..6 {
        let w: Vec<f64> = base
            .iter()
            .zip(obs.iter().zip(&u))
            .map(|(&b, (&o, &x))| if b > 0.0 { b * huber(0.3)(o - x) } else { 0.0 })
            .collect();
        u = dense_smooth(&obs, &w, p);
    }
    let err = rel_diff(&sm.u, &u);
    assert!(err < 1e-10, "robust smoother vs dense IRLS: {err:e}");
    // The outliers are down-weighted, and the smooth part followed.
    assert!(sm.weights[17] < 0.1 && sm.weights[40] == 0.0);
    let off = (0..n)
        .filter(|i| ![17, 60, 61, 95].contains(i))
        .map(|i| (sm.u[i] - (0.05 * i as f64).cos() - 0.05).abs())
        .fold(0.0f64, f64::max);
    assert!(off < 0.25, "worst deviation off the outliers {off}");
}

#[test]
fn the_robust_smoother_stops_once_the_final_loss_settles() {
    let n = 60;
    let obs: Vec<f64> = (0..n).map(|i| 0.01 * i as f64).collect();
    let base = vec![1.0; n];
    let mut sm = Smoother::default();
    let p = Penalty::new(0.0, 0.0, 3.0, 1.0);
    // A schedule that is final only from its third solve on.
    let mut seen = Vec::new();
    let iters = sm
        .solve_robust(&obs, &base, p, 20, 1e-6, |k, r0| {
            seen.push(r0);
            (huber(1.0), k >= 2)
        })
        .expect("solved");
    // Linear data is smoothed exactly, so nothing moves; the loop stops at the first final solve.
    assert_eq!(iters, 3);
    assert!(seen.iter().all(|&r| r == seen[0] && r < 1e-6));
}
