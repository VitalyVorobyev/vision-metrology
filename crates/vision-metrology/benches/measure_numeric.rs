//! Benchmarks of `measure`'s crate-private numerical cores: the robust Whittaker smoother
//! and the layered min-path (ADR-0020).
//!
//! Run with `cargo bench -p vision-metrology --bench measure_numeric`.
//!
//! The cores are not public API, so this bench compiles their source files directly; they
//! depend on nothing else in the crate.
//!
//! - `whittaker_huber_n*`: the smoother on `n` samples, `(ℓ/h)⁴` bending only with
//!   `ℓ = 4h`, one least-squares solve and 5 Huber reweights, every tenth sample a gap and
//!   every 37th an outlier. Linear in `n`: the time per sample is the same at every size.
//! - `minpath_coupled_300layers_65x21x2`: the coupled two-boundary extraction's state space
//!   at its prototype's defaults: 300 stations, 65 centre bins × 21 width bins, a present
//!   and an absent plane, bands of 8 and 2 bins. Solve, max-marginals and path.
//! - `minpath_rim_61rays_113radii`: the bead-end rim, 61 rays of 113 radius bins, a band
//!   of 8 bins, both end rays pinned.
//!
//! The numbers are in `docs/performance.md`.

#[allow(dead_code)]
#[path = "../src/measure/numeric/minpath.rs"]
mod minpath;
#[allow(dead_code)]
#[path = "../src/measure/numeric/whittaker.rs"]
mod whittaker;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use minpath::{Grid, LayeredPath, Steps};
use whittaker::{Penalty, Smoother};

/// A deterministic value in `[0, 1)` for index `i`.
fn hash01(i: usize) -> f64 {
    let x = (i as u64 ^ 0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let x = (x ^ (x >> 31)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (x >> 11) as f64 / (1u64 << 53) as f64
}

fn huber(r: f64) -> f64 {
    let k = 0.5;
    if r.abs() <= k { 1.0 } else { k / r.abs() }
}

fn bench_whittaker(c: &mut Criterion) {
    for n in [1_000usize, 10_000, 100_000] {
        let obs: Vec<f64> = (0..n)
            .map(|i| {
                let smooth = (0.01 * i as f64).sin() * 3.0;
                let outlier = if i % 37 == 0 { 6.0 } else { 0.0 };
                smooth + outlier + 0.2 * (hash01(i) - 0.5)
            })
            .collect();
        let base: Vec<f64> = (0..n)
            .map(|i| if i % 10 == 0 { 0.0 } else { 1.0 })
            .collect();
        let p = Penalty::new(0.0, 0.0, 4.0, 1.0);
        let mut sm = Smoother::default();
        let iters = sm.solve_robust(&obs, &base, p, 5, 0.0, |_, _| (huber, true));
        assert_eq!(iters, Some(5));
        c.bench_function(&format!("whittaker_huber_n{n}"), |b| {
            b.iter(|| {
                let it = sm.solve_robust(black_box(&obs), &base, p, 5, 0.0, |_, _| (huber, true));
                black_box((it, sm.u[n / 2]));
            });
        });
    }
}

fn bench_minpath(c: &mut Criterion) {
    // The coupled extraction: (centre, width) on a present and an absent plane.
    let grid = Grid {
        planes: 2,
        dims: [65, 21],
    };
    let steps = Steps {
        coef: [0.025, 0.05],
        band: [8, 2],
        switch: vec![0.0, 2.0, 2.0, 0.0],
    };
    let layers = 300;
    let mut dp = LayeredPath::default();
    dp.reset(layers, grid, &steps);
    for k in 0..layers {
        // A wall pair that drifts along the stations, under noise; the absent plane flat.
        let c0 = 32.0 + 12.0 * (k as f64 / 40.0).sin();
        for (s, u) in dp.unary_mut(k).iter_mut().enumerate() {
            let (p, i0, i1) = grid.coords(s);
            *u = if p == 1 {
                -0.6
            } else {
                let d = (i0 as f64 - c0).powi(2) / 8.0 + (i1 as f64 - 10.0).powi(2) / 4.0;
                (-2.0 * (-d).exp() + 0.5 * hash01(k * 4096 + s)) as f32
            };
        }
    }
    assert!(dp.solve().is_some());
    c.bench_function("minpath_coupled_300layers_65x21x2", |b| {
        b.iter(|| {
            let best = dp.solve();
            black_box((best, dp.path()[layers / 2], dp.max_marginal(layers / 2, 0)));
        });
    });

    // The bead-end rim: one radius per ray, the end rays pinned.
    let (rays, radii) = (61usize, 113usize);
    let grid = Grid {
        planes: 1,
        dims: [radii, 1],
    };
    let steps = Steps {
        coef: [0.05, 0.0],
        band: [8, 0],
        switch: vec![0.0],
    };
    let mut rim = LayeredPath::default();
    rim.reset(rays, grid, &steps);
    for k in 0..rays {
        let r = 40.0 + 16.0 * (std::f64::consts::PI * k as f64 / (rays - 1) as f64).sin();
        for (i, u) in rim.unary_mut(k).iter_mut().enumerate() {
            *u = (-(-(i as f64 - r).powi(2) / 4.0).exp() + 0.3 * hash01(k * 256 + i)) as f32;
        }
    }
    for k in [0, rays - 1] {
        rim.pin(k, 0, [39, 0], [41, 0], 1e3);
    }
    assert!(rim.solve().is_some());
    c.bench_function("minpath_rim_61rays_113radii", |b| {
        b.iter(|| {
            let best = rim.solve();
            black_box((best, rim.path()[rays / 2]));
        });
    });
}

criterion_group!(benches, bench_whittaker, bench_minpath);
criterion_main!(benches);
