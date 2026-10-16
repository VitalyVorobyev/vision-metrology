//! `metric::rig::RigPoseEstimator` benchmarks: a whole sequence, steps × views.
//!
//! Run with `cargo bench -p vision-metrology --bench rig`.
//!
//! The fixture is `tests/rig/fixture.rs`: affine views at 360°/n spacing 30 mm from the
//! origin, a known per-step increment with a slow rotation, 0.05 px of shift noise, and
//! one view 3 px wrong every tenth step, so the biweight runs from both starts there.
//! `estimate` runs every step and allocates only its output; `smooth` runs the forward
//! filter and the backward pass over the same trajectory. Shifts are generated once,
//! outside the timed loop. Numbers are in `docs/performance.md`.

#[path = "../tests/rig/fixture.rs"]
mod fixture;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use vision_metrology::Vec2f;
use vision_metrology::metric::rig::{RigPoseConfig, RigPoseEstimator, RigSmootherConfig};

fn bench(c: &mut Criterion) {
    let cfg = RigPoseConfig::default();
    for (steps, views) in [(1000, 2), (1000, 3), (1000, 8), (10000, 3)] {
        let truth = fixture::increments(steps);
        let v = fixture::views(views);
        let mut sh = fixture::shifts(views, &truth, 0.05, 1);
        for k in (5..steps).step_by(10) {
            sh[k * views].as_mut().unwrap().d_px += Vec2f::new(3.0, -1.0);
        }
        let mut est = RigPoseEstimator::new();
        c.bench_function(&format!("rig_estimate_{steps}steps_{views}views"), |b| {
            b.iter(|| black_box(est.estimate(&v, black_box(&sh), &cfg).unwrap()))
        });
        if views == 3 {
            let t = est.estimate(&v, &sh, &cfg).unwrap();
            let s = RigSmootherConfig::default();
            c.bench_function(&format!("rig_smooth_{steps}steps"), |b| {
                b.iter(|| black_box(est.smooth(black_box(&t), &s).unwrap()))
            });
        }
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
