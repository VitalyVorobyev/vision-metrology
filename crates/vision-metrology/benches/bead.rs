//! `BeadTracker` benchmarks: one `track` call on a 1280×1024 frame.
//!
//! Run with `cargo bench -p vision-metrology --bench bead`.
//!
//! The scene is the ribbon fixture's S-bend (two arcs of radius 600 px turning 0.9 rad
//! each, 1080 px long), a light bead 50 px wide, blurred by σ 1.2 px, under 2 DN of
//! seeded noise, rendered once outside the timed loop. The prior is the truth from 10 px
//! in from each end, moved 2 px along the normal and bent by a further 1 px sine over its
//! length. `tol` is tiny, so every configured pass runs. The pose count is set through
//! `spacing`; `s15` and `s30` are the tracking reach, `track.max_offset`.
//!
//! The gaps bench removes the bead over 30% of its length, in three gaps, and adds a
//! distractor step beside it. The explain bench runs `diagnostics::explain_bead` on the
//! 300-pose, 3-pass case.
//!
//! ## Measured numbers (2026-10-06, Apple M4 Pro, release, `lto = "thin"`, `codegen-units = 1`)
//!
//! | Benchmark                                               | Time      |
//! |----------------------------------------------------------|-----------|
//! | `bead_track_1280x1024_100poses_s15_1pass`                 | ~0.44 ms  |
//! | `bead_track_1280x1024_300poses_s15_1pass`                 | ~1.30 ms  |
//! | `bead_track_1280x1024_300poses_s15_3pass`                 | ~2.54 ms  |
//! | `bead_track_1280x1024_300poses_s30_3pass`                 | ~2.97 ms  |
//! | `bead_track_1280x1024_300poses_s15_3pass_30pct_gaps`      | ~2.58 ms  |
//! | `bead_explain_1280x1024_300poses_s15_3pass`               | ~3.60 ms  |
//!
//! A call measures one strip per station in each pass and once more in the final stage,
//! about 2.1 µs a strip at a reach of 15 px. Profiled on the 300-pose, 3-pass case, the
//! strip calipers take 96% of the time (profile sampling 75%, smoothing and edge location
//! 21%, of which refilling the smoothing kernel at 955 of the 1200 strips is 1.4%), and the
//! solves, the pairing and the curve geometry about 3%. A 30 px reach lengthens every
//! tracking strip; a gap's strips cost what the bead's do; explaining allocates a caliper
//! trace per station and pass.
//!
//! Re-run and update this table, and `docs/performance.md`, whenever the tracker's or the
//! strip caliper's hot path changes.

#[path = "../examples/common/ribbon.rs"]
mod ribbon;

use std::hint::black_box;
use std::num::NonZeroUsize;

use criterion::{Criterion, criterion_group, criterion_main};
use ribbon::{Curve, P2, Ribbon, Scene, Step, Width};
use vision_metrology::measure::diagnostics::explain_bead;
use vision_metrology::measure::{BeadCaliper, BeadConfig, BeadTracker, BeadTuning};
use vision_metrology::{Image, Point2f};

const SIZE: (usize, usize) = (1280, 1024);

fn s_bend() -> Ribbon {
    Ribbon::new(
        Curve::s_bend(P2::new(150.0, 740.0), 0.0, 600.0, -0.9),
        Width::Const(50.0),
    )
}

/// The rendered scene, under 2 DN of noise.
fn render(scene: Scene) -> Image<u8> {
    scene.render().noisy(2.0, 7).to_u8()
}

/// The prior: 10 px in from each end, a vertex every 4 px, 2 px off and bent by 1 px.
fn prior(ribbon: &Ribbon) -> Vec<Point2f> {
    let len = ribbon.length();
    let n = ((len - 20.0) / 4.0).ceil() as usize;
    (0..=n)
        .map(|k| {
            let s = 10.0 + (len - 20.0) * k as f64 / n as f64;
            let bend = 2.0 + (std::f64::consts::TAU * s / len).sin();
            (ribbon.curve.center(s) + ribbon.curve.normal(s) * bend).point()
        })
        .collect()
}

/// The polyline length of `prior`, px.
fn length(prior: &[Point2f]) -> f32 {
    prior.windows(2).map(|w| (w[1] - w[0]).norm()).sum()
}

/// The default config placing `poses` stations on `prior`, every one of `passes` run,
/// with a tracking reach of `reach` px.
fn config(prior: &[Point2f], poses: usize, passes: usize, reach: f32) -> BeadConfig {
    let d = BeadConfig::default();
    BeadConfig {
        spacing: length(prior) / (poses - 1) as f32,
        track: BeadCaliper {
            max_offset: reach,
            ..d.track
        },
        tuning: BeadTuning {
            passes: NonZeroUsize::new(passes).expect("at least one pass"),
            tol: 1e-6,
            ..d.tuning
        },
        ..d
    }
}

fn bench_track(c: &mut Criterion) {
    let ribbon = s_bend();
    let img = render(Scene::new(SIZE.0, SIZE.1, ribbon.clone(), 40.0, 180.0, 1.2));
    let view = img.as_view();
    let prior = prior(&ribbon);

    for (id, poses, passes, reach) in [
        ("bead_track_1280x1024_100poses_s15_1pass", 100, 1, 15.0),
        ("bead_track_1280x1024_300poses_s15_1pass", 300, 1, 15.0),
        ("bead_track_1280x1024_300poses_s15_3pass", 300, 3, 15.0),
        ("bead_track_1280x1024_300poses_s30_3pass", 300, 3, 30.0),
    ] {
        let mut tracker = BeadTracker::new(config(&prior, poses, passes, reach)).expect("valid");
        let got = tracker.track(&view, &prior).expect("tracked");
        assert_eq!(got.samples.len(), poses, "{id}");
        assert_eq!(got.track.passes.len(), passes, "{id}");
        assert_eq!(got.summary.support, 1.0, "{id}");
        c.bench_function(id, |b| {
            b.iter(|| {
                let bead = tracker.track(black_box(&view), black_box(&prior));
                black_box(bead.expect("tracked").samples.len());
            });
        });
    }
}

fn bench_track_gaps(c: &mut Criterion) {
    let len = s_bend().length();
    let mut ribbon = s_bend();
    for (a, b) in [(0.15, 0.25), (0.45, 0.55), (0.75, 0.85)] {
        ribbon = ribbon.gap(a * len, b * len);
    }
    // A step 60 DN brighter 8 px beyond the bead's +n edge at a third of its length, along
    // the tangent there: the bead curves away from it on both sides.
    let s = len / 3.0;
    let n = ribbon.curve.normal(s);
    let step = Step {
        point: ribbon.curve.center(s) + n * (25.0 + 8.0),
        normal: n,
        contrast: 60.0,
    };
    let img = render(Scene::new(SIZE.0, SIZE.1, ribbon.clone(), 40.0, 180.0, 1.2).step(step));
    let view = img.as_view();
    let prior = prior(&ribbon);
    let mut tracker = BeadTracker::new(config(&prior, 300, 3, 15.0)).expect("valid");
    let got = tracker.track(&view, &prior).expect("tracked");
    assert_eq!(got.track.passes.len(), 3);
    assert!(
        got.summary.support > 0.6 && got.summary.support < 0.75,
        "support {}",
        got.summary.support
    );
    c.bench_function("bead_track_1280x1024_300poses_s15_3pass_30pct_gaps", |b| {
        b.iter(|| {
            let bead = tracker.track(black_box(&view), black_box(&prior));
            black_box(bead.expect("tracked").samples.len());
        });
    });
}

fn bench_explain(c: &mut Criterion) {
    let ribbon = s_bend();
    let img = render(Scene::new(SIZE.0, SIZE.1, ribbon.clone(), 40.0, 180.0, 1.2));
    let view = img.as_view();
    let prior = prior(&ribbon);
    let mut tracker = BeadTracker::new(config(&prior, 300, 3, 15.0)).expect("valid");
    c.bench_function("bead_explain_1280x1024_300poses_s15_3pass", |b| {
        b.iter(|| {
            let trace = explain_bead(&mut tracker, black_box(&view), black_box(&prior));
            black_box(trace.expect("tracked").measure.len());
        });
    });
}

criterion_group!(benches, bench_track, bench_track_gaps, bench_explain);
criterion_main!(benches);
