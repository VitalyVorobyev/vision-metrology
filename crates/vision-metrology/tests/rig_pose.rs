//! `metric::rig::RigPoseEstimator` on a synthetic three-view rig with known truth.
//!
//! The fixture (`rig/fixture.rs`): three affine views 120° apart around the plane origin,
//! 30 mm out, at 20 px/mm, and a known per-step `(tx, ty, θ)` with a slow rotation.
//!
//! - **Recovery.** Noise-free shifts give the increments to rounding; under 0.05 px of
//!   shift noise the increment errors have the bias and σ the accuracy row pins, and the
//!   final pose error stays inside its reported covariance.
//! - **A bridged step.** A step with no usable view carries the previous increment, says
//!   `Bridged`, grows the covariance, and the chain continues.
//! - **A gross view** is down-weighted by the median start and Huber, and given zero
//!   weight by the graduated biweight.
//! - **A single-view step** holds the rotation at the prior, `Previous` or `Zero`.
//! - **Provenance** per step and view: outcomes, weights, lift points, residuals.
//! - **Smoothing** lowers the increment error, interpolates a bridged gap, and gates a
//!   spike.
//! - **Scale.** A plane-height error leaves the views agreeing, and only the path-length
//!   ratio sees it.

#[path = "rig/fixture.rs"]
mod fixture;

use fixture::{Inc, increments, poses, shifts, views, views_scaled};
use vision_metrology::metric::rig::{
    RigPoseConfig, RigPoseEstimator, RigSmootherConfig, RigStep, RigTrajectory, RotationHold,
    ViewOutcome, ViewShift,
};
use vision_metrology::{Error, Point2f, Vec2f};

const N: usize = 3;

fn run(sh: &[Option<ViewShift>], cfg: &RigPoseConfig) -> RigTrajectory {
    RigPoseEstimator::new()
        .estimate(&views(N), sh, cfg)
        .unwrap()
}

fn inc_error(t: &RigTrajectory, truth: &[Inc], frame: usize) -> (f64, f64, f64) {
    let f = &t.frames[frame];
    let i = &truth[frame - 1];
    (
        f.increment.translation.x as f64 - i.tx,
        f.increment.translation.y as f64 - i.ty,
        f.increment.rotation.angle() as f64 - i.theta,
    )
}

#[test]
fn noise_free_shifts_recover_the_increments() {
    let truth = increments(200);
    let t = run(&shifts(N, &truth, 0.0, 1), &RigPoseConfig::default());
    assert_eq!(t.frames.len(), 201);
    assert_eq!(t.stats.measured, 200);
    for k in 1..=200 {
        let (ex, ey, et) = inc_error(&t, &truth, k);
        assert!(
            ex.abs() < 2e-5 && ey.abs() < 2e-5,
            "step {k}: ({ex}, {ey}) mm"
        );
        assert!(et.abs() < 2e-6, "step {k}: θ error {et} rad");
    }
    let end = poses(&truth)[200];
    let got = t.frames[200].pose;
    assert!(((got.translation.x as f64 - end.0).hypot(got.translation.y as f64 - end.1)) < 2e-3);
    assert!((got.rotation.angle() as f64 - end.2).abs() < 1e-5);
    assert!(t.stats.residual_max_mm.unwrap() < 1e-4);
}

#[test]
fn noisy_shifts_stay_inside_the_reported_covariance() {
    let truth = increments(300);
    let (mut worst_t, mut worst_r) = (0.0f64, 0.0f64);
    for seed in 1..=8 {
        let t = run(&shifts(N, &truth, 0.05, seed), &RigPoseConfig::default());
        let end = poses(&truth)[300];
        let f = t.frames[300];
        let (ex, ey) = (
            f.pose.translation.x as f64 - end.0,
            f.pose.translation.y as f64 - end.1,
        );
        let er = f.pose.rotation.angle() as f64 - end.2;
        let c = f.covariance;
        worst_t = worst_t.max((ex / (c[(0, 0)] as f64).sqrt()).abs());
        worst_t = worst_t.max((ey / (c[(1, 1)] as f64).sqrt()).abs());
        worst_r = worst_r.max((er / (c[(2, 2)] as f64).sqrt()).abs());
    }
    assert!(
        worst_t < 4.0 && worst_r < 4.0,
        "final pose error in σ: translation {worst_t:.2}, rotation {worst_r:.2}"
    );
}

#[test]
fn a_step_with_no_usable_view_is_bridged() {
    let truth = increments(60);
    let mut sh = shifts(N, &truth, 0.02, 3);
    // Step 30 (frame 31) and 40-41 (frames 41, 42) measure nothing.
    for s in &mut sh[30 * N..31 * N] {
        *s = None;
    }
    for s in &mut sh[40 * N..42 * N] {
        s.as_mut().unwrap().accepted = false;
    }
    let t = run(&sh, &RigPoseConfig::default());
    assert_eq!(t.frames.len(), 61, "the chain is never split");
    assert_eq!(t.stats.bridged, 3);
    assert_eq!(t.stats.longest_bridge, 2);
    let (prev, bridged) = (t.frames[30], t.frames[31]);
    assert_eq!(bridged.step, RigStep::Bridged);
    assert_eq!(bridged.views_used, 0);
    assert_eq!(bridged.residual_rms_mm, None);
    assert_eq!(bridged.increment, prev.increment, "constant velocity");
    for k in [0, 1, 2] {
        assert!(bridged.increment_covariance[(k, k)] > prev.increment_covariance[(k, k)]);
        assert!(bridged.covariance[(k, k)] > prev.covariance[(k, k)]);
    }
    let (b1, b2) = (t.frames[41], t.frames[42]);
    assert_eq!(b2.step, RigStep::Bridged);
    assert!(b2.increment_covariance[(0, 0)] > b1.increment_covariance[(0, 0)]);
    assert!(
        t.views(41)
            .iter()
            .all(|u| u.outcome == ViewOutcome::NotAccepted && u.lever_mm.is_some())
    );
    assert!(
        t.views(31)
            .iter()
            .all(|u| u.outcome == ViewOutcome::Missing)
    );
    // The bridge costs only the truth's change of velocity over the gap.
    let end = poses(&truth)[60];
    let f = t.frames[60].pose;
    let err = (f.translation.x as f64 - end.0).hypot(f.translation.y as f64 - end.1);
    assert!(err < 0.05, "pose error after the bridges {err} mm");
    assert_eq!(t.frames[43].step, RigStep::Measured);
}

#[test]
fn a_grossly_wrong_view_is_down_weighted() {
    let truth = increments(40);
    let mut sh = shifts(N, &truth, 0.05, 5);
    // View 1 is 3 px (0.15 mm, ~60σ) wrong on step 20.
    sh[20 * N + 1].as_mut().unwrap().d_px += Vec2f::new(3.0, -1.0);

    // Huber alone bounds the pull but, with three views, cannot remove it.
    let huber = run(
        &sh,
        &RigPoseConfig {
            tukey_c: None,
            ..Default::default()
        },
    );
    let uses = huber.views(21);
    assert!(uses[1].weight < 0.1, "gross weight {}", uses[1].weight);
    assert_eq!(uses[1].outcome, ViewOutcome::Used);
    let (hx, hy, _) = inc_error(&huber, &truth, 21);

    // The default biweight gives it zero weight and leaves the good views whole.
    let t = run(&sh, &RigPoseConfig::default());
    let uses = t.views(21);
    assert_eq!(t.frames[21].step, RigStep::Measured);
    assert_eq!(uses[1].outcome, ViewOutcome::Rejected);
    assert_eq!(uses[1].weight, 0.0);
    assert!(uses[1].residual_mm.unwrap() > 0.1);
    assert!(uses[0].weight > 0.9 && uses[2].weight > 0.9, "{uses:?}");
    assert_eq!(t.frames[21].views_used, 2);
    let (ex, ey, et) = inc_error(&t, &truth, 21);
    assert!(ex.hypot(ey) < 0.004, "error ({ex}, {ey}) mm");
    assert!(et.abs() < 1e-4, "θ error {et} rad");
    assert!(
        ex.hypot(ey) < 0.5 * hx.hypot(hy),
        "biweight ({ex}, {ey}) vs Huber ({hx}, {hy})"
    );
    // Clean steps keep every view.
    assert_eq!(t.stats.measured, 40);
    assert!(
        (1..=40)
            .filter(|&k| k != 21)
            .all(|k| t.frames[k].views_used == 3)
    );
}

#[test]
fn a_single_view_step_holds_rotation_at_the_prior() {
    let truth = increments(30);
    let mut sh = shifts(N, &truth, 0.0, 7);
    sh[15 * N] = None;
    sh[15 * N + 2].as_mut().unwrap().accepted = false;
    for (hold, expect) in [
        (RotationHold::Previous, None),
        (RotationHold::Zero, Some(0.0f32)),
    ] {
        let t = run(
            &sh,
            &RigPoseConfig {
                rotation_hold: hold,
                ..Default::default()
            },
        );
        let (prev, single) = (t.frames[15], t.frames[16]);
        assert_eq!(single.step, RigStep::SingleView);
        assert_eq!(single.views_used, 1);
        assert_eq!(single.residual_rms_mm, None, "no redundancy");
        let theta = single.increment.rotation.angle();
        let want = expect.unwrap_or(prev.increment.rotation.angle());
        assert!(
            (theta - want).abs() < 1e-7,
            "{hold:?}: θ {theta}, prior {want}"
        );
        assert!((single.increment_covariance[(2, 2)] - 1e-6).abs() < 1e-9);
        assert_eq!(t.views(16)[1].outcome, ViewOutcome::Used);
        assert_eq!(t.views(16)[1].weight, 1.0);
        // The translation error is the rotation error times the lever to the origin.
        let (ex, ey, et) = inc_error(&t, &truth, 16);
        assert!(
            ex.hypot(ey) < 1.1 * et.abs() * 30.0 + 1e-5,
            "({ex}, {ey}), θ {et}"
        );
    }
}

#[test]
fn provenance_is_complete_and_the_closure_gate_applies() {
    let truth = increments(10);
    let mut sh = shifts(N, &truth, 0.02, 9);
    sh[4 * N + 2].as_mut().unwrap().closure_px = Some(0.8);
    sh[5 * N].as_mut().unwrap().d_px = Vec2f::new(f32::NAN, 0.0);
    let cfg = RigPoseConfig {
        max_closure_px: Some(0.5),
        ..Default::default()
    };
    let t = run(&sh, &cfg);
    assert_eq!(t.view_count(), N);
    assert!(t.views(0).is_empty() && t.views(11).is_empty());
    assert_eq!(t.frames[0].step, RigStep::Anchor);
    assert_eq!(t.views(5)[2].outcome, ViewOutcome::ClosureFailed);
    assert!(
        t.views(5)[2].residual_mm.is_some(),
        "still lifted for diagnostics"
    );
    assert_eq!(t.views(6)[0].outcome, ViewOutcome::Unliftable);
    for k in 1..=10 {
        let f = &t.frames[k];
        let used = t
            .views(k)
            .iter()
            .filter(|u| u.outcome == ViewOutcome::Used)
            .count();
        assert_eq!(f.views_used, used);
        for u in t.views(k).iter().filter(|u| u.outcome == ViewOutcome::Used) {
            let lever = u.lever_mm.unwrap();
            assert!((lever.x.hypot(lever.y) - 30.0).abs() < 2.0, "{lever:?}");
            assert!(u.displacement_mm.is_some() && u.residual_mm.is_some());
        }
    }
}

#[test]
fn smoothing_lowers_the_error_fills_a_gap_and_gates_a_spike() {
    let truth = increments(200);
    // Step 150: every view agrees on a 0.5 mm jump that did not happen.
    let mut seen = truth.clone();
    seen[150].tx += 0.5;
    let mut sh = shifts(N, &seen, 0.2, 11);
    for s in &mut sh[100 * N..103 * N] {
        *s = None;
    }
    let mut est = RigPoseEstimator::new();
    let t = est
        .estimate(&views(N), &sh, &RigPoseConfig::default())
        .unwrap();
    let cfg = RigSmootherConfig {
        motion_noise: vision_metrology::metric::rig::MotionNoise {
            translation_mm: 0.005,
            rotation_rad: 2e-5,
        },
        ..Default::default()
    };
    let s = est.smooth(&t, &cfg).unwrap();
    assert_eq!(s.frames.len(), t.frames.len());
    assert_eq!(s.gated, 1);
    assert!(s.frames[151].gated);
    let rms = |get: &dyn Fn(usize) -> (f64, f64)| {
        let skip = [101, 102, 103, 151];
        let ks: Vec<usize> = (1..=200).filter(|k| !skip.contains(k)).collect();
        let sum: f64 = ks
            .iter()
            .map(|&k| {
                let (x, y) = get(k);
                let i = truth[k - 1];
                (x - i.tx).powi(2) + (y - i.ty).powi(2)
            })
            .sum();
        (sum / ks.len() as f64).sqrt()
    };
    let raw = rms(&|k| {
        let i = t.frames[k].increment.translation;
        (i.x as f64, i.y as f64)
    });
    let smooth = rms(&|k| {
        let i = s.frames[k].increment.translation;
        (i.x as f64, i.y as f64)
    });
    assert!(smooth < 0.6 * raw, "smoothed {smooth} mm vs raw {raw} mm");
    for k in [101, 102, 103, 151] {
        let i = s.frames[k].increment.translation;
        let e = (i.x as f64 - truth[k - 1].tx).hypot(i.y as f64 - truth[k - 1].ty);
        assert!(e < 0.02, "smoothed step {k} error {e} mm");
    }
}

#[test]
fn a_plane_height_error_passes_agreement_and_fails_the_path_ratio() {
    let truth = increments(100);
    let sh = shifts(N, &truth, 0.0, 13);
    // The estimator believes the plane is 1% further away than it is.
    let t = RigPoseEstimator::new()
        .estimate(&views_scaled(N, 1.01), &sh, &RigPoseConfig::default())
        .unwrap();
    assert!(
        t.stats.residual_max_mm.unwrap() < 1e-3,
        "the views still agree: {:?}",
        t.stats.residual_max_mm
    );
    let external: Vec<Point2f> = poses(&truth)
        .iter()
        .map(|p| Point2f::new(p.0 as f32, p.1 as f32))
        .collect();
    let ratio = t
        .path_length_ratio(Point2f::origin(), &external)
        .unwrap()
        .unwrap();
    assert!((ratio - 1.0 / 1.01).abs() < 1e-4, "ratio {ratio}");
    let good = run(&sh, &RigPoseConfig::default());
    let ratio = good
        .path_length_ratio(Point2f::origin(), &external)
        .unwrap()
        .unwrap();
    assert!((ratio - 1.0).abs() < 1e-4, "ratio {ratio}");
    assert!(matches!(
        good.path_length_ratio(Point2f::origin(), &external[1..]),
        Err(Error::SizeMismatch { .. })
    ));
    assert_eq!(
        good.path_length_ratio(Point2f::origin(), &vec![Point2f::origin(); 101])
            .unwrap(),
        None
    );
}

#[test]
fn estimation_is_deterministic_and_the_estimator_reusable() {
    let truth = increments(50);
    let sh = shifts(N, &truth, 0.1, 17);
    let mut est = RigPoseEstimator::new();
    let a = est
        .estimate(&views(N), &sh, &RigPoseConfig::default())
        .unwrap();
    let _ = est
        .estimate(
            &views(4),
            &shifts(4, &truth, 0.1, 1),
            &RigPoseConfig::default(),
        )
        .unwrap();
    let b = est
        .estimate(&views(N), &sh, &RigPoseConfig::default())
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(a, run(&sh, &RigPoseConfig::default()));
}

#[test]
fn bad_input_is_an_error() {
    let mut est = RigPoseEstimator::new();
    let mut uses = vec![Default::default(); N];
    assert!(
        est.step(&[None; N], &mut uses).is_err(),
        "no sequence begun"
    );
    est.begin(&views(N), &RigPoseConfig::default()).unwrap();
    assert!(matches!(
        est.step(&[None; 2], &mut uses),
        Err(Error::SizeMismatch { .. })
    ));
    assert!(matches!(
        est.estimate(&views(N), &[None; 4], &RigPoseConfig::default()),
        Err(Error::SizeMismatch { .. })
    ));
    assert!(est.begin(&[], &RigPoseConfig::default()).is_err());
    let bad = RigPoseConfig {
        huber_k: 0.0,
        ..Default::default()
    };
    assert!(matches!(
        est.begin(&views(N), &bad),
        Err(Error::InvalidConfig(_))
    ));
    let mut v = views(N);
    v[1].plane_to_image = vision_metrology::Projective2f::from_matrix_unchecked(
        nalgebra::Matrix3::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    );
    assert!(matches!(
        est.begin(&v, &RigPoseConfig::default()),
        Err(Error::Degenerate(_))
    ));
}

/// The per-component numbers behind the accuracy rows (`accuracy/rig.rs`): bias and σ of
/// the per-step increment error over 300 steps and eight seeds at 0.05 px.
#[test]
fn increment_error_per_component() {
    let truth = increments(300);
    let mut est = RigPoseEstimator::new();
    let mut err: [Vec<f64>; 3] = Default::default();
    for seed in 1..=8 {
        let t = est
            .estimate(
                &views(N),
                &shifts(N, &truth, 0.05, seed),
                &RigPoseConfig::default(),
            )
            .unwrap();
        for k in 1..=300 {
            let (ex, ey, et) = inc_error(&t, &truth, k);
            err[0].push(ex);
            err[1].push(ey);
            err[2].push(et.to_degrees());
        }
    }
    let stats: Vec<(f64, f64)> = err
        .iter()
        .map(|e| {
            let n = e.len() as f64;
            let m = e.iter().sum::<f64>() / n;
            (
                m,
                (e.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n).sqrt(),
            )
        })
        .collect();
    eprintln!(
        "tx {:.5}±{:.5} mm, ty {:.5}±{:.5} mm, θ {:.5}±{:.5}°",
        stats[0].0, stats[0].1, stats[1].0, stats[1].1, stats[2].0, stats[2].1
    );
    for (i, (bias, sigma)) in stats.iter().enumerate() {
        let (b, s) = if i < 2 {
            (3e-4, 2.5e-3)
        } else {
            (3e-4, 4.5e-3)
        };
        assert!(
            bias.abs() < b && *sigma < s,
            "component {i}: {bias} ± {sigma}"
        );
    }
}
