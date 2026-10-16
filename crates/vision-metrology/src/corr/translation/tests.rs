use super::*;
use vm_primitives::Image;

fn texture(x: f64, y: f64) -> f32 {
    (100.0
        + 25.0 * (x * 0.17 + y * 0.13).sin()
        + 20.0 * (x * 0.08 - y * 0.21).cos()
        + 15.0 * (x * 0.31 + y * 0.07).sin()
        + 10.0 * (x * 0.03 + y * 0.39).cos()) as f32
}

fn image(w: usize, h: usize, shift: [f64; 2], gain: f32, offset: f32, angle: f64) -> Image<f32> {
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (
                x as f64 - shift[0] - w as f64 / 2.0,
                y as f64 - shift[1] - h as f64 / 2.0,
            );
            let (s, c) = angle.sin_cos();
            data.push(
                gain * texture(
                    c * u + s * v + w as f64 / 2.0,
                    -s * u + c * v + h as f64 / 2.0,
                ) + offset,
            );
        }
    }
    Image::from_vec(w, h, data).unwrap()
}

#[test]
fn subpixel_photometry_and_closure() {
    let mut estimator = TranslationEstimator::new();
    let cfg = TranslationConfig::default();
    let a = image(161, 97, [0.0, 0.0], 1.0, 0.0, 0.0);
    for shift in [[0.7, -0.3], [-3.2, 1.8], [5.4, -2.6]] {
        let b = image(161, 97, shift, 1.3, 17.0, 0.0);
        let f = estimator
            .estimate(&a.as_view(), &b.as_view(), None, None, None, &cfg)
            .unwrap();
        assert_eq!(f.status, TranslationStatus::Accepted, "{f:?}");
        let p = f.shift.unwrap();
        let err = (p.x as f64 - shift[0]).hypot(p.y as f64 - shift[1]);
        assert!(err < 0.1, "subpixel translation error {err}: {f:?}");
        assert!((f.gain.unwrap() - 1.0 / 1.3).abs() < 0.02, "{f:?}");
        assert!((f.offset.unwrap() + 17.0 / 1.3).abs() < 1.0, "{f:?}");
        let r = estimator
            .estimate(&b.as_view(), &a.as_view(), None, None, None, &cfg)
            .unwrap();
        assert_eq!(r.status, TranslationStatus::Accepted, "{r:?}");
        assert!((p + r.shift.unwrap()).norm() < 0.05, "closure: {f:?} {r:?}");
        let cov = f.covariance.unwrap();
        assert!(cov[0] >= 0.0 && cov[2] >= 0.0 && cov[0] * cov[2] >= cov[1] * cov[1]);
    }
}

#[test]
fn masks_exclude_static_distractors_and_nonfinite_pixels() {
    let mut a = image(160, 96, [0.0, 0.0], 1.0, 0.0, 0.0);
    let mut b = image(160, 96, [1.3, -0.7], 1.0, 0.0, 0.0);
    let mut mask = Image::new_fill(160, 96, 1u8);
    for y in 20..75 {
        for x in 40..125 {
            let i = y * 160 + x;
            mask.data_mut()[i] = 0;
            a.data_mut()[i] = f32::NAN;
            b.data_mut()[i] = 10000.0 * ((x + y) % 2) as f32;
        }
    }
    let f = TranslationEstimator::new()
        .estimate(
            &a.as_view(),
            &b.as_view(),
            Some(&mask.as_view()),
            Some(&mask.as_view()),
            None,
            &TranslationConfig::default(),
        )
        .unwrap();
    assert_eq!(f.status, TranslationStatus::Accepted, "{f:?}");
    assert!(
        (f.shift.unwrap() - Vec2f::new(1.3, -0.7)).norm() < 0.1,
        "{f:?}"
    );
    let mut p = Plane::default();
    p.load(&b.as_view(), Some(&mask.as_view())).unwrap();
    assert!(
        p.sample(39.0, 25.0).is_none(),
        "even zero-weight masked gradient taps are excluded"
    );
}

#[test]
fn failures_are_explicit() {
    let a = image(160, 96, [0.0, 0.0], 1.0, 0.0, 0.0);
    let flat = Image::new_fill(160, 96, 4.0f32);
    let none = Image::new_fill(160, 96, 0u8);
    let mut e = TranslationEstimator::new();
    let cfg = TranslationConfig::default();
    assert_eq!(
        e.estimate(&flat.as_view(), &flat.as_view(), None, None, None, &cfg)
            .unwrap()
            .status,
        TranslationStatus::LowTexture
    );
    assert_eq!(
        e.estimate(
            &a.as_view(),
            &a.as_view(),
            Some(&none.as_view()),
            None,
            None,
            &cfg
        )
        .unwrap()
        .status,
        TranslationStatus::InsufficientSupport
    );
    let b = image(160, 96, [12.0, 0.0], 1.0, 0.0, 0.0);
    assert_eq!(
        e.estimate(&a.as_view(), &b.as_view(), None, None, None, &cfg)
            .unwrap()
            .status,
        TranslationStatus::SearchBoundary
    );
    let bad = Image::new_fill(4, 4, 1u8);
    assert!(
        e.estimate(
            &a.as_view(),
            &a.as_view(),
            Some(&bad.as_view()),
            None,
            None,
            &cfg
        )
        .is_err()
    );
    assert!(
        e.estimate(
            &a.as_view(),
            &a.as_view(),
            None,
            None,
            Some(Vec2f::new(f32::NAN, 0.0)),
            &cfg
        )
        .is_err()
    );
    let periodic = Image::from_vec(
        160,
        96,
        (0..160 * 96)
            .map(|i| ((i % 160 % 4) as f32 * 1.57).sin() + (i / 160 % 4) as f32)
            .collect(),
    )
    .unwrap();
    assert_eq!(
        e.estimate(
            &periodic.as_view(),
            &periodic.as_view(),
            None,
            None,
            None,
            &cfg
        )
        .unwrap()
        .status,
        TranslationStatus::Ambiguous
    );
}

#[test]
fn observable_rotation_is_rejected() {
    let a = image(320, 97, [0.0, 0.0], 1.0, 0.0, 0.0);
    let b = image(320, 97, [1.2, 0.5], 1.0, 0.0, 0.01);
    let cfg = TranslationConfig {
        min_margin: 0.0,
        ..Default::default()
    };
    let f = TranslationEstimator::new()
        .estimate(&a.as_view(), &b.as_view(), None, None, None, &cfg)
        .unwrap();
    assert_eq!(f.status, TranslationStatus::ModelMismatch, "{f:?}");
    assert!(f.region_error.unwrap() > 0.5);
}

#[test]
fn regional_check_can_be_unavailable() {
    let a = image(160, 96, [0.0, 0.0], 1.0, 0.0, 0.0);
    let mut mask = Image::new_fill(160, 96, 0u8);
    for y in 4..40 {
        for x in 4..70 {
            mask.data_mut()[y * 160 + x] = 1;
        }
    }
    let f = TranslationEstimator::new()
        .estimate(
            &a.as_view(),
            &a.as_view(),
            Some(&mask.as_view()),
            Some(&mask.as_view()),
            None,
            &TranslationConfig::default(),
        )
        .unwrap();
    assert_eq!(f.status, TranslationStatus::Accepted, "{f:?}");
    assert_eq!(f.regions, 1);
    assert_eq!(f.region_error, None);
}

#[test]
fn strided_views_and_scratch_capacity() {
    let a = image(160, 96, [0.0, 0.0], 1.0, 0.0, 0.0);
    let b = image(160, 96, [1.7, 0.4], 1.0, 0.0, 0.0);
    let mut padded = vec![0.0; 170 * 96];
    for y in 0..96 {
        padded[y * 170..y * 170 + 160].copy_from_slice(a.as_view().row(y));
    }
    let view = ImageView::from_slice(160, 96, 170, &padded).unwrap();
    let mut e = TranslationEstimator::new();
    let cfg = TranslationConfig::default();
    let first = e
        .estimate(&view, &b.as_view(), None, None, None, &cfg)
        .unwrap();
    let capacity = (
        e.scores.capacity(),
        e.scratch.residuals.capacity(),
        e.previous[0].data.capacity(),
    );
    let next = e
        .estimate(&view, &b.as_view(), None, None, None, &cfg)
        .unwrap();
    assert_eq!(first, next);
    assert_eq!(
        capacity,
        (
            e.scores.capacity(),
            e.scratch.residuals.capacity(),
            e.previous[0].data.capacity()
        )
    );
    assert_eq!(first.status, TranslationStatus::Accepted, "{first:?}");
}

#[test]
fn prediction_and_partial_overlap() {
    let a = image(121, 75, [0.0, 0.0], 1.0, 0.0, 0.0);
    let b = image(151, 81, [20.7, -3.4], 1.0, 0.0, 0.0);
    let f = TranslationEstimator::new()
        .estimate(
            &a.as_view(),
            &b.as_view(),
            None,
            None,
            Some(Vec2f::new(20.0, -3.0)),
            &TranslationConfig::default(),
        )
        .unwrap();
    assert_eq!(f.status, TranslationStatus::Accepted, "{f:?}");
    assert!(
        (f.shift.unwrap() - Vec2f::new(20.7, -3.4)).norm() < 0.1,
        "{f:?}"
    );
}

#[test]
fn covariance_responds_to_noise_and_pixel_units() {
    let a = image(161, 97, [0.0, 0.0], 1.0, 0.0, 0.0);
    let clean = image(161, 97, [0.7, -0.3], 1.0, 0.0, 0.0);
    let mut noisy = clean.clone();
    for (i, v) in noisy.data_mut().iter_mut().enumerate() {
        // Incommensurate deterministic frequencies avoid a displacement-like perturbation.
        *v += 2.0 * ((i as f64 * 1.731).sin() + (i as f64 * 0.719).cos()) as f32;
    }
    let mut e = TranslationEstimator::new();
    let cfg = TranslationConfig::default();
    let c = e
        .estimate(&a.as_view(), &clean.as_view(), None, None, None, &cfg)
        .unwrap();
    let n = e
        .estimate(&a.as_view(), &noisy.as_view(), None, None, None, &cfg)
        .unwrap();
    assert_eq!(c.status, TranslationStatus::Accepted, "{c:?}");
    assert_eq!(n.status, TranslationStatus::Accepted, "{n:?}");
    assert!(
        n.covariance.unwrap()[0] > c.covariance.unwrap()[0],
        "{c:?} {n:?}"
    );
    let scaled = |img: &Image<f32>| {
        Image::from_vec(
            img.width(),
            img.height(),
            img.data().iter().map(|v| v / 255.0).collect(),
        )
        .unwrap()
    };
    let s = e
        .estimate(
            &scaled(&a).as_view(),
            &scaled(&noisy).as_view(),
            None,
            None,
            None,
            &cfg,
        )
        .unwrap();
    assert_eq!(s.status, n.status);
    assert!((s.shift.unwrap() - n.shift.unwrap()).norm() < 1e-4);
    assert!((s.covariance.unwrap()[0] / n.covariance.unwrap()[0] - 1.0).abs() < 1e-3);
}

#[test]
fn invalid_config_and_refinement_budget() {
    let a = image(161, 97, [0.0, 0.0], 1.0, 0.0, 0.0);
    let b = image(161, 97, [0.7, -0.3], 1.3, 17.0, 0.0);
    let mut cfg = TranslationConfig::default();
    cfg.tuning.iterations = std::num::NonZeroUsize::new(1).unwrap();
    let f = TranslationEstimator::new()
        .estimate(&a.as_view(), &b.as_view(), None, None, None, &cfg)
        .unwrap();
    assert_eq!(f.status, TranslationStatus::RefinementFailed);
    cfg.min_score = f32::NAN;
    assert!(
        TranslationEstimator::new()
            .estimate(&a.as_view(), &b.as_view(), None, None, None, &cfg)
            .is_err()
    );
}

#[test]
fn unrepresentable_photometry_is_a_failed_result_not_infinity() {
    let scale = |img: Image<f32>, factor: f32| {
        Image::from_vec(
            img.width(),
            img.height(),
            img.data().iter().map(|v| v * factor).collect(),
        )
        .unwrap()
    };
    let a = scale(image(161, 97, [0.0, 0.0], 1.0, 0.0, 0.0), 1e36);
    let b = scale(image(161, 97, [0.7, -0.3], 1.0, 0.0, 0.0), 1e-35);
    let result = TranslationEstimator::new()
        .estimate(
            &a.as_view(),
            &b.as_view(),
            None,
            None,
            None,
            &TranslationConfig::default(),
        )
        .unwrap();
    assert_eq!(result.status, TranslationStatus::RefinementFailed);
    assert_eq!(result.gain, None);
    assert_eq!(result.residual_rms, None);
}

#[test]
fn speckled_masks_near_integer_shifts_converge() {
    // A masked DoG of a speckled mask, and shifts within 0.01 px of an integer: taps
    // enter and leave validity as the shift crosses pixel boundaries. Refinement over a
    // re-derived observation set with a re-estimated Huber scale chattered here until its
    // budget ran out (`RefinementFailed`).
    use crate::corr::fixtures::{bandpass, speckle};
    let (w, h) = (320, 97);
    let mut e = TranslationEstimator::new();
    let cfg = TranslationConfig {
        region_check: None,
        ..Default::default()
    };
    let mut outcomes = Vec::new();
    for k in 0..12u64 {
        let shift = [
            (k % 5) as f64 - 2.0 + 0.003 * (k as f64 - 6.0) / 6.0,
            (k % 3) as f64 - 1.0 - 0.004 * (k as f64 - 5.0) / 6.0,
        ];
        let (ma, mb) = (speckle(w, h, 0.1, 2 * k), speckle(w, h, 0.1, 2 * k + 1));
        let a = bandpass(&image(w, h, [0.0, 0.0], 1.0, 0.0, 0.0), &ma);
        let b = bandpass(&image(w, h, shift, 1.0, 0.0, 0.0), &mb);
        let f = e
            .estimate(
                &a.as_view(),
                &b.as_view(),
                Some(&ma.as_view()),
                Some(&mb.as_view()),
                None,
                &cfg,
            )
            .unwrap();
        let error = f
            .shift
            .map(|p| (p.x as f64 - shift[0]).hypot(p.y as f64 - shift[1]));
        outcomes.push((f.status, error));
    }
    assert!(
        outcomes
            .iter()
            .all(|(s, e)| *s == TranslationStatus::Accepted && e.unwrap() < 0.1),
        "{outcomes:?}"
    );
}

#[test]
fn tap_window_matches_a_brute_force_block_check() {
    let (w, h) = (23, 17);
    let mask = crate::corr::fixtures::speckle(w, h, 0.08, 5);
    let mut p = Plane::default();
    p.load(&Image::new_fill(w, h, 1.0f32).as_view(), Some(&mask.as_view()))
        .unwrap();
    for y in -2..h as i64 + 2 {
        for x in -2..w as i64 + 2 {
            let brute = (y - 1..=y + 2).all(|yy| {
                (x - 1..=x + 2).all(|xx| {
                    xx >= 0
                        && yy >= 0
                        && xx < w as i64
                        && yy < h as i64
                        && p.valid[yy as usize * w + xx as usize]
                })
            });
            assert_eq!(p.window_valid(x as f64 + 0.3, y as f64 + 0.7), brute, "{x} {y}");
            if brute {
                for (dx, dy) in [(-0.99, -0.99), (0.99, 0.99), (-0.99, 0.99)] {
                    let (sx, sy) = (x as f64 + 0.3 + dx, y as f64 + 0.7 + dy);
                    assert!(p.sample(sx, sy).is_some(), "{sx} {sy}");
                }
            }
        }
    }
}
