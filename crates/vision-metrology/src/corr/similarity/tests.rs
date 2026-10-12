use super::*;
use vm_primitives::{Image, Point2f, transform_point};

fn texture(x: f64, y: f64) -> f32 {
    (100.
        + 25. * (x * 0.17 + y * 0.13).sin()
        + 20. * (x * 0.08 - y * 0.21).cos()
        + 15. * (x * 0.31 + y * 0.07).sin()
        + 10. * (x * 0.03 + y * 0.39).cos()) as f32
}
fn render(w: usize, h: usize, map: Similarity2f, gain: f32, offset: f32) -> Image<f32> {
    let inverse = map.inverse();
    Image::from_vec(
        w,
        h,
        (0..w * h)
            .map(|i| {
                let p = transform_point(&inverse, Point2f::new((i % w) as f32, (i / w) as f32));
                gain * texture(p.x as f64, p.y as f64) + offset
            })
            .collect(),
    )
    .unwrap()
}
fn centered(w: usize, h: usize, t: [f32; 2], angle: f32, scale: f32) -> Similarity2f {
    let c = Point2f::new((w - 1) as f32 * 0.5, (h - 1) as f32 * 0.5);
    let linear = Similarity2f::new(Vec2f::zeros(), angle, scale);
    let q = transform_point(&linear, c);
    Similarity2f::new(c - q + Vec2f::new(t[0], t[1]), angle, scale)
}
fn error(a: Similarity2f, b: Similarity2f, w: usize, h: usize) -> f32 {
    [
        (0., 0.),
        ((w - 1) as f32, 0.),
        (0., (h - 1) as f32),
        ((w - 1) as f32, (h - 1) as f32),
    ]
    .into_iter()
    .map(|(x, y)| {
        let p = Point2f::new(x, y);
        (transform_point(&a, p) - transform_point(&b, p)).norm()
    })
    .fold(0., f32::max)
}
#[test]
fn known_similarity_photometry_and_inverse_closure() {
    let (w, h) = (161, 97);
    let a = render(w, h, Similarity2f::identity(), 1., 0.);
    let mut estimator = SimilarityEstimator::new();
    for (angle, scale) in [(-0.02, 0.985), (0.01, 1.01), (0.03, 1.02)] {
        let map = centered(w, h, [0.7, -0.3], angle, scale);
        let b = render(w, h, map, 1.3, 17.);
        let f = estimator
            .estimate(
                &a.as_view(),
                &b.as_view(),
                None,
                None,
                None,
                &SimilarityConfig::default(),
            )
            .unwrap();
        assert_eq!(f.status, TranslationStatus::Accepted, "{f:?}");
        assert!(error(f.transform.unwrap(), map, w, h) < 0.1, "{f:?}");
        assert!((f.gain.unwrap() - 1. / 1.3).abs() < 0.02);
        assert!((f.offset.unwrap() + 17. / 1.3).abs() < 1.);
        let reverse = estimator
            .estimate(
                &b.as_view(),
                &a.as_view(),
                None,
                None,
                None,
                &SimilarityConfig::default(),
            )
            .unwrap();
        assert_eq!(reverse.status, TranslationStatus::Accepted, "{reverse:?}");
        assert!(
            error(
                reverse.transform.unwrap() * f.transform.unwrap(),
                Similarity2f::identity(),
                w,
                h
            ) < 0.1
        );
        let covariance = f.covariance.unwrap();
        for (i, row) in covariance.iter().enumerate() {
            assert!(row[i] >= 0.);
            for (j, value) in row.iter().enumerate() {
                assert!((*value - covariance[j][i]).abs() < 1e-6);
            }
        }
    }
}
#[test]
fn prediction_masks_stride_and_absence() {
    let (w, h) = (161, 97);
    let truth = centered(w, h, [18.7, -0.3], 0.03, 1.02);
    let prediction = centered(w, h, [18., 0.], 0.02, 1.01);
    let a = render(w, h, Similarity2f::identity(), 1., 0.);
    let mut b = render(w, h, truth, 1., 0.);
    let mut mask = Image::new_fill(w, h, 1u8);
    for y in 20..60 {
        for x in 40..100 {
            mask.data_mut()[y * w + x] = 0;
            b.data_mut()[y * w + x] = f32::NAN;
        }
    }
    let mut padded = vec![0.; (w + 7) * h];
    for y in 0..h {
        padded[y * (w + 7)..y * (w + 7) + w].copy_from_slice(a.as_view().row(y));
    }
    let a = ImageView::from_slice(w, h, w + 7, &padded).unwrap();
    let mut e = SimilarityEstimator::new();
    let f = e
        .estimate(
            &a,
            &b.as_view(),
            None,
            Some(&mask.as_view()),
            Some(prediction),
            &SimilarityConfig::default(),
        )
        .unwrap();
    assert_eq!(f.status, TranslationStatus::Accepted, "{f:?}");
    assert!(error(f.transform.unwrap(), truth, w, h) < 0.1, "{f:?}");
    let flat = Image::new_fill(w, h, 1f32);
    assert_eq!(
        e.estimate(
            &flat.as_view(),
            &flat.as_view(),
            None,
            None,
            None,
            &SimilarityConfig::default()
        )
        .unwrap()
        .status,
        TranslationStatus::LowTexture
    );
    let empty = Image::new_fill(w, h, 0u8);
    assert_eq!(
        e.estimate(
            &a,
            &b.as_view(),
            None,
            Some(&empty.as_view()),
            None,
            &SimilarityConfig::default()
        )
        .unwrap()
        .status,
        TranslationStatus::InsufficientSupport
    );
    assert!(
        e.estimate(
            &a,
            &b.as_view(),
            None,
            None,
            None,
            &SimilarityConfig::default()
        )
        .is_err()
    );
}
#[test]
fn bounds_rank_and_regional_unavailability() {
    let (w, h) = (161, 97);
    let a = render(w, h, Similarity2f::identity(), 1., 0.);
    let b = render(w, h, centered(w, h, [0.7, -0.3], 0.02, 1.01), 1., 0.);
    let mut cfg = SimilarityConfig {
        max_angle: 0.005,
        ..Default::default()
    };
    let mut e = SimilarityEstimator::new();
    assert_eq!(
        e.estimate(&a.as_view(), &b.as_view(), None, None, None, &cfg)
            .unwrap()
            .status,
        TranslationStatus::SearchBoundary
    );
    cfg = SimilarityConfig::default();
    cfg.translation.max_region_error = None;
    let f = e
        .estimate(&a.as_view(), &b.as_view(), None, None, None, &cfg)
        .unwrap();
    assert_eq!(f.status, TranslationStatus::Accepted);
    assert_eq!(f.region_error, None);
    assert_eq!(f.regions, 0);
    let line = Image::from_vec(w, h, (0..w * h).map(|i| (i % w) as f32).collect()).unwrap();
    assert_ne!(
        e.estimate(&line.as_view(), &line.as_view(), None, None, None, &cfg)
            .unwrap()
            .status,
        TranslationStatus::Accepted
    );
    cfg.max_log_scale = f32::NAN;
    assert!(
        e.estimate(&a.as_view(), &b.as_view(), None, None, None, &cfg)
            .is_err()
    );
}

#[test]
fn native_rank_loss_and_single_region_are_explicit() {
    let (w, h) = (161, 97);
    let line = Image::from_vec(
        w,
        h,
        (0..w * h).map(|i| ((i % w) as f32 * 0.17).sin()).collect(),
    )
    .unwrap();
    let mut a = Plane::default();
    a.load(&line.as_view(), None).unwrap();
    let pose = Pose {
        center: [80., 48.],
        radius: 100.,
        p: SVector::<f64, 6>::new(0., 0., 100., 0., 1., 0.),
    };
    assert!(matches!(
        refine::refine(
            &a,
            &a,
            pose,
            [0, w, 0, h],
            400,
            &SimilarityConfig::default().translation.tuning,
            &mut Vec::new()
        ),
        Err(TranslationStatus::LowTexture)
    ));
    let a = render(w, h, Similarity2f::identity(), 1., 0.);
    let truth = centered(w, h, [0.7, -0.3], 0.01, 1.01);
    let b = render(w, h, truth, 1., 0.);
    let mut mask = Image::new_fill(w, h, 0u8);
    for y in 4..44 {
        for x in 4..75 {
            mask.data_mut()[y * w + x] = 1;
        }
    }
    let f = SimilarityEstimator::new()
        .estimate(
            &a.as_view(),
            &b.as_view(),
            Some(&mask.as_view()),
            Some(&mask.as_view()),
            None,
            &SimilarityConfig::default(),
        )
        .unwrap();
    assert_eq!(f.status, TranslationStatus::Accepted, "{f:?}");
    assert_eq!(f.regions, 1);
    assert_eq!(f.region_error, None);
}

#[test]
fn incompatible_regions_do_not_become_a_similarity_pose() {
    let (w, h) = (161, 97);
    let a = render(w, h, Similarity2f::identity(), 1., 0.);
    let b = Image::from_vec(
        w,
        h,
        (0..w * h)
            .map(|i| {
                let y = (i / w) as f64;
                let x = (i % w) as f64;
                texture(x - if y < 48. { 0.8 } else { -0.8 }, y)
            })
            .collect(),
    )
    .unwrap();
    let f = SimilarityEstimator::new()
        .estimate(
            &a.as_view(),
            &b.as_view(),
            None,
            None,
            None,
            &SimilarityConfig::default(),
        )
        .unwrap();
    assert_eq!(f.status, TranslationStatus::ModelMismatch, "{f:?}");
    assert!(f.region_error.unwrap() > 0.5);
}
