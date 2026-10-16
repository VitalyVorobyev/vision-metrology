//! Known analytic similarity transforms with photometry, masking and seeded noise.
use super::{Image, Lcg, Measured, mean_std};
use vision_metrology::corr::{SimilarityConfig, SimilarityEstimator, TranslationStatus};
use vision_metrology::{Point2f, Similarity2f, Vec2f};
use vm_primitives::{similarity_parts, transform_point, wrap_angle};
fn texture(x: f64, y: f64) -> f32 {
    (100.
        + 25. * (x * 0.17 + y * 0.13).sin()
        + 20. * (x * 0.08 - y * 0.21).cos()
        + 15. * (x * 0.31 + y * 0.07).sin()
        + 10. * (x * 0.03 + y * 0.39).cos()) as f32
}
fn render(map: Similarity2f, gain: f32, offset: f32, noise: f32, seed: u64) -> Image<f32> {
    let mut rng = Lcg(seed);
    let inverse = map.inverse();
    Image::from_vec(
        161,
        97,
        (0..161 * 97)
            .map(|i| {
                let p = transform_point(&inverse, Point2f::new((i % 161) as f32, (i / 161) as f32));
                gain * texture(p.x as f64, p.y as f64) + offset + rng.signed(noise)
            })
            .collect(),
    )
    .unwrap()
}
fn results() -> &'static [Measured; 4] {
    static RESULTS: std::sync::OnceLock<[Measured; 4]> = std::sync::OnceLock::new();
    RESULTS.get_or_init(|| {
        let mut cells: [Vec<(f32, f32)>; 4] = std::array::from_fn(|_| Vec::new());
        let cfg = SimilarityConfig::default();
        let mut e = SimilarityEstimator::new();
        let mut mask = Image::new_fill(161, 97, 1u8);
        for y in 25..55 {
            for x in 40..100 {
                mask.data_mut()[y * 161 + x] = 0;
            }
        }
        for noise in [0., 2., 5.] {
            for angle in [-0.02, 0.01, 0.03] {
                for scale in [0.985, 1.02] {
                    let c = Point2f::new(80., 48.);
                    let linear = Similarity2f::new(Vec2f::zeros(), angle, scale);
                    let truth = Similarity2f::new(
                        c - transform_point(&linear, c) + Vec2f::new(0.7, -0.3),
                        angle,
                        scale,
                    );
                    let mut errors: [Vec<f32>; 10] = std::array::from_fn(|_| Vec::new());
                    for seed in 1..=5 {
                        let a = render(Similarity2f::identity(), 1., 0., noise, seed * 971);
                        let b = render(truth, 1.3, 17., noise, seed * 7919);
                        let f = e
                            .estimate(
                                &a.as_view(),
                                &b.as_view(),
                                Some(&mask.as_view()),
                                Some(&mask.as_view()),
                                None,
                                &cfg,
                            )
                            .unwrap();
                        assert_eq!(f.status, TranslationStatus::Accepted, "{f:?}");
                        let estimate = f.transform.unwrap();
                        for (i, p) in [
                            Point2f::new(0., 0.),
                            Point2f::new(160., 0.),
                            Point2f::new(0., 96.),
                            Point2f::new(160., 96.),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            let d = transform_point(&estimate, p) - transform_point(&truth, p);
                            errors[2 * i].push(d.x);
                            errors[2 * i + 1].push(d.y);
                        }
                        let (_, a, s) = similarity_parts(&estimate);
                        errors[8].push(wrap_angle(a - angle));
                        errors[9].push((s / scale).ln());
                    }
                    for errors in &errors[..8] {
                        let cell = mean_std(errors);
                        cells[1].push(cell);
                        if noise == 0. {
                            cells[0].push(cell);
                        }
                    }
                    cells[2].push(mean_std(&errors[8]));
                    cells[3].push(mean_std(&errors[9]));
                }
            }
        }
        cells.map(|cells| Measured::worst_of(cells.into_iter()))
    })
}
pub(super) fn clean() -> Measured {
    results()[0]
}
pub(super) fn point() -> Measured {
    results()[1]
}
pub(super) fn angle() -> Measured {
    results()[2]
}
pub(super) fn log_scale() -> Measured {
    results()[3]
}
