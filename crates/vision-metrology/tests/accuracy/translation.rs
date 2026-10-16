//! Masked native translation sweep with analytic texture and seeded noise.
use super::{Image, Lcg, Measured, mean_std};
use vision_metrology::corr::{TranslationConfig, TranslationEstimator, TranslationStatus};

fn render(shift: (f32, f32), gain: f32, offset: f32, noise: f32, seed: u64) -> Image<f32> {
    let mut rng = Lcg(seed);
    Image::from_vec(
        161,
        97,
        (0..161 * 97)
            .map(|i| {
                let x = (i % 161) as f32 - shift.0;
                let y = (i / 161) as f32 - shift.1;
                gain * (100.0
                    + 25.0 * (x * 0.17 + y * 0.13).sin()
                    + 20.0 * (x * 0.08 - y * 0.21).cos()
                    + 15.0 * (x * 0.31 + y * 0.07).sin()
                    + 10.0 * (x * 0.03 + y * 0.39).cos())
                    + offset
                    + rng.signed(noise)
            })
            .collect(),
    )
    .unwrap()
}

pub(super) fn sweep() -> Measured {
    measure(&[0.0, 2.0, 5.0])
}

pub(super) fn clean() -> Measured {
    measure(&[0.0])
}

fn measure(noises: &[f32]) -> Measured {
    let mut cells = Vec::new();
    let cfg = TranslationConfig::default();
    let mut estimator = TranslationEstimator::new();
    let mut mask = Image::new_fill(161, 97, 1u8);
    for y in 20..60 {
        for x in 40..100 {
            mask.data_mut()[y * 161 + x] = 0;
        }
    }
    for &noise in noises {
        for &dx in &[-0.8, 0.2, 0.7] {
            for &dy in &[-0.6, 0.3, 0.8] {
                let (mut ex, mut ey) = (Vec::new(), Vec::new());
                for seed in 1..=5 {
                    let a = render((0.0, 0.0), 1.0, 0.0, noise, seed * 971);
                    let b = render((dx, dy), 1.3, 17.0, noise, seed * 7919);
                    let result = estimator
                        .estimate(
                            &a.as_view(),
                            &b.as_view(),
                            Some(&mask.as_view()),
                            Some(&mask.as_view()),
                            None,
                            &cfg,
                        )
                        .unwrap();
                    assert_eq!(result.status, TranslationStatus::Accepted, "{result:?}");
                    let p = result.shift.unwrap();
                    ex.push(p.x - dx);
                    ey.push(p.y - dy);
                }
                cells.push(mean_std(&ex));
                cells.push(mean_std(&ey));
            }
        }
    }
    Measured::worst_of(cells.into_iter())
}
