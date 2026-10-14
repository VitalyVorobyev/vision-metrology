//! `corr` benchmarks: `find` over a VGA scene with a 64x64 template
//! (rotation off and on), and `displacement` over a 320x97 strip window —
//! the glue-rig strip shape.
//!
//! Run with `cargo bench -p vision-metrology --bench corr`.
//!
//! ## Measured numbers (2026-08-20, release, `lto = "thin"`, `codegen-units = 1`)
//!
//! | Benchmark                            | Time      |
//! |---------------------------------------|-----------|
//! | `corr_find_640x480_64x64_rotoff`      | ~4.60 ms  |
//! | `corr_find_640x480_64x64_roton`       | ~22.1 ms  |
//! | `corr_displacement_320x97_quadratic`  | ~1.60 ms  |
//! | `corr_displacement_320x97_lk`         | ~1.71 ms  |
//!
//! Rotation search costs ~4.8x translation-only here (an angle bank swept
//! at the coarse level, corrmatch's own cost — this wrapper adds nothing
//! per candidate). `displacement`'s Lucas-Kanade stage adds ~7% over the
//! quadratic-only baseline: 3 iterations over a 200x65 window, each one
//! bilinear sample of `curr` per template pixel plus a 2x2 solve — cheap
//! next to stage 1's own bounded ZNCC search.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vision_metrology::corr::{
    CorrConfig, CorrTemplate, CorrTemplateConfig, DisplacementConfig, Refine, TranslationConfig,
    TranslationEstimator, displacement, find,
};
use vision_metrology::{Image, Rect2f};

/// A textured ramp-plus-checker pattern: enough high-frequency content for
/// ZNCC to have a sharp peak, cheap to render, deterministic.
fn textured(w: usize, h: usize, phase: (f32, f32)) -> Image<u8> {
    let mut data = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 - phase.0;
            let fy = y as f32 - phase.1;
            let v = 128.0
                + 60.0 * (fx * 0.11).sin()
                + 40.0 * (fy * 0.17).cos()
                + 30.0 * ((fx * 0.05 + fy * 0.05).sin());
            data[y * w + x] = v.clamp(0.0, 255.0) as u8;
        }
    }
    Image::from_vec(w, h, data).expect("valid image")
}

fn bench_find(c: &mut Criterion) {
    let (w, h) = (640usize, 480usize);
    let reference = textured(w, h, (0.0, 0.0));
    let scene = textured(w, h, (3.0, -2.0));
    let rect = Rect2f {
        x: 288.0,
        y: 208.0,
        width: 64.0,
        height: 64.0,
    };

    let tpl_rot_off = CorrTemplate::from_image(
        &reference.as_view(),
        rect,
        &CorrTemplateConfig {
            rotation: false,
            ..Default::default()
        },
    )
    .expect("template compiles");
    let cfg_rot_off = CorrConfig {
        rotation: false,
        ..Default::default()
    };
    c.bench_function("corr_find_640x480_64x64_rotoff", |b| {
        b.iter(|| {
            let m = find(&tpl_rot_off, black_box(&scene.as_view()), &cfg_rot_off)
                .expect("find succeeds");
            black_box(m);
        });
    });

    let tpl_rot_on = CorrTemplate::from_image(
        &reference.as_view(),
        rect,
        &CorrTemplateConfig {
            rotation: true,
            ..Default::default()
        },
    )
    .expect("template compiles");
    let cfg_rot_on = CorrConfig {
        rotation: true,
        ..Default::default()
    };
    c.bench_function("corr_find_640x480_64x64_roton", |b| {
        b.iter(|| {
            let m =
                find(&tpl_rot_on, black_box(&scene.as_view()), &cfg_rot_on).expect("find succeeds");
            black_box(m);
        });
    });
}

fn bench_displacement(c: &mut Criterion) {
    let (w, h) = (320usize, 97usize);
    let prev = textured(w, h, (0.0, 0.0));
    let curr = textured(w, h, (0.7, 0.3));
    let window = Rect2f {
        x: 40.0,
        y: 16.0,
        width: 200.0,
        height: 65.0,
    };

    let cfg_quad = DisplacementConfig {
        window,
        search: (12, 12),
        refine: Refine::None,
        min_score: 0.0,
    };
    c.bench_function("corr_displacement_320x97_quadratic", |b| {
        b.iter(|| {
            let d = displacement(
                black_box(&prev.as_view()),
                black_box(&curr.as_view()),
                &cfg_quad,
            )
            .expect("displacement succeeds");
            black_box(d);
        });
    });

    let cfg_lk = DisplacementConfig {
        window,
        search: (12, 12),
        refine: Refine::LucasKanade { iters: 3 },
        min_score: 0.0,
    };
    c.bench_function("corr_displacement_320x97_lk", |b| {
        b.iter(|| {
            let d = displacement(
                black_box(&prev.as_view()),
                black_box(&curr.as_view()),
                &cfg_lk,
            )
            .expect("displacement succeeds");
            black_box(d);
        });
    });
}

fn bench_translation(c: &mut Criterion) {
    for (w, h) in [(320, 97), (1280, 1024)] {
        let texture = |dx: f32, dy: f32| {
            Image::from_vec(
                w,
                h,
                (0..w * h)
                    .map(|i| {
                        let x = (i % w) as f32 - dx;
                        let y = (i / w) as f32 - dy;
                        (100.0
                            + 25.0 * (x * 0.17 + y * 0.13).sin()
                            + 20.0 * (x * 0.08 - y * 0.21).cos()
                            + 15.0 * (x * 0.31 + y * 0.07).sin()
                            + 10.0 * (x * 0.03 + y * 0.39).cos())
                        .round() as u8
                    })
                    .collect(),
            )
            .unwrap()
        };
        let a = texture(0.0, 0.0);
        let b = texture(0.7, -0.3);
        let mut mask = Image::new_fill(w, h, 1u8);
        for y in h / 3..2 * h / 3 {
            for x in w / 3..2 * w / 3 {
                mask.data_mut()[y * w + x] = 0;
            }
        }
        let cfg = TranslationConfig::default();
        let mut estimator = TranslationEstimator::new();
        let prepared = estimator
            .estimate(
                &a.as_view(),
                &b.as_view(),
                Some(&mask.as_view()),
                Some(&mask.as_view()),
                None,
                &cfg,
            )
            .unwrap();
        assert_eq!(
            prepared.status,
            vision_metrology::corr::TranslationStatus::Accepted,
            "benchmark must exercise successful refinement: {prepared:?}"
        );
        c.bench_function(&format!("corr_translation_masked_{w}x{h}"), |bench| {
            bench.iter(|| {
                black_box(
                    estimator
                        .estimate(
                            black_box(&a.as_view()),
                            black_box(&b.as_view()),
                            Some(&mask.as_view()),
                            Some(&mask.as_view()),
                            None,
                            &cfg,
                        )
                        .unwrap(),
                )
            });
        });
    }
}

#[cfg(feature = "corr-experimental")]
fn bench_similarity(c: &mut Criterion) {
    use vision_metrology::corr::{SimilarityConfig, SimilarityEstimator, TranslationStatus};
    use vision_metrology::{Point2f, Similarity2f, Vec2f, transform_point};
    for (w, h) in [(320, 97), (1280, 1024)] {
        let center = Point2f::new((w - 1) as f32 * 0.5, (h - 1) as f32 * 0.5);
        let linear = Similarity2f::new(Vec2f::zeros(), 0.01, 1.005);
        let map = Similarity2f::new(
            center - transform_point(&linear, center) + Vec2f::new(0.7, -0.3),
            0.01,
            1.005,
        );
        let render = |map: Similarity2f| {
            let inverse = map.inverse();
            Image::from_vec(
                w,
                h,
                (0..w * h)
                    .map(|i| {
                        let p =
                            transform_point(&inverse, Point2f::new((i % w) as f32, (i / w) as f32));
                        let (x, y) = (p.x, p.y);
                        100. + 25. * (x * 0.17 + y * 0.13).sin()
                            + 20. * (x * 0.08 - y * 0.21).cos()
                            + 15. * (x * 0.31 + y * 0.07).sin()
                            + 10. * (x * 0.03 + y * 0.39).cos()
                    })
                    .collect(),
            )
            .unwrap()
        };
        let a = render(Similarity2f::identity());
        let b = render(map);
        let mut mask = Image::new_fill(w, h, 1u8);
        for y in h / 3..2 * h / 3 {
            for x in w / 3..2 * w / 3 {
                mask.data_mut()[y * w + x] = 0;
            }
        }
        let prediction = if w > 320 {
            let linear = Similarity2f::new(Vec2f::zeros(), 0.008, 1.004);
            Some(Similarity2f::new(
                center - transform_point(&linear, center) + Vec2f::new(0.4, -0.2),
                0.008,
                1.004,
            ))
        } else {
            None
        };
        let cfg = SimilarityConfig::default();
        let mut e = SimilarityEstimator::new();
        let estimate = e
            .estimate(
                &a.as_view(),
                &b.as_view(),
                Some(&mask.as_view()),
                Some(&mask.as_view()),
                prediction,
                &cfg,
            )
            .unwrap();
        assert_eq!(
            estimate.status,
            TranslationStatus::Accepted,
            "benchmark must refine successfully: {estimate:?}"
        );
        c.bench_function(
            &format!(
                "corr_similarity_masked_{w}x{h}{}",
                if prediction.is_some() {
                    "_predicted"
                } else {
                    ""
                }
            ),
            |bench| {
                bench.iter(|| {
                    black_box(
                        e.estimate(
                            black_box(&a.as_view()),
                            black_box(&b.as_view()),
                            Some(&mask.as_view()),
                            Some(&mask.as_view()),
                            prediction,
                            &cfg,
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }
}
#[cfg(feature = "corr-experimental")]
criterion_group!(
    benches,
    bench_find,
    bench_displacement,
    bench_translation,
    bench_similarity
);
#[cfg(not(feature = "corr-experimental"))]
criterion_group!(benches, bench_find, bench_displacement, bench_translation);
criterion_main!(benches);
