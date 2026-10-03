//! `measure` benchmarks: single calipers and a full `MetrologyModel::apply`.
//!
//! Run with `cargo bench -p vision-metrology --bench measure`.
//!
//! ## Measured numbers (2026-10-03, release, `lto = "thin"`, `codegen-units = 1`)
//!
//! | Benchmark                                | Time      |
//! |-------------------------------------------|-----------|
//! | `caliper_rect_pos_1280x1024`               | ~1.40 µs  |
//! | `metrology_model_apply_96_calipers`        | ~215 µs   |
//! | `caliper_strip_40px_81s_parabolic`         | ~0.52 µs  |
//! | `caliper_strip_40px_81s_midpoint`          | ~0.48 µs  |
//! | `caliper_strip_40px_81s_half_contrast`     | ~0.79 µs  |
//! | `caliper_strip_400px_801s_w15_a15`         | ~34.4 µs  |
//!
//! The single-caliper number is the cost of one `Caliper::measure` scan on a
//! 1280×1024 synthetic edge scene — a caliper only touches the pixels under
//! its own footprint (`2·half_len+1` samples × `2·half_width+1` averaging
//! rows), so this is independent of image size beyond cache effects.
//! `metrology_model_apply_96_calipers` is the cost of a full circle object
//! (96 calipers around a nominal 300 px-radius circle) run through `apply`,
//! including the robust `fit_circle` at the end: about 2.2 µs per caliper,
//! the single-caliper number plus the fit's own share.
//!
//! The strip benches use the textbook settings on `f32` bar images: a 40 px strip of
//! 81 samples on one line, and a 400 px strip of 801 samples averaged over 15 lines
//! (12 015 `f64` bilinear samples, about 2.8 ns each). The half-contrast bench refines
//! the parabolic bench's two edges, so the difference is the cost of a smoothing pass
//! and two flank-and-crossing iterations; the midpoint bench runs the same strip over a
//! step, since a bar returns to its starting level.
//!
//! Re-run and update this table whenever `measure`'s hot path changes.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::num::NonZeroUsize;
use vision_metrology::measure::{
    Caliper, Derivative, Locate, MeasureConfig, MeasureRect, MeasureStrip, MetrologyModel,
    MetrologyObject, MetrologyShape, OffImage, ProfileConfig,
};
use vision_metrology::{Image, Point2f, Similarity2f, Vec2f};

/// 1280×1024 scene: a single vertical step edge, antialiased over ~1 px so
/// there is a real subpixel position for the caliper to find, plus a large
/// bright disc (nominal radius 300, centred in-frame) for the model bench.
fn synthetic_edge_scene(w: usize, h: usize) -> Image<u8> {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let radius = 300.0f32;
    let mut data = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let d = (dx * dx + dy * dy).sqrt();
            // Antialiased disc edge at exactly `radius`.
            let cover = (radius + 0.5 - d).clamp(0.0, 1.0);
            data[y * w + x] = (20.0 + 180.0 * cover).round() as u8;
        }
    }
    Image::from_vec(w, h, data).expect("valid image")
}

fn bench_caliper_rect_pos(c: &mut Criterion) {
    let (w, h) = (1280usize, 1024usize);
    let img = synthetic_edge_scene(w, h);
    let view = img.as_view();

    // A radial caliper crossing the disc's rim at its top, where the edge is
    // locally horizontal — a representative single-caliper placement.
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let rect = MeasureRect {
        center: Point2f::new(cx, cy - 300.0),
        angle: std::f32::consts::FRAC_PI_2,
        half_len: 10.0,
        half_width: 5.0,
    };
    let mut cal = Caliper::rect(rect, MeasureConfig::default());

    c.bench_function("caliper_rect_pos_1280x1024", |b| {
        b.iter(|| {
            let edges = cal
                .measure(black_box(&view))
                .expect("edge under the caliper");
            black_box(edges.len());
        });
    });
}

fn bench_metrology_model_apply_96_calipers(c: &mut Criterion) {
    let (w, h) = (1280usize, 1024usize);
    let img = synthetic_edge_scene(w, h);
    let view = img.as_view();
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);

    let mut model = MetrologyModel::new();
    model.add(MetrologyObject {
        n_calipers: 96,
        caliper_len: 10.0,
        caliper_width: 5.0,
        ..MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(0.0, 0.0),
            radius: 300.0,
            arc: None,
        })
    });

    let fixture = Similarity2f::new(Vec2f::new(cx, cy), 0.0, 1.0);

    c.bench_function("metrology_model_apply_96_calipers", |b| {
        b.iter(|| {
            let results = model.apply(black_box(&view), black_box(&fixture));
            black_box(results.len());
        });
    });
}

/// A `w × h` f32 image in [0, 1] with a bright bar on columns `x0..x1`, blurred by
/// a 3-tap box so each edge has a subpixel position.
fn bar_scene_f32(w: usize, h: usize, x0: usize, x1: usize) -> Image<f32> {
    let row: Vec<f32> = (0..w)
        .map(|x| {
            let at = |x: isize| f32::from(u8::from((x0 as isize..x1 as isize).contains(&x)));
            let x = x as isize;
            (at(x - 1) + at(x) + at(x + 1)) / 3.0
        })
        .collect();
    let data = (0..h).flat_map(|_| row.iter().copied()).collect();
    Image::from_vec(w, h, data).expect("valid image")
}

/// The textbook settings for a strip sampled every `spacing` pixels: σ of one sample,
/// a radius-3 Gaussian, central differences, a three-point parabola, strict bounds.
fn strip_config(spacing: f32) -> MeasureConfig {
    MeasureConfig {
        threshold: 0.01,
        profile: ProfileConfig {
            sigma: spacing,
            derivative: Derivative::SmoothThenCentral {
                radius_px: 3.0 * spacing,
            },
            off_image: OffImage::Reject,
            ..ProfileConfig::default()
        },
        ..MeasureConfig::default()
    }
}

fn strip(start: (f32, f32), end: (f32, f32), half_width: f32, n: usize, a: usize) -> MeasureStrip {
    MeasureStrip {
        start: Point2f::new(start.0, start.1),
        end: Point2f::new(end.0, end.1),
        half_width,
        samples: NonZeroUsize::new(n),
        across: NonZeroUsize::new(a),
    }
}

fn bench_caliper_strip(c: &mut Criterion) {
    // 40 px, 81 samples (0.5 px apart), a single line: the shape of a typical
    // CaliperBench request.
    let small = bar_scene_f32(96, 96, 40, 60);
    let small_view = small.as_view();
    let mut cal = Caliper::strip(
        strip((28.0, 48.3), (68.0, 48.3), 0.0, 81, 1),
        strip_config(0.5),
    );
    c.bench_function("caliper_strip_40px_81s_parabolic", |b| {
        b.iter(|| {
            let edges = cal.measure(black_box(&small_view)).expect("two edges");
            black_box(edges.len());
        });
    });

    // The same strip, each gradient edge then moved to its local half-contrast level.
    cal.set_config(MeasureConfig {
        locate: Locate::HalfContrast {
            flank_near_px: 3.0,
            flank_far_px: 8.0,
            tol_px: 0.01,
            max_iter: NonZeroUsize::new(5).expect("nonzero"),
            min_contrast: 0.0,
        },
        ..strip_config(0.5)
    });
    c.bench_function("caliper_strip_40px_81s_half_contrast", |b| {
        b.iter(|| {
            let edges = cal.measure(black_box(&small_view)).expect("two edges");
            black_box(edges.len());
        });
    });

    // The midpoint needs different levels at the two ends: the same strip over a step.
    let step = bar_scene_f32(96, 96, 40, 96);
    let step_view = step.as_view();
    let mut cal = Caliper::strip(
        strip((28.0, 48.3), (68.0, 48.3), 0.0, 81, 1),
        MeasureConfig {
            locate: Locate::MidpointCrossing {
                endpoint_samples: NonZeroUsize::new(3).expect("nonzero"),
                min_contrast: 0.05,
            },
            ..strip_config(0.5)
        },
    );
    c.bench_function("caliper_strip_40px_81s_midpoint", |b| {
        b.iter(|| {
            let edges = cal.measure(black_box(&step_view)).expect("one edge");
            black_box(edges.len());
        });
    });

    // 400 px, 801 samples, 15 lines over a 15 px width: the sampling budget of a
    // wide, dense strip.
    let large = bar_scene_f32(512, 64, 150, 350);
    let large_view = large.as_view();
    let geom = strip((50.0, 31.6), (450.0, 33.1), 7.0, 801, 15);
    let mut cal = Caliper::strip(geom, strip_config(geom.length() / 800.0));
    c.bench_function("caliper_strip_400px_801s_w15_a15", |b| {
        b.iter(|| {
            let edges = cal.measure(black_box(&large_view)).expect("two edges");
            black_box(edges.len());
        });
    });
}

criterion_group!(
    benches,
    bench_caliper_rect_pos,
    bench_metrology_model_apply_96_calipers,
    bench_caliper_strip
);
criterion_main!(benches);
