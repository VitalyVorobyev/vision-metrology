//! `measure` benchmarks: a single caliper and a full `MetrologyModel::apply`.
//!
//! Run with `cargo bench -p vision-metrology --bench measure`.
//!
//! ## Measured numbers (2026-08-20, release, `lto = "thin"`, `codegen-units = 1`)
//!
//! | Benchmark                              | Time      |
//! |-----------------------------------------|-----------|
//! | `caliper_rect_pos_1280x1024`             | ~1.75 µs  |
//! | `metrology_model_apply_96_calipers`      | ~215 µs   |
//!
//! The single-caliper number is the cost of one `Caliper::measure` scan on a
//! 1280×1024 synthetic edge scene — a caliper only touches the pixels under
//! its own footprint (`2·half_len+1` samples × `2·half_width+1` averaging
//! rows), so this is independent of image size beyond cache effects.
//! `metrology_model_apply_96_calipers` is the cost of a full circle object
//! (96 calipers around a nominal 300 px-radius circle) run through `apply`,
//! including the robust `fit_circle` at the end: 215 µs / 96 ≈ 2.24 µs per
//! caliper, close to the single-caliper number plus the fit's own share.
//!
//! Re-run and update this table whenever `measure`'s hot path changes.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vision_metrology::measure::{
    Caliper, MeasureConfig, MeasureRect, MetrologyModel, MetrologyObject, MetrologyShape,
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

criterion_group!(
    benches,
    bench_caliper_rect_pos,
    bench_metrology_model_apply_96_calipers
);
criterion_main!(benches);
