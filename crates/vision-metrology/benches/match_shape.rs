use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vision_metrology::{
    Image, Rect2f, ShapeMatcher, ShapeModel, ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig,
};

const W: usize = 1280;
const H: usize = 1024;

/// Anti-aliased L-bracket rendered from its signed distance function, so the
/// benchmark scene has the same edge profile a real optic would produce.
fn render_bracket(w: usize, h: usize, cx: f32, cy: f32, angle: f32, scale: f32) -> Image<u8> {
    let (sn, cs) = angle.sin_cos();
    let mut data = vec![40u8; w * h];
    for y in 0..h {
        for x in 0..w {
            // Into model frame.
            let dx = (x as f32 - cx) / scale;
            let dy = (y as f32 - cy) / scale;
            let mx = cs * dx + sn * dy;
            let my = -sn * dx + cs * dy;
            let inside = ((-90.0..90.0).contains(&mx) && (-90.0..-30.0).contains(&my))
                || ((-90.0..-30.0).contains(&mx) && (-30.0..90.0).contains(&my));
            if inside {
                data[y * w + x] = 210;
            }
        }
    }
    Image::from_vec(w, h, data).expect("valid image")
}

fn model_and_scene() -> (ShapeModel, Image<u8>) {
    let reference = render_bracket(W, H, 640.0, 512.0, 0.0, 1.0);
    let roi = Rect2f {
        x: 520.0,
        y: 392.0,
        width: 240.0,
        height: 240.0,
    };
    let cfg = ShapeModelConfig {
        max_points: 800,
        ..Default::default()
    };
    let model = ShapeModelBuilder::new()
        .build_u8(&reference.as_view(), roi, &cfg)
        .expect("model builds");
    let scene = render_bracket(W, H, 700.0, 470.0, 0.9, 1.0);
    (model, scene)
}

fn bench_create(c: &mut Criterion) {
    let reference = render_bracket(W, H, 640.0, 512.0, 0.0, 1.0);
    let view = reference.as_view();
    let roi = Rect2f {
        x: 520.0,
        y: 392.0,
        width: 240.0,
        height: 240.0,
    };
    let cfg = ShapeModelConfig {
        max_points: 800,
        ..Default::default()
    };
    let mut builder = ShapeModelBuilder::new();
    c.bench_function("shape_model_create_1280x1024", |b| {
        b.iter(|| {
            let m = builder
                .build_u8(black_box(&view), roi, &cfg)
                .expect("model builds");
            black_box(m.point_count(0));
        });
    });
}

fn bench_find_360(c: &mut Criterion) {
    let (model, scene) = model_and_scene();
    let view = scene.as_view();
    let cfg = ShapeSearchConfig::default();
    let mut matcher = ShapeMatcher::new();
    c.bench_function("shape_find_1280x1024_360deg", |b| {
        b.iter(|| {
            let out = matcher.find_u8(black_box(&view), &model, &cfg);
            black_box(out.len());
        });
    });
}

/// The exhaustive reference: quantifies exactly what greediness buys.
fn bench_find_greedy0(c: &mut Criterion) {
    let (model, scene) = model_and_scene();
    let view = scene.as_view();
    let cfg = ShapeSearchConfig {
        greediness: 0.0,
        ..Default::default()
    };
    let mut matcher = ShapeMatcher::new();
    c.bench_function("shape_find_1280x1024_360deg_greedy0", |b| {
        b.iter(|| {
            let out = matcher.find_u8(black_box(&view), &model, &cfg);
            black_box(out.len());
        });
    });
}

fn bench_find_scale(c: &mut Criterion) {
    let reference = render_bracket(W, H, 640.0, 512.0, 0.0, 1.0);
    let roi = Rect2f {
        x: 520.0,
        y: 392.0,
        width: 240.0,
        height: 240.0,
    };
    let cfg = ShapeModelConfig {
        max_points: 800,
        scale_range: (0.8, 1.25),
        ..Default::default()
    };
    let model = ShapeModelBuilder::new()
        .build_u8(&reference.as_view(), roi, &cfg)
        .expect("model builds");
    let scene = render_bracket(W, H, 700.0, 470.0, 0.9, 1.1);
    let view = scene.as_view();
    let search = ShapeSearchConfig::default();
    let mut matcher = ShapeMatcher::new();
    c.bench_function("shape_find_1280x1024_scale_0p8_1p25", |b| {
        b.iter(|| {
            let out = matcher.find_u8(black_box(&view), &model, &search);
            black_box(out.len());
        });
    });
}

criterion_group!(
    benches,
    bench_create,
    bench_find_360,
    bench_find_greedy0,
    bench_find_scale
);
criterion_main!(benches);
