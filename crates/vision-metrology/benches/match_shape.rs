use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vision_metrology::matching::{
    ShapeMatcher, ShapeModel, ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig,
};
use vision_metrology::{Image, Rect2f};

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
        .build(&reference.as_view(), roi, &cfg)
        .expect("model builds");
    let scene = render_bracket(W, H, 700.0, 470.0, 0.9, 1.0);
    (model, scene)
}

/// The bracket in a scene with real clutter statistics: seeded LCG texture,
/// gradient shading and a field of distractor rectangles. Greedy early
/// termination behaves very differently here than on a flat background, so
/// perf numbers quoted for the matcher must come from this fixture too.
fn cluttered_scene() -> Image<u8> {
    let base = render_bracket(W, H, 700.0, 470.0, 0.9, 1.0);
    let mut data = base.data().to_vec();

    // Deterministic texture + illumination gradient.
    let mut state = 0x1234_5678_9abc_def0u64;
    for y in 0..H {
        for x in 0..W {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = ((state >> 33) & 0x1f) as i32 - 16; // ±16 DN
            let shade = (x as f32 * 0.01) as i32; // slow horizontal ramp
            let v = i32::from(data[y * W + x]) + noise + shade;
            data[y * W + x] = v.clamp(0, 255) as u8;
        }
    }

    // Distractor rectangles away from the object.
    let mut st = 0xfeed_beef_dead_c0deu64;
    for _ in 0..40 {
        let mut next = |m: usize| {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((st >> 33) as usize) % m
        };
        let (rx, ry) = (next(W - 80), next(H - 80));
        if (rx as f32 - 700.0).abs() < 260.0 && (ry as f32 - 470.0).abs() < 260.0 {
            continue; // keep the object unoccluded
        }
        let (rw, rh) = (20 + next(50), 20 + next(50));
        let bright = 80 + next(120) as i32;
        for y in ry..(ry + rh).min(H) {
            for x in rx..(rx + rw).min(W) {
                data[y * W + x] = bright as u8;
            }
        }
    }

    Image::from_vec(W, H, data).expect("valid image")
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
                .build(black_box(&view), roi, &cfg)
                .expect("model builds");
            black_box(m.point_count(0));
        });
    });
}

/// Tracking mode: the pose from the previous frame bounds both the position
/// (±60 px ROI) and the angle (±10°). This is the production steady state on
/// a conveyor line — the full-360° benches are the cold-start case.
fn bench_find_tracked(c: &mut Criterion) {
    let (model, scene) = model_and_scene();
    let view = scene.as_view();
    let cfg = ShapeSearchConfig {
        roi: Some(Rect2f {
            x: 640.0,
            y: 410.0,
            width: 120.0,
            height: 120.0,
        }),
        angle_range: Some((0.9 - 0.17, 0.9 + 0.17)),
        ..Default::default()
    };
    let mut matcher = ShapeMatcher::new();
    assert!(!matcher.find(&view, &model, &cfg).is_empty());
    c.bench_function("shape_find_1280x1024_tracked_roi", |b| {
        b.iter(|| {
            let out = matcher.find(black_box(&view), black_box(&model), black_box(&cfg));
            black_box(out.len());
        });
    });
}

fn bench_find_360_clutter(c: &mut Criterion) {
    let (model, _) = model_and_scene();
    let scene = cluttered_scene();
    let view = scene.as_view();
    let cfg = ShapeSearchConfig::default();
    let mut matcher = ShapeMatcher::new();
    // Fail loudly if the fixture stops finding the object -- a perf number
    // for a search that no longer succeeds would be meaningless.
    assert!(
        !matcher.find(&view, &model, &cfg).is_empty(),
        "cluttered fixture must still contain a findable object"
    );
    c.bench_function("shape_find_1280x1024_360deg_clutter", |b| {
        b.iter(|| {
            let out = matcher.find(black_box(&view), black_box(&model), black_box(&cfg));
            black_box(out.len());
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
            let out = matcher.find(black_box(&view), &model, &cfg);
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
            let out = matcher.find(black_box(&view), &model, &cfg);
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
        .build(&reference.as_view(), roi, &cfg)
        .expect("model builds");
    let scene = render_bracket(W, H, 700.0, 470.0, 0.9, 1.1);
    let view = scene.as_view();
    let search = ShapeSearchConfig::default();
    let mut matcher = ShapeMatcher::new();
    c.bench_function("shape_find_1280x1024_scale_0p8_1p25", |b| {
        b.iter(|| {
            let out = matcher.find(black_box(&view), &model, &search);
            black_box(out.len());
        });
    });
}

criterion_group!(
    benches,
    bench_create,
    bench_find_360,
    bench_find_360_clutter,
    bench_find_tracked,
    bench_find_greedy0,
    bench_find_scale
);
criterion_main!(benches);
