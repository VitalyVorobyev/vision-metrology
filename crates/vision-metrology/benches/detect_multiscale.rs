use criterion::{Criterion, black_box, criterion_group, criterion_main};
use vision_metrology::Image;
use vision_metrology::{MultiScaleConfig, MultiScaleEdgeDetector};

fn build_step_image(w: usize, h: usize) -> Image<u8> {
    let data: Vec<u8> = (0..w * h)
        .map(|i| if (i % w) >= w / 2 { 200u8 } else { 0u8 })
        .collect();
    Image::from_vec(w, h, data).expect("valid image")
}

fn bench_multiscale_detect_u8_1280x1024_3levels(c: &mut Criterion) {
    let img = build_step_image(1280, 1024);
    let view = img.as_view();
    let mut det = MultiScaleEdgeDetector::new();
    let cfg = MultiScaleConfig {
        num_levels: 3,
        merge_duplicates: true,
        ..MultiScaleConfig::default()
    };

    // Warm-up: ensure scratch buffers are allocated.
    let _ = det.detect_u8(&view, &cfg);

    c.bench_function("multiscale_detect_u8_1280x1024_3levels", |b| {
        b.iter(|| {
            let edgels = det.detect_u8(black_box(&view), black_box(&cfg));
            black_box(edgels)
        })
    });
}

fn bench_multiscale_detect_u8_1280x1024_1level(c: &mut Criterion) {
    let img = build_step_image(1280, 1024);
    let view = img.as_view();
    let mut det = MultiScaleEdgeDetector::new();
    let cfg = MultiScaleConfig {
        num_levels: 1,
        merge_duplicates: false,
        ..MultiScaleConfig::default()
    };

    let _ = det.detect_u8(&view, &cfg);

    c.bench_function("multiscale_detect_u8_1280x1024_1level", |b| {
        b.iter(|| {
            let edgels = det.detect_u8(black_box(&view), black_box(&cfg));
            black_box(edgels)
        })
    });
}

criterion_group!(
    benches,
    bench_multiscale_detect_u8_1280x1024_3levels,
    bench_multiscale_detect_u8_1280x1024_1level,
);
criterion_main!(benches);
