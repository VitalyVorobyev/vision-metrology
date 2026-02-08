use criterion::{Criterion, black_box, criterion_group, criterion_main};
use vm_core::Image;
use vm_edge::{Edge2DConfig, Edge2DDetector};

fn build_slanted_u8(width: usize, height: usize) -> Image<u8> {
    let theta = 20.0f32.to_radians();
    let nx = theta.cos();
    let ny = theta.sin();
    let t = nx * (0.5 * width as f32) + ny * (0.5 * height as f32);

    let mut data = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let d = nx * x as f32 + ny * y as f32 - t;
            data[y * width + x] = if d >= 0.0 { 255 } else { 0 };
        }
    }

    Image::from_vec(width, height, data).expect("valid image")
}

fn bench_edge2d_u8(c: &mut Criterion) {
    let img = build_slanted_u8(1280, 1024);
    let view = img.as_view();
    let cfg = Edge2DConfig::default();
    let mut det = Edge2DDetector::new();

    c.bench_function("edge2d_detect_u8_1280x1024", |b| {
        b.iter(|| {
            let out = det.detect_u8(black_box(&view), black_box(&cfg));
            black_box(out.len());
        });
    });
}

criterion_group!(benches, bench_edge2d_u8);
criterion_main!(benches);
