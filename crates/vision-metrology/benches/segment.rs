use criterion::{Criterion, black_box, criterion_group, criterion_main};
use vision_metrology::Connectivity;
use vision_metrology::Image;
use vision_metrology::{
    AdaptiveThreshConfig, adaptive_threshold_u8, label_connected_components_u8, otsu_threshold_u8,
    watershed,
};

const W: usize = 1280;
const H: usize = 1024;

/// Build a binary image with a checkerboard pattern of 32×32 blocks.
fn checkerboard(w: usize, h: usize) -> Image<u8> {
    let data: Vec<u8> = (0..w * h)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            if ((x / 32) + (y / 32)).is_multiple_of(2) {
                255
            } else {
                0
            }
        })
        .collect();
    Image::from_vec(w, h, data).unwrap()
}

fn bench_otsu(c: &mut Criterion) {
    let img = checkerboard(W, H);
    c.bench_function("otsu_1280x1024", |b| {
        b.iter(|| otsu_threshold_u8(&black_box(img.as_view())))
    });
}

fn bench_adaptive(c: &mut Criterion) {
    let img = checkerboard(W, H);
    let cfg = AdaptiveThreshConfig {
        radius: 15,
        offset: 0.0,
    };
    c.bench_function("adaptive_threshold_1280x1024", |b| {
        b.iter(|| adaptive_threshold_u8(&black_box(img.as_view()), &cfg))
    });
}

fn bench_ccl_1280x1024(c: &mut Criterion) {
    let img = checkerboard(W, H);
    c.bench_function("ccl_1280x1024", |b| {
        b.iter(|| label_connected_components_u8(&black_box(img.as_view()), Connectivity::C8))
    });
}

fn bench_watershed(c: &mut Criterion) {
    // Flat gradient image, seeds at four corners.
    let data = vec![0.0f32; W * H];
    let grad = Image::from_vec(W, H, data).unwrap();
    let seeds = vec![(0usize, 0usize), (W - 1, 0), (0, H - 1), (W - 1, H - 1)];
    c.bench_function("watershed_1280x1024_4seeds", |b| {
        b.iter(|| watershed(&black_box(grad.as_view()), black_box(&seeds)))
    });
}

criterion_group!(
    benches,
    bench_otsu,
    bench_adaptive,
    bench_ccl_1280x1024,
    bench_watershed
);
criterion_main!(benches);
