use criterion::{Criterion, black_box, criterion_group, criterion_main};
use vm_core::Image;
use vm_pyr::{PyramidF32, downsample2x2_mean_u8_to_f32};

fn bench_downsample_u8_to_f32(c: &mut Criterion) {
    let width = 1280usize;
    let height = 1024usize;
    let mut data = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        data.push((i % 251) as u8);
    }
    let img = Image::from_vec(width, height, data).expect("valid image");
    let view = img.as_view();

    c.bench_function("downsample2x2_mean_u8_to_f32_1280x1024", |b| {
        b.iter(|| {
            let out = downsample2x2_mean_u8_to_f32(black_box(&view));
            black_box(out);
        });
    });
}

fn bench_pyramid_build(c: &mut Criterion) {
    let width = 1280usize;
    let height = 1024usize;
    let mut data = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        data.push((i % 251) as u8);
    }
    let img = Image::from_vec(width, height, data).expect("valid image");
    let view = img.as_view();
    let mut pyr = PyramidF32::new();

    c.bench_function("pyramid_build_u8_6_levels_1280x1024", |b| {
        b.iter(|| {
            pyr.build_from_u8(black_box(&view), 6);
            black_box(pyr.num_levels());
        });
    });
}

criterion_group!(benches, bench_downsample_u8_to_f32, bench_pyramid_build);
criterion_main!(benches);
