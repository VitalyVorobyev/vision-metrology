use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vm_primitives::{Image, Pixel, Pyramid};

fn ramp_u8(width: usize, height: usize) -> Image<u8> {
    Image::from_vec(
        width,
        height,
        (0..width * height).map(|i| (i % 251) as u8).collect(),
    )
    .expect("valid image")
}

/// One level of decimation, per source pixel type: the level-0 widening to
/// `f32` plus one 2×2 box mean, on a reused pyramid so it measures steady
/// state rather than allocation.
///
/// `Pyramid` is the only way in: the kernel is crate-private, and in
/// production it only ever reads `f32` (every level after 0 is `f32`).
fn bench_downsample_kernel(c: &mut Criterion) {
    let (w, h) = (1280usize, 1024usize);
    let src = ramp_u8(w, h);

    fn case<P: Pixel>(c: &mut Criterion, name: &str, src: &Image<P>) {
        let view = src.as_view();
        let mut pyr = Pyramid::new();
        c.bench_function(name, |b| {
            b.iter(|| {
                pyr.build(black_box(&view), 2);
                black_box(pyr.num_levels());
            });
        });
    }

    case(c, "downsample2x2_to_f32_u8_1280x1024", &src);

    let src16 = Image::from_vec(w, h, src.data().iter().map(|&v| u16::from(v)).collect())
        .expect("valid image");
    case(c, "downsample2x2_to_f32_u16_1280x1024", &src16);

    let src32 = Image::from_vec(w, h, src.data().iter().map(|&v| f32::from(v)).collect())
        .expect("valid image");
    case(c, "downsample2x2_to_f32_f32_1280x1024", &src32);
}

/// Full build on a reused pyramid: steady-state, allocation-free.
fn bench_pyramid_build(c: &mut Criterion) {
    let img = ramp_u8(1280, 1024);
    let view = img.as_view();
    let mut pyr = Pyramid::new();

    c.bench_function("pyramid_build_u8_6_levels_1280x1024", |b| {
        b.iter(|| {
            pyr.build(black_box(&view), 6);
            black_box(pyr.num_levels());
        });
    });
}

criterion_group!(benches, bench_downsample_kernel, bench_pyramid_build);
criterion_main!(benches);
