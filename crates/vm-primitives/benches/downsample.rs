use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vm_primitives::{Image, Pixel, Pyramid, downsample2x2_mean_to_f32_into};

fn ramp_u8(width: usize, height: usize) -> Image<u8> {
    Image::from_vec(
        width,
        height,
        (0..width * height).map(|i| (i % 251) as u8).collect(),
    )
    .expect("valid image")
}

/// The single-level kernel the pyramid is built from, measured without the
/// destination allocation so it reports kernel throughput rather than malloc.
///
/// One case per pixel type: the generic loop should be within noise of each
/// other for `u8`/`u16` (same `u32` accumulator) and slightly cheaper for `f32`
/// (no widening).
fn bench_downsample_kernel(c: &mut Criterion) {
    let (w, h) = (1280usize, 1024usize);
    let src = ramp_u8(w, h);
    let mut dst = Image::new_fill(w / 2, h / 2, 0.0f32);

    fn case<P: Pixel>(c: &mut Criterion, name: &str, src: &Image<P>, dst: &mut Image<f32>) {
        let view = src.as_view();
        c.bench_function(name, |b| {
            b.iter(|| {
                let mut out = dst.as_view_mut();
                downsample2x2_mean_to_f32_into(black_box(&view), &mut out).expect("dims");
            });
        });
    }

    case(c, "downsample2x2_to_f32_u8_1280x1024", &src, &mut dst);

    let src16 = Image::from_vec(w, h, src.data().iter().map(|&v| u16::from(v)).collect())
        .expect("valid image");
    case(c, "downsample2x2_to_f32_u16_1280x1024", &src16, &mut dst);

    let src32 = Image::from_vec(w, h, src.data().iter().map(|&v| f32::from(v)).collect())
        .expect("valid image");
    case(c, "downsample2x2_to_f32_f32_1280x1024", &src32, &mut dst);
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
