//! `warp` benchmarks: VGA affine and polar `Map::apply`, bilinear, the map
//! built once outside the timed loop — the "build once, apply per frame"
//! usage this module is designed for.
//!
//! Run with `cargo bench -p vision-metrology --bench warp`.
//!
//! ## Measured numbers (2026-08-20, release, `lto = "thin"`, `codegen-units = 1`)
//!
//! | Benchmark                        | Time      |
//! |-----------------------------------|-----------|
//! | `affine_apply_640x480_bilinear`   | ~510 µs   |
//! | `polar_apply_640x480_bilinear`    | ~494 µs   |
//!
//! Both are dominated by the per-pixel bilinear gather over 640×480 =
//! 307 200 destination pixels (~1.6-1.7 ns/pixel) — `Map::apply` reads four
//! taps and blends per pixel with no per-pixel allocation, matching or
//! trigonometry (that work happened once, in the constructor). The two
//! numbers are close because `apply` does not know or care how its
//! coordinates were built; the cost is the same interior-gather loop either
//! way, and the small delta between them is where `affine`'s coordinates
//! trend more often into the border-fallback branch (a full-circle `polar`
//! map with its radius entirely inside a large source image never leaves
//! the fast path, while the affine case here has an identity-ish map with
//! some out-of-source corner pixels).

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vision_metrology::warp::{Interp, Map};
use vision_metrology::{Affine2f, BorderMode, Image, Point2f};

fn vga_ramp() -> Image<u8> {
    let (w, h) = (640usize, 480usize);
    let data: Vec<u8> = (0..w * h).map(|i| ((i % w) ^ (i / w)) as u8).collect();
    Image::from_vec(w, h, data).expect("valid image")
}

fn bench_affine_apply(c: &mut Criterion) {
    let (w, h) = (640usize, 480usize);
    let src = vga_ramp();
    let view = src.as_view();

    let rot = nalgebra::Rotation2::new(0.1f32);
    let t = nalgebra::Translation2::new(3.0f32, -2.0);
    let sim = t * rot;
    let m: Affine2f = nalgebra::convert(sim);
    let map = Map::affine(w, h, &m);
    let mut dst = vec![0u8; w * h];

    c.bench_function("affine_apply_640x480_bilinear", |b| {
        b.iter(|| {
            map.apply(
                black_box(&view),
                &mut dst,
                Interp::Bilinear,
                BorderMode::Clamp,
            )
            .expect("apply succeeds");
            black_box(&dst);
        });
    });
}

fn bench_polar_apply(c: &mut Criterion) {
    let (w, h) = (640usize, 480usize);
    let src = vga_ramp();
    let view = src.as_view();

    let map = Map::polar(
        Point2f::new(320.0, 240.0),
        10.0..220.0,
        0.0..core::f32::consts::TAU,
        w,
        h,
    );
    let mut dst = vec![0u8; w * h];

    c.bench_function("polar_apply_640x480_bilinear", |b| {
        b.iter(|| {
            map.apply(
                black_box(&view),
                &mut dst,
                Interp::Bilinear,
                BorderMode::Clamp,
            )
            .expect("apply succeeds");
            black_box(&dst);
        });
    });
}

criterion_group!(benches, bench_affine_apply, bench_polar_apply);
criterion_main!(benches);
