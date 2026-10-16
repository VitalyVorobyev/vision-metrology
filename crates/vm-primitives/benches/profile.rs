//! `ProfileSampler` benchmarks: stations × offsets × σ on a 1280×1024 `u8` image.
//!
//! Run with `cargo bench -p vm-primitives --bench profile`. The numbers are in
//! `docs/performance.md`.
//!
//! The stations sit on a circle of radius 300 px around the image centre, their
//! directions radial, so the profiles cross the disc's rim the way calipers along a
//! curve's normals do. Every sample is inside the image. The cost of one sample grows
//! with the support, about `π (4σ)²` pixels.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use vm_primitives::edge::{ProfileSampler, ProfileSamplerConfig, ProfileStation};
use vm_primitives::{Image, Point2f, Vec2f};

/// A bright disc of radius 300 px, anti-aliased, on a dark 1280×1024 frame.
fn disc_scene(w: usize, h: usize) -> Image<u8> {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let data = (0..w * h)
        .map(|i| {
            let (dx, dy) = ((i % w) as f32 - cx, (i / w) as f32 - cy);
            let cover = (300.5 - (dx * dx + dy * dy).sqrt()).clamp(0.0, 1.0);
            (20.0 + 180.0 * cover).round() as u8
        })
        .collect();
    Image::from_vec(w, h, data).expect("valid image")
}

fn stations(count: usize, w: usize, h: usize) -> Vec<ProfileStation> {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    (0..count)
        .map(|k| {
            let a = std::f32::consts::TAU * k as f32 / count as f32;
            let (s, c) = a.sin_cos();
            ProfileStation {
                p: Point2f::new(cx + 300.0 * c, cy + 300.0 * s),
                n: Vec2f::new(c, s),
            }
        })
        .collect()
}

fn bench_profile(c: &mut Criterion) {
    let (w, h) = (1280usize, 1024usize);
    let img = disc_scene(w, h);
    let view = img.as_view();
    let mut sampler = ProfileSampler::new();
    for &count in &[100usize, 400] {
        let st = stations(count, w, h);
        for &n_off in &[11usize, 41] {
            // Offsets 0.5 px apart, centred on the rim.
            let offsets: Vec<f32> = (0..n_off)
                .map(|j| 0.5 * (j as f32 - (n_off - 1) as f32 / 2.0))
                .collect();
            let mut out = vec![None; count * n_off];
            for &sigma in &[1.0f32, 2.5] {
                let cfg = ProfileSamplerConfig { sigma };
                let id = format!("profile_sample_{count}st_{n_off}off_sigma{sigma}_1280x1024");
                c.bench_function(&id, |b| {
                    b.iter(|| {
                        sampler
                            .sample(black_box(&view), None, &st, &offsets, &cfg, &mut out)
                            .expect("valid input");
                        black_box(out[0]);
                    });
                });
            }
        }
    }
}

criterion_group!(benches, bench_profile);
criterion_main!(benches);
