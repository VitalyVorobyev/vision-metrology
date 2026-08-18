use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vm_primitives::{Image, chamfer_distance_u8, open3x3_binary_u8, thin_binary_u8};

/// A deterministic binary scene with structure at several scales: filled
/// discs, a thick diagonal band and thin lines — enough foreground to make
/// the two-pass chamfer sweep and the iterative thinning do real work.
fn binary_scene(width: usize, height: usize) -> Image<u8> {
    let mut data = vec![0u8; width * height];
    let put = |data: &mut Vec<u8>, x: usize, y: usize| {
        data[y * width + x] = 255;
    };

    // Discs on a coarse grid.
    for cy in (60..height).step_by(160) {
        for cx in (60..width).step_by(160) {
            let r = 24i32;
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dy * dy <= r * r {
                        let (x, y) = (cx as i32 + dx, cy as i32 + dy);
                        if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
                            put(&mut data, x as usize, y as usize);
                        }
                    }
                }
            }
        }
    }
    // Thick diagonal band.
    for y in 0..height {
        for x in 0..width {
            let d = x as i32 - y as i32;
            if (100..140).contains(&d) {
                put(&mut data, x, y);
            }
        }
    }
    // Thin horizontal lines.
    for y in (30..height).step_by(97) {
        for x in 0..width {
            put(&mut data, x, y);
        }
    }

    Image::from_vec(width, height, data).expect("valid image")
}

fn bench_chamfer(c: &mut Criterion) {
    let img = binary_scene(1280, 1024);
    let view = img.as_view();
    c.bench_function("morph_chamfer_distance_u8_1280x1024", |b| {
        b.iter(|| {
            let dt = chamfer_distance_u8(black_box(&view));
            black_box(dt.data()[0]);
        });
    });
}

fn bench_thin(c: &mut Criterion) {
    let img = binary_scene(1280, 1024);
    let view = img.as_view();
    c.bench_function("morph_thin_zhang_suen_1280x1024", |b| {
        b.iter(|| {
            let sk = thin_binary_u8(black_box(&view));
            black_box(sk.data()[0]);
        });
    });
}

fn bench_open3x3(c: &mut Criterion) {
    let img = binary_scene(1280, 1024);
    let view = img.as_view();
    c.bench_function("morph_open3x3_1280x1024", |b| {
        b.iter(|| {
            let out = open3x3_binary_u8(black_box(&view));
            black_box(out.data()[0]);
        });
    });
}

criterion_group!(benches, bench_chamfer, bench_thin, bench_open3x3);
criterion_main!(benches);
