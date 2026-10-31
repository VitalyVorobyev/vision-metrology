//! Phase 0 headroom bench (ADR-0023): the prototype against libblur 0.24, on one image.
//!
//! Run from `tools/filter_bench`: `cargo run --release`. Prints a markdown table; the
//! numbers and their machine are recorded in `README.md`.
//!
//! Fairness: the headline comparison is single-threaded on both sides. libblur's adaptive
//! threading is reported separately, because the prototype is single-threaded.

use libblur::{
    BlurImage, BlurImageMut, BoxBlurParameters, EdgeMode, EdgeMode2D, FastBlurChannels,
    GaussianBlurParams, IeeeBinaryConvolutionMode, ThreadingPolicy, box_blur, gaussian_blur_f32,
};
use std::hint::black_box;
use std::time::Instant;

mod proto;
use proto::{BoxU8, GaussF32};

const W: usize = 1280;
const H: usize = 1024;
const WARMUP: usize = 3;
const RUNS: usize = 25;

/// Best (minimum) wall time in milliseconds over `RUNS` timed calls, after `WARMUP` calls.
/// Minimum, not median: on a shared machine interference only adds time, so the minimum is
/// the closest thing to the kernel's own cost. The spread is printed beside it.
fn best_ms<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut t: Vec<f64> = (0..RUNS)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(|a, b| a.total_cmp(b));
    t[0]
}

fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn box_case(src8: &[u8], radius: usize) {
    // Exactness: the prototype must equal the naive integer reference bit for bit.
    let naive = proto::box_u8_naive(src8, W, H, radius);
    let mut scratch = BoxU8::new(W, H, radius);
    let mut mine = vec![0u8; W * H];
    scratch.apply(src8, W, H, radius, &mut mine);
    assert_eq!(mine, naive, "prototype box is not exact at r={radius}");

    let mut proto_t = best_ms(|| {
        scratch.apply(black_box(src8), W, H, radius, &mut mine);
        black_box(&mine);
    });
    proto_t = proto_t.max(0.0);

    // libblur: u8 plane, single-threaded, into a preallocated destination.
    let mut blur_dst = vec![0u8; W * H];
    let kernel = (2 * radius + 1) as u32;
    let params = BoxBlurParameters::new(kernel);
    let mut blur_err = vec![0u8; W * H];
    {
        let src = BlurImage::borrow(src8, W as u32, H as u32, FastBlurChannels::Plane);
        let mut dst =
            BlurImageMut::borrow(&mut blur_dst, W as u32, H as u32, FastBlurChannels::Plane);
        box_blur(&src, &mut dst, params, ThreadingPolicy::Single).expect("box_blur");
    }
    blur_err.copy_from_slice(&blur_dst);
    let blur_t = {
        let src = BlurImage::borrow(src8, W as u32, H as u32, FastBlurChannels::Plane);
        best_ms(|| {
            let mut dst =
                BlurImageMut::borrow(&mut blur_dst, W as u32, H as u32, FastBlurChannels::Plane);
            box_blur(&src, &mut dst, params, ThreadingPolicy::Single).expect("box_blur");
            black_box(&blur_dst);
        })
    };
    let blur_adaptive_t = {
        let src = BlurImage::borrow(src8, W as u32, H as u32, FastBlurChannels::Plane);
        best_ms(|| {
            let mut dst =
                BlurImageMut::borrow(&mut blur_dst, W as u32, H as u32, FastBlurChannels::Plane);
            box_blur(&src, &mut dst, params, ThreadingPolicy::Adaptive).expect("box_blur");
            black_box(&blur_dst);
        })
    };
    let blur_err_max = naive
        .iter()
        .zip(&blur_err)
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .max()
        .unwrap_or(0);

    println!(
        "| box u8 r={radius} (exact) | {proto_t:.3} | {blur_t:.3} | {:.2}x | {blur_adaptive_t:.3} | 0 (exact) | {blur_err_max} |",
        blur_t / proto_t
    );
}

fn gauss_case(src32: &[f32], ref64: &[f64], sigma: f64) {
    let mut scratch = GaussF32::new(W, H, sigma);
    let r = scratch.radius();
    let mut mine = vec![0f32; W * H];
    scratch.apply(src32, W, H, &mut mine);
    let mine64: Vec<f64> = mine.iter().map(|&v| v as f64).collect();
    let proto_err = max_abs_err(&mine64, ref64);
    let proto_t = best_ms(|| {
        scratch.apply(black_box(src32), W, H, &mut mine);
        black_box(&mine);
    });
    let mut saxpy = vec![0f32; W * H];
    scratch.apply_saxpy(src32, W, H, &mut saxpy);
    let saxpy64: Vec<f64> = saxpy.iter().map(|&v| v as f64).collect();
    let saxpy_err = max_abs_err(&saxpy64, ref64);
    let mut fused = vec![0f32; W * H];
    scratch.apply_fused(src32, W, H, &mut fused);
    let fused64: Vec<f64> = fused.iter().map(|&v| v as f64).collect();
    let fused_err = max_abs_err(&fused64, ref64);
    let fused_t = best_ms(|| {
        scratch.apply_fused(black_box(src32), W, H, &mut fused);
        black_box(&fused);
    });
    println!("| (fused mul_add) gauss sigma={sigma} | {fused_t:.3} | | | | {fused_err:.1e} | |");
    let saxpy_t = best_ms(|| {
        scratch.apply_saxpy(black_box(src32), W, H, &mut saxpy);
        black_box(&saxpy);
    });
    println!("| (saxpy baseline) gauss sigma={sigma} | {saxpy_t:.3} | | | | {saxpy_err:.1e} | |");

    // libblur: exact Gaussian with f64 accumulation (IeeeBinaryConvolutionMode::Normal is
    // its f32 path; see the gaussian_hint docs), the same support as the prototype.
    let kernel = (2 * r + 1) as u32;
    let params = GaussianBlurParams::new(kernel, sigma);
    let edge = EdgeMode2D::new(EdgeMode::Clamp);
    let mut blur_dst = vec![0f32; W * H];
    let run = |dst: &mut [f32], policy: ThreadingPolicy| {
        let src = BlurImage::borrow(src32, W as u32, H as u32, FastBlurChannels::Plane);
        let mut d = BlurImageMut::borrow(dst, W as u32, H as u32, FastBlurChannels::Plane);
        gaussian_blur_f32(
            &src,
            &mut d,
            params,
            edge,
            policy,
            IeeeBinaryConvolutionMode::Normal,
        )
        .expect("gaussian_blur_f32");
    };
    run(&mut blur_dst, ThreadingPolicy::Single);
    let blur64: Vec<f64> = blur_dst.iter().map(|&v| v as f64).collect();
    let blur_err = max_abs_err(&blur64, ref64);
    let blur_t = best_ms(|| {
        run(&mut blur_dst, ThreadingPolicy::Single);
        black_box(&blur_dst);
    });
    let blur_adaptive_t = best_ms(|| {
        run(&mut blur_dst, ThreadingPolicy::Adaptive);
        black_box(&blur_dst);
    });

    println!(
        "| gauss f32 sigma={sigma} (R={r}) | {proto_t:.3} | {blur_t:.3} | {:.2}x | {blur_adaptive_t:.3} | {proto_err:.1e} | {blur_err:.1e} |",
        blur_t / proto_t
    );
}

fn main() {
    let src8 = proto::test_image(W, H);
    let src32: Vec<f32> = src8.iter().map(|&v| v as f32).collect();

    println!(
        "image {W}x{H}, {RUNS} timed runs after {WARMUP} warm-up, best-of ms, single thread unless noted"
    );
    println!(
        "| case | prototype ms | libblur ms (1 thread) | libblur / prototype | libblur ms (adaptive) | prototype max err | libblur max err |"
    );
    println!("|---|---|---|---|---|---|---|");

    for r in [1usize, 2, 4, 8] {
        box_case(&src8, r);
    }
    for sigma in [1.0f64, 2.0, 4.0, 6.0] {
        let ref64 = proto::gaussian_f64_reference(&src32, W, H, sigma);
        gauss_case(&src32, &ref64, sigma);
    }

    // Context, not a comparison: libblur's approximate u8 Gaussian (`fast_gaussian`, an
    // O(1)-per-pixel recursive approximation with a radius, not a sigma). It bounds what a
    // large-sigma path can reach. Timed including a 1.3 MB copy into the in-place buffer,
    // so it reads a little pessimistic; no error column because its sigma is not exposed.
    for radius in [6u32, 18] {
        let mut buf = src8.clone();
        let t = best_ms(|| {
            buf.copy_from_slice(&src8);
            let mut img =
                BlurImageMut::borrow(&mut buf, W as u32, H as u32, FastBlurChannels::Plane);
            libblur::fast_gaussian(
                &mut img,
                libblur::AnisotropicRadius::new(radius),
                ThreadingPolicy::Single,
                EdgeMode2D::new(EdgeMode::Clamp),
            )
            .expect("fast_gaussian");
            black_box(&buf);
        });
        println!(
            "| (context) libblur fast_gaussian u8 radius={radius}, approximate | | {t:.3} | | | | |"
        );
    }
}
