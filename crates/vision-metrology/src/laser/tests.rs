//! End-to-end tests for the laser extractor across scan modes and pixel types.
//!
//! Everything here drives the public API only, so the tests live once at the
//! module level rather than per submodule.

use vm_primitives::Image;
use vm_primitives::{DoGKernel1D, edge::convolve_f32};

use crate::Error;
use crate::laser::{ColAccess, LaserExtractConfig, LaserExtractor, ScanAxis};

fn frac_overlap(i: usize, left: f32, right: f32) -> f32 {
    let x0 = i as f32 - 0.5;
    let x1 = i as f32 + 0.5;
    (x1.min(right) - x0.max(left)).clamp(0.0, 1.0)
}

fn blur_rows(img_f: &mut [f32], width: usize, height: usize, sigma: f32) {
    let k = DoGKernel1D::new(sigma);
    let mut tmp = vec![0.0f32; width];
    for y in 0..height {
        let row = &img_f[y * width..(y + 1) * width];
        convolve_f32(
            row,
            &k.g,
            k.radius,
            vm_primitives::BorderMode::Clamp,
            &mut tmp,
        );
        img_f[y * width..(y + 1) * width].copy_from_slice(&tmp);
    }
}

fn blur_cols(img_f: &mut [f32], width: usize, height: usize, sigma: f32) {
    let k = DoGKernel1D::new(sigma);
    let mut col = vec![0.0f32; height];
    let mut out = vec![0.0f32; height];
    for x in 0..width {
        for y in 0..height {
            col[y] = img_f[y * width + x];
        }
        convolve_f32(
            &col,
            &k.g,
            k.radius,
            vm_primitives::BorderMode::Clamp,
            &mut out,
        );
        for y in 0..height {
            img_f[y * width + x] = out[y];
        }
    }
}

fn transpose_u8(img: &Image<u8>) -> Image<u8> {
    let w = img.width();
    let h = img.height();
    let mut out = vec![0u8; w * h];
    let src = img.data();
    for y in 0..h {
        for x in 0..w {
            out[x * h + y] = src[y * w + x];
        }
    }
    Image::from_vec(h, w, out).expect("valid transposed image")
}

#[test]
fn rows_mode_basic() {
    let (w, h) = (64usize, 40usize);
    let (x_l, x_r) = (20.3f32, 25.7f32);

    let mut img_f = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            img_f[y * w + x] = 255.0 * frac_overlap(x, x_l, x_r);
        }
    }
    blur_rows(&mut img_f, w, h, 0.8);

    let img_u8: Vec<u8> = img_f
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    let img = Image::from_vec(w, h, img_u8).expect("valid image");

    let mut ext = LaserExtractor::new(1.2);
    let cfg = LaserExtractConfig {
        axis: ScanAxis::Rows,
        ..LaserExtractConfig::default()
    };

    let line = ext
        .extract_line(&img.as_view(), 0..h, &cfg, None)
        .expect("valid extractor arguments");

    let exp_c = 0.5 * (x_l + x_r);
    let exp_w = x_r - x_l;
    let mut valid = 0usize;
    let mut sum_err = 0.0f32;
    let mut sum_werr = 0.0f32;
    for s in &line.samples {
        if s.valid {
            valid += 1;
            sum_err += (s.center - exp_c).abs();
            sum_werr += (s.width - exp_w).abs();
        }
    }

    assert!(valid >= h - 2);
    let mean_err = sum_err / valid as f32;
    let mean_werr = sum_werr / valid as f32;
    assert!(mean_err <= 0.15);
    assert!(mean_werr <= 0.4);
}

#[test]
fn cols_mode_basic_gather() {
    let (w, h) = (64usize, 40usize);
    let (y_l, y_r) = (10.2f32, 15.6f32);

    let mut img_f = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            img_f[y * w + x] = 255.0 * frac_overlap(y, y_l, y_r);
        }
    }
    blur_cols(&mut img_f, w, h, 0.8);

    let img_u8: Vec<u8> = img_f
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    let img = Image::from_vec(w, h, img_u8).expect("valid image");

    let mut ext = LaserExtractor::new(1.2);
    let cfg = LaserExtractConfig {
        axis: ScanAxis::Cols {
            access: ColAccess::Gather,
        },
        ..LaserExtractConfig::default()
    };

    let line = ext
        .extract_line(&img.as_view(), 0..w, &cfg, None)
        .expect("valid extractor arguments");

    let exp_c = 0.5 * (y_l + y_r);
    let mut valid = 0usize;
    let mut sum_err = 0.0f32;
    for s in &line.samples {
        if s.valid {
            valid += 1;
            sum_err += (s.center - exp_c).abs();
        }
    }

    assert!(valid >= w - 2);
    assert!(sum_err / valid as f32 <= 0.15);
}

#[test]
fn sloped_gaps_reflections_rows_and_cols() {
    let (w, h) = (96usize, 72usize);
    let stripe_w = 5.2f32;

    let mut img_rows = vec![0.0f32; w * h];
    for y in 0..h {
        let main_center = 18.0 + 0.22 * (y as f32);
        let x_l = main_center - stripe_w * 0.5;
        let x_r = main_center + stripe_w * 0.5;

        if !(28..=34).contains(&y) {
            for x in 0..w {
                img_rows[y * w + x] += 255.0 * frac_overlap(x, x_l, x_r);
            }
        }

        if y % 3 == 0 {
            let refl_c = main_center + 13.0;
            let rl = refl_c - stripe_w * 0.4;
            let rr = refl_c + stripe_w * 0.4;
            for x in 0..w {
                img_rows[y * w + x] += 90.0 * frac_overlap(x, rl, rr);
            }
        }
    }
    blur_rows(&mut img_rows, w, h, 0.8);
    let img_rows_u8: Vec<u8> = img_rows
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    let img_r = Image::from_vec(w, h, img_rows_u8).expect("valid image");

    let mut ext = LaserExtractor::new(1.2);
    let cfg_rows = LaserExtractConfig {
        axis: ScanAxis::Rows,
        max_gap_scans: 4,
        max_jump_px: 8.0,
        ..LaserExtractConfig::default()
    };
    let line_r = ext
        .extract_line(&img_r.as_view(), 0..h, &cfg_rows, None)
        .expect("valid extractor arguments");

    let gap_valid = line_r.samples[28..=34].iter().filter(|s| s.valid).count();
    assert!(gap_valid <= 2);
    assert!(line_r.samples[40..].iter().filter(|s| s.valid).count() >= 20);

    let mut err_sum = 0.0f32;
    let mut err_count = 0usize;
    for s in &line_r.samples {
        if s.valid && !(28..=34).contains(&s.scan_i) {
            let true_c = 18.0 + 0.22 * (s.scan_i as f32);
            err_sum += (s.center - true_c).abs();
            err_count += 1;
        }
    }
    assert!(err_count > 20);
    assert!(err_sum / err_count as f32 <= 0.5);

    // Build the analogous horizontal/sloped case for Cols.
    let mut img_cols = vec![0.0f32; w * h];
    for x in 0..w {
        let main_center = 12.0 + 0.16 * (x as f32);
        let y_l = main_center - stripe_w * 0.5;
        let y_r = main_center + stripe_w * 0.5;

        if !(22..=28).contains(&x) {
            for y in 0..h {
                img_cols[y * w + x] += 255.0 * frac_overlap(y, y_l, y_r);
            }
        }

        if x % 4 == 0 {
            let refl_c = main_center + 11.5;
            let rl = refl_c - stripe_w * 0.4;
            let rr = refl_c + stripe_w * 0.4;
            for y in 0..h {
                img_cols[y * w + x] += 90.0 * frac_overlap(y, rl, rr);
            }
        }
    }
    blur_cols(&mut img_cols, w, h, 0.8);
    let img_cols_u8: Vec<u8> = img_cols
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    let img_c = Image::from_vec(w, h, img_cols_u8).expect("valid image");

    let cfg_cols = LaserExtractConfig {
        axis: ScanAxis::Cols {
            access: ColAccess::Gather,
        },
        max_gap_scans: 4,
        max_jump_px: 8.0,
        ..LaserExtractConfig::default()
    };
    let line_c = ext
        .extract_line(&img_c.as_view(), 0..w, &cfg_cols, None)
        .expect("valid extractor arguments");

    let gap_valid_c = line_c.samples[22..=28].iter().filter(|s| s.valid).count();
    assert!(gap_valid_c <= 2);
    assert!(line_c.samples[34..].iter().filter(|s| s.valid).count() >= 30);
}

#[test]
fn cols_transposed_matches_gather() {
    let (w, h) = (64usize, 40usize);
    let (y_l, y_r) = (10.2f32, 15.6f32);

    let mut img_f = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            img_f[y * w + x] = 255.0 * frac_overlap(y, y_l, y_r);
        }
    }
    blur_cols(&mut img_f, w, h, 0.8);

    let img_u8: Vec<u8> = img_f
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    let img = Image::from_vec(w, h, img_u8).expect("valid image");
    let img_t = transpose_u8(&img);

    let mut ext = LaserExtractor::new(1.2);
    let cfg_g = LaserExtractConfig {
        axis: ScanAxis::Cols {
            access: ColAccess::Gather,
        },
        ..LaserExtractConfig::default()
    };
    let out_g = ext
        .extract_line(&img.as_view(), 0..w, &cfg_g, None)
        .expect("valid extractor arguments");

    let cfg_t = LaserExtractConfig {
        axis: ScanAxis::Cols {
            access: ColAccess::Transposed,
        },
        ..LaserExtractConfig::default()
    };
    let out_t = ext
        .extract_line(&img.as_view(), 0..w, &cfg_t, Some(&img_t.as_view()))
        .expect("valid extractor arguments");

    assert_eq!(out_g.samples.len(), out_t.samples.len());
    for (a, b) in out_g.samples.iter().zip(out_t.samples.iter()) {
        assert_eq!(a.valid, b.valid);
        if a.valid {
            assert!((a.center - b.center).abs() <= 0.05);
        }
    }
}

#[test]
fn transposed_access_without_a_transposed_image_is_an_error() {
    // Forgetting the argument is ordinary API misuse, not a broken internal
    // invariant, so it is reported rather than asserted.
    let img = Image::from_vec(8, 4, vec![0u8; 8 * 4]).expect("valid image");
    let mut ext = LaserExtractor::new(1.0);
    let cfg = LaserExtractConfig {
        axis: ScanAxis::Cols {
            access: ColAccess::Transposed,
        },
        ..Default::default()
    };
    assert!(matches!(
        ext.extract_line(&img.as_view(), 0..8, &cfg, None),
        Err(Error::InvalidConfig(_))
    ));
}

#[test]
fn transposed_image_with_wrong_dimensions_is_an_error() {
    let img = Image::from_vec(8, 4, vec![0u8; 8 * 4]).expect("valid image");
    // Correct transpose would be 4x8; this is not.
    let bad = Image::from_vec(8, 4, vec![0u8; 8 * 4]).expect("valid image");
    let mut ext = LaserExtractor::new(1.0);
    let cfg = LaserExtractConfig {
        axis: ScanAxis::Cols {
            access: ColAccess::Transposed,
        },
        ..Default::default()
    };
    assert!(matches!(
        ext.extract_line(&img.as_view(), 0..8, &cfg, Some(&bad.as_view())),
        Err(Error::InvalidConfig(_))
    ));
}

/// Builds the same anti-aliased stripe used by `rows_mode_basic`, as f32
/// in [0, 255].
fn stripe_f32(w: usize, h: usize, x_l: f32, x_r: f32) -> Vec<f32> {
    let mut img_f = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            img_f[y * w + x] = 255.0 * frac_overlap(x, x_l, x_r);
        }
    }
    blur_rows(&mut img_f, w, h, 0.8);
    img_f
}

#[test]
fn u16_and_f32_paths_agree_with_u8() {
    // `extract_line_u16` and `extract_line_f32` had no test at all, and the
    // row scanners they call hold the unsafe fast paths. The same stripe
    // through all three entry points must give the same centre, since the
    // only difference is the input scalar type.
    let (w, h) = (64usize, 40usize);
    let (x_l, x_r) = (20.3f32, 25.7f32);
    let img_f = stripe_f32(w, h, x_l, x_r);

    let img_u8 = Image::from_vec(
        w,
        h,
        img_f
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect::<Vec<u8>>(),
    )
    .expect("valid image");
    // Same signal, scaled to the u16 range, so centres must still match.
    let img_u16 = Image::from_vec(
        w,
        h,
        img_f
            .iter()
            .map(|&v| (v * 257.0).round().clamp(0.0, 65535.0) as u16)
            .collect::<Vec<u16>>(),
    )
    .expect("valid image");
    let img_f32 = Image::from_vec(w, h, img_f.clone()).expect("valid image");

    let cfg = LaserExtractConfig {
        axis: ScanAxis::Rows,
        ..Default::default()
    };
    let mut ext = LaserExtractor::new(1.2);
    let a = ext
        .extract_line(&img_u8.as_view(), 0..h, &cfg, None)
        .expect("valid extractor arguments");
    let b = ext
        .extract_line(&img_u16.as_view(), 0..h, &cfg, None)
        .expect("valid extractor arguments");
    let c = ext
        .extract_line(&img_f32.as_view(), 0..h, &cfg, None)
        .expect("valid extractor arguments");

    assert_eq!(a.samples.len(), h);
    assert_eq!(b.samples.len(), h);
    assert_eq!(c.samples.len(), h);

    let expected = 0.5 * (x_l + x_r);
    let mut compared = 0usize;
    for i in 0..h {
        assert_eq!(
            b.samples[i].valid, a.samples[i].valid,
            "u16 validity must track u8 at row {i}"
        );
        assert_eq!(
            c.samples[i].valid, a.samples[i].valid,
            "f32 validity must track u8 at row {i}"
        );
        if !a.samples[i].valid {
            continue;
        }
        compared += 1;
        assert!(
            (a.samples[i].center - expected).abs() < 0.2,
            "u8 centre at row {i} is {}, expected ~{expected}",
            a.samples[i].center
        );
        // u8 is quantised where u16 and f32 are not, so allow a small gap.
        assert!(
            (b.samples[i].center - a.samples[i].center).abs() < 0.05,
            "u16 centre {} differs from u8 {} at row {i}",
            b.samples[i].center,
            a.samples[i].center
        );
        assert!(
            (c.samples[i].center - a.samples[i].center).abs() < 0.05,
            "f32 centre {} differs from u8 {} at row {i}",
            c.samples[i].center,
            a.samples[i].center
        );
    }
    assert!(compared > h / 2, "most rows should yield a valid sample");
}

#[test]
fn u16_transposed_access_matches_gather() {
    // Exercises the u16 transposed path, which reaches the same row scanner
    // through a different route than column gathering.
    let (w, h) = (48usize, 32usize);
    let img_f = stripe_f32(w, h, 18.4f32, 23.1f32);
    let data: Vec<u16> = img_f
        .iter()
        .map(|&v| (v * 257.0).round().clamp(0.0, 65535.0) as u16)
        .collect();
    let img = Image::from_vec(w, h, data).expect("valid image");

    // Transpose for the ColAccess::Transposed route.
    let src = img.data();
    let mut t = vec![0u16; w * h];
    for y in 0..h {
        for x in 0..w {
            t[x * h + y] = src[y * w + x];
        }
    }
    let img_t = Image::from_vec(h, w, t).expect("valid transposed image");

    let mut ext = LaserExtractor::new(1.2);
    let gather = ext
        .extract_line(
            &img.as_view(),
            0..w,
            &LaserExtractConfig {
                axis: ScanAxis::Cols {
                    access: ColAccess::Gather,
                },
                ..Default::default()
            },
            None,
        )
        .expect("valid extractor arguments");
    let transposed = ext
        .extract_line(
            &img.as_view(),
            0..w,
            &LaserExtractConfig {
                axis: ScanAxis::Cols {
                    access: ColAccess::Transposed,
                },
                ..Default::default()
            },
            Some(&img_t.as_view()),
        )
        .expect("valid extractor arguments");

    assert_eq!(gather.samples.len(), transposed.samples.len());
    for (i, (g, t)) in gather
        .samples
        .iter()
        .zip(transposed.samples.iter())
        .enumerate()
    {
        assert_eq!(g.valid, t.valid, "validity differs at column {i}");
        if g.valid {
            assert!(
                (g.center - t.center).abs() < 1e-4,
                "gather {} vs transposed {} at column {i}",
                g.center,
                t.center
            );
        }
    }
}
