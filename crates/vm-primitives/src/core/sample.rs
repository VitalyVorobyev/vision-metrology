use super::border::{BorderMode, map_index};
use super::image::ImageView;
use super::pixel::Pixel;

/// Sample image `img` at continuous position `(x, y)` using nearest-neighbor interpolation.
///
/// Rounds `(x, y)` to the nearest integer index and applies `border` for
/// out-of-bounds coordinates.
///
/// # Panics
/// Panics when the image is empty and `border` is not [`BorderMode::Constant`].
pub fn sample_nearest<T: Copy>(img: &ImageView<'_, T>, x: f32, y: f32, border: BorderMode<T>) -> T {
    let xi = x.round() as isize;
    let yi = y.round() as isize;

    if img.width() == 0 || img.height() == 0 {
        if let BorderMode::Constant(v) = border {
            return v;
        }
        panic!("cannot sample an empty image with non-constant border");
    }

    match border {
        BorderMode::Constant(v) => {
            if xi < 0 || yi < 0 || xi >= img.width() as isize || yi >= img.height() as isize {
                return v;
            }
            // SAFETY: Bounds are checked immediately above.
            unsafe { *img.get_unchecked(xi as usize, yi as usize) }
        }
        mode @ (BorderMode::Clamp | BorderMode::Reflect101) => {
            let mx =
                map_index(xi, img.width(), &mode).expect("valid mapped index for non-empty image");
            let my =
                map_index(yi, img.height(), &mode).expect("valid mapped index for non-empty image");
            // SAFETY: `map_index` returns indices in `[0, len)` for non-empty images.
            unsafe { *img.get_unchecked(mx, my) }
        }
    }
}

/// Sample image `img` at continuous position `(x, y)` using bilinear interpolation.
///
/// Uses floor-based 2×2 neighborhood. Pixel type `T` is converted to `f32`
/// via `Into<f32>` before blending. Applies `border` for out-of-bounds pixels.
///
/// # Panics
/// Panics when the image is empty and `border` is not [`BorderMode::Constant`].
pub fn sample_bilinear_f32<T: Pixel>(
    img: &ImageView<'_, T>,
    x: f32,
    y: f32,
    border: BorderMode<f32>,
) -> f32 {
    if img.width() == 0 || img.height() == 0 {
        if let BorderMode::Constant(v) = border {
            return v;
        }
        panic!("cannot sample an empty image with non-constant border");
    }

    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let p00 = sample_at_f32(img, x0, y0, &border);
    let p10 = sample_at_f32(img, x1, y0, &border);
    let p01 = sample_at_f32(img, x0, y1, &border);
    let p11 = sample_at_f32(img, x1, y1, &border);

    let top = p00 * (1.0 - dx) + p10 * dx;
    let bottom = p01 * (1.0 - dx) + p11 * dx;
    top * (1.0 - dy) + bottom * dy
}

fn sample_at_f32<T: Pixel>(
    img: &ImageView<'_, T>,
    x: isize,
    y: isize,
    border: &BorderMode<f32>,
) -> f32 {
    match border {
        BorderMode::Constant(c) => {
            if x < 0 || y < 0 || x >= img.width() as isize || y >= img.height() as isize {
                *c
            } else {
                // SAFETY: Bounds are checked immediately above.
                unsafe { img.get_unchecked(x as usize, y as usize).to_f32() }
            }
        }
        BorderMode::Clamp => {
            let xi = map_index(x, img.width(), border).expect("mapped x index should exist");
            let yi = map_index(y, img.height(), border).expect("mapped y index should exist");
            // SAFETY: `map_index` returns indices in `[0, len)` for non-empty images.
            unsafe { img.get_unchecked(xi, yi).to_f32() }
        }
        BorderMode::Reflect101 => {
            let xi = map_index(x, img.width(), border).expect("mapped x index should exist");
            let yi = map_index(y, img.height(), border).expect("mapped y index should exist");
            // SAFETY: `map_index` returns indices in `[0, len)` for non-empty images.
            unsafe { img.get_unchecked(xi, yi).to_f32() }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::BorderMode;
    use crate::Image;
    use crate::{sample_bilinear_f32, sample_nearest};

    #[test]
    fn nearest_on_3x3_with_clamp_and_constant() {
        let img = Image::from_vec(
            3,
            3,
            vec![
                0u8, 1, 2, // row 0
                10, 11, 12, // row 1
                20, 21, 22, // row 2
            ],
        )
        .expect("valid image");
        let view = img.as_view();

        assert_eq!(sample_nearest(&view, 1.2, 1.6, BorderMode::Clamp), 21);
        assert_eq!(sample_nearest(&view, -2.0, 1.0, BorderMode::Clamp), 10);
        assert_eq!(sample_nearest(&view, 9.0, 9.0, BorderMode::Clamp), 22);

        assert_eq!(
            sample_nearest(&view, -0.6, 1.0, BorderMode::Constant(99u8)),
            99
        );
        assert_eq!(
            sample_nearest(&view, 0.49, 0.49, BorderMode::Constant(99u8)),
            0
        );
    }

    #[test]
    fn bilinear_on_2x2_center_and_border_modes() {
        let img = Image::from_vec(2, 2, vec![0u8, 10, 20, 30]).expect("valid image");
        let view = img.as_view();

        let center = sample_bilinear_f32(&view, 0.5, 0.5, BorderMode::Clamp);
        assert!((center - 15.0).abs() < 1e-6);

        let neg_clamp = sample_bilinear_f32(&view, -0.25, -0.25, BorderMode::Clamp);
        assert!((neg_clamp - 0.0).abs() < 1e-6);

        let neg_constant = sample_bilinear_f32(&view, -0.25, -0.25, BorderMode::Constant(100.0));
        // floor-based bilinear at (-0.25, -0.25):
        // p00/p10/p01 are constant(100), p11 is image(0) -> 43.75
        assert!((neg_constant - 43.75).abs() < 1e-6);
    }

    /// 4x3 ramp where every pixel equals `10*y + x`, so any interpolation
    /// result is predictable in closed form.
    fn ramp() -> Image<u8> {
        let (w, h) = (4usize, 3usize);
        let data: Vec<u8> = (0..w * h).map(|i| (10 * (i / w) + (i % w)) as u8).collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    #[test]
    fn nearest_rounds_half_away_from_zero() {
        // `f32::round` is half-away-from-zero, so 0.5 lands on pixel 1 and
        // 1.5 on pixel 2. Pinning this stops the convention drifting.
        let img = ramp();
        let v = img.as_view();
        assert_eq!(sample_nearest(&v, 0.5, 0.0, BorderMode::Clamp), 1);
        assert_eq!(sample_nearest(&v, 1.5, 0.0, BorderMode::Clamp), 2);
        assert_eq!(sample_nearest(&v, 0.49, 0.0, BorderMode::Clamp), 0);
    }

    #[test]
    fn nearest_reflect101_mirrors_without_repeating_the_edge() {
        // Row 0 of the ramp is [0, 1, 2, 3]. Reflect101 around index 0 gives
        // the virtual sequence ... 2 1 0 1 2 3 2 1 ...
        let img = ramp();
        let v = img.as_view();
        assert_eq!(sample_nearest(&v, -1.0, 0.0, BorderMode::Reflect101), 1);
        assert_eq!(sample_nearest(&v, -2.0, 0.0, BorderMode::Reflect101), 2);
        // And around the far edge (width 4, last index 3).
        assert_eq!(sample_nearest(&v, 4.0, 0.0, BorderMode::Reflect101), 2);
        assert_eq!(sample_nearest(&v, 5.0, 0.0, BorderMode::Reflect101), 1);
    }

    #[test]
    fn bilinear_reproduces_a_linear_ramp_exactly() {
        // Bilinear interpolation is exact for functions linear in x and y, so
        // sampling the ramp anywhere inside must return 10*y + x to rounding.
        let img = ramp();
        let v = img.as_view();
        for yi in 0..21 {
            for xi in 0..31 {
                let (x, y) = (xi as f32 / 10.0, yi as f32 / 10.0);
                let got = sample_bilinear_f32(&v, x, y, BorderMode::Clamp);
                let want = 10.0 * y + x;
                assert!(
                    (got - want).abs() < 1e-4,
                    "at ({x}, {y}): got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn bilinear_agrees_with_nearest_at_pixel_centres() {
        let img = ramp();
        let v = img.as_view();
        for y in 0..3 {
            for x in 0..4 {
                let b = sample_bilinear_f32(&v, x as f32, y as f32, BorderMode::Clamp);
                let n = f32::from(sample_nearest(&v, x as f32, y as f32, BorderMode::Clamp));
                assert!((b - n).abs() < 1e-6, "mismatch at ({x}, {y}): {b} vs {n}");
            }
        }
    }

    #[test]
    fn clamp_saturates_far_outside_the_image() {
        let img = ramp();
        let v = img.as_view();
        // Far out in every direction returns the corresponding corner.
        assert_eq!(sample_nearest(&v, -50.0, -50.0, BorderMode::Clamp), 0);
        assert_eq!(sample_nearest(&v, 50.0, -50.0, BorderMode::Clamp), 3);
        assert_eq!(sample_nearest(&v, -50.0, 50.0, BorderMode::Clamp), 20);
        assert_eq!(sample_nearest(&v, 50.0, 50.0, BorderMode::Clamp), 23);
        assert!((sample_bilinear_f32(&v, -50.0, -50.0, BorderMode::Clamp) - 0.0).abs() < 1e-6);
        assert!((sample_bilinear_f32(&v, 50.0, 50.0, BorderMode::Clamp) - 23.0).abs() < 1e-6);
    }

    #[test]
    fn constant_border_applies_outside_only() {
        let img = ramp();
        let v = img.as_view();
        assert_eq!(
            sample_nearest(&v, -1.0, 0.0, BorderMode::Constant(77u8)),
            77
        );
        assert_eq!(sample_nearest(&v, 4.0, 0.0, BorderMode::Constant(77u8)), 77);
        assert_eq!(
            sample_nearest(&v, 0.0, -1.0, BorderMode::Constant(77u8)),
            77
        );
        assert_eq!(sample_nearest(&v, 0.0, 3.0, BorderMode::Constant(77u8)), 77);
        // Just inside is unaffected.
        assert_eq!(sample_nearest(&v, 3.0, 2.0, BorderMode::Constant(77u8)), 23);
    }

    #[test]
    fn empty_image_returns_the_constant() {
        let img = Image::from_vec(0, 0, Vec::<u8>::new()).expect("valid empty image");
        let v = img.as_view();
        assert_eq!(sample_nearest(&v, 0.0, 0.0, BorderMode::Constant(5u8)), 5);
        assert!((sample_bilinear_f32(&v, 0.0, 0.0, BorderMode::Constant(5.0)) - 5.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "cannot sample an empty image")]
    fn empty_image_panics_under_clamp() {
        // Documented behaviour: there is no pixel to clamp to.
        let img = Image::from_vec(0, 0, Vec::<u8>::new()).expect("valid empty image");
        let _ = sample_nearest(&img.as_view(), 0.0, 0.0, BorderMode::Clamp);
    }

    #[test]
    #[should_panic(expected = "cannot sample an empty image")]
    fn empty_image_panics_under_clamp_bilinear() {
        let img = Image::from_vec(0, 0, Vec::<u8>::new()).expect("valid empty image");
        let _ = sample_bilinear_f32(&img.as_view(), 0.0, 0.0, BorderMode::Clamp);
    }

    #[test]
    fn single_pixel_image_is_sampled_everywhere_under_clamp() {
        // Degenerate but legal: every coordinate maps to the one pixel.
        let img = Image::from_vec(1, 1, vec![42u8]).expect("valid image");
        let v = img.as_view();
        for &(x, y) in &[(0.0f32, 0.0f32), (-3.0, 2.0), (7.5, -7.5)] {
            assert_eq!(sample_nearest(&v, x, y, BorderMode::Clamp), 42);
            assert!((sample_bilinear_f32(&v, x, y, BorderMode::Clamp) - 42.0).abs() < 1e-6);
            assert_eq!(sample_nearest(&v, x, y, BorderMode::Reflect101), 42);
        }
    }
}
