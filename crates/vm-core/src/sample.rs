use crate::border::{BorderMode, map_index};
use crate::image::ImageView;

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

pub fn sample_bilinear_f32<T: Copy + Into<f32>>(
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

fn sample_at_f32<T: Copy + Into<f32>>(
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
                unsafe { (*img.get_unchecked(x as usize, y as usize)).into() }
            }
        }
        BorderMode::Clamp => {
            let xi = map_index(x, img.width(), border).expect("mapped x index should exist");
            let yi = map_index(y, img.height(), border).expect("mapped y index should exist");
            // SAFETY: `map_index` returns indices in `[0, len)` for non-empty images.
            unsafe { (*img.get_unchecked(xi, yi)).into() }
        }
        BorderMode::Reflect101 => {
            let xi = map_index(x, img.width(), border).expect("mapped x index should exist");
            let yi = map_index(y, img.height(), border).expect("mapped y index should exist");
            // SAFETY: `map_index` returns indices in `[0, len)` for non-empty images.
            unsafe { (*img.get_unchecked(xi, yi)).into() }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::border::BorderMode;
    use crate::image::Image;
    use crate::sample::{sample_bilinear_f32, sample_nearest};

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
}
