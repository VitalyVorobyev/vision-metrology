//! Minimal binary morphology helpers.
//!
//! Pixels are treated as binary with threshold `> 0`.
//! Outputs are `0` or `255` in `u8`.

use vm_core::{Image, ImageView};

pub fn erode3x3_binary_u8(src: &ImageView<'_, u8>) -> Image<u8> {
    let mut out = Image::new_fill(src.width(), src.height(), 0u8);
    if src.width() == 0 || src.height() == 0 {
        return out;
    }

    for y in 0..src.height() {
        for x in 0..src.width() {
            let mut all_set = true;
            for dy in -1isize..=1 {
                let ny = y as isize + dy;
                if ny < 0 || ny >= src.height() as isize {
                    all_set = false;
                    break;
                }

                for dx in -1isize..=1 {
                    let nx = x as isize + dx;
                    if nx < 0 || nx >= src.width() as isize {
                        all_set = false;
                        break;
                    }

                    let v = src
                        .get(nx as usize, ny as usize)
                        .expect("in-bounds neighborhood access");
                    if *v == 0 {
                        all_set = false;
                        break;
                    }
                }

                if !all_set {
                    break;
                }
            }

            *out.as_view_mut()
                .get_mut(x, y)
                .expect("in-bounds write in erode3x3_binary_u8") = if all_set { 255 } else { 0 };
        }
    }

    out
}

pub fn dilate3x3_binary_u8(src: &ImageView<'_, u8>) -> Image<u8> {
    let mut out = Image::new_fill(src.width(), src.height(), 0u8);
    if src.width() == 0 || src.height() == 0 {
        return out;
    }

    for y in 0..src.height() {
        for x in 0..src.width() {
            let mut any_set = false;
            for dy in -1isize..=1 {
                let ny = y as isize + dy;
                if ny < 0 || ny >= src.height() as isize {
                    continue;
                }

                for dx in -1isize..=1 {
                    let nx = x as isize + dx;
                    if nx < 0 || nx >= src.width() as isize {
                        continue;
                    }

                    let v = src
                        .get(nx as usize, ny as usize)
                        .expect("in-bounds neighborhood access");
                    if *v != 0 {
                        any_set = true;
                        break;
                    }
                }

                if any_set {
                    break;
                }
            }

            *out.as_view_mut()
                .get_mut(x, y)
                .expect("in-bounds write in dilate3x3_binary_u8") = if any_set { 255 } else { 0 };
        }
    }

    out
}

pub fn open3x3_binary_u8(src: &ImageView<'_, u8>) -> Image<u8> {
    let eroded = erode3x3_binary_u8(src);
    dilate3x3_binary_u8(&eroded.as_view())
}

pub fn close3x3_binary_u8(src: &ImageView<'_, u8>) -> Image<u8> {
    let dilated = dilate3x3_binary_u8(src);
    erode3x3_binary_u8(&dilated.as_view())
}

#[cfg(test)]
mod tests {
    use vm_core::Image;

    use crate::{close3x3_binary_u8, open3x3_binary_u8};

    #[test]
    fn open_removes_single_pixel_speck() {
        let mut data = vec![0u8; 25];
        data[12] = 255;
        let img = Image::from_vec(5, 5, data).expect("valid image");

        let out = open3x3_binary_u8(&img.as_view());
        assert!(out.data().iter().all(|&v| v == 0));
    }

    #[test]
    fn close_fills_single_pixel_hole() {
        let mut data = vec![255u8; 25];
        data[12] = 0;
        let img = Image::from_vec(5, 5, data).expect("valid image");

        let out = close3x3_binary_u8(&img.as_view());
        assert_eq!(out.data()[12], 255);
    }
}
