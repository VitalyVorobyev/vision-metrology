//! Binary morphology helpers for machine-vision metrology.
//!
//! Pixels are treated as binary with threshold `> 0`.
//! Binary outputs use `0` (background) and `255` (foreground) in `u8`.
//!
//! ## Morphological operations
//! - [`erode_binary_u8`] / [`dilate_binary_u8`] — parameterized by a [`StructuringElement`].
//! - [`erode3x3_binary_u8`] / [`dilate3x3_binary_u8`] — convenience wrappers for the
//!   classic 3×3 square (backward-compatible).
//! - [`open_binary_u8`] / [`close_binary_u8`] — opening and closing.
//!
//! ## Distance transform
//! - [`chamfer_distance_u8`] — 3-4-5 chamfer approximation to Euclidean distance.
//!
//! ## Skeletonization
//! - [`thin_binary_u8`] — Zhang-Suen iterative thinning.

mod distance;
mod se;
mod thin;

pub use distance::chamfer_distance_u8;
pub use se::StructuringElement;
pub use thin::thin_binary_u8;

use vm_core::{Image, ImageView};

// ---------------------------------------------------------------------------
// Parameterized erode / dilate
// ---------------------------------------------------------------------------

/// Erode a binary image using the given structuring element.
///
/// A pixel is set in the output only if **all** pixels in the structuring
/// element's support are foreground (`> 0`) in the source.
/// Out-of-bounds positions are treated as background (conservative: any
/// border pixel touching the edge will be cleared by erosion).
pub fn erode_binary_u8(src: &ImageView<'_, u8>, se: &StructuringElement) -> Image<u8> {
    let w = src.width();
    let h = src.height();
    let mut out = Image::new_fill(w, h, 0u8);
    let r = se.radius() as isize;

    for y in 0..h {
        for x in 0..w {
            let mut all_set = true;
            'se: for dy in -r..=r {
                for dx in -r..=r {
                    if !se.contains(dx, dy) {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        all_set = false;
                        break 'se;
                    }
                    if *src
                        .get(nx as usize, ny as usize)
                        .expect("in-bounds checked above")
                        == 0
                    {
                        all_set = false;
                        break 'se;
                    }
                }
            }
            *out.as_view_mut().get_mut(x, y).expect("in-bounds") = if all_set { 255 } else { 0 };
        }
    }
    out
}

/// Dilate a binary image using the given structuring element.
///
/// A pixel is set in the output if **any** pixel in the structuring element's
/// support is foreground (`> 0`) in the source. Out-of-bounds neighbors are
/// ignored (treated as background).
pub fn dilate_binary_u8(src: &ImageView<'_, u8>, se: &StructuringElement) -> Image<u8> {
    let w = src.width();
    let h = src.height();
    let mut out = Image::new_fill(w, h, 0u8);
    let r = se.radius() as isize;

    for y in 0..h {
        for x in 0..w {
            let mut any_set = false;
            'se: for dy in -r..=r {
                for dx in -r..=r {
                    if !se.contains(dx, dy) {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    if *src
                        .get(nx as usize, ny as usize)
                        .expect("in-bounds checked above")
                        > 0
                    {
                        any_set = true;
                        break 'se;
                    }
                }
            }
            *out.as_view_mut().get_mut(x, y).expect("in-bounds") = if any_set { 255 } else { 0 };
        }
    }
    out
}

/// Open = erode then dilate (removes small foreground specks).
pub fn open_binary_u8(src: &ImageView<'_, u8>, se: &StructuringElement) -> Image<u8> {
    let eroded = erode_binary_u8(src, se);
    dilate_binary_u8(&eroded.as_view(), se)
}

/// Close = dilate then erode (fills small holes in foreground regions).
pub fn close_binary_u8(src: &ImageView<'_, u8>, se: &StructuringElement) -> Image<u8> {
    let dilated = dilate_binary_u8(src, se);
    erode_binary_u8(&dilated.as_view(), se)
}

// ---------------------------------------------------------------------------
// Backward-compatible 3×3 wrappers
// ---------------------------------------------------------------------------

/// Erode with the classic 3×3 square structuring element.
///
/// Equivalent to `erode_binary_u8(src, &StructuringElement::Square(1))`.
pub fn erode3x3_binary_u8(src: &ImageView<'_, u8>) -> Image<u8> {
    erode_binary_u8(src, &StructuringElement::Square(1))
}

/// Dilate with the classic 3×3 square structuring element.
///
/// Equivalent to `dilate_binary_u8(src, &StructuringElement::Square(1))`.
pub fn dilate3x3_binary_u8(src: &ImageView<'_, u8>) -> Image<u8> {
    dilate_binary_u8(src, &StructuringElement::Square(1))
}

/// Open with the classic 3×3 square structuring element.
pub fn open3x3_binary_u8(src: &ImageView<'_, u8>) -> Image<u8> {
    open_binary_u8(src, &StructuringElement::Square(1))
}

/// Close with the classic 3×3 square structuring element.
pub fn close3x3_binary_u8(src: &ImageView<'_, u8>) -> Image<u8> {
    close_binary_u8(src, &StructuringElement::Square(1))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use vm_core::Image;

    use crate::{
        StructuringElement, close3x3_binary_u8, erode_binary_u8, open_binary_u8, open3x3_binary_u8,
    };

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

    #[test]
    fn disk_se_larger_than_square() {
        // A single foreground pixel eroded by a disk(2) should vanish
        // (it does not have enough surrounding foreground).
        let mut data = vec![255u8; 7 * 7];
        // Clear border so the inner pixels survive a conservative erosion,
        // then put a lone pixel and confirm it vanishes.
        data = vec![0u8; 7 * 7];
        data[3 * 7 + 3] = 255;
        let img = Image::from_vec(7, 7, data).unwrap();
        let out = erode_binary_u8(&img.as_view(), &StructuringElement::Disk(2));
        // Single isolated pixel cannot survive disk(2) erosion.
        assert!(out.data().iter().all(|&v| v == 0));
    }

    #[test]
    fn square5x5_open_removes_small_speck() {
        // A 3×3 foreground block should be removed by Square(2) open.
        let mut data = vec![0u8; 11 * 11];
        for dy in 0..3 {
            for dx in 0..3 {
                data[(4 + dy) * 11 + (4 + dx)] = 255;
            }
        }
        let img = Image::from_vec(11, 11, data).unwrap();
        let out = open_binary_u8(&img.as_view(), &StructuringElement::Square(2));
        assert!(out.data().iter().all(|&v| v == 0));
    }

    #[test]
    fn se_contains_disk_vs_square() {
        let disk = StructuringElement::Disk(2);
        let sq = StructuringElement::Square(2);
        // (0,0) is always inside.
        assert!(disk.contains(0, 0));
        assert!(sq.contains(0, 0));
        // (2,2) is inside the square but outside the disk (dist = 2√2 > 2).
        assert!(!disk.contains(2, 2));
        assert!(sq.contains(2, 2));
    }
}
