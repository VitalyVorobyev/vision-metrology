//! Chamfer distance transform on binary images.
//!
//! Implements the classic 3-4-5 chamfer approximation (Borgefors 1986):
//! - Distance to a horizontal/vertical neighbor costs **3** (≈1.0 px scaled by 3).
//! - Distance to a diagonal neighbor costs **4** (≈√2 px scaled by 3).
//!
//! The raw output is divided by 3.0 to convert to pixel units, giving a maximum
//! absolute error of about 0.07 px versus true Euclidean distance.
//!
//! Foreground pixels (`src > 0`) get distance 0; background pixels get the
//! approximate Euclidean distance to the nearest foreground pixel.
//!
//! # Example
//! ```
//! use vm_primitives::{Image, chamfer_distance_u8};
//!
//! let mut data = vec![0u8; 7 * 7];
//! data[3 * 7 + 3] = 255; // single foreground pixel at center
//! let img = Image::from_vec(7, 7, data).unwrap();
//! let dist = chamfer_distance_u8(&img.as_view());
//! // Center pixel: distance = 0
//! assert_eq!(*dist.as_view().get(3, 3).unwrap(), 0.0);
//! // Horizontal neighbor: distance ≈ 1.0
//! assert!((*dist.as_view().get(4, 3).unwrap() - 1.0).abs() < 0.1);
//! ```

use crate::core::{Image, ImageView};

const INF: f32 = f32::MAX / 2.0;

/// Chamfer 3-4-5 distance transform.
///
/// Returns an `Image<f32>` where each pixel value is the approximate Euclidean
/// distance (in pixels) to the nearest foreground (`> 0`) pixel in `src`.
/// Foreground pixels have distance `0.0`.
pub fn chamfer_distance_u8(src: &ImageView<'_, u8>) -> Image<f32> {
    let w = src.width();
    let h = src.height();
    let mut dist = vec![INF; w * h];

    // Seed: foreground pixels have distance 0.
    for y in 0..h {
        for x in 0..w {
            if *src.get(x, y).expect("in-bounds") > 0 {
                dist[y * w + x] = 0.0;
            }
        }
    }

    // Forward pass (top-left → bottom-right).
    // Neighbors examined: (-1,-1)×4, (0,-1)×3, (1,-1)×4, (-1,0)×3
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let mut d = dist[idx];
            if y > 0 {
                if x > 0 {
                    d = d.min(dist[(y - 1) * w + (x - 1)] + 4.0);
                }
                d = d.min(dist[(y - 1) * w + x] + 3.0);
                if x + 1 < w {
                    d = d.min(dist[(y - 1) * w + (x + 1)] + 4.0);
                }
            }
            if x > 0 {
                d = d.min(dist[y * w + (x - 1)] + 3.0);
            }
            dist[idx] = d;
        }
    }

    // Backward pass (bottom-right → top-left).
    // Neighbors examined: (1,1)×4, (0,1)×3, (-1,1)×4, (1,0)×3
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let idx = y * w + x;
            let mut d = dist[idx];
            if y + 1 < h {
                if x + 1 < w {
                    d = d.min(dist[(y + 1) * w + (x + 1)] + 4.0);
                }
                d = d.min(dist[(y + 1) * w + x] + 3.0);
                if x > 0 {
                    d = d.min(dist[(y + 1) * w + (x - 1)] + 4.0);
                }
            }
            if x + 1 < w {
                d = d.min(dist[y * w + (x + 1)] + 3.0);
            }
            dist[idx] = d;
        }
    }

    // Scale from chamfer units (×3) back to pixels.
    let data: Vec<f32> = dist.iter().map(|&v| v / 3.0).collect();
    Image::from_vec(w, h, data).expect("dimensions unchanged")
}

#[cfg(test)]
mod tests {
    use crate::core::Image;

    use super::chamfer_distance_u8;

    #[test]
    fn foreground_pixel_has_zero_distance() {
        // A single foreground pixel at (3, 3) in a 7×7 image.
        let mut data = vec![0u8; 7 * 7];
        data[3 * 7 + 3] = 255;
        let img = Image::from_vec(7, 7, data).unwrap();
        let d = chamfer_distance_u8(&img.as_view());
        assert_eq!(*d.as_view().get(3, 3).unwrap(), 0.0);
    }

    #[test]
    fn horizontal_neighbor_approx_one_pixel() {
        let mut data = vec![0u8; 7 * 7];
        data[3 * 7 + 3] = 255;
        let img = Image::from_vec(7, 7, data).unwrap();
        let d = chamfer_distance_u8(&img.as_view());
        // Horizontal neighbor is 1 px away; chamfer error < 0.1.
        let dist = *d.as_view().get(4, 3).unwrap();
        assert!((dist - 1.0).abs() < 0.1, "expected ~1.0, got {dist}");
    }

    #[test]
    fn diagonal_neighbor_approx_sqrt2() {
        let mut data = vec![0u8; 9 * 9];
        data[4 * 9 + 4] = 255;
        let img = Image::from_vec(9, 9, data).unwrap();
        let d = chamfer_distance_u8(&img.as_view());
        let dist = *d.as_view().get(5, 5).unwrap();
        let expected = std::f32::consts::SQRT_2;
        assert!(
            (dist - expected).abs() < 0.1,
            "expected ~{expected:.3}, got {dist}"
        );
    }

    #[test]
    fn all_foreground_gives_zero_everywhere() {
        let data = vec![255u8; 5 * 5];
        let img = Image::from_vec(5, 5, data).unwrap();
        let d = chamfer_distance_u8(&img.as_view());
        assert!(d.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn monotone_from_single_foreground() {
        // Distances should increase monotonically moving away from the foreground pixel.
        let mut data = vec![0u8; 7 * 7];
        data[3 * 7 + 3] = 255;
        let img = Image::from_vec(7, 7, data).unwrap();
        let d = chamfer_distance_u8(&img.as_view());
        let v = d.as_view();
        // Along the top row (y=0), x=3 should be closer than x=0.
        assert!(*v.get(3, 0).unwrap() < *v.get(0, 0).unwrap());
    }
}
