//! Zhang-Suen binary thinning (skeletonization).
//!
//! Iteratively removes border pixels from foreground regions until only a
//! one-pixel-wide skeleton remains. The algorithm preserves connectivity.
//!
//! Reference: Zhang, T.Y. and Suen, C.Y. (1984). "A fast parallel algorithm
//! for thinning digital patterns." *Communications of the ACM*, 27(3), 236–239.
//!
//! # Example
//! ```
//! use vm_core::{Image, ImageView};
//! use vm_morph::thin_binary_u8;
//!
//! // A 3-pixel-wide horizontal bar.
//! let mut data = vec![0u8; 5 * 7];
//! for y in 1..=3 {
//!     for x in 0..7 {
//!         data[y * 7 + x] = 255;
//!     }
//! }
//! let img = Image::from_vec(7, 5, data).unwrap();
//! let skeleton = thin_binary_u8(&img.as_view());
//! // The result is at most 1 pixel thick in the vertical direction.
//! for x in 1..6 {
//!     let col_count = (0..5)
//!         .filter(|&y| *skeleton.as_view().get(x, y).unwrap() > 0)
//!         .count();
//!     assert!(col_count <= 1, "column {x} has {col_count} foreground pixels");
//! }
//! ```

use vm_core::{Image, ImageView};

/// Zhang-Suen binary thinning.
///
/// Input: binary `u8` image where `> 0` is foreground.
/// Output: thinned skeleton as a binary `u8` image (`255` = foreground, `0` = background).
pub fn thin_binary_u8(src: &ImageView<'_, u8>) -> Image<u8> {
    let w = src.width();
    let h = src.height();

    if w == 0 || h == 0 {
        return Image::new_fill(w, h, 0u8);
    }

    // Working buffer: 1 = foreground, 0 = background (compact, stride-free).
    let mut buf: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| {
            if *src.get(x, y).expect("in-bounds") > 0 {
                1
            } else {
                0
            }
        })
        .collect();

    loop {
        let mut changed = false;

        // Two sub-iterations per Zhang-Suen cycle.
        for sub in 0..2u8 {
            let mut to_remove = Vec::new();

            for y in 1..(h - 1) {
                for x in 1..(w - 1) {
                    let p = buf[y * w + x];
                    if p == 0 {
                        continue;
                    }

                    // 8-neighbors in order: N, NE, E, SE, S, SW, W, NW
                    // Indices: p2..p9 per the original paper's notation.
                    let p2 = buf[(y - 1) * w + x];
                    let p3 = buf[(y - 1) * w + (x + 1)];
                    let p4 = buf[y * w + (x + 1)];
                    let p5 = buf[(y + 1) * w + (x + 1)];
                    let p6 = buf[(y + 1) * w + x];
                    let p7 = buf[(y + 1) * w + (x - 1)];
                    let p8 = buf[y * w + (x - 1)];
                    let p9 = buf[(y - 1) * w + (x - 1)];

                    let neighbors = [p2, p3, p4, p5, p6, p7, p8, p9];
                    let b = neighbors.iter().map(|&v| v as u32).sum::<u32>(); // count of set neighbors

                    // Condition A: 2 ≤ B(P1) ≤ 6
                    if !(2..=6).contains(&b) {
                        continue;
                    }

                    // Condition B: A(P1) == 1 (exactly one 0→1 transition in CCW order)
                    let a = count_01_transitions(&neighbors);
                    if a != 1 {
                        continue;
                    }

                    // Sub-iteration 1: P2·P4·P6 == 0  AND  P4·P6·P8 == 0
                    // Sub-iteration 2: P2·P4·P8 == 0  AND  P2·P6·P8 == 0
                    let cond = if sub == 0 {
                        p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0
                    } else {
                        p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0
                    };

                    if cond {
                        to_remove.push(y * w + x);
                    }
                }
            }

            if to_remove.is_empty() {
                continue;
            }

            changed = true;
            for idx in to_remove {
                buf[idx] = 0;
            }
        }

        if !changed {
            break;
        }
    }

    // Convert back to u8 (0 / 255).
    let data: Vec<u8> = buf.iter().map(|&v| if v > 0 { 255 } else { 0 }).collect();
    Image::from_vec(w, h, data).expect("dimensions unchanged")
}

/// Count 0→1 transitions in the circular sequence of 8 neighbors.
#[inline]
fn count_01_transitions(n: &[u8; 8]) -> u32 {
    let mut count = 0u32;
    for i in 0..8 {
        let a = n[i];
        let b = n[(i + 1) % 8];
        if a == 0 && b != 0 {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use vm_core::Image;

    use super::thin_binary_u8;

    fn make_hbar(w: usize, h: usize, y0: usize, y1: usize) -> Image<u8> {
        let mut data = vec![0u8; w * h];
        for y in y0..=y1 {
            for x in 0..w {
                data[y * w + x] = 255;
            }
        }
        Image::from_vec(w, h, data).unwrap()
    }

    #[test]
    fn thin_horizontal_bar_to_one_pixel() {
        // A 3-pixel-wide horizontal bar in a 9×7 image (y rows 2..=4).
        let img = make_hbar(9, 7, 2, 4);
        let skel = thin_binary_u8(&img.as_view());
        let v = skel.as_view();

        // Inner columns (x = 1..7) must have at most 1 foreground row.
        for x in 1..8 {
            let count = (0..7).filter(|&y| *v.get(x, y).unwrap() > 0).count();
            assert!(
                count <= 1,
                "column {x} has {count} fg pixels after thinning"
            );
        }
    }

    #[test]
    fn empty_image_returns_empty() {
        let img = Image::from_vec(5, 5, vec![0u8; 25]).unwrap();
        let skel = thin_binary_u8(&img.as_view());
        assert!(skel.data().iter().all(|&v| v == 0));
    }

    #[test]
    fn single_pixel_unchanged() {
        let mut data = vec![0u8; 5 * 5];
        data[2 * 5 + 2] = 255;
        let img = Image::from_vec(5, 5, data).unwrap();
        let skel = thin_binary_u8(&img.as_view());
        assert_eq!(*skel.as_view().get(2, 2).unwrap(), 255);
    }
}
