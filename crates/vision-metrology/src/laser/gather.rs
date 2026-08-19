//! Column gathering into a reusable scratch buffer.
//!
//! This is the only unsafe code in the laser pipeline. The unchecked indexing
//! is load-bearing for column-scan throughput (it removes two bounds checks
//! per element in a per-scanline loop), and its correctness rests entirely on
//! the two `assert!`s at the top of [`gather_col_segment`]: the asserts, the
//! `// SAFETY:` comments and the unsafe blocks form one unit and must move
//! together in any future refactor.
//!
//! The gather copies the pixel type as-is rather than widening to `f32`.
//! Widening here was measured at +21% on `laser_extract_cols_gather_512x1280`
//! (151 -> 183 us): it quadruples the write traffic on every gather, while the
//! widening it avoids happens only over the much smaller ROI segment inside the
//! 1-D detector.

use vm_primitives::{ImageView, Pixel};

/// Copy `img[x, y0..y1]` into `out` and return the filled slice.
pub(super) fn gather_col_segment<'a, P: Pixel>(
    img: &ImageView<'_, P>,
    x: usize,
    y0: usize,
    y1: usize,
    out: &'a mut Vec<P>,
) -> &'a [P] {
    assert!(x < img.width(), "x out of bounds");
    assert!(y0 <= y1 && y1 <= img.height(), "invalid y-range");

    let len = y1 - y0;
    out.resize(len, P::ZERO);

    if img.is_contiguous()
        && let Some(data) = img.as_contiguous_slice()
    {
        let w = img.width();
        // SAFETY:
        // - `x < w`; `y in [y0, y1)` and `y < img.height()`.
        // - index `y*w + x` is in-bounds for contiguous image backing.
        // - `out` was just resized to `len`, and `i < len`.
        unsafe {
            for (i, y) in (y0..y1).enumerate() {
                *out.get_unchecked_mut(i) = *data.get_unchecked(y * w + x);
            }
        }
        return &out[..len];
    }

    for (i, y) in (y0..y1).enumerate() {
        // SAFETY: bounded by the asserts above; `i < len == out.len()`.
        unsafe {
            *out.get_unchecked_mut(i) = *img.get_unchecked(x, y);
        }
    }
    &out[..len]
}
