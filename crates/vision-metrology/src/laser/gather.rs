//! Column gathering into a reusable scratch buffer.
//!
//! This is the only unsafe code in the laser pipeline. The unchecked indexing
//! is load-bearing for column-scan throughput (it removes two bounds checks
//! per element in a per-scanline loop), and its correctness rests entirely on
//! the two `assert!`s at the top of [`gather_col_segment`]: the asserts, the
//! `// SAFETY:` comments and the unsafe blocks form one unit and must move
//! together in any future refactor.

use vm_primitives::ImageView;

use super::scan::ScanPixel;

/// Copy `img[x, y0..y1]` into `out` and return the filled slice.
pub(super) fn gather_col_segment<'a, T: ScanPixel>(
    img: &ImageView<'_, T>,
    x: usize,
    y0: usize,
    y1: usize,
    out: &'a mut Vec<T>,
) -> &'a [T] {
    assert!(x < img.width(), "x out of bounds");
    assert!(y0 <= y1 && y1 <= img.height(), "invalid y-range");

    let len = y1 - y0;
    out.resize(len, T::ZERO);

    if img.is_contiguous()
        && let Some(data) = img.as_contiguous_slice()
    {
        let w = img.width();
        // SAFETY:
        // - `x < w`; `y in [y0, y1)` and `y < img.height()`.
        // - index `y*w + x` is in-bounds for contiguous image backing.
        unsafe {
            for (i, y) in (y0..y1).enumerate() {
                *out.get_unchecked_mut(i) = *data.get_unchecked(y * w + x);
            }
        }
        return &out[..len];
    }

    for (i, y) in (y0..y1).enumerate() {
        // SAFETY: bounded by asserts above.
        unsafe {
            *out.get_unchecked_mut(i) = *img.get_unchecked(x, y);
        }
    }
    &out[..len]
}
