//! 2×2 box-mean downsampling, generic over [`Pixel`].
//!
//! This used to be ten hand-written functions — five pixel-type/output-type
//! combinations, each with a raw-pointer "contiguous and even" fast path and a
//! safe fallback. [`Pixel::to_acc`] carries the only part that actually
//! differed (accumulator width), so one generic loop now covers all of them,
//! with no `unsafe`.
//!
//! Only the `f32`-destination kernel survives: every pyramid level above 0 is
//! `f32`, and the same-type variants had no caller. It is `pub(crate)` — the
//! public entry point is [`Pyramid`](super::Pyramid).
//!
//! The `chunks_exact(2)` formulation gives the optimiser a compile-time-known
//! chunk length, so the bounds checks fold away and the loop vectorises; it
//! benchmarks at parity with the raw-pointer version it replaced.

use crate::core::{Error, ImageView, ImageViewMut, Pixel};

/// Destination dimensions for a 2×2 downsample: integer halving, dropping a
/// trailing odd column or row.
#[inline]
pub(crate) fn dst_dims(src_w: usize, src_h: usize) -> (usize, usize) {
    (src_w / 2, src_h / 2)
}

/// Checks that `dst` is exactly the 2×2-downsampled size of `src`.
fn check_dims(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Result<(), Error> {
    let (w, h) = dst_dims(src_w, src_h);
    if dst_w != w || dst_h != h {
        return Err(Error::SizeMismatch {
            expected: w * h,
            actual: dst_w * dst_h,
        });
    }
    Ok(())
}

/// 2×2 box mean of `src` into an `f32` destination.
///
/// The sum is accumulated in [`Pixel::Acc`] and converted once, so an integer
/// source loses nothing to intermediate rounding. This is the kernel every
/// pyramid level above 0 is built with.
///
/// # Errors
/// Returns [`Error::SizeMismatch`] unless `dst` is exactly
/// `(src.width() / 2, src.height() / 2)`.
pub(crate) fn downsample2x2_mean_to_f32_into<P: Pixel>(
    src: &ImageView<'_, P>,
    dst: &mut ImageViewMut<'_, f32>,
) -> Result<(), Error> {
    let (dst_w, dst_h) = (dst.width(), dst.height());
    check_dims(src.width(), src.height(), dst_w, dst_h)?;

    for y in 0..dst_h {
        let (r0, r1) = (src.row(2 * y), src.row(2 * y + 1));
        let out = dst.row_mut(y);
        for ((o, a), b) in out
            .iter_mut()
            .zip(r0[..2 * dst_w].chunks_exact(2))
            .zip(r1[..2 * dst_w].chunks_exact(2))
        {
            let sum = a[0].to_acc() + a[1].to_acc() + b[0].to_acc() + b[1].to_acc();
            *o = P::acc_to_f32(sum) * 0.25;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{downsample2x2_mean_to_f32_into, dst_dims};
    use crate::core::{Image, ImageView, Pixel};

    fn img<P: Pixel>(w: usize, h: usize, d: Vec<P>) -> Image<P> {
        Image::from_vec(w, h, d).expect("valid image")
    }

    /// Allocating helper: the kernel writes into a caller-owned destination, so
    /// the tests need somewhere to put it.
    fn down<P: Pixel>(src: &ImageView<'_, P>) -> Image<f32> {
        let (w, h) = dst_dims(src.width(), src.height());
        let mut out = Image::new_fill(w, h, 0.0f32);
        downsample2x2_mean_to_f32_into(src, &mut out.as_view_mut()).expect("dims");
        out
    }

    #[test]
    fn f32_block_mean_is_the_arithmetic_mean() {
        // 4x2 -> 2x1. Blocks are [1,2 / 5,6] and [3,4 / 7,8].
        let src = img(4, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(down(&src.as_view()).data(), &[3.5, 5.5]);
    }

    #[test]
    fn odd_dimensions_drop_the_last_row_and_column() {
        // 5x3 -> 2x1: the last column and the last row are ignored.
        let data: Vec<u8> = (0..15).collect();
        let src = img(5, 3, data);
        let out = down(&src.as_view());
        assert_eq!((out.width(), out.height()), (2, 1));
        // Blocks [0,1 / 5,6] -> 12/4 = 3 ; [2,3 / 7,8] -> 20/4 = 5
        assert_eq!(out.data(), &[3.0f32, 5.0]);
    }

    /// A strided (non-contiguous) source must give the same answer as a packed one.
    #[test]
    fn strided_source_matches_packed() {
        // 4x2 payload embedded in a stride-6 buffer with junk in the padding.
        let padded = vec![
            10u8, 20, 30, 40, 99, 99, // row 0
            50, 60, 70, 80, 98, 98, // row 1
        ];
        let strided = ImageView::from_slice(4, 2, 6, &padded).expect("valid view");
        let packed = img(4, 2, vec![10u8, 20, 30, 40, 50, 60, 70, 80]);

        assert_eq!(down(&strided).data(), down(&packed.as_view()).data());
        // (10+20+50+60)/4 = 35 ; (30+40+70+80)/4 = 55
        assert_eq!(down(&strided).data(), &[35.0f32, 55.0]);
    }

    /// An integer source accumulates before converting, so no precision is lost.
    #[test]
    fn u16_to_f32_keeps_the_exact_mean() {
        let src = img(2, 2, vec![65535u16, 65535, 65535, 65534]);
        assert_eq!(down(&src.as_view()).data(), &[65534.75]);
    }

    /// D3: the size contract is checked, not merely debug-asserted.
    #[test]
    fn mismatched_destination_is_an_error() {
        let src = img(4, 4, vec![0u8; 16]);
        let mut too_small = Image::new_fill(1, 1, 0.0f32);
        assert!(
            downsample2x2_mean_to_f32_into(&src.as_view(), &mut too_small.as_view_mut()).is_err()
        );
        let mut too_big = Image::new_fill(4, 4, 0.0f32);
        assert!(
            downsample2x2_mean_to_f32_into(&src.as_view(), &mut too_big.as_view_mut()).is_err()
        );
        let mut right = Image::new_fill(2, 2, 0.0f32);
        assert!(downsample2x2_mean_to_f32_into(&src.as_view(), &mut right.as_view_mut()).is_ok());
    }

    #[test]
    fn degenerate_sources_produce_empty_output() {
        let src = img(1, 1, vec![7u8]);
        let out = down(&src.as_view());
        assert_eq!((out.width(), out.height()), (0, 0));
    }
}
