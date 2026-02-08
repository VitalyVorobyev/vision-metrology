use vm_core::{Image, ImageView};

#[inline]
fn dst_dims(src_w: usize, src_h: usize) -> (usize, usize) {
    (src_w / 2, src_h / 2)
}

pub fn downsample2x2_mean_u8_to_f32(src: &ImageView<'_, u8>) -> Image<f32> {
    let (dst_w, dst_h) = dst_dims(src.width(), src.height());
    let mut dst = Image::new_fill(dst_w, dst_h, 0.0f32);
    downsample2x2_mean_u8_to_f32_into(src, &mut dst);
    dst
}

pub fn downsample2x2_mean_u16_to_f32(src: &ImageView<'_, u16>) -> Image<f32> {
    let (dst_w, dst_h) = dst_dims(src.width(), src.height());
    let mut dst = Image::new_fill(dst_w, dst_h, 0.0f32);
    downsample2x2_mean_u16_to_f32_into(src, &mut dst);
    dst
}

pub fn downsample2x2_mean_f32(src: &ImageView<'_, f32>) -> Image<f32> {
    let (dst_w, dst_h) = dst_dims(src.width(), src.height());
    let mut dst = Image::new_fill(dst_w, dst_h, 0.0f32);
    downsample2x2_mean_f32_into(src, &mut dst);
    dst
}

pub fn downsample2x2_mean_u8(src: &ImageView<'_, u8>) -> Image<u8> {
    let (dst_w, dst_h) = dst_dims(src.width(), src.height());
    let mut dst = Image::new_fill(dst_w, dst_h, 0u8);
    downsample2x2_mean_u8_into(src, &mut dst);
    dst
}

pub fn downsample2x2_mean_u16(src: &ImageView<'_, u16>) -> Image<u16> {
    let (dst_w, dst_h) = dst_dims(src.width(), src.height());
    let mut dst = Image::new_fill(dst_w, dst_h, 0u16);
    downsample2x2_mean_u16_into(src, &mut dst);
    dst
}

pub(crate) fn downsample2x2_mean_u8_to_f32_into(src: &ImageView<'_, u8>, dst: &mut Image<f32>) {
    let dst_w = src.width() / 2;
    let dst_h = src.height() / 2;
    debug_assert_eq!(dst.width(), dst_w);
    debug_assert_eq!(dst.height(), dst_h);

    if dst_w == 0 || dst_h == 0 {
        return;
    }

    if src.is_contiguous()
        && src.width().is_multiple_of(2)
        && src.height().is_multiple_of(2)
        && let Some(src_contig) = src.as_contiguous_slice()
    {
        downsample_u8_to_f32_contiguous_even(src_contig, src.width(), dst.data_mut(), dst_w, dst_h);
        return;
    }

    downsample_u8_to_f32_fallback(src, dst.data_mut(), dst_w, dst_h);
}

pub(crate) fn downsample2x2_mean_u16_to_f32_into(src: &ImageView<'_, u16>, dst: &mut Image<f32>) {
    let dst_w = src.width() / 2;
    let dst_h = src.height() / 2;
    debug_assert_eq!(dst.width(), dst_w);
    debug_assert_eq!(dst.height(), dst_h);

    if dst_w == 0 || dst_h == 0 {
        return;
    }

    if src.is_contiguous()
        && src.width().is_multiple_of(2)
        && src.height().is_multiple_of(2)
        && let Some(src_contig) = src.as_contiguous_slice()
    {
        downsample_u16_to_f32_contiguous_even(
            src_contig,
            src.width(),
            dst.data_mut(),
            dst_w,
            dst_h,
        );
        return;
    }

    downsample_u16_to_f32_fallback(src, dst.data_mut(), dst_w, dst_h);
}

pub(crate) fn downsample2x2_mean_f32_into(src: &ImageView<'_, f32>, dst: &mut Image<f32>) {
    let dst_w = src.width() / 2;
    let dst_h = src.height() / 2;
    debug_assert_eq!(dst.width(), dst_w);
    debug_assert_eq!(dst.height(), dst_h);

    if dst_w == 0 || dst_h == 0 {
        return;
    }

    if src.is_contiguous()
        && src.width().is_multiple_of(2)
        && src.height().is_multiple_of(2)
        && let Some(src_contig) = src.as_contiguous_slice()
    {
        downsample_f32_contiguous_even(src_contig, src.width(), dst.data_mut(), dst_w, dst_h);
        return;
    }

    downsample_f32_fallback(src, dst.data_mut(), dst_w, dst_h);
}

fn downsample2x2_mean_u8_into(src: &ImageView<'_, u8>, dst: &mut Image<u8>) {
    let dst_w = src.width() / 2;
    let dst_h = src.height() / 2;
    debug_assert_eq!(dst.width(), dst_w);
    debug_assert_eq!(dst.height(), dst_h);

    if dst_w == 0 || dst_h == 0 {
        return;
    }

    if src.is_contiguous()
        && src.width().is_multiple_of(2)
        && src.height().is_multiple_of(2)
        && let Some(src_contig) = src.as_contiguous_slice()
    {
        downsample_u8_contiguous_even(src_contig, src.width(), dst.data_mut(), dst_w, dst_h);
        return;
    }

    downsample_u8_fallback(src, dst.data_mut(), dst_w, dst_h);
}

fn downsample2x2_mean_u16_into(src: &ImageView<'_, u16>, dst: &mut Image<u16>) {
    let dst_w = src.width() / 2;
    let dst_h = src.height() / 2;
    debug_assert_eq!(dst.width(), dst_w);
    debug_assert_eq!(dst.height(), dst_h);

    if dst_w == 0 || dst_h == 0 {
        return;
    }

    if src.is_contiguous()
        && src.width().is_multiple_of(2)
        && src.height().is_multiple_of(2)
        && let Some(src_contig) = src.as_contiguous_slice()
    {
        downsample_u16_contiguous_even(src_contig, src.width(), dst.data_mut(), dst_w, dst_h);
        return;
    }

    downsample_u16_fallback(src, dst.data_mut(), dst_w, dst_h);
}

fn downsample_u8_to_f32_contiguous_even(
    src: &[u8],
    src_w: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    // SAFETY:
    // - `src` is contiguous with `src.len() == src_w * (dst_h * 2)`.
    // - `dst` has exactly `dst_w * dst_h` elements.
    // - Loops only access in-range addresses derived from these lengths.
    unsafe {
        for y in 0..dst_h {
            let src_row0 = src_ptr.add((2 * y) * src_w);
            let src_row1 = src_ptr.add((2 * y + 1) * src_w);
            let dst_row = dst_ptr.add(y * dst_w);
            for x in 0..dst_w {
                let sx = 2 * x;
                let sum = (*src_row0.add(sx) as u32)
                    + (*src_row0.add(sx + 1) as u32)
                    + (*src_row1.add(sx) as u32)
                    + (*src_row1.add(sx + 1) as u32);
                *dst_row.add(x) = (sum as f32) * 0.25;
            }
        }
    }
}

fn downsample_u16_to_f32_contiguous_even(
    src: &[u16],
    src_w: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    // SAFETY:
    // - `src` is contiguous with `src.len() == src_w * (dst_h * 2)`.
    // - `dst` has exactly `dst_w * dst_h` elements.
    // - Loops only access in-range addresses derived from these lengths.
    unsafe {
        for y in 0..dst_h {
            let src_row0 = src_ptr.add((2 * y) * src_w);
            let src_row1 = src_ptr.add((2 * y + 1) * src_w);
            let dst_row = dst_ptr.add(y * dst_w);
            for x in 0..dst_w {
                let sx = 2 * x;
                let sum = (*src_row0.add(sx) as u64)
                    + (*src_row0.add(sx + 1) as u64)
                    + (*src_row1.add(sx) as u64)
                    + (*src_row1.add(sx + 1) as u64);
                *dst_row.add(x) = (sum as f32) * 0.25;
            }
        }
    }
}

fn downsample_f32_contiguous_even(
    src: &[f32],
    src_w: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    // SAFETY:
    // - `src` is contiguous with `src.len() == src_w * (dst_h * 2)`.
    // - `dst` has exactly `dst_w * dst_h` elements.
    // - Loops only access in-range addresses derived from these lengths.
    unsafe {
        for y in 0..dst_h {
            let src_row0 = src_ptr.add((2 * y) * src_w);
            let src_row1 = src_ptr.add((2 * y + 1) * src_w);
            let dst_row = dst_ptr.add(y * dst_w);
            for x in 0..dst_w {
                let sx = 2 * x;
                let sum = *src_row0.add(sx)
                    + *src_row0.add(sx + 1)
                    + *src_row1.add(sx)
                    + *src_row1.add(sx + 1);
                *dst_row.add(x) = sum * 0.25;
            }
        }
    }
}

fn downsample_u8_contiguous_even(
    src: &[u8],
    src_w: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    // SAFETY:
    // - `src` is contiguous with `src.len() == src_w * (dst_h * 2)`.
    // - `dst` has exactly `dst_w * dst_h` elements.
    // - Loops only access in-range addresses derived from these lengths.
    unsafe {
        for y in 0..dst_h {
            let src_row0 = src_ptr.add((2 * y) * src_w);
            let src_row1 = src_ptr.add((2 * y + 1) * src_w);
            let dst_row = dst_ptr.add(y * dst_w);
            for x in 0..dst_w {
                let sx = 2 * x;
                let sum = (*src_row0.add(sx) as u32)
                    + (*src_row0.add(sx + 1) as u32)
                    + (*src_row1.add(sx) as u32)
                    + (*src_row1.add(sx + 1) as u32);
                *dst_row.add(x) = ((sum + 2) / 4) as u8;
            }
        }
    }
}

fn downsample_u16_contiguous_even(
    src: &[u16],
    src_w: usize,
    dst: &mut [u16],
    dst_w: usize,
    dst_h: usize,
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    // SAFETY:
    // - `src` is contiguous with `src.len() == src_w * (dst_h * 2)`.
    // - `dst` has exactly `dst_w * dst_h` elements.
    // - Loops only access in-range addresses derived from these lengths.
    unsafe {
        for y in 0..dst_h {
            let src_row0 = src_ptr.add((2 * y) * src_w);
            let src_row1 = src_ptr.add((2 * y + 1) * src_w);
            let dst_row = dst_ptr.add(y * dst_w);
            for x in 0..dst_w {
                let sx = 2 * x;
                let sum = (*src_row0.add(sx) as u32)
                    + (*src_row0.add(sx + 1) as u32)
                    + (*src_row1.add(sx) as u32)
                    + (*src_row1.add(sx + 1) as u32);
                *dst_row.add(x) = ((sum + 2) / 4) as u16;
            }
        }
    }
}

fn downsample_u8_to_f32_fallback(
    src: &ImageView<'_, u8>,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
) {
    for y in 0..dst_h {
        let src_row0 = src.row(2 * y);
        let src_row1 = src.row(2 * y + 1);
        let dst_row = &mut dst[y * dst_w..(y + 1) * dst_w];
        for (x, out) in dst_row.iter_mut().enumerate() {
            let sx = 2 * x;
            let sum = (src_row0[sx] as u32)
                + (src_row0[sx + 1] as u32)
                + (src_row1[sx] as u32)
                + (src_row1[sx + 1] as u32);
            *out = (sum as f32) * 0.25;
        }
    }
}

fn downsample_u16_to_f32_fallback(
    src: &ImageView<'_, u16>,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
) {
    for y in 0..dst_h {
        let src_row0 = src.row(2 * y);
        let src_row1 = src.row(2 * y + 1);
        let dst_row = &mut dst[y * dst_w..(y + 1) * dst_w];
        for (x, out) in dst_row.iter_mut().enumerate() {
            let sx = 2 * x;
            let sum = (src_row0[sx] as u64)
                + (src_row0[sx + 1] as u64)
                + (src_row1[sx] as u64)
                + (src_row1[sx + 1] as u64);
            *out = (sum as f32) * 0.25;
        }
    }
}

fn downsample_f32_fallback(src: &ImageView<'_, f32>, dst: &mut [f32], dst_w: usize, dst_h: usize) {
    for y in 0..dst_h {
        let src_row0 = src.row(2 * y);
        let src_row1 = src.row(2 * y + 1);
        let dst_row = &mut dst[y * dst_w..(y + 1) * dst_w];
        for (x, out) in dst_row.iter_mut().enumerate() {
            let sx = 2 * x;
            *out = (src_row0[sx] + src_row0[sx + 1] + src_row1[sx] + src_row1[sx + 1]) * 0.25;
        }
    }
}

fn downsample_u8_fallback(src: &ImageView<'_, u8>, dst: &mut [u8], dst_w: usize, dst_h: usize) {
    for y in 0..dst_h {
        let src_row0 = src.row(2 * y);
        let src_row1 = src.row(2 * y + 1);
        let dst_row = &mut dst[y * dst_w..(y + 1) * dst_w];
        for (x, out) in dst_row.iter_mut().enumerate() {
            let sx = 2 * x;
            let sum = (src_row0[sx] as u32)
                + (src_row0[sx + 1] as u32)
                + (src_row1[sx] as u32)
                + (src_row1[sx + 1] as u32);
            *out = ((sum + 2) / 4) as u8;
        }
    }
}

fn downsample_u16_fallback(src: &ImageView<'_, u16>, dst: &mut [u16], dst_w: usize, dst_h: usize) {
    for y in 0..dst_h {
        let src_row0 = src.row(2 * y);
        let src_row1 = src.row(2 * y + 1);
        let dst_row = &mut dst[y * dst_w..(y + 1) * dst_w];
        for (x, out) in dst_row.iter_mut().enumerate() {
            let sx = 2 * x;
            let sum = (src_row0[sx] as u32)
                + (src_row0[sx + 1] as u32)
                + (src_row1[sx] as u32)
                + (src_row1[sx + 1] as u32);
            *out = ((sum + 2) / 4) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use vm_core::Image;

    use crate::downsample::{downsample2x2_mean_u8_to_f32, downsample2x2_mean_u16};

    #[test]
    fn downsample_u8_to_f32_on_4x4_known_values() {
        let src = Image::from_vec(
            4,
            4,
            vec![
                0u8, 1, 2, 3, //
                4, 5, 6, 7, //
                8, 9, 10, 11, //
                12, 13, 14, 15, //
            ],
        )
        .expect("valid image");

        let dst = downsample2x2_mean_u8_to_f32(&src.as_view());
        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 2);
        assert_eq!(dst.data(), &[2.5, 4.5, 10.5, 12.5]);
    }

    #[test]
    fn odd_dimensions_drop_last_row_col() {
        let src = Image::from_vec(
            5,
            3,
            vec![
                1u8, 2, 3, 4, 5, //
                6, 7, 8, 9, 10, //
                11, 12, 13, 14, 15, //
            ],
        )
        .expect("valid image");

        let dst = downsample2x2_mean_u8_to_f32(&src.as_view());
        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 1);
        assert_eq!(dst.data(), &[4.0, 6.0]);
    }

    #[test]
    fn downsample_non_contiguous_view() {
        let src = Image::from_vec(
            6,
            4,
            vec![
                0u8, 1, 2, 3, 4, 5, //
                6, 7, 8, 9, 10, 11, //
                12, 13, 14, 15, 16, 17, //
                18, 19, 20, 21, 22, 23, //
            ],
        )
        .expect("valid image");

        let sub = src.as_view().subview(1, 1, 4, 2).expect("valid subview");
        assert!(!sub.is_contiguous());

        let dst = downsample2x2_mean_u8_to_f32(&sub);
        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 1);
        assert_eq!(dst.data(), &[10.5, 12.5]);
    }

    #[test]
    fn integer_u16_rounding_rule() {
        let src = Image::from_vec(2, 2, vec![1u16, 2, 2, 3]).expect("valid image");
        let dst = downsample2x2_mean_u16(&src.as_view());
        assert_eq!(dst.data(), &[2u16]);
    }
}
