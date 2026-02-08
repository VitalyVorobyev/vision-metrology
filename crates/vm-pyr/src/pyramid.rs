use vm_core::{Image, ImageView};

use crate::downsample::downsample2x2_mean_f32_into;

/// Reusable f32 image pyramid.
///
/// Level 0 is a f32 copy of the input. Each next level is a 2x2 mean
/// downsample of the previous level.
///
/// If a requested level cannot be built because `width < 2` or `height < 2`,
/// building stops early.
#[derive(Debug, Default, Clone)]
pub struct PyramidF32 {
    levels: Vec<Image<f32>>,
}

impl PyramidF32 {
    pub fn new() -> Self {
        Self { levels: Vec::new() }
    }

    /// Ensures that internal buffers match the size chain implied by
    /// `(base_w, base_h, num_levels)`.
    ///
    /// Level dimensions are computed with integer halving:
    /// `(w, h), (w/2, h/2), ...`.
    pub fn ensure(&mut self, base_w: usize, base_h: usize, num_levels: usize) {
        if num_levels == 0 {
            self.levels.clear();
            return;
        }

        self.levels.truncate(num_levels);
        self.levels
            .resize_with(num_levels, || Image::new_fill(0, 0, 0.0f32));

        let mut w = base_w;
        let mut h = base_h;
        for level in &mut self.levels {
            if level.width() != w || level.height() != h {
                *level = Image::new_fill(w, h, 0.0f32);
            }
            w /= 2;
            h /= 2;
        }
    }

    pub fn build_from_u8(&mut self, src: &ImageView<'_, u8>, num_levels: usize) {
        let build_levels = max_build_levels(src.width(), src.height(), num_levels);
        if build_levels == 0 {
            self.levels.clear();
            return;
        }

        self.ensure(src.width(), src.height(), build_levels);
        copy_u8_to_f32(src, &mut self.levels[0]);

        for level_idx in 1..build_levels {
            let (prev_levels, curr_and_tail) = self.levels.split_at_mut(level_idx);
            let prev = &prev_levels[level_idx - 1];
            let curr = &mut curr_and_tail[0];
            downsample2x2_mean_f32_into(&prev.as_view(), curr);
        }
    }

    pub fn build_from_u16(&mut self, src: &ImageView<'_, u16>, num_levels: usize) {
        let build_levels = max_build_levels(src.width(), src.height(), num_levels);
        if build_levels == 0 {
            self.levels.clear();
            return;
        }

        self.ensure(src.width(), src.height(), build_levels);
        copy_u16_to_f32(src, &mut self.levels[0]);

        for level_idx in 1..build_levels {
            let (prev_levels, curr_and_tail) = self.levels.split_at_mut(level_idx);
            let prev = &prev_levels[level_idx - 1];
            let curr = &mut curr_and_tail[0];
            downsample2x2_mean_f32_into(&prev.as_view(), curr);
        }
    }

    pub fn build_from_f32(&mut self, src: &ImageView<'_, f32>, num_levels: usize) {
        let build_levels = max_build_levels(src.width(), src.height(), num_levels);
        if build_levels == 0 {
            self.levels.clear();
            return;
        }

        self.ensure(src.width(), src.height(), build_levels);
        copy_f32(src, &mut self.levels[0]);

        for level_idx in 1..build_levels {
            let (prev_levels, curr_and_tail) = self.levels.split_at_mut(level_idx);
            let prev = &prev_levels[level_idx - 1];
            let curr = &mut curr_and_tail[0];
            downsample2x2_mean_f32_into(&prev.as_view(), curr);
        }
    }

    pub fn level(&self, i: usize) -> Option<&Image<f32>> {
        self.levels.get(i)
    }

    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }
}

fn max_build_levels(base_w: usize, base_h: usize, requested_levels: usize) -> usize {
    if requested_levels == 0 || base_w == 0 || base_h == 0 {
        return 0;
    }

    let mut levels = 1usize;
    let mut w = base_w;
    let mut h = base_h;
    while levels < requested_levels && w >= 2 && h >= 2 {
        w /= 2;
        h /= 2;
        levels += 1;
    }
    levels
}

fn copy_u8_to_f32(src: &ImageView<'_, u8>, dst: &mut Image<f32>) {
    debug_assert_eq!(src.width(), dst.width());
    debug_assert_eq!(src.height(), dst.height());

    if src.is_contiguous()
        && let Some(src_contig) = src.as_contiguous_slice()
    {
        for (dst_px, &src_px) in dst.data_mut().iter_mut().zip(src_contig.iter()) {
            *dst_px = src_px as f32;
        }
        return;
    }

    let dst_w = dst.width();
    copy_u8_to_f32_fallback(src, dst.data_mut(), dst_w);
}

fn copy_u16_to_f32(src: &ImageView<'_, u16>, dst: &mut Image<f32>) {
    debug_assert_eq!(src.width(), dst.width());
    debug_assert_eq!(src.height(), dst.height());

    if src.is_contiguous()
        && let Some(src_contig) = src.as_contiguous_slice()
    {
        for (dst_px, &src_px) in dst.data_mut().iter_mut().zip(src_contig.iter()) {
            *dst_px = src_px as f32;
        }
        return;
    }

    let dst_w = dst.width();
    copy_u16_to_f32_fallback(src, dst.data_mut(), dst_w);
}

fn copy_f32(src: &ImageView<'_, f32>, dst: &mut Image<f32>) {
    debug_assert_eq!(src.width(), dst.width());
    debug_assert_eq!(src.height(), dst.height());

    if src.is_contiguous()
        && let Some(src_contig) = src.as_contiguous_slice()
    {
        dst.data_mut().copy_from_slice(src_contig);
        return;
    }

    let dst_w = dst.width();
    copy_f32_fallback(src, dst.data_mut(), dst_w);
}

fn copy_u8_to_f32_fallback(src: &ImageView<'_, u8>, dst: &mut [f32], dst_w: usize) {
    for y in 0..src.height() {
        let src_row = src.row(y);
        let dst_row = &mut dst[y * dst_w..(y + 1) * dst_w];
        for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
            *d = s as f32;
        }
    }
}

fn copy_u16_to_f32_fallback(src: &ImageView<'_, u16>, dst: &mut [f32], dst_w: usize) {
    for y in 0..src.height() {
        let src_row = src.row(y);
        let dst_row = &mut dst[y * dst_w..(y + 1) * dst_w];
        for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
            *d = s as f32;
        }
    }
}

fn copy_f32_fallback(src: &ImageView<'_, f32>, dst: &mut [f32], dst_w: usize) {
    for y in 0..src.height() {
        let src_row = src.row(y);
        let dst_row = &mut dst[y * dst_w..(y + 1) * dst_w];
        dst_row.copy_from_slice(src_row);
    }
}

#[cfg(test)]
mod tests {
    use vm_core::Image;

    use crate::PyramidF32;

    #[test]
    fn pyramid_build_from_u8_stops_at_1x1() {
        let mut data = Vec::with_capacity(16 * 16);
        for i in 0..(16 * 16) {
            data.push((i % 251) as u8);
        }
        let src = Image::from_vec(16, 16, data).expect("valid image");

        let mut pyr = PyramidF32::new();
        pyr.build_from_u8(&src.as_view(), 10);

        assert_eq!(pyr.num_levels(), 5);
        let dims: Vec<(usize, usize)> = (0..pyr.num_levels())
            .map(|i| {
                let level = pyr.level(i).expect("level should exist");
                (level.width(), level.height())
            })
            .collect();
        assert_eq!(dims, vec![(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]);
    }

    #[test]
    fn pyramid_level_zero_is_f32_copy() {
        let src = Image::from_vec(3, 2, vec![1u8, 2, 3, 4, 5, 6]).expect("valid image");
        let mut pyr = PyramidF32::new();
        pyr.build_from_u8(&src.as_view(), 3);

        let l0 = pyr.level(0).expect("level 0");
        assert_eq!(l0.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn build_zero_levels_clears_pyramid() {
        let src = Image::from_vec(4, 4, vec![1u8; 16]).expect("valid image");
        let mut pyr = PyramidF32::new();
        pyr.build_from_u8(&src.as_view(), 2);
        assert_eq!(pyr.num_levels(), 2);
        pyr.build_from_u8(&src.as_view(), 0);
        assert_eq!(pyr.num_levels(), 0);
    }
}
