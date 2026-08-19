use crate::core::{Image, ImageView, Pixel};

use super::downsample::{downsample2x2_mean_to_f32_into, dst_dims};

/// Pre-filter applied to a level before it is decimated to the next one.
///
/// A plain 2×2 box mean has no stop-band: content finer than the new Nyquist
/// aliases instead of vanishing, which is how a fine-toothed contour can die at
/// pyramid level 3–4 (backlog item **R3**). A symmetric 3-tap binomial run
/// before each decimation suppresses that.
///
/// The filter is symmetric about the pixel centre, so it does **not** move the
/// level-to-level coordinate mapping — [`level_to_base`] is unaffected.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PreSmooth {
    /// No pre-filter: each level is exactly the 2×2 mean of the one below.
    ///
    /// The default, and the behaviour every stored [`ShapeModel`] built before
    /// this option existed depends on.
    ///
    /// [`ShapeModel`]: https://docs.rs/vision-metrology
    #[default]
    None,
    /// Separable `[1 2 1] / 4` in both axes, with `Clamp` borders.
    Binomial121,
}

/// How a [`Pyramid`] is built.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PyramidConfig {
    /// Pre-filter applied before each decimation step. Default [`PreSmooth::None`].
    pub pre_smooth: PreSmooth,
}

/// Reusable `f32` image pyramid.
///
/// Level 0 is an `f32` copy of the input; each next level is a 2×2 mean of the
/// previous one, so level `l` is the mean over a `2^l × 2^l` block of level 0.
/// A trailing odd column or row is dropped at every step.
///
/// Buffers are retained between calls: rebuilding at the same size allocates
/// nothing. Building stops early when a level would fall below 2×2.
///
/// # Example
/// ```
/// use vm_primitives::{Image, Pyramid};
///
/// let img = Image::from_vec(16, 16, vec![7u8; 256]).unwrap();
/// let mut pyr = Pyramid::new();
/// pyr.build(&img.as_view(), 3);
///
/// assert_eq!(pyr.num_levels(), 3);
/// assert_eq!(pyr.level(2).unwrap().width(), 4);
/// // A constant image survives box-mean downsampling exactly.
/// assert_eq!(pyr.level(2).unwrap().data(), &[7.0; 16]);
/// ```
#[derive(Debug, Default, Clone)]
pub struct Pyramid {
    levels: Vec<Image<f32>>,
    /// Scratch for the separable pre-smooth; empty when `PreSmooth::None`.
    scratch: Vec<f32>,
}

impl Pyramid {
    /// Create an empty pyramid. No allocation until the first build.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from any [`Pixel`] source with the default (box-mean) config.
    pub fn build<P: Pixel>(&mut self, src: &ImageView<'_, P>, num_levels: usize) {
        self.build_with(src, num_levels, &PyramidConfig::default());
    }

    /// Build from any [`Pixel`] source.
    ///
    /// Level 0 converts `src` to `f32`; every level after it is a 2×2 mean of
    /// its predecessor, optionally pre-smoothed per `cfg`. Building stops at
    /// `num_levels` or when the next level would be smaller than 2×2,
    /// whichever comes first — check [`num_levels`](Self::num_levels) for what
    /// was actually built.
    pub fn build_with<P: Pixel>(
        &mut self,
        src: &ImageView<'_, P>,
        num_levels: usize,
        cfg: &PyramidConfig,
    ) {
        let build_levels = max_build_levels(src.width(), src.height(), num_levels);
        if build_levels == 0 {
            self.levels.clear();
            return;
        }

        self.ensure(src.width(), src.height(), build_levels);
        copy_to_f32(src, &mut self.levels[0]);

        for idx in 1..build_levels {
            let (head, tail) = self.levels.split_at_mut(idx);
            let prev = &mut head[idx - 1];
            if cfg.pre_smooth == PreSmooth::Binomial121 {
                binomial121_in_place(prev, &mut self.scratch);
            }
            downsample2x2_mean_to_f32_into(&prev.as_view(), &mut tail[0].as_view_mut())
                .expect("level dimensions come from the same halving chain");
        }
    }

    /// Level `i`, or `None` when `i >= num_levels()`.
    pub fn level(&self, i: usize) -> Option<&Image<f32>> {
        self.levels.get(i)
    }

    /// Number of levels actually built.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Resize the level buffers to the halving chain implied by the arguments,
    /// reusing any level that already has the right dimensions.
    ///
    /// Private on purpose: it does not apply the "stop below 2×2" rule, so
    /// calling it directly with too many levels would construct 0×0 levels.
    /// [`build_with`](Self::build_with) always clamps via [`max_build_levels`]
    /// first.
    fn ensure(&mut self, base_w: usize, base_h: usize, num_levels: usize) {
        self.levels.truncate(num_levels);
        self.levels
            .resize_with(num_levels, || Image::new_fill(0, 0, 0.0f32));

        let (mut w, mut h) = (base_w, base_h);
        for level in &mut self.levels {
            if level.width() != w || level.height() != h {
                *level = Image::new_fill(w, h, 0.0f32);
            }
            (w, h) = dst_dims(w, h);
        }
    }
}

/// Map a level-`level` coordinate to level-0 (base) coordinates.
///
/// `base = v · 2^level + (2^level − 1) / 2`
///
/// The half-pixel term is the centre of the `2^level × 2^level` block that a
/// level-`level` pixel summarises, under the drop-odd 2×2 box mean and the
/// pixel-centre convention. This is **system-design invariant 2** and this is
/// its single implementation — do not re-derive it at call sites.
///
/// # Example
/// ```
/// use vm_primitives::pyr::{base_to_level, level_to_base};
///
/// // Level 0 is the identity.
/// assert_eq!(level_to_base(3.0, 0), 3.0);
/// // A level-1 pixel sits between two base pixels.
/// assert_eq!(level_to_base(0.0, 1), 0.5);
/// // Level 2 spans four: centre of block [0..4) is 1.5.
/// assert_eq!(level_to_base(0.0, 2), 1.5);
/// // The two directions are inverses.
/// assert!((base_to_level(level_to_base(7.25, 3), 3) - 7.25).abs() < 1e-6);
/// ```
#[inline]
pub fn level_to_base(v: f32, level: u32) -> f32 {
    let s = (1u32 << level) as f32;
    v * s + 0.5 * (s - 1.0)
}

/// Map a level-0 (base) coordinate to level `level`. Inverse of [`level_to_base`].
///
/// `v = (base − (2^level − 1) / 2) / 2^level`
#[inline]
pub fn base_to_level(base: f32, level: u32) -> f32 {
    let s = (1u32 << level) as f32;
    (base - 0.5 * (s - 1.0)) / s
}

/// How many levels can be built before one would fall below 2×2.
fn max_build_levels(base_w: usize, base_h: usize, requested: usize) -> usize {
    if requested == 0 || base_w == 0 || base_h == 0 {
        return 0;
    }
    let (mut levels, mut w, mut h) = (1usize, base_w, base_h);
    while levels < requested && w >= 2 && h >= 2 {
        (w, h) = dst_dims(w, h);
        levels += 1;
    }
    levels
}

/// Convert `src` into `dst`, which must already have `src`'s dimensions.
///
/// Level 0 is the largest single write in a pyramid build (5 MB for a
/// 1280×1024 frame), so the common packed case gets one flat loop rather than
/// one loop per row — worth ~4% of the whole build.
fn copy_to_f32<P: Pixel>(src: &ImageView<'_, P>, dst: &mut Image<f32>) {
    debug_assert_eq!((src.width(), src.height()), (dst.width(), dst.height()));

    if let Some(packed) = src.as_contiguous_slice() {
        for (d, s) in dst.data_mut().iter_mut().zip(packed) {
            *d = s.to_f32();
        }
        return;
    }

    let dst_w = dst.width();
    let data = dst.data_mut();
    for y in 0..src.height() {
        let out = &mut data[y * dst_w..(y + 1) * dst_w];
        for (d, s) in out.iter_mut().zip(src.row(y)) {
            *d = s.to_f32();
        }
    }
}

/// Separable `[1 2 1] / 4` in place, `Clamp` borders, using `scratch` for the
/// intermediate horizontal pass.
fn binomial121_in_place(img: &mut Image<f32>, scratch: &mut Vec<f32>) {
    let (w, h) = (img.width(), img.height());
    if w < 3 && h < 3 {
        return;
    }
    scratch.clear();
    scratch.resize(w * h, 0.0);
    let data = img.data_mut();

    // Horizontal pass: data -> scratch.
    if w >= 3 {
        for y in 0..h {
            let row = &data[y * w..(y + 1) * w];
            let out = &mut scratch[y * w..(y + 1) * w];
            out[0] = (row[0] * 3.0 + row[1]) * 0.25;
            for x in 1..w - 1 {
                out[x] = (row[x - 1] + 2.0 * row[x] + row[x + 1]) * 0.25;
            }
            out[w - 1] = (row[w - 2] + row[w - 1] * 3.0) * 0.25;
        }
    } else {
        scratch.copy_from_slice(data);
    }

    // Vertical pass: scratch -> data.
    if h >= 3 {
        for y in 0..h {
            let (ym, yp) = (y.saturating_sub(1), (y + 1).min(h - 1));
            for x in 0..w {
                data[y * w + x] =
                    (scratch[ym * w + x] + 2.0 * scratch[y * w + x] + scratch[yp * w + x]) * 0.25;
            }
        }
    } else {
        data.copy_from_slice(scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::{PreSmooth, Pyramid, PyramidConfig, base_to_level, level_to_base};
    use crate::core::Image;

    fn ramp_u8(w: usize, h: usize) -> Image<u8> {
        Image::from_vec(w, h, (0..w * h).map(|i| (i % 251) as u8).collect()).expect("valid image")
    }

    #[test]
    fn build_halves_until_1x1() {
        let mut pyr = Pyramid::new();
        pyr.build(&ramp_u8(16, 16).as_view(), 10);
        let dims: Vec<_> = (0..pyr.num_levels())
            .map(|i| {
                let l = pyr.level(i).expect("built level");
                (l.width(), l.height())
            })
            .collect();
        assert_eq!(dims, vec![(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]);
    }

    #[test]
    fn level_zero_is_an_f32_copy_of_any_pixel_type() {
        let mut pyr = Pyramid::new();

        pyr.build(
            &Image::from_vec(3, 2, vec![1u8, 2, 3, 4, 5, 6])
                .unwrap()
                .as_view(),
            1,
        );
        assert_eq!(
            pyr.level(0).unwrap().data(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        pyr.build(
            &Image::from_vec(2, 1, vec![1000u16, 65535])
                .unwrap()
                .as_view(),
            1,
        );
        assert_eq!(pyr.level(0).unwrap().data(), &[1000.0, 65535.0]);

        pyr.build(
            &Image::from_vec(2, 1, vec![0.5f32, -1.5]).unwrap().as_view(),
            1,
        );
        assert_eq!(pyr.level(0).unwrap().data(), &[0.5, -1.5]);
    }

    /// The three pixel types must agree once they are on a common scale.
    #[test]
    fn pixel_types_agree_on_the_same_content() {
        let u8_src = ramp_u8(8, 8);
        let f32_src =
            Image::from_vec(8, 8, u8_src.data().iter().map(|&v| f32::from(v)).collect()).unwrap();
        let u16_src =
            Image::from_vec(8, 8, u8_src.data().iter().map(|&v| u16::from(v)).collect()).unwrap();

        let (mut a, mut b, mut c) = (Pyramid::new(), Pyramid::new(), Pyramid::new());
        a.build(&u8_src.as_view(), 4);
        b.build(&f32_src.as_view(), 4);
        c.build(&u16_src.as_view(), 4);

        for l in 0..4 {
            assert_eq!(
                a.level(l).unwrap().data(),
                b.level(l).unwrap().data(),
                "level {l}"
            );
            assert_eq!(
                a.level(l).unwrap().data(),
                c.level(l).unwrap().data(),
                "level {l}"
            );
        }
    }

    #[test]
    fn rebuilding_at_the_same_size_reuses_buffers() {
        let mut pyr = Pyramid::new();
        pyr.build(&ramp_u8(32, 32).as_view(), 4);
        let ptr = pyr.level(1).unwrap().data().as_ptr();
        pyr.build(&ramp_u8(32, 32).as_view(), 4);
        assert_eq!(
            pyr.level(1).unwrap().data().as_ptr(),
            ptr,
            "level 1 reallocated"
        );
    }

    #[test]
    fn zero_levels_clears() {
        let mut pyr = Pyramid::new();
        pyr.build(&ramp_u8(4, 4).as_view(), 2);
        assert_eq!(pyr.num_levels(), 2);
        pyr.build(&ramp_u8(4, 4).as_view(), 0);
        assert_eq!(pyr.num_levels(), 0);
    }

    #[test]
    fn coordinate_mapping_round_trips() {
        for level in 0..5u32 {
            for &v in &[0.0f32, 1.0, 7.5, 123.25] {
                let back = base_to_level(level_to_base(v, level), level);
                assert!((back - v).abs() < 1e-4, "level {level}, v {v}, back {back}");
            }
        }
        // The block a level-l pixel summarises is centred on the mapped point.
        assert_eq!(level_to_base(0.0, 1), 0.5);
        assert_eq!(level_to_base(0.0, 2), 1.5);
        assert_eq!(level_to_base(0.0, 3), 3.5);
    }

    /// The default must stay a plain box mean — stored shape models depend on it.
    #[test]
    fn default_config_is_bit_identical_to_plain_box_mean() {
        let src = ramp_u8(64, 64);
        let (mut plain, mut cfgd) = (Pyramid::new(), Pyramid::new());
        plain.build(&src.as_view(), 5);
        cfgd.build_with(&src.as_view(), 5, &PyramidConfig::default());
        for l in 0..plain.num_levels() {
            assert_eq!(
                plain.level(l).unwrap().data(),
                cfgd.level(l).unwrap().data()
            );
        }
    }

    /// R3, demonstrated: a raw box mean has no stop-band, so fine structure
    /// survives at *full* contrast for one level and then annihilates itself.
    ///
    /// The fixture is a 4 px-period stripe. Under `PreSmooth::None` the measured
    /// intensity span goes `200 → 200 → 0 → 0`: level 1 is undiminished, then
    /// level 2 cancels it completely. That cliff is exactly how a fine-toothed
    /// contour dies mid-descent and takes the true match with it. With
    /// `Binomial121` the same fixture decays `200 → 87.5 → 37.5 → 25`: still
    /// present, and monotonically attenuated, at every level.
    #[test]
    fn pre_smooth_replaces_the_aliasing_cliff_with_a_gentle_decay() {
        let (w, h) = (64usize, 64usize);
        let data: Vec<u8> = (0..w * h)
            .map(|i| if (i % w) % 4 < 2 { 0u8 } else { 200 })
            .collect();
        let src = Image::from_vec(w, h, data).expect("valid image");

        let spans = |pre_smooth: PreSmooth| {
            let mut pyr = Pyramid::new();
            pyr.build_with(&src.as_view(), 4, &PyramidConfig { pre_smooth });
            (0..pyr.num_levels())
                .map(|l| {
                    let d = pyr.level(l).expect("built level").data();
                    let lo = d.iter().copied().fold(f32::MAX, f32::min);
                    let hi = d.iter().copied().fold(f32::MIN, f32::max);
                    hi - lo
                })
                .collect::<Vec<_>>()
        };

        let plain = spans(PreSmooth::None);
        assert_eq!(plain[0], 200.0, "level 0 is a plain copy");
        assert_eq!(plain[1], 200.0, "box mean does not attenuate at all");
        assert_eq!(plain[2], 0.0, "and then the pattern annihilates itself");
        assert_eq!(plain[3], 0.0);

        let smooth = spans(PreSmooth::Binomial121);
        assert_eq!(smooth[0], 200.0, "level 0 is still a plain copy");
        for l in 1..4 {
            assert!(
                smooth[l] < smooth[l - 1],
                "level {l} should be attenuated below level {}: {:?}",
                l - 1,
                smooth
            );
            assert!(
                smooth[l] > 1.0,
                "level {l} should survive, not vanish: {smooth:?}"
            );
        }
    }

    /// A constant field is a fixed point of the pre-filter, including at borders.
    #[test]
    fn pre_smooth_preserves_a_constant_field() {
        let src = Image::from_vec(16, 16, vec![42u8; 256]).expect("valid image");
        let mut pyr = Pyramid::new();
        pyr.build_with(
            &src.as_view(),
            3,
            &PyramidConfig {
                pre_smooth: PreSmooth::Binomial121,
            },
        );
        for l in 0..pyr.num_levels() {
            for &v in pyr.level(l).unwrap().data() {
                assert!((v - 42.0).abs() < 1e-4, "level {l} drifted to {v}");
            }
        }
    }
}
