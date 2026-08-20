//! Resampling a source image through a precomputed [`Map`].

use vm_primitives::{BorderMode, Error, ImageView, Pixel, sample_bilinear_f32, sample_nearest};

use super::map::Map;

/// Interpolation kernel used when gathering through a [`Map`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    /// Round to the nearest source pixel. One tap.
    Nearest,
    /// Bilinear blend of the four surrounding source pixels.
    Bilinear,
}

/// `border` converted to `f32` for the bilinear border-fallback path, which
/// works in `f32` throughout (same as [`sample_bilinear_f32`]).
fn border_as_f32<P: Pixel>(border: BorderMode<P>) -> BorderMode<f32> {
    match border {
        BorderMode::Clamp => BorderMode::Clamp,
        BorderMode::Reflect101 => BorderMode::Reflect101,
        BorderMode::Constant(v) => BorderMode::Constant(v.to_f32()),
    }
}

impl Map {
    /// Resample `src` into `dst` through this map.
    ///
    /// `dst.len()` must equal `width() * height()`. Out-of-source taps are
    /// filled from `border`, exactly like any other sampling call in this
    /// workspace; see [`Map::apply_with_mask`] to also learn *which* pixels
    /// that happened to.
    ///
    /// # Errors
    /// [`Error::SizeMismatch`] if `dst.len() != width() * height()`.
    pub fn apply<P: Pixel>(
        &self,
        src: &ImageView<'_, P>,
        dst: &mut [P],
        interp: Interp,
        border: BorderMode<P>,
    ) -> Result<(), Error> {
        self.gather(src, dst, None, interp, border)
    }

    /// Resample `src` into `dst` through this map, also reporting per-pixel
    /// validity into `mask`.
    ///
    /// `mask[i]` is `255` iff every interpolation tap `dst[i]` read (the one
    /// tap for [`Interp::Nearest`], all four for [`Interp::Bilinear`]) fell
    /// inside `src`, `0` otherwise — see the [module docs](super) for why
    /// this is the property that matters, not just the destination pixel's
    /// own coordinate.
    ///
    /// `dst.len()` and `mask.len()` must both equal `width() * height()`.
    ///
    /// # Errors
    /// [`Error::SizeMismatch`] if either length is wrong.
    pub fn apply_with_mask<P: Pixel>(
        &self,
        src: &ImageView<'_, P>,
        dst: &mut [P],
        mask: &mut [u8],
        interp: Interp,
        border: BorderMode<P>,
    ) -> Result<(), Error> {
        self.gather(src, dst, Some(mask), interp, border)
    }

    fn gather<P: Pixel>(
        &self,
        src: &ImageView<'_, P>,
        dst: &mut [P],
        mut mask: Option<&mut [u8]>,
        interp: Interp,
        border: BorderMode<P>,
    ) -> Result<(), Error> {
        let expected = self.coords.len();
        if dst.len() != expected {
            return Err(Error::SizeMismatch {
                expected,
                actual: dst.len(),
            });
        }
        if let Some(m) = mask.as_deref()
            && m.len() != expected
        {
            return Err(Error::SizeMismatch {
                expected,
                actual: m.len(),
            });
        }

        let (sw, sh) = (src.width() as isize, src.height() as isize);
        let border_f32 = border_as_f32(border);

        for (i, p) in self.coords.iter().enumerate() {
            let (value, valid) = match interp {
                Interp::Nearest => {
                    let xi = p.x.round() as isize;
                    let yi = p.y.round() as isize;
                    if xi >= 0 && yi >= 0 && xi < sw && yi < sh {
                        // SAFETY: bounds just checked against `src`'s own
                        // dimensions.
                        let v = unsafe { *src.get_unchecked(xi as usize, yi as usize) };
                        (v, true)
                    } else {
                        (sample_nearest(src, p.x, p.y, border), false)
                    }
                }
                Interp::Bilinear => {
                    let x0 = p.x.floor() as isize;
                    let y0 = p.y.floor() as isize;
                    let (x1, y1) = (x0 + 1, y0 + 1);
                    if x0 >= 0 && y0 >= 0 && x1 < sw && y1 < sh {
                        let (x0u, y0u, x1u, y1u) =
                            (x0 as usize, y0 as usize, x1 as usize, y1 as usize);
                        let dx = p.x - x0 as f32;
                        let dy = p.y - y0 as f32;
                        // SAFETY: all four taps just checked in bounds.
                        let (p00, p10, p01, p11) = unsafe {
                            (
                                src.get_unchecked(x0u, y0u).to_f32(),
                                src.get_unchecked(x1u, y0u).to_f32(),
                                src.get_unchecked(x0u, y1u).to_f32(),
                                src.get_unchecked(x1u, y1u).to_f32(),
                            )
                        };
                        let top = p00 * (1.0 - dx) + p10 * dx;
                        let bottom = p01 * (1.0 - dx) + p11 * dx;
                        let blended = top * (1.0 - dy) + bottom * dy;
                        (P::from_f32_sat(blended), true)
                    } else {
                        let blended = sample_bilinear_f32(src, p.x, p.y, border_f32);
                        (P::from_f32_sat(blended), false)
                    }
                }
            };

            dst[i] = value;
            if let Some(m) = mask.as_deref_mut() {
                m[i] = if valid { 255 } else { 0 };
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Interp;
    use crate::warp::Map;
    use vm_primitives::{Affine2f, BorderMode, Image, Projective2f};

    /// `w x h` image where pixel `(x, y)` holds `10*y + x` as `f32` — any
    /// bilinear read inside the image has a closed-form expected value.
    fn ramp_f32(w: usize, h: usize) -> Image<f32> {
        let data: Vec<f32> = (0..w * h)
            .map(|i| (10 * (i / w) + (i % w)) as f32)
            .collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    #[test]
    fn identity_affine_is_a_copy() {
        let src = Image::from_vec(4, 3, (0u8..12).collect()).expect("valid image");
        let map = Map::affine(4, 3, &Affine2f::identity());
        let mut dst = vec![0u8; 12];
        map.apply(&src.as_view(), &mut dst, Interp::Nearest, BorderMode::Clamp)
            .expect("apply succeeds");
        assert_eq!(dst, src.data());

        let mut dst_bilinear = vec![0u8; 12];
        map.apply(
            &src.as_view(),
            &mut dst_bilinear,
            Interp::Bilinear,
            BorderMode::Clamp,
        )
        .expect("apply succeeds");
        assert_eq!(dst_bilinear, src.data());
    }

    #[test]
    fn translation_by_half_a_pixel_shifts_a_ramp() {
        let (w, h) = (8usize, 8usize);
        let src = ramp_f32(w, h);
        let t = nalgebra::Translation2::new(0.5f32, 0.0);
        let m: Affine2f = nalgebra::convert(t);
        let map = Map::affine(w, h, &m);

        let mut dst = vec![0.0f32; w * h];
        map.apply(
            &src.as_view(),
            &mut dst,
            Interp::Bilinear,
            BorderMode::Clamp,
        )
        .expect("apply succeeds");

        // Interior pixels: dst(x, y) = src(x + 0.5, y) = 10*y + x + 0.5
        // exactly, since bilinear reproduces a linear ramp exactly.
        for y in 0..h {
            for x in 0..w - 1 {
                let want = 10.0 * y as f32 + x as f32 + 0.5;
                let got = dst[y * w + x];
                assert!((got - want).abs() < 1e-4, "at ({x},{y}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn affine_composed_with_its_inverse_is_the_identity() {
        let (w, h) = (32usize, 32usize);
        let src = ramp_f32(w, h);

        let rot = nalgebra::Rotation2::new(0.3f32);
        let t = nalgebra::Translation2::new(4.0f32, -2.5);
        let sim = t * rot;
        let m: Affine2f = nalgebra::convert(sim);
        let m_inv = m.inverse();

        let forward = Map::affine(w, h, &m);
        let backward = Map::affine(w, h, &m_inv);

        let mut mid = vec![0.0f32; w * h];
        forward
            .apply(
                &src.as_view(),
                &mut mid,
                Interp::Bilinear,
                BorderMode::Clamp,
            )
            .expect("apply succeeds");
        let mid_img = Image::from_vec(w, h, mid).expect("valid image");

        let mut recovered = vec![0.0f32; w * h];
        backward
            .apply(
                &mid_img.as_view(),
                &mut recovered,
                Interp::Bilinear,
                BorderMode::Clamp,
            )
            .expect("apply succeeds");

        // Away from the border (clamp on the intermediate image distorts the
        // edge), the roundtrip must recover the original ramp to numerical
        // precision.
        let margin = 6;
        for y in margin..h - margin {
            for x in margin..w - margin {
                let want = src.data()[y * w + x];
                let got = recovered[y * w + x];
                assert!(
                    (got - want).abs() < 1e-3,
                    "at ({x},{y}): {got} vs {want} (affine ∘ affine⁻¹ should be identity)"
                );
            }
        }
    }

    #[test]
    fn projective_embedding_of_an_affine_matches_the_affine_path() {
        let (w, h) = (16usize, 16usize);
        let src = ramp_f32(w, h);

        let rot = nalgebra::Rotation2::new(0.2f32);
        let t = nalgebra::Translation2::new(1.5f32, 2.5);
        let sim = t * rot;
        let m: Affine2f = nalgebra::convert(sim);
        // Embed the same 3x3 homogeneous matrix as a general projective
        // transform (bottom row already [0, 0, 1] for an affine map).
        let h_ = Projective2f::from_matrix_unchecked(*m.matrix());

        let via_affine = Map::affine(w, h, &m);
        let via_projective = Map::projective(w, h, &h_);

        let mut a = vec![0.0f32; w * h];
        let mut b = vec![0.0f32; w * h];
        via_affine
            .apply(&src.as_view(), &mut a, Interp::Bilinear, BorderMode::Clamp)
            .unwrap();
        via_projective
            .apply(&src.as_view(), &mut b, Interp::Bilinear, BorderMode::Clamp)
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn mask_marks_out_of_source_taps_and_nothing_else() {
        let (w, h) = (6usize, 4usize);
        let src = Image::from_vec(w, h, vec![7u8; w * h]).expect("valid image");
        // Shift everything right by 3: the leftmost 3 columns of dst sample
        // outside the source under Nearest.
        let t = nalgebra::Translation2::new(-3.0f32, 0.0);
        let m: Affine2f = nalgebra::convert(t);
        let map = Map::affine(w, h, &m);

        let mut dst = vec![0u8; w * h];
        let mut mask = vec![0u8; w * h];
        map.apply_with_mask(
            &src.as_view(),
            &mut dst,
            &mut mask,
            Interp::Nearest,
            BorderMode::Constant(0u8),
        )
        .expect("apply succeeds");

        for y in 0..h {
            for x in 0..w {
                let expect_valid = x >= 3;
                assert_eq!(
                    mask[y * w + x],
                    if expect_valid { 255 } else { 0 },
                    "at ({x},{y})"
                );
                if expect_valid {
                    assert_eq!(dst[y * w + x], 7);
                } else {
                    assert_eq!(dst[y * w + x], 0, "border fill at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn polar_followed_by_its_inverse_recovers_a_synthetic_disc() {
        use vm_primitives::Point2f;

        // A disc pattern that varies in both angle and radius, so a bug in
        // either axis of `Map::polar`'s bin-center mapping shows up.
        const N: usize = 240;
        let center = Point2f::new(120.0f32, 120.0);
        let (r_lo, r_hi) = (30.0f32, 90.0f32);
        let intensity = |r: f32, phi: f32| -> f32 {
            128.0 + 60.0 * (3.0 * phi).sin() + 80.0 * ((r - r_lo) / (r_hi - r_lo) - 0.5)
        };

        let mut data = vec![0.0f32; N * N];
        for y in 0..N {
            for x in 0..N {
                let dx = x as f32 - center.x;
                let dy = y as f32 - center.y;
                let r = (dx * dx + dy * dy).sqrt();
                let phi = dy.atan2(dx).rem_euclid(core::f32::consts::TAU);
                data[y * N + x] = intensity(r, phi);
            }
        }
        let src = Image::from_vec(N, N, data).expect("valid image");

        // Forward: cartesian -> polar strip (angle x radius).
        let (sw, sh) = (720usize, 240usize);
        let forward = Map::polar(center, r_lo..r_hi, 0.0..core::f32::consts::TAU, sw, sh);
        let mut strip = vec![0.0f32; sw * sh];
        forward
            .apply(
                &src.as_view(),
                &mut strip,
                Interp::Bilinear,
                BorderMode::Clamp,
            )
            .expect("apply succeeds");
        let strip_img = Image::from_vec(sw, sh, strip).expect("valid image");

        // Inverse: polar strip -> cartesian, undoing `Map::polar`'s own
        // bin-center formula (`u = (x+0.5)/w`, `v = (y+0.5)/h`).
        let (swf, shf) = (sw as f32, sh as f32);
        let backward = Map::from_fn(N, N, move |x, y| {
            let dx = x - center.x;
            let dy = y - center.y;
            let r = (dx * dx + dy * dy).sqrt();
            let phi = dy.atan2(dx).rem_euclid(core::f32::consts::TAU);
            let u = phi / core::f32::consts::TAU * swf - 0.5;
            let v = (r - r_lo) / (r_hi - r_lo) * shf - 0.5;
            (u, v)
        });
        let mut recon = vec![0.0f32; N * N];
        backward
            .apply(
                &strip_img.as_view(),
                &mut recon,
                Interp::Bilinear,
                BorderMode::Clamp,
            )
            .expect("apply succeeds");

        // Compare inside the annulus only, a few pixels shy of both radial
        // edges (`v` runs out of the strip's own bin-center range there,
        // same as any resample near an image border).
        let margin = 3.0f32;
        let mut max_abs_diff = 0.0f32;
        for y in 0..N {
            for x in 0..N {
                let dx = x as f32 - center.x;
                let dy = y as f32 - center.y;
                let r = (dx * dx + dy * dy).sqrt();
                if r < r_lo + margin || r > r_hi - margin {
                    continue;
                }
                let phi = dy.atan2(dx).rem_euclid(core::f32::consts::TAU);
                let want = intensity(r, phi);
                let got = recon[y * N + x];
                max_abs_diff = max_abs_diff.max((got - want).abs());
            }
        }
        // Radial gradient is 80/(r_hi-r_lo) = 1.33 intensity/px and the
        // angular gradient at worst is 60*3/r ~ 6 intensity/px near r_lo —
        // 0.05 px of equivalent position error is comfortably under 1
        // intensity unit either way. Measured max (2026-08-20): 0.787.
        assert!(
            max_abs_diff < 1.0,
            "polar -> inverse-polar roundtrip diff {max_abs_diff} too large"
        );
    }

    #[test]
    fn size_mismatch_is_reported_for_dst_and_mask() {
        let src = Image::from_vec(2, 2, vec![0u8; 4]).expect("valid image");
        let map = Map::affine(2, 2, &Affine2f::identity());

        let mut short_dst = vec![0u8; 3];
        assert!(
            map.apply(
                &src.as_view(),
                &mut short_dst,
                Interp::Nearest,
                BorderMode::Clamp
            )
            .is_err()
        );

        let mut dst = vec![0u8; 4];
        let mut short_mask = vec![0u8; 3];
        assert!(
            map.apply_with_mask(
                &src.as_view(),
                &mut dst,
                &mut short_mask,
                Interp::Nearest,
                BorderMode::Clamp
            )
            .is_err()
        );
    }
}
