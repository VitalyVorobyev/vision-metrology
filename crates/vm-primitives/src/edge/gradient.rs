//! Dense unit gradient direction field.
//!
//! [`DirectionField`] is the scene-side data structure for gradient-orientation
//! similarity measures (shape-based matching, template correlation). It answers
//! one question for every pixel — *which way does the intensity increase here,
//! and is the edge strong enough to be worth believing* — and it answers it
//! with a plain array lookup.
//!
//! # Why this is not `Edge2DDetector`
//!
//! [`Edge2DDetector`](crate::Edge2DDetector) exists to produce a sparse
//! `Vec<Edgel>`: it runs non-maximum suppression, hysteresis, and subpixel
//! refinement on top of the gradient. A matcher that samples the gradient at a
//! few million *arbitrary* positions needs none of that, and cannot use the
//! detector's [`GradientBuffers`](crate::GradientBuffers) either — those borrow
//! the detector mutably, so they cannot be held for more than one pyramid level
//! at a time.
//!
//! # Layout
//!
//! Directions are stored interleaved, `dir[2 * (y * width + x)]` = `nx` and
//! `dir[2 * (y * width + x) + 1]` = `ny`, so one 8-byte load serves an inner
//! loop iteration. Both components are **zero** wherever the gradient magnitude
//! is below `min_mag`; a scoring loop therefore needs no branch and no compare
//! to reject low-contrast pixels — they simply contribute nothing.
//!
//! Magnitudes are kept in a separate `width × height` plane, used by subpixel
//! refinement (which interpolates along the normal) but not by scoring.
//!
//! # Coordinate convention
//!
//! Pixel centres, as everywhere in this crate: `(x, y)` is the centre of pixel
//! `(x, y)`. Directions point **dark to bright**, the same convention as
//! [`Edgel::n`](crate::Edgel).

use crate::core::{Image, ImageView};

use super::edge2d::SmoothKind;

/// Dense per-pixel unit gradient direction with a magnitude gate.
///
/// Build with [`DirectionField::build_u8`] / [`build_u16`](Self::build_u16) /
/// [`build_f32`](Self::build_f32); the internal buffers are reused across
/// calls, so a field kept in a long-lived struct allocates only when the image
/// dimensions change.
///
/// # Example
/// ```
/// use vm_primitives::{DirectionField, Image, SmoothKind};
///
/// // Vertical step edge: dark on the left, bright on the right.
/// let mut data = vec![0u8; 32 * 16];
/// for y in 0..16 {
///     for x in 16..32 {
///         data[y * 32 + x] = 200;
///     }
/// }
/// let img = Image::from_vec(32, 16, data).unwrap();
///
/// let mut field = DirectionField::new();
/// field.build_u8(&img.as_view(), SmoothKind::None, 10.0);
///
/// // On the step the direction points along +x (dark to bright)...
/// let (nx, ny) = field.dir_at(15, 8);
/// assert!(nx > 0.99 && ny.abs() < 1e-6);
/// // ...and in the flat interior it is gated to exactly zero.
/// assert_eq!(field.dir_at(4, 8), (0.0, 0.0));
/// ```
#[derive(Debug, Clone, Default)]
pub struct DirectionField {
    dir: Vec<f32>,
    mag: Vec<f32>,
    tmp: Vec<f32>,
    scratch: Vec<f32>,
    width: usize,
    height: usize,
    min_mag: f32,
}

impl DirectionField {
    /// Create an empty field. Buffers are allocated on the first `build_*` call.
    pub fn new() -> Self {
        Self::default()
    }

    /// Image width in pixels (0 before the first build).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels (0 before the first build).
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Magnitude gate the field was built with.
    #[inline]
    pub fn min_mag(&self) -> f32 {
        self.min_mag
    }

    /// Interleaved `[nx, ny]` directions, `2 · width · height` entries.
    #[inline]
    pub fn dir(&self) -> &[f32] {
        &self.dir
    }

    /// Gradient magnitudes, `width · height` entries, in input pixel units.
    #[inline]
    pub fn mag(&self) -> &[f32] {
        &self.mag
    }

    /// Unit direction at pixel `(x, y)`, or `(0, 0)` outside the image.
    #[inline]
    pub fn dir_at(&self, x: usize, y: usize) -> (f32, f32) {
        if x >= self.width || y >= self.height {
            return (0.0, 0.0);
        }
        let k = 2 * (y * self.width + x);
        (self.dir[k], self.dir[k + 1])
    }

    /// Gradient magnitude at pixel `(x, y)`, or `0` outside the image.
    ///
    /// Unlike [`dir_at`](Self::dir_at) this is **not** gated: the true magnitude
    /// is reported even below `min_mag`.
    #[inline]
    pub fn mag_at(&self, x: usize, y: usize) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.mag[y * self.width + x]
    }

    /// Bilinearly interpolated gradient magnitude at a subpixel position.
    ///
    /// Returns 0 when the 2×2 interpolation neighbourhood is not fully inside
    /// the image, so callers get a smooth "no evidence" answer at the border
    /// rather than a clamped fabrication.
    pub fn sample_mag(&self, x: f32, y: f32) -> f32 {
        if !x.is_finite() || !y.is_finite() || self.width < 2 || self.height < 2 {
            return 0.0;
        }
        let x0 = x.floor();
        let y0 = y.floor();
        if x0 < 0.0 || y0 < 0.0 {
            return 0.0;
        }
        let (xi, yi) = (x0 as usize, y0 as usize);
        if xi + 1 >= self.width || yi + 1 >= self.height {
            return 0.0;
        }
        let fx = x - x0;
        let fy = y - y0;
        let row0 = yi * self.width + xi;
        let row1 = row0 + self.width;
        let top = self.mag[row0] * (1.0 - fx) + self.mag[row0 + 1] * fx;
        let bot = self.mag[row1] * (1.0 - fx) + self.mag[row1 + 1] * fx;
        top * (1.0 - fy) + bot * fy
    }

    /// Build from a `u8` image.
    ///
    /// `min_mag` is expressed in **Scharr response units on the input pixel
    /// scale**: a clean black/white step in `u8` gives `|∇I| ≈ 8·255 ≈ 2000`, so
    /// a gate of 10 sits well above 8-bit sensor noise while keeping faint but
    /// real edges. Re-tune it for `u16` and `f32` inputs, whose pixel scales
    /// differ by orders of magnitude.
    pub fn build_u8(&mut self, img: &ImageView<'_, u8>, smooth: SmoothKind, min_mag: f32) {
        self.ensure(img.width(), img.height());
        for y in 0..img.height() {
            let src = img.row(y);
            let dst = &mut self.tmp[y * self.width..(y + 1) * self.width];
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d = f32::from(s);
            }
        }
        self.finish(smooth, min_mag);
    }

    /// Build from a `u16` image. See [`build_u8`](Self::build_u8) on `min_mag`.
    pub fn build_u16(&mut self, img: &ImageView<'_, u16>, smooth: SmoothKind, min_mag: f32) {
        self.ensure(img.width(), img.height());
        for y in 0..img.height() {
            let src = img.row(y);
            let dst = &mut self.tmp[y * self.width..(y + 1) * self.width];
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d = f32::from(s);
            }
        }
        self.finish(smooth, min_mag);
    }

    /// Build from an `f32` image. See [`build_u8`](Self::build_u8) on `min_mag`.
    pub fn build_f32(&mut self, img: &ImageView<'_, f32>, smooth: SmoothKind, min_mag: f32) {
        self.ensure(img.width(), img.height());
        for y in 0..img.height() {
            let src = img.row(y);
            let dst = &mut self.tmp[y * self.width..(y + 1) * self.width];
            dst.copy_from_slice(src);
        }
        self.finish(smooth, min_mag);
    }

    /// Build from an owned image (convenience for pyramid levels).
    pub fn build_image_f32(&mut self, img: &Image<f32>, smooth: SmoothKind, min_mag: f32) {
        self.build_f32(&img.as_view(), smooth, min_mag);
    }

    fn ensure(&mut self, w: usize, h: usize) {
        if self.width != w || self.height != h {
            self.width = w;
            self.height = h;
            let n = w.saturating_mul(h);
            self.dir = vec![0.0; 2 * n];
            self.mag = vec![0.0; n];
            self.tmp = vec![0.0; n];
            self.scratch = vec![0.0; n];
        }
    }

    fn finish(&mut self, smooth: SmoothKind, min_mag: f32) {
        self.min_mag = min_mag;
        let (w, h) = (self.width, self.height);
        if w == 0 || h == 0 {
            return;
        }
        if smooth == SmoothKind::Binomial3 {
            self.smooth_binomial3();
        }
        self.scharr_normalised(min_mag);
    }

    /// Separable `[1, 2, 1] / 4` smoothing, clamped at the border.
    fn smooth_binomial3(&mut self) {
        let (w, h) = (self.width, self.height);
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let xm1 = x.saturating_sub(1);
                let xp1 = (x + 1).min(w - 1);
                self.scratch[row + x] =
                    0.25 * (self.tmp[row + xm1] + 2.0 * self.tmp[row + x] + self.tmp[row + xp1]);
            }
        }
        for y in 0..h {
            let r0 = y.saturating_sub(1) * w;
            let r1 = y * w;
            let r2 = (y + 1).min(h - 1) * w;
            for x in 0..w {
                self.tmp[r1 + x] = 0.25
                    * (self.scratch[r0 + x] + 2.0 * self.scratch[r1 + x] + self.scratch[r2 + x]);
            }
        }
    }

    /// Scharr 3×3 gradient, magnitude, and gated normalisation in one pass.
    fn scharr_normalised(&mut self, min_mag: f32) {
        let (w, h) = (self.width, self.height);
        let src = &self.tmp;
        for y in 0..h {
            let ym1 = y.saturating_sub(1) * w;
            let y0 = y * w;
            let yp1 = (y + 1).min(h - 1) * w;
            for x in 0..w {
                let xm1 = x.saturating_sub(1);
                let xp1 = (x + 1).min(w - 1);

                let p00 = src[ym1 + xm1];
                let p01 = src[ym1 + x];
                let p02 = src[ym1 + xp1];
                let p10 = src[y0 + xm1];
                let p12 = src[y0 + xp1];
                let p20 = src[yp1 + xm1];
                let p21 = src[yp1 + x];
                let p22 = src[yp1 + xp1];

                let gx =
                    (3.0 * p02 + 10.0 * p12 + 3.0 * p22) - (3.0 * p00 + 10.0 * p10 + 3.0 * p20);
                let gy =
                    (3.0 * p20 + 10.0 * p21 + 3.0 * p22) - (3.0 * p00 + 10.0 * p01 + 3.0 * p02);

                let idx = y0 + x;
                let m = (gx * gx + gy * gy).sqrt();
                self.mag[idx] = m;
                let k = 2 * idx;
                if m >= min_mag && m > 0.0 {
                    let inv = 1.0 / m;
                    self.dir[k] = gx * inv;
                    self.dir[k + 1] = gy * inv;
                } else {
                    self.dir[k] = 0.0;
                    self.dir[k + 1] = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DirectionField, SmoothKind};
    use crate::core::Image;

    /// Disc of radius `r` centred in a `size × size` image, bright on dark.
    fn disc(size: usize, r: f32) -> Image<f32> {
        let c = 0.5 * size as f32;
        let data = (0..size * size)
            .map(|i| {
                let x = (i % size) as f32 - c;
                let y = (i / size) as f32 - c;
                if x * x + y * y <= r * r { 255.0 } else { 0.0 }
            })
            .collect();
        Image::from_vec(size, size, data).unwrap()
    }

    #[test]
    fn step_edge_direction_points_dark_to_bright() {
        let mut data = vec![0.0f32; 24 * 24];
        for y in 0..24 {
            for x in 12..24 {
                data[y * 24 + x] = 100.0;
            }
        }
        let img = Image::from_vec(24, 24, data).unwrap();
        let mut f = DirectionField::new();
        f.build_f32(&img.as_view(), SmoothKind::None, 1.0);

        // The Scharr response straddles the step at x = 11 and x = 12.
        for x in [11usize, 12] {
            let (nx, ny) = f.dir_at(x, 12);
            assert!(nx > 0.999, "x={x} nx={nx}");
            assert!(ny.abs() < 1e-5, "x={x} ny={ny}");
        }
        // Flat regions are gated to exactly zero on both sides.
        assert_eq!(f.dir_at(3, 12), (0.0, 0.0));
        assert_eq!(f.dir_at(20, 12), (0.0, 0.0));
    }

    #[test]
    fn directions_are_unit_length_wherever_they_are_not_gated() {
        let img = disc(64, 20.0);
        let mut f = DirectionField::new();
        f.build_f32(&img.as_view(), SmoothKind::Binomial3, 20.0);

        let mut nonzero = 0usize;
        for i in 0..64 * 64 {
            let (nx, ny) = (f.dir()[2 * i], f.dir()[2 * i + 1]);
            let n = (nx * nx + ny * ny).sqrt();
            if n > 0.0 {
                nonzero += 1;
                assert!((n - 1.0).abs() < 1e-5, "|n| = {n} at {i}");
            }
        }
        // A radius-20 circle has a circumference of ~126 px; the smoothed
        // Scharr response spreads it over a few pixels either side.
        assert!(nonzero > 200, "only {nonzero} gated-in pixels");
    }

    #[test]
    fn disc_directions_point_at_the_centre() {
        // On a bright disc the intensity increases inwards, so the unit
        // gradient at radius r points towards the centre.
        let img = disc(64, 20.0);
        let mut f = DirectionField::new();
        f.build_f32(&img.as_view(), SmoothKind::Binomial3, 50.0);

        let c = 32.0f32;
        let mut checked = 0usize;
        for y in 0..64 {
            for x in 0..64 {
                let (nx, ny) = f.dir_at(x, y);
                if nx == 0.0 && ny == 0.0 {
                    continue;
                }
                let (dx, dy) = (x as f32 - c, y as f32 - c);
                let r = (dx * dx + dy * dy).sqrt();
                if !(18.0..=22.0).contains(&r) {
                    continue;
                }
                // Inward radial direction is (-dx, -dy) / r.
                let dot = nx * (-dx / r) + ny * (-dy / r);
                assert!(dot > 0.85, "dot={dot} at ({x},{y})");
                checked += 1;
            }
        }
        assert!(checked > 100, "only {checked} rim pixels checked");
    }

    #[test]
    fn magnitude_gate_is_the_only_thing_that_zeroes_a_direction() {
        let img = disc(48, 12.0);
        let mut lo = DirectionField::new();
        let mut hi = DirectionField::new();
        lo.build_f32(&img.as_view(), SmoothKind::None, 0.0);
        hi.build_f32(&img.as_view(), SmoothKind::None, 500.0);

        for i in 0..48 * 48 {
            let gated = hi.dir()[2 * i] == 0.0 && hi.dir()[2 * i + 1] == 0.0;
            let weak = lo.mag()[i] < 500.0;
            assert_eq!(gated, weak || lo.mag()[i] == 0.0, "mismatch at {i}");
            // The magnitude plane itself is never gated.
            assert_eq!(lo.mag()[i], hi.mag()[i]);
        }
    }

    #[test]
    fn matches_a_naive_scharr_reference() {
        let img = disc(40, 13.0);
        let mut f = DirectionField::new();
        f.build_f32(&img.as_view(), SmoothKind::None, 0.0);

        let src = img.data();
        let (w, h) = (40usize, 40usize);
        for y in 0..h {
            for x in 0..w {
                let at = |xx: usize, yy: usize| src[yy * w + xx];
                let (xm, xp) = (x.saturating_sub(1), (x + 1).min(w - 1));
                let (ym, yp) = (y.saturating_sub(1), (y + 1).min(h - 1));
                let gx = (3.0 * at(xp, ym) + 10.0 * at(xp, y) + 3.0 * at(xp, yp))
                    - (3.0 * at(xm, ym) + 10.0 * at(xm, y) + 3.0 * at(xm, yp));
                let gy = (3.0 * at(xm, yp) + 10.0 * at(x, yp) + 3.0 * at(xp, yp))
                    - (3.0 * at(xm, ym) + 10.0 * at(x, ym) + 3.0 * at(xp, ym));
                let m = (gx * gx + gy * gy).sqrt();
                assert!((f.mag_at(x, y) - m).abs() < 1e-3, "mag at ({x},{y})");
                if m > 0.0 {
                    let (nx, ny) = f.dir_at(x, y);
                    assert!((nx - gx / m).abs() < 1e-5, "nx at ({x},{y})");
                    assert!((ny - gy / m).abs() < 1e-5, "ny at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn sample_mag_interpolates_and_refuses_the_border() {
        let mut data = vec![0.0f32; 8 * 8];
        // A single ramp row so the interpolation result is predictable.
        for x in 0..8 {
            data[3 * 8 + x] = x as f32;
        }
        let img = Image::from_vec(8, 8, data).unwrap();
        let mut f = DirectionField::new();
        f.build_f32(&img.as_view(), SmoothKind::None, 0.0);

        let a = f.mag_at(3, 3);
        let b = f.mag_at(4, 3);
        let mid = f.sample_mag(3.5, 3.0);
        assert!((mid - 0.5 * (a + b)).abs() < 1e-3, "{mid} vs {a}/{b}");

        // Outside, and inside-but-without-a-full-2x2-neighbourhood, give 0.
        assert_eq!(f.sample_mag(-0.5, 3.0), 0.0);
        assert_eq!(f.sample_mag(7.0, 3.0), 0.0);
        assert_eq!(f.sample_mag(f32::NAN, 3.0), 0.0);
    }

    #[test]
    fn rebuilding_at_a_new_size_reallocates() {
        let mut f = DirectionField::new();
        f.build_f32(&disc(32, 10.0).as_view(), SmoothKind::None, 1.0);
        assert_eq!((f.width(), f.height()), (32, 32));
        assert_eq!(f.dir().len(), 2 * 32 * 32);

        f.build_f32(&disc(16, 5.0).as_view(), SmoothKind::None, 1.0);
        assert_eq!((f.width(), f.height()), (16, 16));
        assert_eq!(f.dir().len(), 2 * 16 * 16);
        assert_eq!(f.mag().len(), 16 * 16);
    }
}
