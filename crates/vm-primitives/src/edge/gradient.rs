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

use crate::core::{Image, ImageView, Pixel};

use super::edge2d::SmoothKind;

/// Dense per-pixel unit gradient direction with a magnitude gate.
///
/// Build with [`DirectionField::build`] for any pixel type, or
/// [`build_image_f32`](Self::build_image_f32) for a pyramid level; the internal
/// buffers are reused across calls, so a field kept in a long-lived struct
/// allocates only when the image dimensions change.
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
/// field.build(&img.as_view(), SmoothKind::None, 10.0);
///
/// // On the step the direction points along +x (dark to bright)...
/// let (nx, ny) = field.dir_at(15, 8);
/// assert!(nx > 0.99 && ny.abs() < 1e-6);
/// // ...and in the flat interior it is gated to exactly zero.
/// assert_eq!(field.dir_at(4, 8), (0.0, 0.0));
/// ```
#[derive(Debug, Clone)]
pub struct DirectionField {
    dir: Vec<f32>,
    mag: Vec<f32>,
    tmp: Vec<f32>,
    scratch: Vec<f32>,
    width: usize,
    height: usize,
    min_mag: f32,
    // ── lazy tiled mode ────────────────────────────────────────────────────
    /// Smoothing recorded when tiled mode was entered, applied per tile.
    smooth: SmoothKind,
    /// Per-tile build stamp; a tile is current iff `tile_stamp[i] == generation`.
    tile_stamp: Vec<u32>,
    /// Tiles built in the current generation (for O(built) cleanup).
    built_tiles: Vec<u32>,
    tiles_x: usize,
    tiles_y: usize,
    generation: u32,
}

impl Default for DirectionField {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectionField {
    /// Create an empty field. Buffers are allocated on the first `build_*` call.
    pub fn new() -> Self {
        Self {
            dir: Vec::new(),
            mag: Vec::new(),
            tmp: Vec::new(),
            scratch: Vec::new(),
            width: 0,
            height: 0,
            min_mag: 0.0,
            smooth: SmoothKind::None,
            tile_stamp: Vec::new(),
            built_tiles: Vec::new(),
            tiles_x: 0,
            tiles_y: 0,
            generation: 0,
        }
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

    /// Build the field from a grayscale image of any [`Pixel`] type.
    ///
    /// `min_mag` is expressed in **Scharr response units on the input pixel
    /// scale**: a clean black/white step in `u8` gives `|∇I| ≈ 8·255 ≈ 2000`, so
    /// a gate of 10 sits well above 8-bit sensor noise while keeping faint but
    /// real edges. Re-tune it for `u16` and `f32` inputs, whose pixel scales
    /// differ by orders of magnitude.
    pub fn build<P: Pixel>(&mut self, img: &ImageView<'_, P>, smooth: SmoothKind, min_mag: f32) {
        self.ensure(img.width(), img.height());
        let width = self.width;
        for y in 0..img.height() {
            let dst = &mut self.tmp[y * width..(y + 1) * width];
            for (d, s) in dst.iter_mut().zip(img.row(y)) {
                *d = s.to_f32();
            }
        }
        self.finish(smooth, min_mag);
    }

    /// Build from an owned image — convenience for pyramid levels.
    pub fn build_image_f32(&mut self, img: &Image<f32>, smooth: SmoothKind, min_mag: f32) {
        self.build(&img.as_view(), smooth, min_mag);
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

    // ── lazy tiled mode ─────────────────────────────────────────────────────
    //
    // The full builds above touch every pixel; a coarse-to-fine matcher reads
    // the fine pyramid levels only in small windows around its candidates. In
    // tiled mode the field is built on demand, one TILE×TILE block at a time,
    // and every built pixel is **bit-identical** to what the full build would
    // have produced: the per-pixel smoothing and Scharr expressions below are
    // the same expression trees as `smooth_binomial3` / `scharr_normalised`,
    // evaluated with image-border clamping in absolute coordinates.
    //
    // Safety net: entering tiled mode zeroes exactly the tiles the previous
    // generation built (O(built), not O(frame)), so a read of a rectangle the
    // caller forgot to ensure sees deterministic zeros — "no evidence", the
    // same contribution as an out-of-image point — never stale data from
    // another frame.

    /// Enter lazy tiled mode for `img`, returning the session that owns it.
    ///
    /// The returned [`TiledField`] borrows both the field and `img`, so the
    /// two cannot drift apart: every `ensure_rect` afterwards necessarily
    /// refers to the image tiled mode was entered with. That used to be a
    /// runtime `assert!` on a second `&Image` argument, and the ordering
    /// requirement — begin, then ensure, then read — was documentation only.
    ///
    /// The field's buffers are current only where
    /// [`TiledField::ensure_rect`] has been called with a covering rectangle;
    /// everywhere else `dir` reads as zero.
    #[must_use = "the session owns ensure_rect; dropping it immediately leaves the field all zeros"]
    pub fn begin_tiled_f32<'a>(
        &'a mut self,
        img: &'a Image<f32>,
        smooth: SmoothKind,
        min_mag: f32,
    ) -> TiledField<'a> {
        self.enter_tiled(img, smooth, min_mag);
        TiledField { field: self, img }
    }

    fn enter_tiled(&mut self, img: &Image<f32>, smooth: SmoothKind, min_mag: f32) {
        let (w, h) = (img.width(), img.height());
        let dims_changed = self.width != w || self.height != h;
        self.ensure(w, h);
        self.smooth = smooth;
        self.min_mag = min_mag;

        let tx = w.div_ceil(Self::TILE);
        let ty = h.div_ceil(Self::TILE);
        if dims_changed || self.tiles_x != tx || self.tiles_y != ty {
            self.tiles_x = tx;
            self.tiles_y = ty;
            self.tile_stamp = vec![0; tx * ty];
            self.built_tiles.clear();
            // `ensure` zero-fills fresh buffers, so nothing stale can survive
            // a reallocation.
        } else {
            // Zero only what the last generation actually built.
            let built = core::mem::take(&mut self.built_tiles);
            for &t in &built {
                self.zero_tile(t as usize);
            }
            self.built_tiles = built;
            self.built_tiles.clear();
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // A wrapped counter would make stale stamps look current.
            self.tile_stamp.fill(0);
            self.generation = 1;
        }
    }

    fn ensure_rect(&mut self, img: &Image<f32>, x0: i32, y0: i32, x1: i32, y1: i32) {
        debug_assert!(
            img.width() == self.width && img.height() == self.height,
            "tiled session image does not match the field"
        );
        if self.width == 0 || self.height == 0 {
            return;
        }
        let x0 = x0.clamp(0, self.width as i32 - 1) as usize / Self::TILE;
        let y0 = y0.clamp(0, self.height as i32 - 1) as usize / Self::TILE;
        let x1 = (x1 - 1).clamp(0, self.width as i32 - 1) as usize / Self::TILE;
        let y1 = (y1 - 1).clamp(0, self.height as i32 - 1) as usize / Self::TILE;

        for ty in y0..=y1 {
            for tx in x0..=x1 {
                let t = ty * self.tiles_x + tx;
                if self.tile_stamp[t] != self.generation {
                    self.build_tile(img, tx, ty);
                    self.tile_stamp[t] = self.generation;
                    self.built_tiles.push(t as u32);
                }
            }
        }
    }

    const TILE: usize = 64;

    fn zero_tile(&mut self, t: usize) {
        let (tx, ty) = (t % self.tiles_x, t / self.tiles_x);
        let px0 = tx * Self::TILE;
        let py0 = ty * Self::TILE;
        let px1 = (px0 + Self::TILE).min(self.width);
        let py1 = (py0 + Self::TILE).min(self.height);
        for y in py0..py1 {
            let row = y * self.width;
            self.dir[2 * (row + px0)..2 * (row + px1)].fill(0.0);
            self.mag[row + px0..row + px1].fill(0.0);
        }
    }

    /// Smoothed source value at absolute pixel `(x, y)`, clamped at the image
    /// border — the same expression tree as `smooth_binomial3`, so the result
    /// is bit-identical to the full build.
    #[inline]
    fn smoothed_at(img: &Image<f32>, x: usize, y: usize, w: usize, h: usize) -> f32 {
        let data = img.data();
        let hs = |yy: usize| {
            let row = yy * w;
            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(w - 1);
            0.25 * (data[row + xm1] + 2.0 * data[row + x] + data[row + xp1])
        };
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(h - 1);
        0.25 * (hs(ym1) + 2.0 * hs(y) + hs(yp1))
    }

    fn build_tile(&mut self, img: &Image<f32>, tx: usize, ty: usize) {
        let (w, h) = (self.width, self.height);
        let px0 = tx * Self::TILE;
        let py0 = ty * Self::TILE;
        let px1 = (px0 + Self::TILE).min(w);
        let py1 = (py0 + Self::TILE).min(h);

        // Smoothed values for the tile plus a 1-pixel apron, in absolute
        // clamped coordinates. `sw × sh` is at most (TILE+2)².
        let ax0 = px0.saturating_sub(1);
        let ay0 = py0.saturating_sub(1);
        let ax1 = (px1 + 1).min(w);
        let ay1 = (py1 + 1).min(h);
        let (sw, sh) = (ax1 - ax0, ay1 - ay0);

        self.scratch.resize(self.scratch.len().max(sw * sh), 0.0);
        let smoothed = &mut self.scratch[..sw * sh];
        match self.smooth {
            SmoothKind::Binomial3 => {
                for y in ay0..ay1 {
                    for x in ax0..ax1 {
                        smoothed[(y - ay0) * sw + (x - ax0)] = Self::smoothed_at(img, x, y, w, h);
                    }
                }
            }
            SmoothKind::None => {
                let data = img.data();
                for y in ay0..ay1 {
                    let src = &data[y * w + ax0..y * w + ax1];
                    smoothed[(y - ay0) * sw..(y - ay0) * sw + sw].copy_from_slice(src);
                }
            }
        }

        let min_mag = self.min_mag;
        for y in py0..py1 {
            // Clamped absolute rows, translated into the apron buffer.
            let ym1 = y.saturating_sub(1).max(ay0) - ay0;
            let yc = y - ay0;
            let yp1 = ((y + 1).min(h - 1)).min(ay1 - 1) - ay0;
            let (rm, rc, rp) = (ym1 * sw, yc * sw, yp1 * sw);
            for x in px0..px1 {
                let xm1 = x.saturating_sub(1).max(ax0) - ax0;
                let xc = x - ax0;
                let xp1 = ((x + 1).min(w - 1)).min(ax1 - 1) - ax0;

                let p00 = smoothed[rm + xm1];
                let p01 = smoothed[rm + xc];
                let p02 = smoothed[rm + xp1];
                let p10 = smoothed[rc + xm1];
                let p12 = smoothed[rc + xp1];
                let p20 = smoothed[rp + xm1];
                let p21 = smoothed[rp + xc];
                let p22 = smoothed[rp + xp1];

                let gx =
                    (3.0 * p02 + 10.0 * p12 + 3.0 * p22) - (3.0 * p00 + 10.0 * p10 + 3.0 * p20);
                let gy =
                    (3.0 * p20 + 10.0 * p21 + 3.0 * p22) - (3.0 * p00 + 10.0 * p01 + 3.0 * p02);

                let idx = y * w + x;
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

/// A lazy-tiling session over a [`DirectionField`] and one image.
///
/// Returned by [`DirectionField::begin_tiled_f32`]. It holds both halves of
/// the protocol together: the field whose tiles are being filled and the image
/// they are filled from. Nothing else can call `ensure_rect`, and `ensure_rect`
/// cannot be handed the wrong image, so the two ways this protocol used to be
/// misusable are gone — no runtime dimension assert, and no way to read a
/// field that was never put into tiled mode at all.
///
/// The session derefs to the field, so scoring code reads through it unchanged.
/// Pixels in tiles that were never ensured read as zero, which the score treats
/// as "no evidence" — the same contribution as an out-of-image point.
pub struct TiledField<'a> {
    field: &'a mut DirectionField,
    img: &'a Image<f32>,
}

impl TiledField<'_> {
    /// Build every tile intersecting the half-open pixel rectangle
    /// `[x0, x1) × [y0, y1)`.
    ///
    /// Clamped to the image; tiles already built in this session are skipped,
    /// so overlapping requests cost only the intersection test.
    #[inline]
    pub fn ensure_rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) {
        self.field.ensure_rect(self.img, x0, y0, x1, y1);
    }

    /// The field being filled, for reading.
    ///
    /// Same as dereferencing; named for call sites that need an explicit
    /// `&DirectionField`.
    #[inline]
    pub fn field(&self) -> &DirectionField {
        self.field
    }
}

impl core::ops::Deref for TiledField<'_> {
    type Target = DirectionField;

    #[inline]
    fn deref(&self) -> &DirectionField {
        self.field
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
        f.build(&img.as_view(), SmoothKind::None, 1.0);

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
        f.build(&img.as_view(), SmoothKind::Binomial3, 20.0);

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
        f.build(&img.as_view(), SmoothKind::Binomial3, 50.0);

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
        lo.build(&img.as_view(), SmoothKind::None, 0.0);
        hi.build(&img.as_view(), SmoothKind::None, 500.0);

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
        f.build(&img.as_view(), SmoothKind::None, 0.0);

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
        f.build(&img.as_view(), SmoothKind::None, 0.0);

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
        f.build(&disc(32, 10.0).as_view(), SmoothKind::None, 1.0);
        assert_eq!((f.width(), f.height()), (32, 32));
        assert_eq!(f.dir().len(), 2 * 32 * 32);

        f.build(&disc(16, 5.0).as_view(), SmoothKind::None, 1.0);
        assert_eq!((f.width(), f.height()), (16, 16));
        assert_eq!(f.dir().len(), 2 * 16 * 16);
        assert_eq!(f.mag().len(), 16 * 16);
    }

    /// Deterministic textured image exercising smooth borders and the gate.
    fn textured(w: usize, h: usize) -> Image<f32> {
        let mut data = vec![0.0f32; w * h];
        let mut state = 0xdead_beef_cafe_f00du64;
        for y in 0..h {
            for x in 0..w {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = ((state >> 33) & 0x3f) as f32;
                let step = if x >= w / 2 { 150.0 } else { 0.0 };
                data[y * w + x] = 40.0 + step + noise + 20.0 * (0.3 * y as f32).sin();
            }
        }
        Image::from_vec(w, h, data).unwrap()
    }

    #[test]
    fn tiled_build_is_bit_identical_to_the_full_build() {
        // Odd dimensions so the last tile row/column is partial, and both
        // smoothing modes, since each has its own tile path.
        for smooth in [SmoothKind::None, SmoothKind::Binomial3] {
            let img = textured(157, 101);

            let mut full = DirectionField::new();
            full.build_image_f32(&img, smooth, 25.0);

            let mut tiled = DirectionField::new();
            let mut session = tiled.begin_tiled_f32(&img, smooth, 25.0);
            session.ensure_rect(0, 0, img.width() as i32, img.height() as i32);

            assert_eq!(full.dir(), tiled.dir(), "dir differs ({smooth:?})");
            assert_eq!(full.mag(), tiled.mag(), "mag differs ({smooth:?})");
        }
    }

    #[test]
    fn unensured_tiles_read_as_deterministic_zero() {
        let img = textured(157, 101);
        let mut tiled = DirectionField::new();

        // Build everything once (generation 1)...
        let mut session = tiled.begin_tiled_f32(&img, SmoothKind::Binomial3, 25.0);
        session.ensure_rect(0, 0, 157, 101);
        assert!(tiled.dir().iter().any(|&v| v != 0.0));

        // ...then begin a new generation and ensure only a corner window.
        let mut session = tiled.begin_tiled_f32(&img, SmoothKind::Binomial3, 25.0);
        session.ensure_rect(0, 0, 40, 40);

        // Inside the window: identical to a full build. Outside: exact zeros,
        // never a stale value from the previous generation.
        let mut full = DirectionField::new();
        full.build_image_f32(&img, SmoothKind::Binomial3, 25.0);
        let w = img.width();
        for y in 0..img.height() {
            for x in 0..w {
                let k = 2 * (y * w + x);
                if x < 64 && y < 64 {
                    assert_eq!(tiled.dir()[k], full.dir()[k]);
                    assert_eq!(tiled.dir()[k + 1], full.dir()[k + 1]);
                } else {
                    assert_eq!(tiled.dir()[k], 0.0, "stale nx at ({x},{y})");
                    assert_eq!(tiled.dir()[k + 1], 0.0, "stale ny at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn tiled_mode_reuses_cleanly_across_generations_and_sizes() {
        let a = textured(157, 101);
        let b = textured(96, 64);
        let mut tiled = DirectionField::new();
        let mut full = DirectionField::new();

        for img in [&a, &b, &a] {
            let mut session = tiled.begin_tiled_f32(img, SmoothKind::Binomial3, 25.0);
            session.ensure_rect(0, 0, img.width() as i32, img.height() as i32);
            full.build_image_f32(img, SmoothKind::Binomial3, 25.0);
            assert_eq!(full.dir(), tiled.dir());
            assert_eq!(full.mag(), tiled.mag());
        }
    }
}
