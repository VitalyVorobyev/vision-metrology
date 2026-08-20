//! [`Map`]: a precomputed `dst → src` coordinate table.

use core::ops::Range;

use vm_primitives::{Affine2f, Point2f, Projective2f};

/// A precomputed `dst → src` coordinate map, built once and applied to as
/// many frames as needed.
///
/// See the [module docs](super) for the `dst → src` convention, the
/// pixel-center rule, and why validity is reported per pixel rather than
/// assumed. `Map` itself is geometry only — call [`Map::apply`] or
/// [`Map::apply_with_mask`] to actually resample an image through it.
#[derive(Debug, Clone, PartialEq)]
pub struct Map {
    pub(super) width: usize,
    pub(super) height: usize,
    /// Source coordinate for destination pixel `(x, y)`, at index
    /// `y * width + x` — the same row-major layout every raster type in this
    /// workspace uses.
    pub(super) coords: Vec<Point2f>,
}

impl Map {
    /// Destination width in pixels.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Destination height in pixels.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Build a map from an arbitrary per-pixel `dst → src` function.
    ///
    /// `f(x, y)` is called once per destination pixel with `x`/`y` equal to
    /// that pixel's **center** coordinate (`x as f32`, `y as f32` — invariant
    /// 1), and must return the source coordinate to sample from. This is the
    /// general escape hatch: [`Map::affine`], [`Map::projective`] and
    /// [`Map::polar`] are all expressed in terms of it.
    ///
    /// ```
    /// use vision_metrology::warp::Map;
    ///
    /// // Same as a +1 px horizontal translation map.
    /// let map = Map::from_fn(4, 4, |x, y| (x + 1.0, y));
    /// assert_eq!(map.width(), 4);
    /// ```
    pub fn from_fn(w: usize, h: usize, f: impl Fn(f32, f32) -> (f32, f32)) -> Self {
        let mut coords = Vec::with_capacity(w.saturating_mul(h));
        for y in 0..h {
            let yf = y as f32;
            for x in 0..w {
                let (sx, sy) = f(x as f32, yf);
                coords.push(Point2f::new(sx, sy));
            }
        }
        Self {
            width: w,
            height: h,
            coords,
        }
    }

    /// Build a map from an affine `dst → src` transform.
    ///
    /// `m` maps a **destination** pixel center to the **source** coordinate
    /// it should sample — see the [module docs](super) on inverting a
    /// forward (source-into-destination) transform first.
    pub fn affine(w: usize, h: usize, m: &Affine2f) -> Self {
        let m = *m;
        Self::from_fn(w, h, move |x, y| {
            let p = m * Point2f::new(x, y);
            (p.x, p.y)
        })
    }

    /// Build a map from a projective `dst → src` transform (a homography).
    ///
    /// Same `dst → src` direction as [`Map::affine`]. An affine transform
    /// embedded as a homography (bottom row `[0, 0, 1]`) produces the
    /// identical map — `projective` is a strict superset.
    pub fn projective(w: usize, h: usize, h_: &Projective2f) -> Self {
        let h_ = *h_;
        Self::from_fn(w, h, move |x, y| {
            let p = h_ * Point2f::new(x, y);
            (p.x, p.y)
        })
    }

    /// Build a polar-unwrap map: destination **x** sweeps `phi`, destination
    /// **y** sweeps `r`, both linearly and at bin centers.
    ///
    /// Destination pixel `(x, y)` (out of a `w × h` destination) samples the
    /// source at:
    ///
    /// ```text
    /// u = (x + 0.5) / w,  v = (y + 0.5) / h
    /// angle  = phi.start + u * (phi.end - phi.start)
    /// radius = r.start   + v * (r.end   - r.start)
    /// src    = center + radius * (cos(angle), sin(angle))
    /// ```
    ///
    /// so neither `phi.end` nor `r.end` is ever sampled exactly — the last
    /// row/column sits half a bin short of it, the same convention a
    /// histogram uses for its bin centers. A `phi` range spanning a full
    /// turn (`2π`) therefore does not duplicate the seam column.
    ///
    /// This is the round-part unwrap: sweep a full circle and every column
    /// of the destination is one angular slice of the source, at every
    /// radius in `r`. Directly useful on canend-style round features.
    pub fn polar(center: Point2f, r: Range<f32>, phi: Range<f32>, w: usize, h: usize) -> Self {
        let r_span = r.end - r.start;
        let phi_span = phi.end - phi.start;
        let wf = (w.max(1)) as f32;
        let hf = (h.max(1)) as f32;
        Self::from_fn(w, h, move |x, y| {
            let angle = phi.start + (x + 0.5) / wf * phi_span;
            let radius = r.start + (y + 0.5) / hf * r_span;
            let (sn, cs) = angle.sin_cos();
            (center.x + radius * cs, center.y + radius * sn)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Map;
    use vm_primitives::{Affine2f, Point2f};

    #[test]
    fn from_fn_stores_dimensions_and_row_major_coords() {
        let map = Map::from_fn(3, 2, |x, y| (x + 10.0, y + 20.0));
        assert_eq!(map.width(), 3);
        assert_eq!(map.height(), 2);
        assert_eq!(map.coords.len(), 6);
        // Row-major: index 1 is (x=1, y=0).
        assert_eq!(map.coords[1], Point2f::new(11.0, 20.0));
        // Index 4 is (x=1, y=1).
        assert_eq!(map.coords[4], Point2f::new(11.0, 21.0));
    }

    #[test]
    fn affine_identity_samples_every_pixel_at_itself() {
        let map = Map::affine(4, 3, &Affine2f::identity());
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(map.coords[y * 4 + x], Point2f::new(x as f32, y as f32));
            }
        }
    }

    #[test]
    fn affine_translation_shifts_every_source_coordinate() {
        let t = nalgebra::Translation2::new(2.5f32, -1.0);
        let m: Affine2f = nalgebra::convert(t);
        let map = Map::affine(2, 2, &m);
        assert_eq!(map.coords[0], Point2f::new(2.5, -1.0));
        assert_eq!(map.coords[1], Point2f::new(3.5, -1.0));
    }

    #[test]
    fn from_fn_matches_affine_for_the_same_map() {
        let t = nalgebra::Translation2::new(3.0f32, 4.0);
        let m: Affine2f = nalgebra::convert(t);
        let via_affine = Map::affine(5, 5, &m);
        let via_from_fn = Map::from_fn(5, 5, |x, y| (x + 3.0, y + 4.0));
        assert_eq!(via_affine, via_from_fn);
    }

    #[test]
    fn polar_seam_and_bin_centers() {
        // A full turn starting at angle 0, radius 10..20, 4x1 destination.
        let map = Map::polar(
            Point2f::new(0.0, 0.0),
            10.0..20.0,
            0.0..core::f32::consts::TAU,
            4,
            1,
        );
        // Column 0 sits at bin-center angle = 0.5/4 turn = pi/4, radius = 15
        // (single row, bin-centered at 0.5/1 = midpoint of [10, 20)).
        let want_angle = 0.5 / 4.0 * core::f32::consts::TAU;
        let want = Point2f::new(15.0 * want_angle.cos(), 15.0 * want_angle.sin());
        let got = map.coords[0];
        assert!((got.x - want.x).abs() < 1e-5);
        assert!((got.y - want.y).abs() < 1e-5);

        // The seam is not duplicated: column 3's bin center sits at angle
        // 7pi/4 (just *before* a full turn), distinctly apart from column
        // 0's pi/4 (just after zero) — a duplicated-seam bug would instead
        // put column 3 at (or very near) angle 0.
        let last = map.coords[3];
        let dist = ((got.x - last.x).powi(2) + (got.y - last.y).powi(2)).sqrt();
        assert!(dist > 1.0, "seam column duplicated: {got:?} vs {last:?}");
    }
}
