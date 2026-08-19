use core::{f32::consts::PI, ops::Neg};

/// A 2-D point in floating-point pixel coordinates.
///
/// An alias for [`nalgebra::Point2<f32>`], so points move between this crate
/// and any other nalgebra-based library without conversion. Construct with
/// `Point2f::new(x, y)`; read or write `.x` / `.y` as usual.
///
/// By convention, integer coordinate `i` refers to the **center** of pixel `i`.
pub type Point2f = nalgebra::Point2<f32>;

/// A 2-D displacement vector (difference of two [`Point2f`] values).
///
/// An alias for [`nalgebra::Vector2<f32>`]. `Point2f - Point2f` yields one, and
/// `Point2f + Vec2f` moves a point.
///
/// See [`Vec2fExt`] for the two operations nalgebra does not provide under
/// these names (`perp`, `cross`) and for
/// [`normalized_or_zero`](Vec2fExt::normalized_or_zero), which callers should
/// prefer over `normalize` whenever the input may be zero.
pub type Vec2f = nalgebra::Vector2<f32>;

/// 2-D vector operations this crate relies on beyond nalgebra's inherent set.
pub trait Vec2fExt: Sized {
    /// Rotate by +90° (counter-clockwise in a y-down image frame): `(x, y) → (−y, x)`.
    ///
    /// Turns an edge tangent into a normal, and builds the rotational column of
    /// a similarity Jacobian.
    fn perp(self) -> Self;

    /// 2-D cross product `self × rhs = self.x·rhs.y − self.y·rhs.x`.
    ///
    /// The signed area of the parallelogram spanned by the two vectors; equal
    /// to `self.perp().dot(&rhs)`.
    fn cross(self, rhs: Self) -> f32;

    /// Unit vector in the same direction, or the **zero vector** when `self` is
    /// zero.
    ///
    /// nalgebra's `normalize` divides unconditionally and yields `NaN` for a
    /// zero input. Several call sites here normalize a possibly-degenerate
    /// gradient or tangent and then reject it with `t.norm() < 0.5` — a test
    /// that `NaN` silently *passes*. Use this instead wherever the input can be
    /// zero.
    fn normalized_or_zero(self) -> Self;
}

impl Vec2fExt for Vec2f {
    #[inline]
    fn perp(self) -> Self {
        Vec2f::new(-self.y, self.x)
    }

    #[inline]
    fn cross(self, rhs: Self) -> f32 {
        self.x * rhs.y - self.y * rhs.x
    }

    #[inline]
    fn normalized_or_zero(self) -> Self {
        self.try_normalize(0.0).unwrap_or_else(Vec2f::zeros)
    }
}

/// A 2-D line represented by a point and a direction vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line2f {
    /// A point on the line.
    pub p: Point2f,
    /// Direction vector (not necessarily unit length).
    pub dir: Vec2f,
}

/// An ordered sequence of 2-D points forming a polyline.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Polyline2f {
    /// Ordered vertices of the polyline.
    pub points: Vec<Point2f>,
}

// ---------------------------------------------------------------------------
// Rect2f
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box in pixel coordinates.
///
/// `(x, y)` is the top-left corner; the box spans
/// `[x, x + width) × [y, y + height)`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rect2f {
    /// Left edge (column) of the rectangle.
    pub x: f32,
    /// Top edge (row) of the rectangle.
    pub y: f32,
    /// Horizontal extent in pixels.
    pub width: f32,
    /// Vertical extent in pixels.
    pub height: f32,
}

impl Rect2f {
    /// Returns `true` if `p` is strictly inside the rectangle.
    #[inline]
    pub fn contains(self, p: Point2f) -> bool {
        p.x >= self.x && p.x < self.x + self.width && p.y >= self.y && p.y < self.y + self.height
    }

    /// Center of the bounding box.
    #[inline]
    pub fn center(self) -> Point2f {
        Point2f::new(self.x + self.width * 0.5, self.y + self.height * 0.5)
    }

    /// Area (`width × height`). Returns 0 for degenerate rectangles.
    #[inline]
    pub fn area(self) -> f32 {
        (self.width * self.height).max(0.0)
    }

    /// Expand uniformly by `margin` on each side.
    #[inline]
    pub fn expanded_by(self, margin: f32) -> Self {
        Self {
            x: self.x - margin,
            y: self.y - margin,
            width: self.width + 2.0 * margin,
            height: self.height + 2.0 * margin,
        }
    }
}

// ---------------------------------------------------------------------------
// Angle
// ---------------------------------------------------------------------------

/// Oriented angle in radians with canonical range `(-π, π]`.
///
/// Use [`Angle::new`] to wrap an arbitrary radian value and
/// [`Angle::diff`] to compute the shortest signed angular distance.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Angle(f32);

impl Angle {
    /// Wrap `radians` into `(-π, π]`.
    pub fn new(radians: f32) -> Self {
        Self(wrap_angle(radians))
    }

    /// Raw radian value in `(-π, π]`.
    #[inline]
    pub fn value(self) -> f32 {
        self.0
    }

    /// Shortest signed angular difference `self - other`, result in `(-π, π]`.
    pub fn diff(self, other: Self) -> Self {
        Self(wrap_angle(self.0 - other.0))
    }

    /// Absolute angular difference in `[0, π]`.
    pub fn abs_diff(self, other: Self) -> f32 {
        self.diff(other).0.abs()
    }
}

impl Neg for Angle {
    type Output = Self;
    fn neg(self) -> Self {
        Self(wrap_angle(-self.0))
    }
}

/// Wrap an angle in radians to `(-π, π]`.
#[inline]
pub fn wrap_angle(a: f32) -> f32 {
    let mut v = a % (2.0 * PI);
    if v <= -PI {
        v += 2.0 * PI;
    } else if v > PI {
        v -= 2.0 * PI;
    }
    v
}

// ---------------------------------------------------------------------------
// nalgebra type aliases
// ---------------------------------------------------------------------------

/// Rigid 2-D transform (rotation + translation, scale = 1).
///
/// Backed by [`nalgebra::Isometry2<f32>`]. Use `nalgebra::Isometry2::new` or
/// `nalgebra::Isometry2::rotation` to construct instances.
pub type Isometry2f = nalgebra::Isometry2<f32>;

/// Similarity 2-D transform (rotation + uniform scale + translation).
///
/// Backed by [`nalgebra::Similarity2<f32>`].
pub type Similarity2f = nalgebra::Similarity2<f32>;

/// General affine 2-D transform.
///
/// Backed by [`nalgebra::Affine2<f32>`].
pub type Affine2f = nalgebra::Affine2<f32>;

/// General projective 2-D transform (homography).
///
/// Backed by [`nalgebra::Projective2<f32>`].
pub type Projective2f = nalgebra::Projective2<f32>;

// ---------------------------------------------------------------------------
// Transform helpers
// ---------------------------------------------------------------------------

/// Apply a similarity transform to a point: `s·R·p + t`.
///
/// # Note
/// This reads the rotation out of `sim` on every call. It is meant for the
/// handful of call sites outside inner loops; a scoring loop that transforms
/// thousands of points per pose must hoist `(cos, sin, tx, ty)` into scalars
/// itself instead of calling this per point.
#[inline]
pub fn transform_point(sim: &Similarity2f, p: Point2f) -> Point2f {
    sim * p
}

/// Apply the linear part of a similarity transform to a displacement: `s·R·v`.
///
/// Translation is deliberately not applied — a displacement has no origin.
#[inline]
pub fn transform_vec(sim: &Similarity2f, v: Vec2f) -> Vec2f {
    sim * v
}

/// Build a [`Similarity2f`] from translation, rotation angle (radians) and
/// uniform scale.
///
/// The resulting map is `p ↦ scale·R(angle)·p + translation`.
#[inline]
pub fn similarity_from_parts(translation: Vec2f, angle: f32, scale: f32) -> Similarity2f {
    Similarity2f::new(translation, angle, scale)
}

/// Decompose a [`Similarity2f`] into `(translation, angle, scale)`.
///
/// Inverse of [`similarity_from_parts`]; the angle is in `(-π, π]`.
#[inline]
pub fn similarity_parts(sim: &Similarity2f) -> (Vec2f, f32, f32) {
    (
        sim.isometry.translation.vector,
        sim.isometry.rotation.angle(),
        sim.scaling(),
    )
}

/// Vertex offset of the parabola through `(-1, ym)`, `(0, y0)`, `(1, yp)`.
///
/// Returns the offset from the centre sample to the extremum, or `None` when
/// the three samples are collinear (or produce a non-finite result), which is
/// the degenerate case where no vertex exists.
///
/// The offset is **not** clamped: `|offset| > 1` means the true extremum lies
/// outside the sampled bracket, which callers usually want to reject rather
/// than saturate.
///
/// ```
/// use vm_primitives::parabolic_peak_offset;
///
/// // Symmetric peak: vertex sits exactly on the centre sample.
/// assert_eq!(parabolic_peak_offset(1.0, 2.0, 1.0), Some(0.0));
/// // Skewed towards the right sample.
/// let t = parabolic_peak_offset(0.0, 2.0, 1.0).unwrap();
/// assert!(t > 0.0 && t < 0.5);
/// // Collinear samples have no vertex.
/// assert_eq!(parabolic_peak_offset(0.0, 1.0, 2.0), None);
/// ```
#[inline]
pub fn parabolic_peak_offset(ym: f32, y0: f32, yp: f32) -> Option<f32> {
    let denom = ym - 2.0 * y0 + yp;
    if denom.abs() <= 1e-12 {
        return None;
    }
    let t = 0.5 * (ym - yp) / denom;
    t.is_finite().then_some(t)
}

#[cfg(test)]
mod tests {
    use super::{Angle, Point2f, Rect2f, Vec2f, Vec2fExt};

    #[test]
    fn vec_ops_and_normalize() {
        let a = Vec2f::new(3.0, 4.0);
        let b = Vec2f::new(1.0, -2.0);

        assert_eq!(a + b, Vec2f::new(4.0, 2.0));
        assert_eq!(a - b, Vec2f::new(2.0, 6.0));
        assert!((a.dot(&b) + 5.0).abs() < 1e-6);
        assert!((a.norm() - 5.0).abs() < 1e-6);

        let n = a.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-6);

        // nalgebra's `normalize` divides unconditionally, so a zero vector
        // yields NaN. `normalized_or_zero` is the guard-friendly form the rest
        // of the workspace uses; the difference matters because `NaN < 0.5`
        // is false, so a bare `normalize` would sail through degeneracy checks.
        assert!(Vec2f::zeros().normalize().x.is_nan());
        assert_eq!(Vec2f::zeros().normalized_or_zero(), Vec2f::zeros());
        assert!((a.normalized_or_zero().norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn point_vec_ops() {
        let p = Point2f::new(2.0, 3.0);
        let v = Vec2f::new(0.5, -1.0);

        assert_eq!(p + v, Point2f::new(2.5, 2.0));
        assert_eq!(p - v, Point2f::new(1.5, 4.0));
        assert_eq!(p - Point2f::new(1.0, 1.0), Vec2f::new(1.0, 2.0));
    }

    #[test]
    fn rect2f_ops() {
        let r = Rect2f {
            x: 1.0,
            y: 2.0,
            width: 4.0,
            height: 3.0,
        };
        assert!((r.area() - 12.0).abs() < 1e-6);
        assert_eq!(r.center(), Point2f::new(3.0, 3.5));
        assert!(r.contains(Point2f::new(2.0, 3.0)));
        assert!(!r.contains(Point2f::new(0.0, 3.0)));
        let e = r.expanded_by(1.0);
        assert_eq!(e.x, 0.0);
        assert_eq!(e.width, 6.0);
    }

    #[test]
    fn angle_wrap_and_diff() {
        use core::f32::consts::PI;
        let a = Angle::new(PI + 0.1);
        // Wraps to ≈ -π + 0.1.
        assert!((a.value() - (-PI + 0.1)).abs() < 1e-5);

        let x = Angle::new(0.3);
        let y = Angle::new(-0.1);
        assert!((x.diff(y).value() - 0.4).abs() < 1e-5);
        assert!((x.abs_diff(y) - 0.4).abs() < 1e-5);
    }

    #[test]
    fn isometry2f_apply() {
        use core::f32::consts::FRAC_PI_2;
        // 90° CCW rotation should map (1, 0) → (0, 1).
        let iso = nalgebra::Isometry2::rotation(FRAC_PI_2);
        let q = iso * Point2f::new(1.0, 0.0);
        assert!((q.x).abs() < 1e-6);
        assert!((q.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity2f_scale_and_rotate() {
        use nalgebra::{Similarity2, Vector2};
        // Scale ×2, no rotation, translate by (1, 0).
        let sim = Similarity2::new(Vector2::new(1.0, 0.0), 0.0, 2.0);
        let q = sim * Point2f::new(1.0, 1.0);
        assert!((q.x - 3.0).abs() < 1e-6); // 2*1 + 1
        assert!((q.y - 2.0).abs() < 1e-6); // 2*1
    }

    #[test]
    fn perp_and_cross_agree() {
        let a = Vec2f::new(3.0, 1.0);
        let b = Vec2f::new(-2.0, 4.0);
        // perp is a +90 degree rotation, so it preserves length and is orthogonal.
        assert!((a.perp().norm() - a.norm()).abs() < 1e-6);
        assert!(a.perp().dot(&a).abs() < 1e-6);
        assert_eq!(a.perp(), Vec2f::new(-1.0, 3.0));
        // cross(a, b) == perp(a) . b by definition.
        assert!((a.cross(b) - a.perp().dot(&b)).abs() < 1e-6);
        // Antisymmetry.
        assert!((a.cross(b) + b.cross(a)).abs() < 1e-6);
    }

    #[test]
    fn similarity_parts_roundtrip() {
        use super::{similarity_from_parts, similarity_parts, transform_point, transform_vec};

        let t = Vec2f::new(12.5, -3.25);
        let angle = 0.7_f32;
        let scale = 1.75_f32;
        let sim = similarity_from_parts(t, angle, scale);

        let (t2, a2, s2) = similarity_parts(&sim);
        assert!((t2.x - t.x).abs() < 1e-5 && (t2.y - t.y).abs() < 1e-5);
        assert!((a2 - angle).abs() < 1e-5);
        assert!((s2 - scale).abs() < 1e-5);

        // transform_point applies scale, then rotation, then translation.
        let p = Point2f::new(1.0, 0.0);
        let q = transform_point(&sim, p);
        assert!((q.x - (scale * angle.cos() + t.x)).abs() < 1e-4);
        assert!((q.y - (scale * angle.sin() + t.y)).abs() < 1e-4);

        // A displacement is transformed by the linear part only, so the
        // difference of two transformed points equals the transformed difference.
        let r = Point2f::new(4.0, -2.0);
        let dv = transform_vec(&sim, r - p);
        let dq = transform_point(&sim, r) - q;
        assert!((dv.x - dq.x).abs() < 1e-4 && (dv.y - dq.y).abs() < 1e-4);
    }

    #[test]
    fn parabolic_peak_offset_locates_a_known_vertex() {
        use super::parabolic_peak_offset;

        // y = -(x - 0.25)^2 sampled at -1, 0, 1 has its vertex at +0.25.
        let f = |x: f32| -(x - 0.25) * (x - 0.25);
        let t = parabolic_peak_offset(f(-1.0), f(0.0), f(1.0)).expect("vertex exists");
        assert!((t - 0.25).abs() < 1e-5, "got {t}");

        // A flat bracket has no vertex.
        assert_eq!(parabolic_peak_offset(1.0, 1.0, 1.0), None);
        // Offsets outside the bracket are reported, not clamped.
        let g = |x: f32| -(x - 3.0) * (x - 3.0);
        let far = parabolic_peak_offset(g(-1.0), g(0.0), g(1.0)).expect("vertex exists");
        assert!(far > 1.0, "got {far}");
    }
}
