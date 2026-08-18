use core::{
    f32::consts::PI,
    ops::{Add, Mul, Neg, Sub},
};

/// A 2-D point in floating-point pixel coordinates.
///
/// By convention, integer coordinate `i` refers to the **center** of pixel `i`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point2f {
    /// Horizontal (column) coordinate in pixels.
    pub x: f32,
    /// Vertical (row) coordinate in pixels.
    pub y: f32,
}

/// A 2-D displacement vector (difference of two [`Point2f`] values).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2f {
    /// Horizontal component in pixels.
    pub x: f32,
    /// Vertical component in pixels.
    pub y: f32,
}

impl Vec2f {
    /// Dot product `self · rhs`.
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// Euclidean length `√(x² + y²)`.
    pub fn norm(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns a unit vector in the same direction, or the zero vector if `norm() == 0`.
    pub fn normalize(self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            Self::default()
        } else {
            self * (1.0 / n)
        }
    }
}

impl Add<Vec2f> for Point2f {
    type Output = Point2f;

    fn add(self, rhs: Vec2f) -> Self::Output {
        Point2f {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub<Vec2f> for Point2f {
    type Output = Point2f;

    fn sub(self, rhs: Vec2f) -> Self::Output {
        Point2f {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Sub<Point2f> for Point2f {
    type Output = Vec2f;

    fn sub(self, rhs: Point2f) -> Self::Output {
        Vec2f {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Add for Vec2f {
    type Output = Vec2f;

    fn add(self, rhs: Vec2f) -> Self::Output {
        Vec2f {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vec2f {
    type Output = Vec2f;

    fn sub(self, rhs: Vec2f) -> Self::Output {
        Vec2f {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Mul<f32> for Vec2f {
    type Output = Vec2f;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec2f {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Mul<Vec2f> for f32 {
    type Output = Vec2f;

    fn mul(self, rhs: Vec2f) -> Self::Output {
        rhs * self
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
        Point2f {
            x: self.x + self.width * 0.5,
            y: self.y + self.height * 0.5,
        }
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
// Conversions between this crate's lightweight types and nalgebra
// ---------------------------------------------------------------------------

/// Convert a [`Point2f`] to a nalgebra `Point2<f32>`.
#[inline]
pub fn to_na_point(p: Point2f) -> nalgebra::Point2<f32> {
    nalgebra::Point2::new(p.x, p.y)
}

/// Convert a nalgebra `Point2<f32>` to a [`Point2f`].
#[inline]
pub fn from_na_point(p: nalgebra::Point2<f32>) -> Point2f {
    Point2f { x: p.x, y: p.y }
}

/// Convert a [`Vec2f`] to a nalgebra `Vector2<f32>`.
#[inline]
pub fn to_na_vec(v: Vec2f) -> nalgebra::Vector2<f32> {
    nalgebra::Vector2::new(v.x, v.y)
}

/// Convert a nalgebra `Vector2<f32>` to a [`Vec2f`].
#[inline]
pub fn from_na_vec(v: nalgebra::Vector2<f32>) -> Vec2f {
    Vec2f { x: v.x, y: v.y }
}

// ---------------------------------------------------------------------------
// Vector helpers used by pose estimation and contour orientation
// ---------------------------------------------------------------------------

impl Vec2f {
    /// Rotate by +90° (counter-clockwise in a y-down image frame): `(x, y) → (−y, x)`.
    ///
    /// Used to turn an edge tangent into a normal and to build the rotational
    /// column of a similarity Jacobian.
    #[inline]
    pub fn perp(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    /// 2-D cross product `self × rhs = self.x·rhs.y − self.y·rhs.x`.
    ///
    /// The signed area of the parallelogram spanned by the two vectors; equal
    /// to `self.perp().dot(rhs)`.
    #[inline]
    pub fn cross(self, rhs: Self) -> f32 {
        self.x * rhs.y - self.y * rhs.x
    }
}

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
    from_na_point(sim * to_na_point(p))
}

/// Apply the linear part of a similarity transform to a displacement: `s·R·v`.
///
/// Translation is deliberately not applied — a displacement has no origin.
#[inline]
pub fn transform_vec(sim: &Similarity2f, v: Vec2f) -> Vec2f {
    from_na_vec(sim * to_na_vec(v))
}

/// Apply a rigid transform to a point: `R·p + t`.
#[inline]
pub fn transform_point_iso(iso: &Isometry2f, p: Point2f) -> Point2f {
    from_na_point(iso * to_na_point(p))
}

/// Build a [`Similarity2f`] from translation, rotation angle (radians) and
/// uniform scale.
///
/// The resulting map is `p ↦ scale·R(angle)·p + translation`.
#[inline]
pub fn similarity_from_parts(translation: Vec2f, angle: f32, scale: f32) -> Similarity2f {
    Similarity2f::new(to_na_vec(translation), angle, scale)
}

/// Decompose a [`Similarity2f`] into `(translation, angle, scale)`.
///
/// Inverse of [`similarity_from_parts`]; the angle is in `(-π, π]`.
#[inline]
pub fn similarity_parts(sim: &Similarity2f) -> (Vec2f, f32, f32) {
    (
        from_na_vec(sim.isometry.translation.vector),
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
    use super::{Angle, Point2f, Rect2f, Vec2f, from_na_point, to_na_point};

    #[test]
    fn vec_ops_and_normalize() {
        let a = Vec2f { x: 3.0, y: 4.0 };
        let b = Vec2f { x: 1.0, y: -2.0 };

        assert_eq!(a + b, Vec2f { x: 4.0, y: 2.0 });
        assert_eq!(a - b, Vec2f { x: 2.0, y: 6.0 });
        assert!((a.dot(b) + 5.0).abs() < 1e-6);
        assert!((a.norm() - 5.0).abs() < 1e-6);

        let n = a.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-6);

        let z = Vec2f::default().normalize();
        assert_eq!(z, Vec2f::default());
    }

    #[test]
    fn point_vec_ops() {
        let p = Point2f { x: 2.0, y: 3.0 };
        let v = Vec2f { x: 0.5, y: -1.0 };

        assert_eq!(p + v, Point2f { x: 2.5, y: 2.0 });
        assert_eq!(p - v, Point2f { x: 1.5, y: 4.0 });
        assert_eq!(p - Point2f { x: 1.0, y: 1.0 }, Vec2f { x: 1.0, y: 2.0 });
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
        assert_eq!(r.center(), Point2f { x: 3.0, y: 3.5 });
        assert!(r.contains(Point2f { x: 2.0, y: 3.0 }));
        assert!(!r.contains(Point2f { x: 0.0, y: 3.0 }));
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
    fn na_conversion_roundtrip() {
        let p = Point2f { x: 1.5, y: -2.3 };
        assert_eq!(from_na_point(to_na_point(p)), p);
    }

    #[test]
    fn isometry2f_apply() {
        use core::f32::consts::FRAC_PI_2;
        // 90° CCW rotation should map (1, 0) → (0, 1).
        let iso = nalgebra::Isometry2::rotation(FRAC_PI_2);
        let q = iso * to_na_point(Point2f { x: 1.0, y: 0.0 });
        assert!((q.x).abs() < 1e-6);
        assert!((q.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity2f_scale_and_rotate() {
        use nalgebra::{Similarity2, Vector2};
        // Scale ×2, no rotation, translate by (1, 0).
        let sim = Similarity2::new(Vector2::new(1.0, 0.0), 0.0, 2.0);
        let q = sim * to_na_point(Point2f { x: 1.0, y: 1.0 });
        assert!((q.x - 3.0).abs() < 1e-6); // 2*1 + 1
        assert!((q.y - 2.0).abs() < 1e-6); // 2*1
    }

    #[test]
    fn perp_and_cross_agree() {
        let a = Vec2f { x: 3.0, y: 1.0 };
        let b = Vec2f { x: -2.0, y: 4.0 };
        // perp is a +90 degree rotation, so it preserves length and is orthogonal.
        assert!((a.perp().norm() - a.norm()).abs() < 1e-6);
        assert!(a.perp().dot(a).abs() < 1e-6);
        assert_eq!(a.perp(), Vec2f { x: -1.0, y: 3.0 });
        // cross(a, b) == perp(a) . b by definition.
        assert!((a.cross(b) - a.perp().dot(b)).abs() < 1e-6);
        // Antisymmetry.
        assert!((a.cross(b) + b.cross(a)).abs() < 1e-6);
    }

    #[test]
    fn similarity_parts_roundtrip() {
        use super::{similarity_from_parts, similarity_parts, transform_point, transform_vec};

        let t = Vec2f { x: 12.5, y: -3.25 };
        let angle = 0.7_f32;
        let scale = 1.75_f32;
        let sim = similarity_from_parts(t, angle, scale);

        let (t2, a2, s2) = similarity_parts(&sim);
        assert!((t2.x - t.x).abs() < 1e-5 && (t2.y - t.y).abs() < 1e-5);
        assert!((a2 - angle).abs() < 1e-5);
        assert!((s2 - scale).abs() < 1e-5);

        // transform_point applies scale, then rotation, then translation.
        let p = Point2f { x: 1.0, y: 0.0 };
        let q = transform_point(&sim, p);
        assert!((q.x - (scale * angle.cos() + t.x)).abs() < 1e-4);
        assert!((q.y - (scale * angle.sin() + t.y)).abs() < 1e-4);

        // A displacement is transformed by the linear part only, so the
        // difference of two transformed points equals the transformed difference.
        let r = Point2f { x: 4.0, y: -2.0 };
        let dv = transform_vec(&sim, r - p);
        let dq = transform_point(&sim, r) - q;
        assert!((dv.x - dq.x).abs() < 1e-4 && (dv.y - dq.y).abs() < 1e-4);
    }

    #[test]
    fn transform_point_iso_has_unit_scale() {
        use super::transform_point_iso;
        use core::f32::consts::FRAC_PI_2;

        let iso = nalgebra::Isometry2::new(nalgebra::Vector2::new(2.0, 3.0), FRAC_PI_2);
        let q = transform_point_iso(&iso, Point2f { x: 1.0, y: 0.0 });
        assert!((q.x - 2.0).abs() < 1e-5);
        assert!((q.y - 4.0).abs() < 1e-5);
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
