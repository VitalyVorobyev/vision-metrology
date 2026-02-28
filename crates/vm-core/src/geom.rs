use core::{
    f32::consts::PI,
    ops::{Add, Mul, Neg, Sub},
};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point2f {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2f {
    pub x: f32,
    pub y: f32,
}

impl Vec2f {
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }

    pub fn norm(self) -> f32 {
        self.dot(self).sqrt()
    }

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line2f {
    pub p: Point2f,
    pub dir: Vec2f,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Polyline2f {
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
    pub x: f32,
    pub y: f32,
    pub width: f32,
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
// Conversions between vm-core lightweight types and nalgebra
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
}
