use core::ops::{Add, Mul, Sub};

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

#[cfg(test)]
mod tests {
    use super::{Point2f, Vec2f};

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
}
