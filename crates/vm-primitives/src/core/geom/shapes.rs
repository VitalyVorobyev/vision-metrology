//! Geometric primitives: [`Circle2f`], [`Ellipse2f`] and the algebraic
//! [`Conic2f`] they convert through.
//!
//! A conic is represented by the general equation
//!
//! ```text
//! A x² + B x y + C y² + D x + E y + F = 0
//! ```
//!
//! stored as `coeffs = [A, B, C, D, E, F]`.
//!
//! # Ellipse condition
//! A conic is an ellipse when its discriminant `B² - 4AC < 0`.
//!
//! # Coordinate convention
//! Pixel centers at integer coordinates. Conic coefficients are dimensionless
//! ratios; scale invariance is maintained by the fitting constraint (Bookstein
//! or Fitzgibbon) applied during fitting.

use super::transform::{Point2f, Vec2f};
use crate::core::raster::Error;

// ---------------------------------------------------------------------------
// Conic2f
// ---------------------------------------------------------------------------

/// General conic section `A x² + B x y + C y² + D x + E y + F = 0`.
///
/// Coefficients `[A, B, C, D, E, F]` define the conic up to a non-zero scalar
/// multiple. Fitting routines normalise the coefficients according to their
/// chosen constraint (Bookstein: `‖c‖ = 1`; Fitzgibbon: `4AC − B² = 1`).
#[derive(Debug, Clone, PartialEq)]
pub struct Conic2f {
    /// Coefficients `[A, B, C, D, E, F]`.
    pub coeffs: [f32; 6],
}

impl Conic2f {
    /// Evaluate the conic polynomial at point `p`.
    ///
    /// Returns `A x² + B x y + C y² + D x + E y + F`. A point lies on the
    /// conic when `eval(p) == 0`.
    #[inline]
    pub fn eval(&self, p: Point2f) -> f32 {
        let [a, b, c, d, e, f] = self.coeffs;
        let x = p.x;
        let y = p.y;
        a * x * x + b * x * y + c * y * y + d * x + e * y + f
    }

    /// Discriminant `B² − 4AC`.
    ///
    /// - `< 0` → ellipse (or circle)
    /// - `= 0` → parabola
    /// - `> 0` → hyperbola
    #[inline]
    pub fn discriminant(&self) -> f32 {
        let [a, b, c, _, _, _] = self.coeffs;
        b * b - 4.0 * a * c
    }

    /// Returns `true` when the conic is an ellipse (`discriminant < 0`).
    #[inline]
    pub fn is_ellipse(&self) -> bool {
        self.discriminant() < 0.0
    }

    /// Attempt to convert the conic to geometric ellipse form.
    ///
    /// Fails with [`Error::Degenerate`] when:
    /// - The conic is not an ellipse (`discriminant ≥ 0`).
    /// - The matrix `M = [[A, B/2], [B/2, C]]` is near-singular.
    /// - The resulting ellipse has zero or negative semi-axis.
    ///
    /// # Returns
    /// `Ok(Ellipse2f)` on success.
    pub fn to_ellipse(&self) -> Result<Ellipse2f, Error> {
        Ellipse2f::from_conic(self)
    }

    /// Approximate algebraic gradient norm at point `p`.
    ///
    /// `‖∇F(p)‖ = ‖(2Ax + By + D, Bx + 2Cy + E)‖`.
    /// Used by RANSAC to compute normalised algebraic distance.
    #[inline]
    pub fn grad_norm(&self, p: Point2f) -> f32 {
        let [a, b, c, d, e, _] = self.coeffs;
        let x = p.x;
        let y = p.y;
        let gx = 2.0 * a * x + b * y + d;
        let gy = b * x + 2.0 * c * y + e;
        (gx * gx + gy * gy).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Circle2f
// ---------------------------------------------------------------------------

/// A circle in pixel coordinates.
///
/// The most-measured primitive in industrial metrology — hole diameters, boss
/// positions, rim fits — and one this crate had no type for until now.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle2f {
    /// Centre in pixel coordinates.
    pub center: Point2f,
    /// Radius in pixels.
    pub radius: f32,
}

impl Circle2f {
    /// Point on the circle at parameter `t` radians.
    #[inline]
    pub fn point_at(&self, t: f32) -> Point2f {
        Point2f::new(
            self.center.x + self.radius * t.cos(),
            self.center.y + self.radius * t.sin(),
        )
    }

    /// Signed distance from `p` to the circle: positive outside, negative inside.
    ///
    /// This is the residual a geometric (orthogonal-distance) fit minimises, so
    /// it is what such a fit's `rms` is computed from.
    #[inline]
    pub fn signed_distance(&self, p: Point2f) -> f32 {
        (p - self.center).norm() - self.radius
    }

    /// Circumference `2πr`.
    #[inline]
    pub fn circumference(&self) -> f32 {
        2.0 * core::f32::consts::PI * self.radius
    }

    /// Area `πr²`.
    #[inline]
    pub fn area(&self) -> f32 {
        core::f32::consts::PI * self.radius * self.radius
    }
}

// ---------------------------------------------------------------------------
// Ellipse2f
// ---------------------------------------------------------------------------

/// Geometric ellipse in pixel coordinates.
///
/// The ellipse is parameterised as:
/// ```text
/// x(t) = cx + a·cos(t)·cos(θ) − b·sin(t)·sin(θ)
/// y(t) = cy + a·cos(t)·sin(θ) + b·sin(t)·cos(θ)
/// ```
/// where `a = semi_major`, `b = semi_minor`, `θ = angle` (rotation of major
/// axis relative to the positive x-axis), and `t ∈ [0, 2π)`.
///
/// Convention: `semi_major ≥ semi_minor > 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ellipse2f {
    /// Centre in pixel coordinates.
    pub center: Point2f,
    /// Semi-axes: `x = semi_major ≥ semi_minor = y > 0`.
    pub semi_axes: Vec2f,
    /// Rotation of the major axis relative to the positive x-axis, in radians.
    pub angle: f32,
}

impl Ellipse2f {
    /// Semi-major axis length (≥ semi-minor).
    #[inline]
    pub fn semi_major(&self) -> f32 {
        self.semi_axes.x
    }

    /// Semi-minor axis length.
    #[inline]
    pub fn semi_minor(&self) -> f32 {
        self.semi_axes.y
    }

    /// Parametric point at angle `t ∈ [0, 2π)`.
    pub fn point_at(&self, t: f32) -> Point2f {
        let (ct, st) = (t.cos(), t.sin());
        let (ca, sa) = (self.angle.cos(), self.angle.sin());
        let a = self.semi_major();
        let b = self.semi_minor();
        Point2f::new(
            self.center.x + a * ct * ca - b * st * sa,
            self.center.y + a * ct * sa + b * st * ca,
        )
    }

    /// Returns `true` when `p` is within `tol` pixels of the ellipse boundary.
    ///
    /// Uses the Sampson algebraic distance approximation (fast, sub-pixel
    /// accuracy within a few pixels of the curve).
    pub fn contains(&self, p: Point2f, tol: f32) -> bool {
        // Convert to conic and use algebraic distance / gradient norm.
        if let Ok(c) = self.to_conic() {
            let d = c.eval(p);
            let g = c.grad_norm(p);
            if g > 1e-8 {
                return (d / g).abs() < tol;
            }
        }
        false
    }

    /// Convert the geometric ellipse back to algebraic conic form.
    ///
    /// The returned coefficients satisfy the Fitzgibbon `4AC − B² = 1`
    /// normalisation to one decimal degree of freedom.
    pub fn to_conic(&self) -> Result<Conic2f, Error> {
        let a = self.semi_major() as f64;
        let b = self.semi_minor() as f64;
        if a <= 0.0 || b <= 0.0 {
            return Err(Error::Degenerate("ellipse has non-positive semi-axis"));
        }
        let ca = self.angle.cos() as f64;
        let sa = self.angle.sin() as f64;
        let cx = self.center.x as f64;
        let cy = self.center.y as f64;

        // Standard algebraic coefficients for a rotated, translated ellipse:
        // F(x,y) = A(x-cx)^2 + B(x-cx)(y-cy) + C(y-cy)^2 - 1 = 0
        // where A = (ca/a)^2 + (sa/b)^2, B = 2*ca*sa*(1/a^2 - 1/b^2), C = (sa/a)^2 + (ca/b)^2
        let a2 = a * a;
        let b2 = b * b;
        let aa = ca * ca / a2 + sa * sa / b2;
        let bb = 2.0 * ca * sa * (1.0 / a2 - 1.0 / b2);
        let cc = sa * sa / a2 + ca * ca / b2;

        // Expand (x - cx), (y - cy):
        // F = aa*x^2 + bb*x*y + cc*y^2 + dd*x + ee*y + ff = 0
        let dd = -2.0 * (aa * cx + bb / 2.0 * cy);
        let ee = -2.0 * (bb / 2.0 * cx + cc * cy);
        let ff = aa * cx * cx + bb * cx * cy + cc * cy * cy - 1.0;

        Ok(Conic2f {
            coeffs: [
                aa as f32, bb as f32, cc as f32, dd as f32, ee as f32, ff as f32,
            ],
        })
    }

    /// Construct from algebraic conic coefficients.
    ///
    /// Returns `Err(Error::Degenerate)` when the conic is not an ellipse or is
    /// numerically degenerate.
    pub fn from_conic(c: &Conic2f) -> Result<Self, Error> {
        if !c.is_ellipse() {
            return Err(Error::Degenerate("conic discriminant ≥ 0: not an ellipse"));
        }
        let [a, b, cc, d, e, f] = c.coeffs.map(|v| v as f64);

        // Centre: solve 2Ax + By + D = 0, Bx + 2Cy + E = 0
        // Determinant of [[2A, B], [B, 2C]]
        let det_m = 4.0 * a * cc - b * b;
        if det_m.abs() < 1e-12 {
            return Err(Error::Degenerate("conic matrix is singular"));
        }
        let cx = (-2.0 * cc * d + b * e) / det_m;
        let cy = (-2.0 * a * e + b * d) / det_m;

        // Value at centre: F(cx, cy). For an ellipse, F(centre) ≠ 0.
        // We normalise by dividing all coefficients by −F(centre) so that the centred
        // quadratic form becomes A'u² + B'uv + C'v² = 1 where (u,v) = (x−cx, y−cy).
        // After dividing, the matrix [[A', B'/2],[B'/2, C']] must be positive-definite
        // (both eigenvalues positive). The sign of F(centre) tells us which factor to use:
        //   F(centre) < 0 → interior is negative → divide by −F(centre) > 0 → A', C' keep sign of A, C → OK.
        //   F(centre) > 0 → divide by −F(centre) < 0 → A', C' flip sign.
        // We handle both by checking the trace after normalisation.
        let f0 = a * cx * cx + b * cx * cy + cc * cy * cy + d * cx + e * cy + f;
        if f0.abs() < 1e-14 {
            return Err(Error::Degenerate(
                "conic value at centre is zero: degenerate ellipse",
            ));
        }
        // First attempt: divide by −f0.
        let inv_f0 = -1.0 / f0;
        let mut an = a * inv_f0;
        let mut bn = b * inv_f0;
        let mut cn = cc * inv_f0;
        // If the centred matrix trace is negative, the normalisation was in the wrong direction.
        // Flip sign (equivalent to dividing by +f0 instead of −f0).
        if an + cn < 0.0 {
            an = -an;
            bn = -bn;
            cn = -cn;
        }

        // Eigendecomposition of [[an, bn/2], [bn/2, cn]]
        // Eigenvalues: λ = ((an + cn) ± sqrt((an - cn)^2 + bn^2)) / 2
        let sum = an + cn;
        let diff = an - cn;
        let disc = (diff * diff + bn * bn).sqrt();
        let lam1 = (sum + disc) / 2.0; // larger eigenvalue
        let lam2 = (sum - disc) / 2.0; // smaller eigenvalue

        if lam1 <= 0.0 || lam2 <= 0.0 {
            return Err(Error::Degenerate(
                "ellipse eigenvalues non-positive: imaginary axes",
            ));
        }

        // Semi-axes: a = 1/√λ_min, b = 1/√λ_max (larger axis = smaller eigenvalue)
        let semi_major = (1.0 / lam2).sqrt();
        let semi_minor = (1.0 / lam1).sqrt();

        // Rotation angle: eigenvector of lam2 (smaller eigenvalue = major axis direction)
        // For [[an, bn/2], [bn/2, cn]] with eigenvalue lam2:
        //   (an - lam2) * v.x + (bn/2) * v.y = 0
        let vx = bn / 2.0;
        let vy = lam2 - an;
        let angle = f64::atan2(vy, vx) as f32;

        Ok(Ellipse2f {
            center: Point2f::new(cx as f32, cy as f32),
            semi_axes: Vec2f::new(semi_major as f32, semi_minor as f32),
            angle,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    fn make_circle(cx: f32, cy: f32, r: f32) -> Ellipse2f {
        Ellipse2f {
            center: Point2f::new(cx, cy),
            semi_axes: Vec2f::new(r, r),
            angle: 0.0,
        }
    }

    #[test]
    fn circle_roundtrip_conic() {
        // A unit circle centred at (5, 3) should survive conic conversion.
        let ell = make_circle(5.0, 3.0, 10.0);
        let conic = ell.to_conic().expect("to_conic");
        assert!(
            conic.is_ellipse(),
            "discriminant must be negative for circle"
        );

        let ell2 = conic.to_ellipse().expect("from_conic");
        assert!((ell2.center.x - ell.center.x).abs() < 1e-3, "cx");
        assert!((ell2.center.y - ell.center.y).abs() < 1e-3, "cy");
        assert!((ell2.semi_major() - 10.0).abs() < 1e-2, "semi_major");
        assert!((ell2.semi_minor() - 10.0).abs() < 1e-2, "semi_minor");
    }

    #[test]
    fn ellipse_roundtrip_rotated() {
        // An ellipse with semi-axes (20, 8), rotated 30°, centred at (50, 40).
        let ell = Ellipse2f {
            center: Point2f::new(50.0, 40.0),
            semi_axes: Vec2f::new(20.0, 8.0),
            angle: PI / 6.0,
        };
        let conic = ell.to_conic().expect("to_conic");
        assert!(conic.is_ellipse());
        let ell2 = conic.to_ellipse().expect("from_conic");

        assert!((ell2.center.x - 50.0).abs() < 0.1, "cx: {}", ell2.center.x);
        assert!((ell2.center.y - 40.0).abs() < 0.1, "cy: {}", ell2.center.y);
        assert!(
            (ell2.semi_major() - 20.0).abs() < 0.05,
            "a: {}",
            ell2.semi_major()
        );
        assert!(
            (ell2.semi_minor() - 8.0).abs() < 0.05,
            "b: {}",
            ell2.semi_minor()
        );
    }

    #[test]
    fn conic_eval_on_ellipse_boundary() {
        // Points exactly on the ellipse boundary should evaluate to near zero.
        let ell = Ellipse2f {
            center: Point2f::new(10.0, 10.0),
            semi_axes: Vec2f::new(5.0, 3.0),
            angle: 0.0,
        };
        let conic = ell.to_conic().expect("to_conic");
        for i in 0..8 {
            let t = (i as f32) * PI / 4.0;
            let p = ell.point_at(t);
            let v = conic.eval(p);
            assert!(v.abs() < 1e-4, "eval at t={t:.2}: {v}");
        }
    }

    #[test]
    fn non_ellipse_conic_to_ellipse_fails() {
        // A hyperbola x^2 - y^2 = 1: A=1, B=0, C=-1, D=0, E=0, F=-1
        let c = Conic2f {
            coeffs: [1.0, 0.0, -1.0, 0.0, 0.0, -1.0],
        };
        assert!(!c.is_ellipse(), "hyperbola should not be ellipse");
        assert!(
            c.to_ellipse().is_err(),
            "to_ellipse should fail for hyperbola"
        );
    }

    #[test]
    fn ellipse_contains_boundary_points() {
        let ell = Ellipse2f {
            center: Point2f::new(0.0, 0.0),
            semi_axes: Vec2f::new(8.0, 4.0),
            angle: 0.0,
        };
        // Points on the boundary (t = 0, π/2, π, 3π/2) should be within 0.5 px.
        for i in 0..4 {
            let t = (i as f32) * PI / 2.0;
            let p = ell.point_at(t);
            assert!(
                ell.contains(p, 0.5),
                "boundary point t={t:.2} not contained"
            );
        }
        // A point far away should not be contained.
        assert!(!ell.contains(Point2f::new(100.0, 100.0), 0.5));
    }
}
