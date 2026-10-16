//! Lifting one view's pixel translation to a plane displacement at one point.

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use vm_primitives::Projective2f;

use super::types::ViewShift;

// Invariant 20: every quantity here is f64.

/// A view's image→plane map, or `None` when the homography is not invertible.
pub(super) fn image_to_plane(h: &Projective2f) -> Option<Matrix3<f64>> {
    let m: Matrix3<f64> = h.matrix().cast::<f64>();
    if !m.iter().all(|v| v.is_finite()) {
        return None;
    }
    let inv = m.try_inverse()?;
    inv.iter().all(|v| v.is_finite()).then_some(inv)
}

/// One view's translation on the plane.
#[derive(Debug, Clone, Copy)]
pub(super) struct Lift {
    /// The region centroid on the plane, mm.
    pub lever: Vector2<f64>,
    /// The plane displacement of that point, mm.
    pub d: Vector2<f64>,
    /// Inverse covariance of `d`, 1/mm².
    pub info: Matrix2<f64>,
}

/// `g · (x, y, 1)`, perspective-divided, with the divisor.
fn map(g: &Matrix3<f64>, x: f64, y: f64) -> Option<(Vector2<f64>, f64)> {
    let p = g * Vector3::new(x, y, 1.0);
    let scale = p.x.abs().max(p.y.abs()).max(1.0);
    if p.z.is_nan() || p.z.abs() <= 1e-12 * scale {
        return None;
    }
    let q = Vector2::new(p.x / p.z, p.y / p.z);
    (q.x.is_finite() && q.y.is_finite()).then_some((q, p.z))
}

/// The Jacobian of `map` at `(x, y)`.
fn jacobian(g: &Matrix3<f64>, x: f64, y: f64) -> Option<Matrix2<f64>> {
    let (q, w) = map(g, x, y)?;
    Some(Matrix2::new(
        (g[(0, 0)] - q.x * g[(2, 0)]) / w,
        (g[(0, 1)] - q.x * g[(2, 1)]) / w,
        (g[(1, 0)] - q.y * g[(2, 0)]) / w,
        (g[(1, 1)] - q.y * g[(2, 1)]) / w,
    ))
}

/// Lift `shift` through the image→plane map `g` at the shift's centroid.
///
/// The displacement is the exact plane distance between `g(c)` and `g(c + d)`. Its
/// covariance is the pixel covariance (floored by `floor_px` in quadrature) through the
/// local Jacobian at the midpoint. `None` for non-finite input, a point sent to
/// infinity, or a covariance that is not positive definite.
pub(super) fn lift(g: &Matrix3<f64>, shift: &ViewShift, floor_px: f64) -> Option<Lift> {
    let (cx, cy) = (shift.centroid.x as f64, shift.centroid.y as f64);
    let (dx, dy) = (shift.d_px.x as f64, shift.d_px.y as f64);
    let [sxx, sxy, syy] = shift.covariance.map(f64::from);
    if ![cx, cy, dx, dy, sxx, sxy, syy]
        .iter()
        .all(|v| v.is_finite())
    {
        return None;
    }
    let (p0, _) = map(g, cx, cy)?;
    let (p1, _) = map(g, cx + dx, cy + dy)?;
    let j = jacobian(g, cx + 0.5 * dx, cy + 0.5 * dy)?;
    let f2 = floor_px * floor_px;
    let cov_px = Matrix2::new(sxx + f2, sxy, sxy, syy + f2);
    let cov = j * cov_px * j.transpose();
    let det = cov[(0, 0)] * cov[(1, 1)] - cov[(0, 1)] * cov[(1, 0)];
    if !(det > 0.0 && cov[(0, 0)] > 0.0) {
        return None;
    }
    let info = cov.try_inverse()?;
    Some(Lift {
        lever: p0,
        d: p1 - p0,
        info: 0.5 * (info + info.transpose()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vm_primitives::{Point2f, Vec2f};

    fn shift(c: (f32, f32), d: (f32, f32), var: f32) -> ViewShift {
        ViewShift {
            d_px: Vec2f::new(d.0, d.1),
            covariance: [var, 0.0, var],
            centroid: Point2f::new(c.0, c.1),
            accepted: true,
            closure_px: None,
        }
    }

    /// An affine view at 20 px/mm: the lift divides by the scale, and the covariance by
    /// its square.
    #[test]
    fn affine_lift_divides_by_the_scale() {
        let h = Projective2f::from_matrix_unchecked(nalgebra::Matrix3::new(
            20.0, 0.0, 320.0, 0.0, 20.0, 240.0, 0.0, 0.0, 1.0,
        ));
        let g = image_to_plane(&h).unwrap();
        let l = lift(&g, &shift((340.0, 260.0), (4.0, -2.0), 0.04), 0.0).unwrap();
        assert!(
            (l.lever - Vector2::new(1.0, 1.0)).norm() < 1e-12,
            "{:?}",
            l.lever
        );
        assert!((l.d - Vector2::new(0.2, -0.1)).norm() < 1e-12, "{:?}", l.d);
        // σ = 0.2 px → 0.01 mm, information 1e4 / mm².
        assert!((l.info[(0, 0)] / 1e4 - 1.0).abs() < 1e-6, "{}", l.info);
    }

    #[test]
    fn a_singular_homography_and_a_point_at_infinity_do_not_lift() {
        let singular = Projective2f::from_matrix_unchecked(nalgebra::Matrix3::new(
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ));
        assert!(image_to_plane(&singular).is_none());
        // Image→plane is the identity with a horizon at x = 100.
        let g = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.01, 0.0, 1.0);
        assert!(lift(&g, &shift((100.0, 0.0), (1.0, 0.0), 0.01), 0.01).is_none());
        assert!(lift(&g, &shift((f32::NAN, 0.0), (1.0, 0.0), 0.01), 0.01).is_none());
    }
}
