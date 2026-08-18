//! Direct algebraic conic fitting.
//!
//! Two methods are provided:
//!
//! ## Bookstein (`use_bookstein = true`)
//! Constraint: `‖c‖ = 1` (unit-norm coefficient vector). Solves the right
//! singular vector of the design matrix `D` for the smallest singular value.
//! Similarity-invariant. Works for any conic type.
//!
//! Input coordinates are translated to their centroid and scaled to unit
//! variance before forming the scatter matrix. The resulting coefficients are
//! mapped back to original coordinate space before returning.
//!
//! ## Fitzgibbon DLS (`use_bookstein = false`)
//! Constraint: `4AC − B² = 1`, which forces an ellipse solution.
//! Reduced 3×3 eigenproblem `M a₁ = λ a₁` where
//! `M = C₁₁⁻¹ (S₁₁ − S₁₂ S₂₂⁻¹ S₂₁)`.
//! After selecting the ellipse eigenvector, the full coefficient vector is
//! rescaled to satisfy `a₁ᵀ C₁₁ a₁ = 1` (the actual constraint).
//!
//! # References
//! - Bookstein, F.L. (1979). Fitting conic sections to scattered data.
//!   Computer Graphics and Image Processing 9(1), 56–71.
//! - Fitzgibbon, A., Pilu, M., Fisher, R.B. (1999). Direct least squares
//!   fitting of ellipses. IEEE TPAMI 21(5), 476–480.

use nalgebra::{Matrix3, Matrix6, SVD, SymmetricEigen, Vector3};
use vm_primitives::{Error, Point2f};

use super::conic::Conic2f;

// ---------------------------------------------------------------------------
// Coordinate normalisation helpers
// ---------------------------------------------------------------------------

/// Compute translation (centroid) and uniform scale such that normalised points
/// have zero mean and RMS distance ≈ 1 from the origin.
///
/// Returns `(cx, cy, scale)`. Apply as: `x' = (x - cx) / scale`, `y' = (y - cy) / scale`.
/// Reject inputs that cannot produce a meaningful fit.
///
/// One non-finite coordinate contaminates the centroid, hence every normalised
/// coordinate, hence the whole scatter matrix. Past that point the eigenvalues
/// are all NaN and any ordering of them is meaningless, so catch it at the
/// boundary and return an error rather than panicking inside the solver.
fn ensure_finite(pts: &[Point2f]) -> Result<(), Error> {
    if pts.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
        return Err(Error::Degenerate("conic fit: input points must be finite"));
    }
    Ok(())
}

fn normalise_params(pts: &[Point2f]) -> (f64, f64, f64) {
    let n = pts.len() as f64;
    let cx: f64 = pts.iter().map(|p| p.x as f64).sum::<f64>() / n;
    let cy: f64 = pts.iter().map(|p| p.y as f64).sum::<f64>() / n;
    let rms: f64 = pts
        .iter()
        .map(|p| {
            let dx = p.x as f64 - cx;
            let dy = p.y as f64 - cy;
            dx * dx + dy * dy
        })
        .sum::<f64>()
        / n;
    let scale = rms.sqrt().max(1e-10);
    (cx, cy, scale)
}

// ---------------------------------------------------------------------------
// Design matrix helpers
// ---------------------------------------------------------------------------

/// Build the 6-column design matrix row for point `(x, y)`.
/// Row = `[x², xy, y², x, y, 1]`.
#[inline]
fn design_row_xy(x: f64, y: f64) -> [f64; 6] {
    [x * x, x * y, y * y, x, y, 1.0]
}

/// Accumulate `DᵀD` (6×6 scatter matrix) from a normalised set of points.
fn scatter_matrix_xy(xs: &[f64], ys: &[f64]) -> Matrix6<f64> {
    let mut s = Matrix6::<f64>::zeros();
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let r = design_row_xy(x, y);
        for i in 0..6 {
            for j in i..6 {
                let v = r[i] * r[j];
                s[(i, j)] += v;
                if i != j {
                    s[(j, i)] += v;
                }
            }
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Denormalisation
// ---------------------------------------------------------------------------

/// Transform conic coefficients fitted in normalised space `(x', y')` back to
/// original space `(x, y)` where `x = cx + scale * x'`.
///
/// The conic in normalised space is `a'·(x')² + b'·x'y' + c'·(y')² + d'·x' + e'·y' + f' = 0`.
/// Substituting `x' = (x - cx)/s` and `y' = (y - cy)/s`:
///
/// ```text
/// a = a'/s²        b = b'/s²        c = c'/s²
/// d = (-2a'·cx - b'·cy + d'·s) / s²
/// e = (-b'·cx - 2c'·cy + e'·s) / s²
/// f = (a'·cx² + b'·cx·cy + c'·cy² - d'·s·cx - e'·s·cy + f'·s²) / s²
/// ```
fn denormalise_conic(coeffs: &[f64; 6], cx: f64, cy: f64, scale: f64) -> [f32; 6] {
    let [a, b, c, d, e, f] = *coeffs;
    let s = scale;
    let s2 = s * s;

    let an = a / s2;
    let bn = b / s2;
    let cn = c / s2;
    let dn = (-2.0 * a * cx - b * cy + d * s) / s2;
    let en = (-b * cx - 2.0 * c * cy + e * s) / s2;
    let fn_ = (a * cx * cx + b * cx * cy + c * cy * cy - d * s * cx - e * s * cy + f * s2) / s2;

    [
        an as f32, bn as f32, cn as f32, dn as f32, en as f32, fn_ as f32,
    ]
}

// ---------------------------------------------------------------------------
// Bookstein fit
// ---------------------------------------------------------------------------

/// Fit a conic to `pts` using the Bookstein `‖c‖ = 1` constraint.
///
/// Coordinates are normalised to zero mean and unit RMS radius before fitting,
/// then the result is mapped back to the original coordinate frame.
///
/// # Errors
/// - [`Error::InsufficientData`] when `pts.len() < 5`.
/// - [`Error::Degenerate`] when the scatter matrix is numerically degenerate.
pub(crate) fn fit_bookstein(pts: &[Point2f]) -> Result<Conic2f, Error> {
    if pts.len() < 5 {
        return Err(Error::InsufficientData {
            need: 5,
            got: pts.len(),
        });
    }

    ensure_finite(pts)?;

    let (cx, cy, scale) = normalise_params(pts);
    let xs: Vec<f64> = pts.iter().map(|p| (p.x as f64 - cx) / scale).collect();
    let ys: Vec<f64> = pts.iter().map(|p| (p.y as f64 - cy) / scale).collect();

    let s = scatter_matrix_xy(&xs, &ys);
    // Smallest-eigenvalue eigenvector of DᵀD = Bookstein solution.
    let eig = SymmetricEigen::new(s);
    // nalgebra does NOT guarantee sorted eigenvalues — find the minimum explicitly.
    // Skip non-finite eigenvalues rather than ordering against them: `partial_cmp`
    // returns `None` for NaN, so comparing them at all is a panic. `ensure_finite`
    // rules out the common cause, but a finite input can still be rank-deficient
    // enough for the decomposition to produce NaN, and there is no sensible
    // "smallest" among them.
    let min_col = eig
        .eigenvalues
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("finite compare"))
        .map(|(i, _)| i)
        .ok_or(Error::Degenerate(
            "Bookstein fit: scatter matrix has no finite eigenvalue",
        ))?;
    let v = eig.eigenvectors.column(min_col);
    let raw: [f64; 6] = [v[0], v[1], v[2], v[3], v[4], v[5]];

    // Map back to original coordinate space.
    let coeffs = denormalise_conic(&raw, cx, cy, scale);
    Ok(Conic2f { coeffs })
}

// ---------------------------------------------------------------------------
// Fitzgibbon DLS fit
// ---------------------------------------------------------------------------

/// Fit an ellipse using the Fitzgibbon DLS (`4AC − B² = 1`) constraint.
///
/// Solves the generalized 3×3 eigenproblem
/// `(S₁₁ − S₁₂ S₂₂⁻¹ S₂₁) a₁ = λ C₁₁ a₁`
/// using the Schur decomposition of `M = C₁₁⁻¹ T` (real, generally
/// non-symmetric). For each real eigenvalue `λ > 0`, the corresponding
/// eigenvector is recovered via SVD of `(M − λI)`. The candidate satisfying
/// the ellipse condition `4AC − B² > 0` (positive definite quadratic form) is
/// rescaled to `a₁ᵀ C₁₁ a₁ = 1` before recovering the full 6-vector.
///
/// # Errors
/// - [`Error::InsufficientData`] when `pts.len() < 5`.
/// - [`Error::Degenerate`] when no valid ellipse eigenvector is found or the
///   system is numerically singular.
pub(crate) fn fit_fitzgibbon(pts: &[Point2f]) -> Result<Conic2f, Error> {
    if pts.len() < 5 {
        return Err(Error::InsufficientData {
            need: 5,
            got: pts.len(),
        });
    }

    ensure_finite(pts)?;

    // Normalise coordinates for numerical stability.
    let (cx, cy, scale) = normalise_params(pts);
    let xs: Vec<f64> = pts.iter().map(|p| (p.x as f64 - cx) / scale).collect();
    let ys: Vec<f64> = pts.iter().map(|p| (p.y as f64 - cy) / scale).collect();

    // Scatter matrix in normalised coordinates.
    let s = scatter_matrix_xy(&xs, &ys);

    // Block-partition S into 3×3 sub-blocks:
    //   S = [[S11, S12], [S21, S22]]
    let s11 = s.fixed_view::<3, 3>(0, 0).into_owned();
    let s12 = s.fixed_view::<3, 3>(0, 3).into_owned();
    let s21 = s.fixed_view::<3, 3>(3, 0).into_owned();
    let s22 = s.fixed_view::<3, 3>(3, 3).into_owned();

    // Fitzgibbon constraint matrix top-left block C₁₁ (encodes 4AC − B² = 1):
    //   C₁₁ = [[0, 0, 2],
    //           [0,-1, 0],
    //           [2, 0, 0]]
    let c11 = Matrix3::<f64>::from_row_slice(&[0.0, 0.0, 2.0, 0.0, -1.0, 0.0, 2.0, 0.0, 0.0]);

    let s22_inv = s22
        .try_inverse()
        .ok_or(Error::Degenerate("S22 block is singular in Fitzgibbon fit"))?;
    let c11_inv = c11.try_inverse().ok_or(Error::Degenerate(
        "Fitzgibbon constraint matrix C11 is singular",
    ))?;

    // T = S₁₁ − S₁₂ S₂₂⁻¹ S₂₁  (symmetric, positive semi-definite)
    let t = s11 - s12 * s22_inv * s21;

    // M = C₁₁⁻¹ T  — real but NOT symmetric in general.
    // The generalized eigenproblem T a₁ = λ C₁₁ a₁ has the same eigenvalues
    // as the standard problem M a₁ = λ a₁. Use the Schur decomposition to
    // obtain real eigenvalues, then recover each eigenvector via SVD of (M − λI).
    let m = c11_inv * t;

    // Real Schur decomposition: M = Q T Q^T where T is quasi-upper-triangular.
    // `eigenvalues()` returns Some(v) only when all eigenvalues are real.
    let schur = m.schur();
    let eigenvalues = schur
        .eigenvalues()
        .ok_or(Error::Degenerate("Fitzgibbon fit: complex eigenvalues"))?;

    // For each real eigenvalue λ, recover the eigenvector as the right singular
    // vector of (M − λI) corresponding to the smallest singular value, then
    // apply the ellipse condition check.
    let mut best_a1: Option<Vector3<f64>> = None;
    for &lam in eigenvalues.iter() {
        // Eigenvector via null space of (M − λI).
        let deflated = m - Matrix3::from_diagonal_element(lam);
        // SVD: the last right singular vector (smallest singular value) is the null vector.
        let svd = SVD::new(deflated, false, true);
        let vt = svd
            .v_t
            .ok_or(Error::Degenerate("SVD failed in Fitzgibbon fit"))?;
        // Last row of V^T = last column of V = null-space vector.
        let a1 = Vector3::new(vt[(2, 0)], vt[(2, 1)], vt[(2, 2)]);

        // Ellipse condition: 4AC − B² > 0.
        let cap_a = a1[0];
        let cap_b = a1[1];
        let cap_c = a1[2];
        if 4.0 * cap_a * cap_c - cap_b * cap_b <= 0.0 {
            continue;
        }

        // Rescale to satisfy a₁ᵀ C₁₁ a₁ = 1.
        let constraint_val = a1.dot(&(c11 * a1));
        if constraint_val <= 0.0 {
            continue;
        }
        best_a1 = Some(a1 / constraint_val.sqrt());
        break;
    }

    let a1 = best_a1.ok_or(Error::Degenerate(
        "Fitzgibbon fit: no valid ellipse eigenvector found",
    ))?;

    // Recover unconstrained bottom coefficients: a₂ = −S₂₂⁻¹ S₂₁ a₁.
    let a2 = -(s22_inv * s21) * a1;

    // Assemble full normalised-space coefficient vector.
    let raw_n = [a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]];

    // Ensure leading coefficient A is positive.
    let sign = if raw_n[0] >= 0.0 { 1.0_f64 } else { -1.0_f64 };
    let raw_n = raw_n.map(|v| v * sign);

    // Map back to original coordinate space.
    let coeffs = denormalise_conic(&raw_n, cx, cy, scale);
    Ok(Conic2f { coeffs })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;
    use vm_primitives::{Point2f, Vec2f};

    use crate::shape::conic::Ellipse2f;

    /// Generate exact points on an ellipse.
    fn ellipse_points(ell: &Ellipse2f, n: usize) -> Vec<Point2f> {
        (0..n)
            .map(|i| {
                let t = 2.0 * PI * i as f32 / n as f32;
                ell.point_at(t)
            })
            .collect()
    }

    #[test]
    fn bookstein_fits_circle() {
        // 20 exact points on a circle of radius 15 centred at (30, 25).
        // Bookstein fit should recover centre within 0.5 px and radius within 1%.
        let ell = Ellipse2f {
            center: Point2f { x: 30.0, y: 25.0 },
            semi_axes: Vec2f { x: 15.0, y: 15.0 },
            angle: 0.0,
        };
        let pts = ellipse_points(&ell, 20);
        let conic = fit_bookstein(&pts).expect("bookstein fit");
        assert!(conic.is_ellipse(), "result should be ellipse");
        let recovered = conic.to_ellipse().expect("to_ellipse");
        assert!(
            (recovered.center.x - 30.0).abs() < 0.5,
            "cx={}",
            recovered.center.x
        );
        assert!(
            (recovered.center.y - 25.0).abs() < 0.5,
            "cy={}",
            recovered.center.y
        );
        assert!(
            (recovered.semi_major() - 15.0).abs() < 0.5,
            "a={}",
            recovered.semi_major()
        );
    }

    #[test]
    fn fitzgibbon_fits_rotated_ellipse() {
        // 20 exact points on an ellipse: a=20, b=8, angle=45°, centre=(60,40).
        // Fitzgibbon fit should recover axes within 0.3 px.
        let ell = Ellipse2f {
            center: Point2f { x: 60.0, y: 40.0 },
            semi_axes: Vec2f { x: 20.0, y: 8.0 },
            angle: PI / 4.0,
        };
        let pts = ellipse_points(&ell, 20);
        let conic = fit_fitzgibbon(&pts).expect("fitzgibbon fit");
        assert!(conic.is_ellipse(), "result should be ellipse");
        let recovered = conic.to_ellipse().expect("to_ellipse");
        assert!(
            (recovered.center.x - 60.0).abs() < 0.5,
            "cx={}",
            recovered.center.x
        );
        assert!(
            (recovered.center.y - 40.0).abs() < 0.5,
            "cy={}",
            recovered.center.y
        );
        assert!(
            (recovered.semi_major() - 20.0).abs() < 0.3,
            "a={}",
            recovered.semi_major()
        );
        assert!(
            (recovered.semi_minor() - 8.0).abs() < 0.3,
            "b={}",
            recovered.semi_minor()
        );
    }

    #[test]
    fn insufficient_data_returns_error() {
        let pts = vec![Point2f { x: 1.0, y: 2.0 }; 3];
        assert!(fit_bookstein(&pts).is_err());
        assert!(fit_fitzgibbon(&pts).is_err());
    }

    #[test]
    fn non_finite_points_report_degenerate_not_panic() {
        // A NaN coordinate propagates into the scatter matrix and makes every
        // eigenvalue NaN. `partial_cmp` returns None for NaN, so ranking them
        // with `.unwrap()` used to panic inside library code on ordinary bad
        // input rather than returning an error.
        let pts = vec![
            Point2f { x: 0.0, y: 0.0 },
            Point2f { x: 1.0, y: 0.0 },
            Point2f { x: 2.0, y: 1.0 },
            Point2f {
                x: f32::NAN,
                y: 2.0,
            },
            Point2f { x: 0.0, y: 3.0 },
            Point2f { x: -1.0, y: 1.0 },
        ];
        assert!(
            matches!(fit_bookstein(&pts), Err(Error::Degenerate(_))),
            "non-finite input must be reported as degenerate"
        );
        assert!(
            matches!(fit_fitzgibbon(&pts), Err(Error::Degenerate(_))),
            "non-finite input must be reported as degenerate"
        );
    }

    #[test]
    fn infinite_points_report_degenerate_not_panic() {
        let pts = vec![
            Point2f { x: 0.0, y: 0.0 },
            Point2f { x: 1.0, y: 0.0 },
            Point2f {
                x: f32::INFINITY,
                y: 1.0,
            },
            Point2f { x: 1.0, y: 2.0 },
            Point2f { x: 0.0, y: 3.0 },
            Point2f { x: -1.0, y: 1.0 },
        ];
        assert!(matches!(fit_bookstein(&pts), Err(Error::Degenerate(_))));
        assert!(matches!(fit_fitzgibbon(&pts), Err(Error::Degenerate(_))));
    }
}
