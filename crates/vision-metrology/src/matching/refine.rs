//! Subpixel pose refinement.

use nalgebra::{Matrix2, Matrix3, Matrix4, Vector2, Vector3, Vector4};
use vm_primitives::{DirectionField, parabolic_peak_offset};

use super::config::Polarity;
use super::model::ModelPoint;
use super::score::{Bound, RotPoint, rotate_into, score_at};

/// A pose in one pyramid level's coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pose {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub scale: f32,
}

/// Fit a quadratic to the score surface around the grid maximum.
///
/// The 2-D fit uses all nine samples of the 3×3 neighbourhood rather than two
/// independent 1-D parabolas, which matters whenever the score ridge is
/// elongated — a long straight edge gives exactly that. Angle and scale are
/// refined by 1-D parabolas through the neighbouring steps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn interpolate(
    field: &DirectionField,
    points: &[ModelPoint],
    start: Pose,
    angle_step: f32,
    scale_step: f32,
    polarity: Polarity,
    rot: &mut Vec<RotPoint>,
) -> Pose {
    let (qx, qy) = (start.x.round() as i32, start.y.round() as i32);
    rotate_into(points, start.angle, start.scale, rot);

    let mut v = [[0.0f32; 3]; 3];
    for (j, row) in v.iter_mut().enumerate() {
        for (i, cell) in row.iter_mut().enumerate() {
            *cell = score_at(
                field,
                rot,
                qx + i as i32 - 1,
                qy + j as i32 - 1,
                polarity,
                Bound::never(),
            )
            .unwrap_or(f32::NEG_INFINITY);
        }
    }
    let (dx, dy) = quadratic_vertex(&v);

    // Angle and scale each get their own 1-D parabola at the grid position,
    // which keeps them independent of the translation fit above.
    let centre = v[1][1];
    let angle = start.angle
        + angle_step
            * axis_offset(field, points, start, polarity, rot, centre, |p, t| Pose {
                angle: p.angle + t * angle_step,
                ..p
            });
    let scale_rel = axis_offset(field, points, start, polarity, rot, centre, |p, t| Pose {
        scale: p.scale * (1.0 + t * scale_step),
        ..p
    });

    Pose {
        x: start.x + dx,
        y: start.y + dy,
        angle,
        scale: start.scale * (1.0 + scale_rel * scale_step),
    }
}

/// Parabolic vertex along one pose axis, sampled at ±1 step.
fn axis_offset(
    field: &DirectionField,
    points: &[ModelPoint],
    start: Pose,
    polarity: Polarity,
    rot: &mut Vec<RotPoint>,
    centre: f32,
    at: impl Fn(Pose, f32) -> Pose,
) -> f32 {
    let (qx, qy) = (start.x.round() as i32, start.y.round() as i32);
    let mut sample = |t: f32| {
        let p = at(start, t);
        rotate_into(points, p.angle, p.scale, rot);
        score_at(field, rot, qx, qy, polarity, Bound::never()).unwrap_or(f32::NEG_INFINITY)
    };
    let minus = sample(-1.0);
    let plus = sample(1.0);
    if !minus.is_finite() || !plus.is_finite() {
        return 0.0;
    }
    parabolic_peak_offset(minus, centre, plus)
        .filter(|t| t.abs() <= 1.0)
        .unwrap_or(0.0)
}

/// Least-squares vertex of a 3×3 score patch, falling back to two 1-D
/// parabolas when the fitted surface is not a well-formed maximum.
fn quadratic_vertex(v: &[[f32; 3]; 3]) -> (f32, f32) {
    if v.iter().flatten().any(|s| !s.is_finite()) {
        return (0.0, 0.0);
    }
    let col = |i: usize| v[0][i] + v[1][i] + v[2][i];
    let row = |j: usize| v[j][0] + v[j][1] + v[j][2];

    let c1 = (col(2) - col(0)) / 6.0;
    let c2 = (row(2) - row(0)) / 6.0;
    let c3 = (col(2) + col(0) - 2.0 * col(1)) / 6.0;
    let c4 = (row(2) + row(0) - 2.0 * row(1)) / 6.0;
    let c5 = (v[2][2] - v[2][0] - v[0][2] + v[0][0]) / 4.0;

    let hess = Matrix2::new(2.0 * c3, c5, c5, 2.0 * c4);
    let det = hess.determinant();
    // Negative-definite Hessian == the fit really is a maximum.
    if c3 < 0.0 && c4 < 0.0 && det > 1e-12 {
        let sol = hess
            .try_inverse()
            .map(|inv| inv * Vector2::new(-c1, -c2))
            .unwrap_or_else(Vector2::zeros);
        if sol.x.abs() <= 1.0 && sol.y.abs() <= 1.0 && sol.x.is_finite() && sol.y.is_finite() {
            return (sol.x, sol.y);
        }
    }

    let dx = parabolic_peak_offset(v[1][0], v[1][1], v[1][2])
        .filter(|t| t.abs() <= 1.0)
        .unwrap_or(0.0);
    let dy = parabolic_peak_offset(v[0][1], v[1][1], v[2][1])
        .filter(|t| t.abs() <= 1.0)
        .unwrap_or(0.0);
    (dx, dy)
}

/// Cosine of the maximum angle between a model normal and the scene gradient
/// for the sample to be trusted (30°).
const MIN_DIR_COS: f32 = 0.866;
/// Huber transition, in pixels.
const HUBER_K: f32 = 0.5;
/// Largest per-iteration correction accepted before the step is treated as a
/// divergence and discarded.
const MAX_STEP_PX: f64 = 3.0;
const MAX_STEP_ANG: f64 = 0.3;

/// Correspondence-free similarity refinement.
///
/// For each model point the gradient magnitude is sampled at `p − n`, `p` and
/// `p + n` and a parabola gives the signed distance to the true edge along the
/// normal. That validity gate — contrast, a well-formed peak, an offset inside
/// the bracket, and a direction within 30° — replaces nearest-neighbour search
/// entirely, so the cost is `O(|model|)` with no scene edgel list and no
/// per-iteration allocation.
///
/// Only the normal component of the displacement is observable (moving along an
/// edge changes nothing), so each point gives one equation:
/// `aᵢᵀ x = δᵢ` with `aᵢ = (nₓ, n_y, cross(vᵢ, nᵢ), dot(vᵢ, nᵢ))` and
/// `x = (Δtₓ, Δt_y, ω, λ)`.
pub(crate) fn least_squares(
    field: &DirectionField,
    points: &[ModelPoint],
    start: Pose,
    radius: f32,
    min_contrast: f32,
    polarity: Polarity,
) -> Pose {
    let mut pose = start;
    let r0 = radius.max(1.0);

    // Two outer iterations are enough from a sub-pixel start; each runs three
    // IRLS re-weightings.
    for _ in 0..2 {
        let mut scale_w = 1.0f32;
        for _ in 0..3 {
            let (sn, cs) = pose.angle.sin_cos();
            let mut ata = Matrix4::<f64>::zeros();
            let mut atb = Vector4::<f64>::zeros();
            let mut used = 0usize;

            for p in points {
                let wx = pose.scale * (cs * p.d.x - sn * p.d.y);
                let wy = pose.scale * (sn * p.d.x + cs * p.d.y);
                let (px, py) = (pose.x + wx, pose.y + wy);
                let nx = cs * p.t.x - sn * p.t.y;
                let ny = sn * p.t.x + cs * p.t.y;

                let m0 = field.sample_mag(px, py);
                if m0 < min_contrast || m0 <= 0.0 {
                    continue;
                }
                let mp = field.sample_mag(px + nx, py + ny);
                let mm = field.sample_mag(px - nx, py - ny);
                if mm - 2.0 * m0 + mp >= 0.0 {
                    continue; // not a ridge along the normal
                }
                let Some(delta) = parabolic_peak_offset(mm, m0, mp) else {
                    continue;
                };
                // `parabolic_peak_offset` only returns finite values, so a plain
                // comparison is enough here.
                if delta.abs() >= 1.0 {
                    continue;
                }

                let ex = px + delta * nx;
                let ey = py + delta * ny;
                let (gx, gy) =
                    field.dir_at(ex.round().max(0.0) as usize, ey.round().max(0.0) as usize);
                let dot = gx * nx + gy * ny;
                let agree = if polarity == Polarity::Match {
                    dot
                } else {
                    dot.abs()
                };
                if agree < MIN_DIR_COS {
                    continue;
                }

                let (vx, vy) = (wx / r0, wy / r0);
                let a = [
                    f64::from(nx),
                    f64::from(ny),
                    f64::from(vx * ny - vy * nx),
                    f64::from(vx * nx + vy * ny),
                ];
                let huber = if delta.abs() <= HUBER_K * scale_w {
                    1.0
                } else {
                    HUBER_K * scale_w / delta.abs()
                };
                let w = f64::from(agree * huber);
                for (i, &ai) in a.iter().enumerate() {
                    atb[i] += w * ai * f64::from(delta);
                    for (j, &aj) in a.iter().enumerate() {
                        ata[(i, j)] += w * ai * aj;
                    }
                }
                used += 1;
            }

            if used < 6 {
                return pose;
            }
            let Some(x) = solve_graceful(&ata, &atb) else {
                return pose;
            };
            if x[0].abs() > MAX_STEP_PX
                || x[1].abs() > MAX_STEP_PX
                || x[2].abs() > MAX_STEP_ANG * f64::from(r0)
                || x[3].abs() > MAX_STEP_ANG * f64::from(r0)
            {
                return pose;
            }

            pose.x += x[0] as f32;
            pose.y += x[1] as f32;
            pose.angle += (x[2] / f64::from(r0)) as f32;
            pose.scale *= 1.0 + (x[3] / f64::from(r0)) as f32;
            scale_w *= 0.7;
        }
    }
    pose
}

/// Solve the normal equations, dropping degrees of freedom that the data does
/// not constrain.
///
/// Rank deficiency here is normal, not exceptional: a circular part has
/// `cross(vᵢ, nᵢ) = 0` at every point because its normals are radial, so its
/// rotation is genuinely unobservable — a circle has no orientation. The
/// fallback chain solves for `(Δt, ω, λ)`, then `(Δt, ω)`, then `Δt` alone, and
/// finally gives up rather than inventing a pose.
fn solve_graceful(ata: &Matrix4<f64>, atb: &Vector4<f64>) -> Option<Vector4<f64>> {
    if let Some(ch) = ata.cholesky() {
        let x = ch.solve(atb);
        if x.iter().all(|v| v.is_finite()) {
            return Some(x);
        }
    }
    let m3 = Matrix3::from_fn(|i, j| ata[(i, j)]);
    if let Some(ch) = m3.cholesky() {
        let x = ch.solve(&Vector3::new(atb[0], atb[1], atb[2]));
        if x.iter().all(|v| v.is_finite()) {
            return Some(Vector4::new(x[0], x[1], x[2], 0.0));
        }
    }
    let m2 = Matrix2::from_fn(|i, j| ata[(i, j)]);
    let ch = m2.cholesky()?;
    let x = ch.solve(&Vector2::new(atb[0], atb[1]));
    x.iter()
        .all(|v| v.is_finite())
        .then(|| Vector4::new(x[0], x[1], 0.0, 0.0))
}

#[cfg(test)]
mod tests {
    use super::{Matrix4, Vector4, quadratic_vertex, solve_graceful};

    #[test]
    fn quadratic_vertex_recovers_a_known_offset() {
        // f(x, y) = -( (x - 0.3)^2 + (y + 0.2)^2 ) sampled on the 3x3 grid.
        let f = |x: f32, y: f32| -((x - 0.3) * (x - 0.3) + (y + 0.2) * (y + 0.2));
        let mut v = [[0.0f32; 3]; 3];
        for (j, row) in v.iter_mut().enumerate() {
            for (i, cell) in row.iter_mut().enumerate() {
                *cell = f(i as f32 - 1.0, j as f32 - 1.0);
            }
        }
        let (dx, dy) = quadratic_vertex(&v);
        assert!((dx - 0.3).abs() < 1e-4, "dx = {dx}");
        assert!((dy + 0.2).abs() < 1e-4, "dy = {dy}");
    }

    #[test]
    fn quadratic_vertex_refuses_a_saddle() {
        // A saddle is not a maximum; the fallback 1-D parabolas also find no
        // interior vertex here, so the offset stays at the grid position.
        let f = |x: f32, y: f32| x * x - y * y;
        let mut v = [[0.0f32; 3]; 3];
        for (j, row) in v.iter_mut().enumerate() {
            for (i, cell) in row.iter_mut().enumerate() {
                *cell = f(i as f32 - 1.0, j as f32 - 1.0);
            }
        }
        let (dx, dy) = quadratic_vertex(&v);
        assert!(dx.abs() < 1e-6 && dy.abs() < 1e-6, "({dx}, {dy})");
    }

    #[test]
    fn solve_graceful_drops_unobservable_degrees_of_freedom() {
        // Only the first two parameters are constrained, as for a shape whose
        // rotation and scale leave the score unchanged.
        let mut ata = Matrix4::<f64>::zeros();
        ata[(0, 0)] = 4.0;
        ata[(1, 1)] = 9.0;
        let atb = Vector4::new(8.0, 27.0, 5.0, -3.0);

        let x = solve_graceful(&ata, &atb).expect("2-DOF fallback");
        assert!((x[0] - 2.0).abs() < 1e-9);
        assert!((x[1] - 3.0).abs() < 1e-9);
        assert_eq!(x[2], 0.0, "rotation must be reported as unmoved, not NaN");
        assert_eq!(x[3], 0.0);
    }

    #[test]
    fn solve_graceful_uses_the_full_system_when_it_is_well_posed() {
        let ata = Matrix4::<f64>::from_diagonal(&Vector4::new(1.0, 2.0, 4.0, 8.0));
        let atb = Vector4::new(1.0, 2.0, 4.0, 8.0);
        let x = solve_graceful(&ata, &atb).expect("full rank");
        for v in x.iter() {
            assert!((v - 1.0).abs() < 1e-9);
        }
    }
}
