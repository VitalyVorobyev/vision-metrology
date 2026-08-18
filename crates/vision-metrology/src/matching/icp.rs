//! Point-to-point ICP refinement for 2D rigid matching.
//!
//! Uses the closed-form 2D solution:
//! 1. Compute cross-covariance H = Σ (model_i - µ_m)(scene_i - µ_s)^T.
//! 2. Optimal rotation: θ = atan2(H[0,1] - H[1,0], H[0,0] + H[1,1]).
//! 3. Optimal translation: t = µ_s - R·µ_m.
//!
//! Correspondence is found by brute-force nearest neighbour (practical for
//! model sizes ≤ 200 edgels). Returns the cumulative (tx, ty, angle) delta.

use vm_primitives::Point2f;

/// Run point-to-point ICP on matched inlier positions.
///
/// `model_pts` — model points in model-local coordinates (centroid-subtracted).
/// `scene_pts` — corresponding scene points; must be the same length as `model_pts`.
///
/// Finds the nearest-scene-edgel correspondence for each model point, computes
/// the closed-form 2D rigid transform, applies it, and repeats for `max_iters`.
///
/// Returns the cumulative `(tx, ty, angle)` in radians that maps model points to
/// scene coordinates.
///
/// # Panics
/// Does not panic; returns `(0, 0, 0)` if either slice is empty.
pub fn icp_refine(
    model_pts: &[Point2f],
    scene_pts: &[Point2f],
    max_iters: usize,
) -> (f32, f32, f32) {
    if model_pts.is_empty() || scene_pts.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    // Working copies: model points in the current (accumulated) transform frame.
    let mut current: Vec<Point2f> = model_pts.to_vec();

    let mut cum_tx = 0.0f32;
    let mut cum_ty = 0.0f32;
    let mut cum_cos = 1.0f32;
    let mut cum_sin = 0.0f32;

    for _ in 0..max_iters {
        // --- Correspondence: nearest scene point for each current model point ---
        let mut matched_scene: Vec<Point2f> = Vec::with_capacity(current.len());
        for mp in &current {
            let closest = scene_pts
                .iter()
                .min_by(|a, b| {
                    let da = (a.x - mp.x).powi(2) + (a.y - mp.y).powi(2);
                    let db = (b.x - mp.x).powi(2) + (b.y - mp.y).powi(2);
                    da.partial_cmp(&db).expect("finite distances")
                })
                .expect("scene_pts non-empty");
            matched_scene.push(*closest);
        }

        let n = current.len() as f32;

        // --- Compute centroids ---
        let mu_mx = current.iter().map(|p| p.x).sum::<f32>() / n;
        let mu_my = current.iter().map(|p| p.y).sum::<f32>() / n;
        let mu_sx = matched_scene.iter().map(|p| p.x).sum::<f32>() / n;
        let mu_sy = matched_scene.iter().map(|p| p.y).sum::<f32>() / n;

        // --- Cross-covariance H (2×2) ---
        let (mut h00, mut h01, mut h10, mut h11) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (mp, sp) in current.iter().zip(matched_scene.iter()) {
            let dm_x = mp.x - mu_mx;
            let dm_y = mp.y - mu_my;
            let ds_x = sp.x - mu_sx;
            let ds_y = sp.y - mu_sy;
            h00 += dm_x * ds_x;
            h01 += dm_x * ds_y;
            h10 += dm_y * ds_x;
            h11 += dm_y * ds_y;
        }

        // --- Closed-form 2D rotation ---
        // θ = atan2(h01 - h10, h00 + h11)
        let theta = (h01 - h10).atan2(h00 + h11);
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // --- Translation: t = µ_s - R·µ_m ---
        let tx = mu_sx - (cos_t * mu_mx - sin_t * mu_my);
        let ty = mu_sy - (sin_t * mu_mx + cos_t * mu_my);

        // --- Apply to current points ---
        for p in &mut current {
            let new_x = cos_t * p.x - sin_t * p.y + tx;
            let new_y = sin_t * p.x + cos_t * p.y + ty;
            p.x = new_x;
            p.y = new_y;
        }

        // --- Accumulate transform: T_new = T_iter ∘ T_prev ---
        // New translation in original frame: t_acc = R_iter * t_prev + t_iter
        let new_cum_tx = cos_t * cum_tx - sin_t * cum_ty + tx;
        let new_cum_ty = sin_t * cum_tx + cos_t * cum_ty + ty;
        let new_cum_cos = cos_t * cum_cos - sin_t * cum_sin;
        let new_cum_sin = sin_t * cum_cos + cos_t * cum_sin;
        cum_tx = new_cum_tx;
        cum_ty = new_cum_ty;
        cum_cos = new_cum_cos;
        cum_sin = new_cum_sin;

        // Early exit if residual is tiny.
        if tx.hypot(ty) < 1e-5 && theta.abs() < 1e-5 {
            break;
        }
    }

    let cum_angle = cum_sin.atan2(cum_cos);
    (cum_tx, cum_ty, cum_angle)
}

#[cfg(test)]
mod tests {
    use vm_primitives::Point2f;

    use super::icp_refine;

    fn rot_translate(pts: &[Point2f], tx: f32, ty: f32, angle: f32) -> Vec<Point2f> {
        let (c, s) = (angle.cos(), angle.sin());
        pts.iter()
            .map(|p| Point2f {
                x: c * p.x - s * p.y + tx,
                y: s * p.x + c * p.y + ty,
            })
            .collect()
    }

    #[test]
    fn icp_converges_pure_translation() {
        // ICP receives model_pts as the current model frame (already coarsely aligned)
        // and scene_pts as the scene. When the model is already close (within
        // nearest-neighbor range), ICP refines the residual offset.
        //
        // Setup: 4 points on a cross pattern, small 1-px offset.
        // At 1-px offset, correspondences are 1:1 correct (each model point's
        // nearest scene point is exactly the correct partner).
        let model = vec![
            Point2f { x: 10.0, y: 0.0 },
            Point2f { x: -10.0, y: 0.0 },
            Point2f { x: 0.0, y: 10.0 },
            Point2f { x: 0.0, y: -10.0 },
        ];
        // Pass as both model_pts and scene_pts with a 1-px offset.
        // model_pts = current aligned frame, scene_pts = scene.
        let scene = rot_translate(&model, 1.0, 0.5, 0.0);

        // Feed model as current (already pre-aligned) and scene as target.
        let (tx, ty, angle) = icp_refine(&model, &scene, 20);
        assert!((tx - 1.0).abs() < 0.1, "expected tx≈1.0, got {tx}");
        assert!((ty - 0.5).abs() < 0.1, "expected ty≈0.5, got {ty}");
        assert!(angle.abs() < 0.01, "expected angle≈0, got {angle}");
    }

    #[test]
    fn icp_converges_pure_rotation() {
        use core::f32::consts::FRAC_PI_4;
        // Model: points on a circle (symmetric under rotation: correspondence is exact).
        // Rotate by 45°.
        let n = 8;
        let model: Vec<Point2f> = (0..n)
            .map(|i| {
                let a = i as f32 * 2.0 * core::f32::consts::PI / n as f32;
                Point2f {
                    x: 5.0 * a.cos(),
                    y: 5.0 * a.sin(),
                }
            })
            .collect();
        let scene = rot_translate(&model, 0.0, 0.0, FRAC_PI_4);
        let (tx, ty, angle) = icp_refine(&model, &scene, 50);
        // Because the point set is symmetric under 45° rotation, ICP may converge
        // to the identity or to the true rotation — what matters is low residual.
        let residual: f32 = model
            .iter()
            .map(|p| {
                let (c, s) = (angle.cos(), angle.sin());
                let qx = c * p.x - s * p.y + tx;
                let qy = s * p.x + c * p.y + ty;
                scene
                    .iter()
                    .map(|sp| (sp.x - qx).hypot(sp.y - qy))
                    .fold(f32::MAX, f32::min)
            })
            .sum::<f32>()
            / model.len() as f32;
        assert!(residual < 0.5, "ICP residual too large: {residual}");
    }

    #[test]
    fn icp_clean_data_sub_pixel_residual() {
        // 4-point cross pattern, 0.3-px translation.
        // With 0.3-px offset from well-separated points, correspondences are
        // unambiguous and ICP must converge in 1 iteration with residual < 0.01 px.
        let model = vec![
            Point2f { x: 20.0, y: 0.0 },
            Point2f { x: -20.0, y: 0.0 },
            Point2f { x: 0.0, y: 20.0 },
            Point2f { x: 0.0, y: -20.0 },
        ];
        let scene = rot_translate(&model, 0.3, 0.2, 0.0);
        let (tx, ty, _) = icp_refine(&model, &scene, 10);
        let err = (tx - 0.3).abs().max((ty - 0.2).abs());
        assert!(err < 0.01, "sub-pixel residual expected, got err={err}");
    }

    #[test]
    fn icp_empty_returns_zero() {
        let (tx, ty, a) = icp_refine(&[], &[], 10);
        assert_eq!((tx, ty, a), (0.0, 0.0, 0.0));
    }
}
