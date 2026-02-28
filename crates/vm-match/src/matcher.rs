//! Top-level rigid edge matcher combining coarse grid search and ICP refinement.

use nalgebra::Isometry2;
use vm_core::{Image, Isometry2f};
use vm_edge::edge2d::Edgel;

use crate::{
    icp::icp_refine,
    model::EdgeModel,
    rigid::RigidMatchConfig,
    score::{build_scene_chamfer, chamfer_score, normal_score, transform_points},
};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a successful rigid edge match.
#[derive(Debug, Clone)]
pub struct RigidMatchResult {
    /// Rigid transform (rotation + translation) that maps model-local coordinates
    /// to scene coordinates.
    pub transform: Isometry2f,
    /// Inlier fraction after ICP refinement: proportion of model edgels within
    /// `chamfer_threshold` of a scene edgel.
    pub score: f32,
    /// Number of inlier edgels.
    pub inlier_count: usize,
    /// Mean chamfer distance of inlier edgels (pixels).
    pub chamfer_mean: f32,
}

// ---------------------------------------------------------------------------
// Coarse candidate
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Candidate {
    tx: f32,
    ty: f32,
    angle: f32,
    coarse_score: f32, // mean chamfer distance (lower = better)
}

// ---------------------------------------------------------------------------
// RigidEdgeMatcher
// ---------------------------------------------------------------------------

/// Rigid edge matcher that holds a scene chamfer map scratch buffer.
///
/// Call [`RigidEdgeMatcher::match_model`] to search for a model in a scene.
/// The internal chamfer map is rebuilt for each new scene.
pub struct RigidEdgeMatcher {
    scene_chamfer: Image<f32>,
}

impl RigidEdgeMatcher {
    /// Create a new matcher with an empty internal buffer.
    pub fn new() -> Self {
        Self {
            scene_chamfer: Image::from_vec(1, 1, vec![0.0f32]).expect("1×1 image"),
        }
    }

    /// Match `model` against `scene_edgels` using the given configuration.
    ///
    /// Steps:
    /// 1. Build the scene chamfer distance map from `scene_edgels`.
    /// 2. Grid-search over `angle_range / angle_step` and `position_search`
    ///    (translation step = `chamfer_threshold / 2`).
    /// 3. Keep the top-K candidates by mean chamfer distance.
    /// 4. Optionally run ICP refinement on each top-K candidate.
    /// 5. Score with normal coherence; return the best result above `min_score`.
    ///
    /// Returns `None` if no candidate exceeds `min_score`.
    pub fn match_model(
        &mut self,
        model: &EdgeModel,
        scene_edgels: &[Edgel],
        cfg: &RigidMatchConfig,
    ) -> Option<RigidMatchResult> {
        if model.edgels.is_empty() || scene_edgels.is_empty() {
            return None;
        }

        // --- Step 1: scene chamfer map ---
        // Determine scene bounding box from scene edgels.
        let scene_w = scene_edgels.iter().map(|e| e.idx.0).max().unwrap_or(0) + 1;
        let scene_h = scene_edgels.iter().map(|e| e.idx.1).max().unwrap_or(0) + 1;
        let scene_w =
            scene_w.max((cfg.position_search.x + cfg.position_search.width).ceil() as usize + 1);
        let scene_h =
            scene_h.max((cfg.position_search.y + cfg.position_search.height).ceil() as usize + 1);
        self.scene_chamfer = build_scene_chamfer(scene_edgels, scene_w, scene_h);
        let chamfer_view = self.scene_chamfer.as_view();

        // --- Step 2: grid search ---
        let t_step = (cfg.chamfer_threshold / 2.0).max(1.0);
        let angle_min = cfg.angle_range.0.min(cfg.angle_range.1);
        let angle_max = cfg.angle_range.0.max(cfg.angle_range.1);
        let angle_step = cfg.angle_step.max(1e-4);

        let n_angles = ((angle_max - angle_min) / angle_step).ceil() as usize + 1;
        let n_tx = ((cfg.position_search.width) / t_step).ceil() as usize + 1;
        let n_ty = ((cfg.position_search.height) / t_step).ceil() as usize + 1;

        let mut candidates: Vec<Candidate> = Vec::new();
        // Keep a fixed-capacity top-K heap (max-heap by coarse_score so we can evict worst).
        let top_k = cfg.top_k.max(1);

        for ai in 0..n_angles {
            let angle = angle_min + ai as f32 * angle_step;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            for tyi in 0..n_ty {
                let ty = cfg.position_search.y + tyi as f32 * t_step;
                for txi in 0..n_tx {
                    let tx = cfg.position_search.x + txi as f32 * t_step;

                    let score = chamfer_score(model, &chamfer_view, tx, ty, cos_a, sin_a);

                    // Maintain top-K by lowest chamfer score.
                    if candidates.len() < top_k {
                        candidates.push(Candidate {
                            tx,
                            ty,
                            angle,
                            coarse_score: score,
                        });
                        if candidates.len() == top_k {
                            // Sort descending by score (worst first for eviction).
                            candidates.sort_unstable_by(|a, b| {
                                b.coarse_score
                                    .partial_cmp(&a.coarse_score)
                                    .unwrap_or(core::cmp::Ordering::Equal)
                            });
                        }
                    } else {
                        // candidates[0] is the worst (highest chamfer).
                        if score < candidates[0].coarse_score {
                            candidates[0] = Candidate {
                                tx,
                                ty,
                                angle,
                                coarse_score: score,
                            };
                            // Re-sort to maintain max-heap invariant at index 0.
                            candidates.sort_unstable_by(|a, b| {
                                b.coarse_score
                                    .partial_cmp(&a.coarse_score)
                                    .unwrap_or(core::cmp::Ordering::Equal)
                            });
                        }
                    }
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Sort best-first (lowest chamfer score).
        candidates.sort_unstable_by(|a, b| {
            a.coarse_score
                .partial_cmp(&b.coarse_score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // --- Steps 3+4: optional ICP refinement and final scoring ---
        let mut best: Option<RigidMatchResult> = None;

        let _model_pts: Vec<_> = model.edgels.iter().map(|e| e.p).collect();
        let scene_pts: Vec<_> = scene_edgels.iter().map(|e| e.p).collect();

        for cand in &candidates {
            let (tx, ty, angle) = if cfg.refine_icp {
                // Transform model points to scene frame and refine.
                let cos_a = cand.angle.cos();
                let sin_a = cand.angle.sin();
                let transformed = transform_points(model, cand.tx, cand.ty, cos_a, sin_a);
                let (dtx, dty, da) = icp_refine(&transformed, &scene_pts, 20);
                (cand.tx + dtx, cand.ty + dty, cand.angle + da)
            } else {
                (cand.tx, cand.ty, cand.angle)
            };

            let cos_a = angle.cos();
            let sin_a = angle.sin();

            // Count inliers and compute mean chamfer.
            let mut inlier_count = 0usize;
            let mut chamfer_sum = 0.0f32;
            for e in &model.edgels {
                let px = cos_a * e.p.x - sin_a * e.p.y + tx;
                let py = sin_a * e.p.x + cos_a * e.p.y + ty;
                let ix = px.round() as isize;
                let iy = py.round() as isize;
                let sw = self.scene_chamfer.width();
                let sh = self.scene_chamfer.height();
                let d = if ix >= 0 && iy >= 0 && (ix as usize) < sw && (iy as usize) < sh {
                    *self
                        .scene_chamfer
                        .as_view()
                        .get(ix as usize, iy as usize)
                        .expect("in-bounds")
                } else {
                    f32::MAX
                };
                if d <= cfg.chamfer_threshold {
                    inlier_count += 1;
                    chamfer_sum += d;
                }
            }

            let score = inlier_count as f32 / model.edgels.len() as f32;
            if score < cfg.min_score {
                continue;
            }

            let chamfer_mean = if inlier_count > 0 {
                chamfer_sum / inlier_count as f32
            } else {
                f32::MAX
            };

            // Normal coherence filter: reject if score < 0 (flipped normals).
            let ns = normal_score(
                model,
                scene_edgels,
                tx,
                ty,
                cos_a,
                sin_a,
                cfg.chamfer_threshold * 2.0,
            );
            if ns < 0.0 {
                continue;
            }

            let transform = Isometry2::new(nalgebra::Vector2::new(tx, ty), angle);

            let result = RigidMatchResult {
                transform,
                score,
                inlier_count,
                chamfer_mean,
            };

            let is_better = match &best {
                None => true,
                Some(prev) => score > prev.score,
            };

            if is_better {
                best = Some(result);
            }
        }

        best
    }
}

impl Default for RigidEdgeMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use vm_core::{Point2f, Rect2f, Vec2f};
    use vm_edge::edge2d::Edgel;

    use crate::{model::EdgeModel, rigid::RigidMatchConfig};

    use super::RigidEdgeMatcher;

    fn make_edgel(x: f32, y: f32, nx: f32, ny: f32) -> Edgel {
        Edgel {
            p: Point2f { x, y },
            n: Vec2f { x: nx, y: ny },
            strength: 1.0,
            idx: (x as usize, y as usize),
        }
    }

    /// Build a rectangular ring of edgels (perimeter pixels of a w×h box at (ox, oy)).
    fn rect_edgels(ox: f32, oy: f32, w: f32, h: f32) -> Vec<Edgel> {
        let mut v = Vec::new();
        let steps = ((w + h) * 2.0) as usize;
        for i in 0..steps {
            let t = i as f32 / steps as f32;
            // parametric rectangle boundary
            let (x, y, nx, ny): (f32, f32, f32, f32) = if t < 0.25 {
                // top edge, left → right
                (ox + t * 4.0 * w, oy, 0.0, -1.0)
            } else if t < 0.5 {
                // right edge, top → bottom
                (ox + w, oy + (t - 0.25) * 4.0 * h, 1.0, 0.0)
            } else if t < 0.75 {
                // bottom edge, right → left
                (ox + w - (t - 0.5) * 4.0 * w, oy + h, 0.0, 1.0)
            } else {
                // left edge, bottom → top
                (ox, oy + h - (t - 0.75) * 4.0 * h, -1.0, 0.0)
            };
            let nx2 = (nx * nx + ny * ny).sqrt();
            v.push(make_edgel(x, y, nx / nx2, ny / nx2));
        }
        v
    }

    #[test]
    fn match_model_finds_translated_rectangle() {
        // Model: 20×10 rectangle centred at origin (will be centred by from_edgels).
        let model_raw = rect_edgels(0.0, 0.0, 20.0, 10.0);
        let model = EdgeModel::from_edgels(model_raw, 5);

        // Scene: same rectangle translated to (50, 40).
        let scene_edgels = rect_edgels(50.0, 40.0, 20.0, 10.0);
        let scene_edgels_with_idx: Vec<Edgel> = scene_edgels
            .iter()
            .map(|e| Edgel {
                p: e.p,
                n: e.n,
                strength: e.strength,
                idx: (e.p.x.round() as usize, e.p.y.round() as usize),
            })
            .collect();

        let cfg = RigidMatchConfig {
            angle_range: (-0.1, 0.1), // near-zero rotation search
            angle_step: 0.05,
            position_search: Rect2f {
                x: 30.0,
                y: 20.0,
                width: 50.0,
                height: 50.0,
            },
            chamfer_threshold: 5.0,
            min_score: 0.3,
            refine_icp: true,
            top_k: 5,
            resolution_factor: 1.0,
        };

        let mut matcher = RigidEdgeMatcher::new();
        let result = matcher.match_model(&model, &scene_edgels_with_idx, &cfg);
        assert!(result.is_some(), "should find the rectangle in the scene");
        let r = result.unwrap();
        assert!(
            r.score >= 0.3,
            "inlier fraction should be ≥ 0.3, got {}",
            r.score
        );
    }

    #[test]
    fn flipped_normals_rejected() {
        // Model with all normals pointing +x.
        let model_edgels: Vec<Edgel> = (0..10)
            .map(|i| make_edgel(i as f32, 0.0, 1.0, 0.0))
            .collect();
        let model = EdgeModel::from_edgels(model_edgels.clone(), 3);

        // Scene with all normals pointing −x (flipped).
        let scene_edgels: Vec<Edgel> = (0..10)
            .map(|i| Edgel {
                p: Point2f {
                    x: i as f32,
                    y: 0.0,
                },
                n: Vec2f { x: -1.0, y: 0.0 },
                strength: 1.0,
                idx: (i, 0),
            })
            .collect();

        let cfg = RigidMatchConfig {
            angle_range: (-0.01, 0.01),
            angle_step: 0.01,
            position_search: Rect2f {
                x: 0.0,
                y: 0.0,
                width: 5.0,
                height: 5.0,
            },
            chamfer_threshold: 3.0,
            min_score: 0.5,
            refine_icp: false,
            top_k: 3,
            resolution_factor: 1.0,
        };

        let mut matcher = RigidEdgeMatcher::new();
        let result = matcher.match_model(&model, &scene_edgels, &cfg);
        // Flipped normals → normal_score < 0 → rejected.
        assert!(
            result.is_none(),
            "flipped normals should be rejected by normal-coherence filter"
        );
    }
}
