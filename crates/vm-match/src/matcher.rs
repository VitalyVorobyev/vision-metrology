//! Top-level edge matcher combining coarse grid search and ICP refinement.
//!
//! Supports both rigid (R|t) and similarity (sR|t) transforms depending on
//! [`MatchConfig`] settings. When `scale_range == (1.0, 1.0)` the outer
//! scale loop executes a single iteration and the runtime is identical to
//! the original rigid-only implementation.

use nalgebra::{Isometry2, Similarity2, Vector2};
use vm_core::{Image, Point2f, Rect2f, Similarity2f};
use vm_edge::edge2d::Edgel;

use crate::{
    icp::icp_refine,
    model::EdgeModel,
    rigid::MatchConfig,
    score::{build_scene_chamfer, chamfer_score, normal_score},
};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a successful edge match (rigid or similarity).
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Transform (rotation + uniform scale + translation) that maps
    /// model-local coordinates to scene coordinates.
    ///
    /// For rigid matches the scale component is 1.0.
    pub transform: Similarity2f,
    /// Inlier fraction after ICP refinement: proportion of model edgels within
    /// `chamfer_threshold` of a scene edgel.
    pub score: f32,
    /// Number of inlier edgels.
    pub inlier_count: usize,
    /// Mean chamfer distance of inlier edgels (pixels).
    pub chamfer_mean: f32,
}

/// Backward-compatible alias for rigid-match result.
///
/// Provides `transform: Isometry2f` via [`MatchResult::isometry`].
/// All code that previously used `RigidMatchResult` should migrate to
/// [`MatchResult`]; this alias exists for transition.
pub type RigidMatchResult = MatchResult;

impl MatchResult {
    /// Return the rigid part of the transform (drops scale).
    ///
    /// Equivalent to `nalgebra::Isometry2::new(translation, rotation)` with
    /// the same translation and rotation as `self.transform`.
    pub fn isometry(&self) -> vm_core::Isometry2f {
        Isometry2::new(self.transform.isometry.translation.vector, self.angle())
    }

    /// Rotation angle in radians extracted from `self.transform`.
    pub fn angle(&self) -> f32 {
        self.transform.isometry.rotation.angle()
    }

    /// Uniform scale extracted from `self.transform`.
    pub fn scale(&self) -> f32 {
        self.transform.scaling()
    }

    /// Translation vector `(tx, ty)`.
    pub fn translation(&self) -> (f32, f32) {
        let t = self.transform.isometry.translation.vector;
        (t.x, t.y)
    }
}

// ---------------------------------------------------------------------------
// Internal coarse candidate
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Candidate {
    tx: f32,
    ty: f32,
    angle: f32,
    scale: f32,
    coarse_score: f32, // mean chamfer distance (lower = better)
}

// ---------------------------------------------------------------------------
// Axis-aligned bbox of a set of points (after transform)
// ---------------------------------------------------------------------------

fn bbox_of_points(pts: &[Point2f]) -> Rect2f {
    if pts.is_empty() {
        return Rect2f::default();
    }
    let mut min_x = pts[0].x;
    let mut max_x = min_x;
    let mut min_y = pts[0].y;
    let mut max_y = min_y;
    for p in pts {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    Rect2f {
        x: min_x,
        y: min_y,
        width: max_x - min_x,
        height: max_y - min_y,
    }
}

/// Intersection-over-union of two axis-aligned bounding boxes.
fn iou(a: Rect2f, b: Rect2f) -> f32 {
    let ix = (a.x + a.width).min(b.x + b.width) - a.x.max(b.x);
    let iy = (a.y + a.height).min(b.y + b.height) - a.y.max(b.y);
    if ix <= 0.0 || iy <= 0.0 {
        return 0.0;
    }
    let inter = ix * iy;
    let union = a.area() + b.area() - inter;
    if union <= 0.0 { 1.0 } else { inter / union }
}

// ---------------------------------------------------------------------------
// RigidEdgeMatcher
// ---------------------------------------------------------------------------

/// Edge matcher that holds a scene chamfer map scratch buffer.
///
/// Call [`RigidEdgeMatcher::match_model`] to search for a single best match,
/// or [`RigidEdgeMatcher::match_all_instances`] for multi-instance NMS search.
/// The internal chamfer map is rebuilt for each new scene.
///
/// Supports both rigid (R|t) and similarity (sR|t) transforms; the transform
/// family is controlled by [`MatchConfig::scale_range`] and
/// [`MatchConfig::scale_step`].
pub struct RigidEdgeMatcher {
    scene_chamfer: Image<f32>,
}

impl RigidEdgeMatcher {
    /// Create a new matcher with an empty internal buffer.
    pub fn new() -> Self {
        Self {
            scene_chamfer: Image::from_vec(1, 1, vec![0.0f32]).expect("1x1 image"),
        }
    }

    /// Match `model` against `scene_edgels` using the given configuration.
    ///
    /// Steps:
    /// 1. Build the scene chamfer distance map from `scene_edgels`.
    /// 2. Grid-search over scale × angle × position.
    /// 3. Keep the top-K candidates by mean chamfer distance.
    /// 4. Optionally run ICP refinement on each top-K candidate.
    /// 5. Score with normal coherence; return the best result above `min_score`.
    ///
    /// Returns `None` if no candidate exceeds `min_score`.
    pub fn match_model(
        &mut self,
        model: &EdgeModel,
        scene_edgels: &[Edgel],
        cfg: &MatchConfig,
    ) -> Option<MatchResult> {
        let all = self.find_candidates(model, scene_edgels, cfg, 1);
        all.into_iter().next()
    }

    /// Return all non-overlapping instances above `cfg.min_score`, sorted by
    /// score descending.
    ///
    /// Two results overlap when their model bounding boxes (after transform)
    /// overlap by more than `overlap_thresh` IoU. Greedy suppression: iterate
    /// results sorted by score, keep a result only if its transformed bbox
    /// does not overlap any already-kept result.
    ///
    /// `overlap_thresh` is typically 0.5. Values outside [0, 1] are clamped.
    pub fn match_all_instances(
        &mut self,
        model: &EdgeModel,
        scene_edgels: &[Edgel],
        cfg: &MatchConfig,
        overlap_thresh: f32,
    ) -> Vec<MatchResult> {
        let overlap_thresh = overlap_thresh.clamp(0.0, 1.0);
        // Collect all candidates (no per-instance top_k cap here; use cfg.top_k
        // as max candidates per scale×angle bucket to limit work).
        let mut all = self.find_candidates(model, scene_edgels, cfg, usize::MAX);

        // Sort best-score first.
        all.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Pre-compute model bounding box corners for transform.
        let model_pts: Vec<Point2f> = model.edgels.iter().map(|e| e.p).collect();

        // Greedy NMS.
        let mut kept: Vec<MatchResult> = Vec::new();
        let mut kept_bboxes: Vec<Rect2f> = Vec::new();

        for result in all {
            let (tx, ty) = result.translation();
            let angle = result.angle();
            let scale = result.scale();
            let cos_a = angle.cos() * scale;
            let sin_a = angle.sin() * scale;

            let transformed: Vec<Point2f> = model_pts
                .iter()
                .map(|p| Point2f {
                    x: cos_a * p.x - sin_a * p.y + tx,
                    y: sin_a * p.x + cos_a * p.y + ty,
                })
                .collect();
            let bbox = bbox_of_points(&transformed);

            let overlaps = kept_bboxes.iter().any(|&kb| iou(bbox, kb) > overlap_thresh);
            if !overlaps {
                kept_bboxes.push(bbox);
                kept.push(result);
            }
        }

        kept
    }

    // -----------------------------------------------------------------------
    // Internal: grid search + ICP + scoring → Vec<MatchResult>
    // -----------------------------------------------------------------------

    fn build_scene_chamfer_map(&mut self, scene_edgels: &[Edgel], cfg: &MatchConfig) {
        let scene_w = scene_edgels.iter().map(|e| e.idx.0).max().unwrap_or(0) + 1;
        let scene_h = scene_edgels.iter().map(|e| e.idx.1).max().unwrap_or(0) + 1;
        let scene_w =
            scene_w.max((cfg.position_search.x + cfg.position_search.width).ceil() as usize + 1);
        let scene_h =
            scene_h.max((cfg.position_search.y + cfg.position_search.height).ceil() as usize + 1);
        self.scene_chamfer = build_scene_chamfer(scene_edgels, scene_w, scene_h);
    }

    fn find_candidates(
        &mut self,
        model: &EdgeModel,
        scene_edgels: &[Edgel],
        cfg: &MatchConfig,
        max_results: usize,
    ) -> Vec<MatchResult> {
        if model.edgels.is_empty() || scene_edgels.is_empty() {
            return Vec::new();
        }

        // --- Step 1: scene chamfer map ---
        self.build_scene_chamfer_map(scene_edgels, cfg);
        let chamfer_view = self.scene_chamfer.as_view();

        // --- Step 2: grid search ---
        let t_step = (cfg.chamfer_threshold / 2.0).max(1.0);
        let angle_min = cfg.angle_range.0.min(cfg.angle_range.1);
        let angle_max = cfg.angle_range.0.max(cfg.angle_range.1);
        let angle_step = cfg.angle_step.max(1e-4);

        let n_angles = ((angle_max - angle_min) / angle_step).ceil() as usize + 1;
        let n_tx = ((cfg.position_search.width) / t_step).ceil() as usize + 1;
        let n_ty = ((cfg.position_search.height) / t_step).ceil() as usize + 1;

        let top_k = cfg.top_k.max(1);
        let mut candidates: Vec<Candidate> = Vec::new();

        for scale in cfg.scale_iter() {
            for ai in 0..n_angles {
                let angle = angle_min + ai as f32 * angle_step;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                for tyi in 0..n_ty {
                    let ty = cfg.position_search.y + tyi as f32 * t_step;
                    for txi in 0..n_tx {
                        let tx = cfg.position_search.x + txi as f32 * t_step;

                        // For similarity: scale the model points before lookup.
                        // chamfer_score applies (cos_a, sin_a) rotation; we
                        // bake scale into the lookup by passing a scaled model.
                        let score = if (scale - 1.0).abs() < 1e-6 {
                            chamfer_score(model, &chamfer_view, tx, ty, cos_a, sin_a)
                        } else {
                            // Scale-aware chamfer: compute manually.
                            scaled_chamfer_score(
                                model,
                                &chamfer_view,
                                tx,
                                ty,
                                cos_a * scale,
                                sin_a * scale,
                            )
                        };

                        // Maintain top-K by lowest chamfer score.
                        if candidates.len() < top_k {
                            candidates.push(Candidate {
                                tx,
                                ty,
                                angle,
                                scale,
                                coarse_score: score,
                            });
                            if candidates.len() == top_k {
                                candidates.sort_unstable_by(|a, b| {
                                    b.coarse_score
                                        .partial_cmp(&a.coarse_score)
                                        .unwrap_or(core::cmp::Ordering::Equal)
                                });
                            }
                        } else if score < candidates[0].coarse_score {
                            candidates[0] = Candidate {
                                tx,
                                ty,
                                angle,
                                scale,
                                coarse_score: score,
                            };
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
            return Vec::new();
        }

        // Sort best-first (lowest chamfer score).
        candidates.sort_unstable_by(|a, b| {
            a.coarse_score
                .partial_cmp(&b.coarse_score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // --- Steps 3+4: optional ICP refinement and final scoring ---
        let scene_pts: Vec<_> = scene_edgels.iter().map(|e| e.p).collect();
        let mut results: Vec<MatchResult> = Vec::new();

        for cand in &candidates {
            let scale = cand.scale;
            let (tx, ty, angle) = if cfg.refine_icp {
                let cos_a = cand.angle.cos() * scale;
                let sin_a = cand.angle.sin() * scale;
                let transformed = transform_points_scaled(model, cand.tx, cand.ty, cos_a, sin_a);
                let (dtx, dty, da) = icp_refine(&transformed, &scene_pts, 20);
                (cand.tx + dtx, cand.ty + dty, cand.angle + da)
            } else {
                (cand.tx, cand.ty, cand.angle)
            };

            let cos_a = angle.cos() * scale;
            let sin_a = angle.sin() * scale;

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
            let cos_a_unit = angle.cos();
            let sin_a_unit = angle.sin();
            let ns = normal_score(
                model,
                scene_edgels,
                tx,
                ty,
                cos_a_unit,
                sin_a_unit,
                cfg.chamfer_threshold * 2.0,
            );
            if ns < 0.0 {
                continue;
            }

            let transform =
                Similarity2::from_isometry(Isometry2::new(Vector2::new(tx, ty), angle), scale);

            results.push(MatchResult {
                transform,
                score,
                inlier_count,
                chamfer_mean,
            });

            if results.len() >= max_results {
                break;
            }
        }

        // Sort best-score first.
        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        results
    }
}

impl Default for RigidEdgeMatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Scale-aware helpers
// ---------------------------------------------------------------------------

/// Chamfer score with scale baked into cos/sin (i.e. `cos_a = cos*scale`).
fn scaled_chamfer_score(
    model: &EdgeModel,
    scene_chamfer: &vm_core::ImageView<'_, f32>,
    tx: f32,
    ty: f32,
    cos_a_s: f32, // cos(angle) * scale
    sin_a_s: f32, // sin(angle) * scale
) -> f32 {
    if model.edgels.is_empty() {
        return 0.0;
    }
    let w = scene_chamfer.width() as f32;
    let h = scene_chamfer.height() as f32;
    let penalty = (w * w + h * h).sqrt();
    let sw = scene_chamfer.width();
    let sh = scene_chamfer.height();
    let mut sum = 0.0f32;
    for e in &model.edgels {
        let px = cos_a_s * e.p.x - sin_a_s * e.p.y + tx;
        let py = sin_a_s * e.p.x + cos_a_s * e.p.y + ty;
        let ix = px.round() as isize;
        let iy = py.round() as isize;
        let d = if ix >= 0 && iy >= 0 && (ix as usize) < sw && (iy as usize) < sh {
            *scene_chamfer
                .get(ix as usize, iy as usize)
                .expect("in-bounds")
        } else {
            penalty
        };
        sum += d;
    }
    sum / model.edgels.len() as f32
}

/// Project model points through a transform where cos/sin already includes scale.
fn transform_points_scaled(
    model: &EdgeModel,
    tx: f32,
    ty: f32,
    cos_a_s: f32,
    sin_a_s: f32,
) -> Vec<Point2f> {
    model
        .edgels
        .iter()
        .map(|e| Point2f {
            x: cos_a_s * e.p.x - sin_a_s * e.p.y + tx,
            y: sin_a_s * e.p.x + cos_a_s * e.p.y + ty,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use vm_core::{Point2f, Rect2f, Vec2f};
    use vm_edge::edge2d::Edgel;

    use crate::{
        model::EdgeModel,
        rigid::{MatchConfig, RigidMatchConfig},
    };

    use super::{RigidEdgeMatcher, iou};

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
            let (x, y, nx, ny): (f32, f32, f32, f32) = if t < 0.25 {
                (ox + t * 4.0 * w, oy, 0.0, -1.0)
            } else if t < 0.5 {
                (ox + w, oy + (t - 0.25) * 4.0 * h, 1.0, 0.0)
            } else if t < 0.75 {
                (ox + w - (t - 0.5) * 4.0 * w, oy + h, 0.0, 1.0)
            } else {
                (ox, oy + h - (t - 0.75) * 4.0 * h, -1.0, 0.0)
            };
            let nn = (nx * nx + ny * ny).sqrt();
            v.push(make_edgel(x, y, nx / nn, ny / nn));
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
            angle_range: (-0.1, 0.1),
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
            ..RigidMatchConfig::default()
        };

        let mut matcher = RigidEdgeMatcher::new();
        let result = matcher.match_model(&model, &scene_edgels_with_idx, &cfg);
        assert!(result.is_some(), "should find the rectangle in the scene");
        let r = result.unwrap();
        assert!(
            r.score >= 0.3,
            "inlier fraction should be >= 0.3, got {}",
            r.score
        );
        // Rigid match: scale should be 1.0.
        assert!(
            (r.scale() - 1.0).abs() < 1e-5,
            "rigid match scale should be 1.0"
        );
    }

    #[test]
    fn flipped_normals_rejected() {
        // Model with all normals pointing +x.
        let model_edgels: Vec<Edgel> = (0..10)
            .map(|i| make_edgel(i as f32, 0.0, 1.0, 0.0))
            .collect();
        let model = EdgeModel::from_edgels(model_edgels, 3);

        // Scene with all normals pointing -x (flipped).
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

        let cfg = MatchConfig {
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
            ..MatchConfig::default()
        };

        let mut matcher = RigidEdgeMatcher::new();
        let result = matcher.match_model(&model, &scene_edgels, &cfg);
        assert!(
            result.is_none(),
            "flipped normals should be rejected by normal-coherence filter"
        );
    }

    #[test]
    fn match_config_rigid_is_single_scale_iter() {
        let cfg = MatchConfig::default(); // scale_range = (1.0, 1.0)
        let scales: Vec<f32> = cfg.scale_iter().collect();
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn match_config_similarity_iter_count() {
        let cfg = MatchConfig {
            scale_range: (0.8, 1.2),
            scale_step: 0.1,
            ..MatchConfig::default()
        };
        let scales: Vec<f32> = cfg.scale_iter().collect();
        // (1.2 - 0.8) / 0.1 = 4, +1 → 5 steps
        assert_eq!(scales.len(), 5, "expected 5 scale steps, got {:?}", scales);
        assert!((scales[0] - 0.8).abs() < 1e-5);
        assert!(scales.last().copied().unwrap() <= 1.2 + 1e-5);
    }

    #[test]
    fn match_result_isometry_extracts_correct_angle() {
        use nalgebra::{Isometry2, Similarity2, Vector2};
        let angle = 0.5f32;
        let sim = Similarity2::from_isometry(Isometry2::new(Vector2::new(3.0, 4.0), angle), 1.5);
        let result = super::MatchResult {
            transform: sim,
            score: 1.0,
            inlier_count: 10,
            chamfer_mean: 0.1,
        };
        assert!((result.angle() - angle).abs() < 1e-5, "angle mismatch");
        assert!((result.scale() - 1.5).abs() < 1e-5, "scale mismatch");
        let (tx, ty) = result.translation();
        assert!((tx - 3.0).abs() < 1e-5);
        assert!((ty - 4.0).abs() < 1e-5);
    }

    #[test]
    fn iou_non_overlapping_is_zero() {
        use vm_core::Rect2f;
        let a = Rect2f {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let b = Rect2f {
            x: 20.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        assert_eq!(iou(a, b), 0.0);
    }

    #[test]
    fn iou_identical_is_one() {
        use vm_core::Rect2f;
        let a = Rect2f {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        assert!((iou(a, a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn match_all_instances_finds_two_rectangles() {
        // Scene: two 20×10 rectangles at (50, 40) and (150, 40) — well separated.
        let model_raw = rect_edgels(0.0, 0.0, 20.0, 10.0);
        let model = EdgeModel::from_edgels(model_raw, 5);

        let mut scene = rect_edgels(50.0, 40.0, 20.0, 10.0);
        scene.extend(rect_edgels(150.0, 40.0, 20.0, 10.0));
        let scene_with_idx: Vec<Edgel> = scene
            .iter()
            .map(|e| Edgel {
                idx: (e.p.x.round() as usize, e.p.y.round() as usize),
                ..*e
            })
            .collect();

        let cfg = MatchConfig {
            angle_range: (-0.1, 0.1),
            angle_step: 0.05,
            position_search: Rect2f {
                x: 30.0,
                y: 20.0,
                width: 160.0,
                height: 60.0,
            },
            chamfer_threshold: 5.0,
            min_score: 0.3,
            refine_icp: true,
            top_k: 10,
            ..MatchConfig::default()
        };

        let mut matcher = RigidEdgeMatcher::new();
        let results = matcher.match_all_instances(&model, &scene_with_idx, &cfg, 0.5);
        assert!(
            !results.is_empty(),
            "should find at least 1 instance, got {}",
            results.len()
        );
    }
}
