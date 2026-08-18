//! Scoring functions for rigid edge matching.
//!
//! Two scoring modes:
//! 1. **Chamfer score** — mean chamfer distance of rotated+translated model edgels
//!    looked up in the scene's chamfer distance map. Lower is better; 0 = perfect.
//! 2. **Normal-coherence score** — mean dot product of rotated model normals against
//!    the nearest scene edgel's normal. Range [−1, 1]; 1 = perfect alignment,
//!    negative = inverted polarity (mirror-image false positive).

use vm_primitives::edge::edge2d::Edgel;
use vm_primitives::{ImageView, Point2f, Vec2f};

use super::model::EdgeModel;

// ---------------------------------------------------------------------------
// Chamfer score
// ---------------------------------------------------------------------------

/// Mean chamfer distance of model edgels projected into the scene chamfer map.
///
/// Applies rotation `(cos_a, sin_a)` and translation `(tx, ty)` to each model
/// edgel position and looks up its approximate distance in `scene_chamfer`.
/// Out-of-bounds positions contribute a penalty equal to the image diagonal.
///
/// Returns a non-negative mean distance in pixels. **Lower is better.**
///
/// # Arguments
/// - `model`: edge model in model-local coordinates.
/// - `scene_chamfer`: chamfer distance map computed from scene edgels.
/// - `tx`, `ty`: translation of the model centroid in scene coordinates.
/// - `cos_a`, `sin_a`: precomputed `cos(angle)` and `sin(angle)`.
pub fn chamfer_score(
    model: &EdgeModel,
    scene_chamfer: &ImageView<'_, f32>,
    tx: f32,
    ty: f32,
    cos_a: f32,
    sin_a: f32,
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
        // Rotate then translate.
        let px = cos_a * e.p.x - sin_a * e.p.y + tx;
        let py = sin_a * e.p.x + cos_a * e.p.y + ty;

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

// ---------------------------------------------------------------------------
// Normal-coherence score
// ---------------------------------------------------------------------------

/// Directed normal-coherence score.
///
/// For each model edgel (after rotation), finds the nearest scene edgel within
/// `max_dist` pixels (brute-force linear scan) and computes the dot product of
/// the rotated model normal with the scene edgel normal. Returns the mean dot
/// product over matched edgels, in the range `[-1, 1]`.
///
/// Returns `0.0` if no edgels are within `max_dist` or if either list is empty.
///
/// **Higher is better** (1 = perfectly aligned normals, −1 = opposite normals).
pub fn normal_score(
    model: &EdgeModel,
    scene_edgels: &[Edgel],
    tx: f32,
    ty: f32,
    cos_a: f32,
    sin_a: f32,
    max_dist: f32,
) -> f32 {
    if model.edgels.is_empty() || scene_edgels.is_empty() {
        return 0.0;
    }

    let max_dist2 = max_dist * max_dist;
    let mut total = 0.0f32;
    let mut count = 0usize;

    for e in &model.edgels {
        // Rotate model position and normal.
        let px = cos_a * e.p.x - sin_a * e.p.y + tx;
        let py = sin_a * e.p.x + cos_a * e.p.y + ty;
        let nx = cos_a * e.n.x - sin_a * e.n.y;
        let ny = sin_a * e.n.x + cos_a * e.n.y;
        let model_normal = Vec2f { x: nx, y: ny };

        // Find nearest scene edgel within max_dist.
        let mut best_dist2 = max_dist2 + 1.0;
        let mut best_dot = 0.0f32;

        for se in scene_edgels {
            let dx = se.p.x - px;
            let dy = se.p.y - py;
            let d2 = dx * dx + dy * dy;
            if d2 < best_dist2 {
                best_dist2 = d2;
                best_dot = model_normal.dot(se.n);
            }
        }

        if best_dist2 <= max_dist2 {
            total += best_dot;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

// ---------------------------------------------------------------------------
// Helper: build scene chamfer map from edgels
// ---------------------------------------------------------------------------

/// Rasterise `scene_edgels` into a binary mask of size `w × h` and compute
/// the chamfer distance map.
///
/// Returns `Image<f32>` where each pixel is the approximate Euclidean distance
/// to the nearest scene edgel. Pixels with no edgels get a large distance.
pub fn build_scene_chamfer(
    scene_edgels: &[Edgel],
    w: usize,
    h: usize,
) -> vm_primitives::Image<f32> {
    use vm_primitives::chamfer_distance_u8;

    let mut mask = vec![0u8; w * h];
    for e in scene_edgels {
        let x = e.idx.0;
        let y = e.idx.1;
        if x < w && y < h {
            mask[y * w + x] = 255;
        }
    }
    let mask_img = vm_primitives::Image::from_vec(w, h, mask).expect("valid dims");
    chamfer_distance_u8(&mask_img.as_view())
}

/// Project a model edgel set (model-local coordinates) through a rigid transform
/// and return the set of scene-space positions.
pub fn transform_points(
    model: &EdgeModel,
    tx: f32,
    ty: f32,
    cos_a: f32,
    sin_a: f32,
) -> Vec<Point2f> {
    model
        .edgels
        .iter()
        .map(|e| Point2f {
            x: cos_a * e.p.x - sin_a * e.p.y + tx,
            y: sin_a * e.p.x + cos_a * e.p.y + ty,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use vm_primitives::edge::edge2d::Edgel;
    use vm_primitives::{Point2f, Vec2f};

    use crate::matching::model::EdgeModel;

    use super::{build_scene_chamfer, chamfer_score, normal_score};

    fn make_edgel(x: f32, y: f32, nx: f32, ny: f32) -> Edgel {
        Edgel {
            p: Point2f { x, y },
            n: Vec2f { x: nx, y: ny },
            strength: 1.0,
            idx: (x as usize, y as usize),
        }
    }

    #[test]
    fn chamfer_score_zero_at_exact_position() {
        // Model is a single edgel at the origin (after centring).
        // Scene chamfer map has foreground at (10, 10).
        // Place the model at (10, 10) with no rotation → chamfer distance ≈ 0.
        let scene_edgels = vec![make_edgel(10.0, 10.0, 1.0, 0.0)];
        let chamfer = build_scene_chamfer(&scene_edgels, 20, 20);

        let model_edgels = vec![make_edgel(0.0, 0.0, 1.0, 0.0)]; // centred at origin
        let model = EdgeModel::from_edgels(model_edgels, 3);

        let score = chamfer_score(&model, &chamfer.as_view(), 10.0, 10.0, 1.0, 0.0);
        assert!(
            score < 0.5,
            "chamfer score at exact position should be ~0, got {score}"
        );
    }

    #[test]
    fn chamfer_score_increases_with_offset() {
        let scene_edgels = vec![make_edgel(10.0, 10.0, 1.0, 0.0)];
        let chamfer = build_scene_chamfer(&scene_edgels, 30, 30);

        let model_edgels = vec![make_edgel(0.0, 0.0, 1.0, 0.0)];
        let model = EdgeModel::from_edgels(model_edgels, 3);

        let s0 = chamfer_score(&model, &chamfer.as_view(), 10.0, 10.0, 1.0, 0.0);
        let s5 = chamfer_score(&model, &chamfer.as_view(), 15.0, 10.0, 1.0, 0.0);
        assert!(
            s5 > s0,
            "chamfer score should increase as model moves away from scene edgel"
        );
    }

    #[test]
    fn normal_score_aligned_normals_positive() {
        // Model and scene edgel at same position with aligned normals (1, 0).
        let model_edgels = vec![make_edgel(0.0, 0.0, 1.0, 0.0)];
        let model = EdgeModel::from_edgels(model_edgels, 2);
        let scene_edgels = vec![make_edgel(0.0, 0.0, 1.0, 0.0)]; // tx=0, ty=0 in scene

        let score = normal_score(&model, &scene_edgels, 0.0, 0.0, 1.0, 0.0, 5.0);
        assert!(
            score > 0.9,
            "aligned normals: normal score should be ~1, got {score}"
        );
    }

    #[test]
    fn normal_score_flipped_normals_negative() {
        // Flipped model normal (-1, 0) vs scene normal (1, 0): dot product = -1.
        let model_edgels = vec![make_edgel(0.0, 0.0, -1.0, 0.0)];
        let model = EdgeModel::from_edgels(model_edgels, 2);
        let scene_edgels = vec![make_edgel(0.0, 0.0, 1.0, 0.0)];

        let score = normal_score(&model, &scene_edgels, 0.0, 0.0, 1.0, 0.0, 5.0);
        assert!(
            score < -0.9,
            "flipped normals: normal score should be ~-1, got {score}"
        );
    }

    #[test]
    fn normal_score_out_of_range_returns_zero() {
        // Scene edgel far away (> max_dist): no correspondence found → score = 0.
        let model_edgels = vec![make_edgel(0.0, 0.0, 1.0, 0.0)];
        let model = EdgeModel::from_edgels(model_edgels, 2);
        let scene_edgels = vec![make_edgel(100.0, 100.0, 1.0, 0.0)];

        let score = normal_score(&model, &scene_edgels, 0.0, 0.0, 1.0, 0.0, 5.0);
        assert_eq!(score, 0.0, "no correspondence within max_dist → 0");
    }
}
