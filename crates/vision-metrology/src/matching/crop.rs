//! Rectifying a [`ShapeMatch`] into a canonical, model-frame crop.
//!
//! `warp` moves pixels; `matching` says where the part is. [`CropSpec`] +
//! [`ShapeMatch::model_frame_map`] is the seam between them: a fixed
//! rectangle in the model's own reference-image frame, resampled to
//! [`CropSpec::output_size`] pixels through the found pose. The point is
//! **identical tensor shapes across frames** — an anomaly model trained on
//! rectified crops needs every crop to be the same `w × h`, whatever pose the
//! part was found at, which is why the output size is a property of the spec
//! alone and never of the match.

use vm_primitives::{Point2f, Rect2f, Similarity2f, Vec2f};

use super::matcher::ShapeMatch;
use crate::warp::Map;

/// Fixed crop geometry for [`ShapeMatch::model_frame_map`].
///
/// Read this once, reuse it for every frame: the same `CropSpec` fixes both
/// which part of the model is rectified and the exact pixel size of the
/// result, so crops from different matches — different frames, different
/// found poses — are directly comparable pixel-for-pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropSpec {
    /// Crop rectangle in **model-frame coordinates** — the same
    /// reference-image frame [`ShapeModel::reference_points`](super::ShapeModel::reference_points)
    /// reports and [`ShapeMatch::pose`] consumes directly (`pose * p` for a
    /// reference-image point `p`, no origin bookkeeping needed).
    pub rect: Rect2f,
    /// Destination pixels per one model-frame unit (pixel). `1.0` keeps the
    /// crop at native reference-image density; `2.0` doubles it.
    pub px_per_unit: f32,
    /// `true` (default): render the crop at **model scale** — the found
    /// pose's scale factor is dropped, so a part found 20% larger than
    /// taught still fills the crop exactly as the untouched reference would.
    /// `false`: render at the **found** scale, faithfully reproducing what
    /// was actually seen.
    ///
    /// This only changes *content*; [`CropSpec::output_size`] — and
    /// therefore the crop's pixel dimensions — is the same either way.
    pub normalize_scale: bool,
}

impl CropSpec {
    /// A spec over `rect` at `px_per_unit` resolution, canonical
    /// (`normalize_scale = true`) by default.
    #[inline]
    pub fn new(rect: Rect2f, px_per_unit: f32) -> Self {
        Self {
            rect,
            px_per_unit,
            normalize_scale: true,
        }
    }

    /// Output `(width, height)` in pixels, fixed by the spec alone —
    /// `(rect.width, rect.height) * px_per_unit`, rounded and floored at one
    /// pixel per side. Identical for every match rectified with this spec,
    /// which is the whole point (see [`CropSpec`]'s own docs).
    #[inline]
    pub fn output_size(&self) -> (usize, usize) {
        let w = (self.rect.width * self.px_per_unit).round().max(1.0) as usize;
        let h = (self.rect.height * self.px_per_unit).round().max(1.0) as usize;
        (w, h)
    }
}

impl ShapeMatch {
    /// The `dst → scene` [`Similarity2f`] [`model_frame_map`](Self::model_frame_map)
    /// uses to place a model-frame point into the scene.
    ///
    /// `spec.normalize_scale` decides whether the found scale factor
    /// participates: `true` keeps this match's rotation and translation but
    /// forces scale to `1.0` (canonical); `false` returns [`Self::pose`]
    /// unchanged (found scale). Recovering this transform is what lets a
    /// downstream detection made on the rectified crop be mapped back into
    /// scene coordinates: `scene_pt = model_frame_pose(spec) * crop_model_pt`.
    #[inline]
    pub fn model_frame_pose(&self, spec: &CropSpec) -> Similarity2f {
        if spec.normalize_scale {
            Similarity2f::from_isometry(self.pose.isometry, 1.0)
        } else {
            self.pose
        }
    }

    /// Build the `dst → src` [`Map`] that rectifies this match into a
    /// canonical, model-frame crop.
    ///
    /// Destination pixel `(x, y)` (out of `spec.output_size()`) samples the
    /// scene at:
    ///
    /// ```text
    /// model_pt = spec.rect.min + (x, y) / spec.px_per_unit   // model frame
    /// scene_pt = model_frame_pose(spec) * model_pt           // scene frame
    /// ```
    ///
    /// Apply the returned map with [`Map::apply_with_mask`] and
    /// `BorderMode::Constant` (**not** the crate's usual `Clamp` default) —
    /// a crop pixel that falls outside the scene has no real content, and
    /// `Clamp` would fabricate texture by repeating the scene's edge pixel,
    /// which an anomaly model trained on these crops would then learn as
    /// normal. Read the mask: `0` marks a pixel the scene never covered, not
    /// image content, and must not feed a downstream measurement or model
    /// without being told apart from real data.
    pub fn model_frame_map(&self, spec: &CropSpec) -> Map {
        let (w, h) = spec.output_size();
        let sim = self.model_frame_pose(spec);
        let origin = Point2f::new(spec.rect.x, spec.rect.y);
        let inv_scale = 1.0 / spec.px_per_unit;
        Map::from_fn(w, h, move |x, y| {
            let model_pt = origin + Vec2f::new(x * inv_scale, y * inv_scale);
            let scene_pt = sim * model_pt;
            (scene_pt.x, scene_pt.y)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vm_primitives::similarity_from_parts;

    fn match_at(position: Point2f, angle: f32, scale: f32) -> ShapeMatch {
        // Model origin at (0, 0) so `pose` reduces to a plain similarity —
        // enough to test the crop geometry in isolation from `ShapeModel`.
        let pose = similarity_from_parts(Vec2f::new(position.x, position.y), angle, scale);
        ShapeMatch {
            pose,
            position,
            score: 1.0,
            support: 0,
            level: 0,
        }
    }

    #[test]
    fn output_size_depends_only_on_spec() {
        let rect = Rect2f {
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 15.0,
        };
        let spec = CropSpec::new(rect, 2.0);
        assert_eq!(spec.output_size(), (60, 30));

        // Independent of any match: two matches at wildly different poses
        // still produce the same map dimensions.
        let m1 = match_at(Point2f::new(0.0, 0.0), 0.0, 1.0);
        let m2 = match_at(Point2f::new(500.0, -300.0), 1.2, 1.8);
        assert_eq!(
            (
                m1.model_frame_map(&spec).width(),
                m1.model_frame_map(&spec).height()
            ),
            spec.output_size()
        );
        assert_eq!(
            (
                m2.model_frame_map(&spec).width(),
                m2.model_frame_map(&spec).height()
            ),
            spec.output_size()
        );
    }

    #[test]
    fn identity_pose_maps_crop_pixels_straight_to_rect() {
        let rect = Rect2f {
            x: 5.0,
            y: 5.0,
            width: 4.0,
            height: 4.0,
        };
        let spec = CropSpec::new(rect, 1.0);
        let m = match_at(Point2f::new(0.0, 0.0), 0.0, 1.0);

        let map = m.model_frame_map(&spec);
        // dst (0,0) -> model (5,5) -> scene (5,5) under identity pose.
        let mut dst = vec![0.0f32; 16];
        let src =
            vm_primitives::Image::from_vec(16, 16, (0..256).map(|v| v as f32).collect()).unwrap();
        map.apply(
            &src.as_view(),
            &mut dst,
            crate::warp::Interp::Nearest,
            vm_primitives::BorderMode::Constant(0.0),
        )
        .unwrap();
        // src(5,5) = 5*16 + 5 = 85.
        assert_eq!(dst[0], 85.0);
    }

    #[test]
    fn normalize_scale_drops_found_scale_but_keeps_pose() {
        let rect = Rect2f {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        };
        let spec_norm = CropSpec::new(rect, 1.0);
        let mut spec_found = spec_norm;
        spec_found.normalize_scale = false;

        let m = match_at(Point2f::new(10.0, 20.0), 0.0, 2.0);

        let normed = m.model_frame_pose(&spec_norm);
        assert!((normed.scaling() - 1.0).abs() < 1e-6);
        assert!((normed.isometry.translation.vector.x - 10.0).abs() < 1e-6);
        assert!((normed.isometry.translation.vector.y - 20.0).abs() < 1e-6);

        let found = m.model_frame_pose(&spec_found);
        assert!((found.scaling() - 2.0).abs() < 1e-6);
        assert_eq!(found, m.pose);
    }
}
