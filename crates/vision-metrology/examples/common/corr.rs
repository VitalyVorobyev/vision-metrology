//! Bridge to `corrmatch` for independent pose validation.
//!
//! Two jobs:
//! - [`zncc_at_pose`] — score a pose *recovered by the shape matcher* with
//!   masked ZNCC of the reference patch, rotated to the recovered angle,
//!   against the scene. No search: this is an independent measure computed by
//!   a different algorithm over different data (raw intensities instead of
//!   gradient directions).
//! - [`corr_to_center`] / [`center_to_corr_topleft`] — coordinate conversion
//!   between corrmatch's convention (template **top-left**, rotation about
//!   the **template centre**) and this crate's (model **origin** = level-0
//!   centroid, pixel-centre coordinates). Convention mismatches are the
//!   classic silent bug in cross-library comparison, so the round trip is
//!   pinned by `tests/corrmatch_bridge.rs`.
//!
//! Validity note: corrmatch has no scale search and the plans below rasterise
//! the reference at scale 1, so the ZNCC is meaningful at `scale ≈ 1` only.

// `#[path]`-included from the pose_audit example and tests/corrmatch_bridge;
// each root uses a subset, so per-root dead-code analysis is meaningless.
#![allow(dead_code)]

use corrmatch::ImageView as CorrView;
use corrmatch::lowlevel::{MaskedTemplatePlan, rotate_u8_bilinear_masked, score_masked_zncc_at};
use vision_metrology::{Image, Point2f, Rect2f};

/// Where corrmatch would report a match, for an object whose *centre of the
/// template rectangle* lands at `center` after rotating by `angle_deg`.
///
/// corrmatch rotates the template about its centre and reports the top-left
/// of the (unrotated-size) placement rectangle, so:
/// `top_left = center − (tw/2 − 0.5, th/2 − 0.5)` in pixel-centre coordinates.
pub fn center_to_corr_topleft(center: Point2f, tw: usize, th: usize) -> Point2f {
    Point2f {
        x: center.x - 0.5 * (tw as f32 - 1.0),
        y: center.y - 0.5 * (th as f32 - 1.0),
    }
}

/// Inverse of [`center_to_corr_topleft`]: corrmatch `Match{x, y}` → the scene
/// position of the template centre.
pub fn corr_to_center(x: f32, y: f32, tw: usize, th: usize) -> Point2f {
    Point2f {
        x: x + 0.5 * (tw as f32 - 1.0),
        y: y + 0.5 * (th as f32 - 1.0),
    }
}

/// Extract the reference ROI as an owned patch (u8, row-major).
pub fn roi_patch(reference: &Image<u8>, roi: Rect2f) -> (Vec<u8>, usize, usize) {
    let x0 = roi.x.round().max(0.0) as usize;
    let y0 = roi.y.round().max(0.0) as usize;
    let x1 = ((roi.x + roi.width).round() as usize).min(reference.width());
    let y1 = ((roi.y + roi.height).round() as usize).min(reference.height());
    let (tw, th) = (x1 - x0, y1 - y0);
    let view = reference.as_view();
    let mut buf = Vec::with_capacity(tw * th);
    for y in y0..y1 {
        buf.extend_from_slice(&view.row(y)[x0..x1]);
    }
    (buf, tw, th)
}

/// Independent ZNCC of the recovered pose, in `[-1, 1]`.
///
/// Builds a masked plan of the reference ROI rotated to `angle_deg` and
/// scores it at the integer placement nearest to where the shape matcher put
/// the ROI centre. Returns `None` when the placement does not fit inside the
/// scene or the template is degenerate.
pub fn zncc_at_pose(
    scene: &Image<u8>,
    reference: &Image<u8>,
    roi: Rect2f,
    roi_center_in_scene: Point2f,
    angle_deg: f32,
) -> Option<f32> {
    let (tpl, tw, th) = roi_patch(reference, roi);
    let tpl_view = CorrView::new(&tpl, tw, th, tw).ok()?;

    // Rotate the reference patch about its centre; pixels whose bilinear
    // support falls outside the patch are masked out of the ZNCC statistics.
    let (rotated, mask) = rotate_u8_bilinear_masked(tpl_view, angle_deg, 0);
    let plan = MaskedTemplatePlan::from_rotated_u8(rotated.view(), mask, angle_deg).ok()?;

    // The rotated raster keeps the template dimensions; its centre stays the
    // template centre, so the placement is centre-anchored.
    let (pw, ph) = (rotated.width(), rotated.height());
    let top_left = Point2f {
        x: roi_center_in_scene.x - 0.5 * (pw as f32 - 1.0),
        y: roi_center_in_scene.y - 0.5 * (ph as f32 - 1.0),
    };
    let x = top_left.x.round();
    let y = top_left.y.round();
    if x < 0.0 || y < 0.0 {
        return None;
    }
    let (x, y) = (x as usize, y as usize);
    if x + pw > scene.width() || y + ph > scene.height() {
        return None;
    }

    let data = scene.data();
    let scene_view = CorrView::new(data, scene.width(), scene.height(), scene.width()).ok()?;
    Some(score_masked_zncc_at(scene_view, &plan, x, y, 1e-8))
}
