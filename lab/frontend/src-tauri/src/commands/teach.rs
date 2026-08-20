//! `teach_preview` — the candidate contours a caller curates *before* a model
//! is built, and the mask that turns that curation into a model.
//!
//! Teaching used to be one shot: drag a rectangle, get a model, and read a
//! point count. Nothing showed what had been learned, so nothing could be
//! corrected — and on a round or L-shaped part a rectangle always learns some
//! background, whose points then dilute every score (the model's own point
//! count is the score's denominator).
//!
//! This module splits it in two. `teach_preview` runs exactly the edge
//! extraction `ShapeModelBuilder` runs and links the result into contours the
//! caller can see and pick from; `mask_for_contours` turns a chosen subset
//! back into the inclusion mask `ShapeModelBuilder::build_with_mask` takes.
//!
//! **The two halves must agree**, and they do so by recomputing rather than by
//! caching: extraction is deterministic (invariant 12), so `models_create`
//! re-running `contours_in_roi` with the same arguments gets the same contours
//! with the same ids the preview handed out. A cache would be a second source
//! of truth to keep in step, and its staleness would show up as a model built
//! from contours the user did not choose.

use vision_metrology::contour::{ContourBuildConfig, build_graph_from_edgels};
use vision_metrology::matching::Contrast;
use vm_primitives::{Edge2DConfig, Edge2DDetector, Image, ImageView, Rect2f};

use crate::error::AppResult;
use crate::state::AppState;
use crate::types::{ContourOut, TeachPreviewRequest, TeachPreviewResponse};

/// How far the inclusion mask is grown around a kept contour, in level-0
/// pixels.
///
/// The mask is tested at the level-0 position a point was aggregated from, and
/// a coarse level's point can sit up to about half its own pixel from the fine
/// edge that produced it. Six pixels covers that for the levels a model
/// actually builds (at most five), and being generous costs only the odd
/// neighbouring edgel — being stingy silently deletes the coarse levels, which
/// is far worse: the search would still run, just from the wrong pyramid.
const MASK_DILATION: i32 = 6;

/// The ROI crop a preview and a build must share.
///
/// `ShapeModelBuilder` measures a fractional `min_contrast` against the ROI,
/// not the whole frame, so the preview has to do the same or it offers
/// contours the build then discards.
fn roi_crop<'a>(
    image: &'a Image<u8>,
    roi: [f32; 4],
) -> AppResult<(ImageView<'a, u8>, usize, usize)> {
    let x0 = roi[0].floor().max(0.0) as usize;
    let y0 = roi[1].floor().max(0.0) as usize;
    let x1 = ((roi[0] + roi[2]).ceil() as usize).min(image.width());
    let y1 = ((roi[1] + roi[3]).ceil() as usize).min(image.height());
    if x1 <= x0 || y1 <= y0 {
        return Err("roi must have positive extent inside the image".into());
    }
    let view = image.as_view().subview(x0, y0, x1 - x0, y1 - y0)?;
    Ok((view, x0, y0))
}

/// The candidate contours inside `roi`, in **image** coordinates.
///
/// Deterministic in its arguments — see this module's own docs for why that
/// matters.
pub fn contours_in_roi(
    image: &Image<u8>,
    roi: [f32; 4],
    min_contrast: f32,
) -> AppResult<Vec<ContourOut>> {
    let (view, x0, y0) = roi_crop(image, roi)?;
    let floor = Contrast::FractionOfRange(min_contrast).resolve_for(&view);

    let edge_cfg = Edge2DConfig::default();
    let mut detector = Edge2DDetector::new();
    let edgels: Vec<_> = detector
        .detect(&view, &edge_cfg)
        .into_iter()
        .filter(|e| e.strength >= floor)
        .collect();

    let graph = build_graph_from_edgels(
        view.width(),
        view.height(),
        &edgels,
        &ContourBuildConfig {
            record_strengths: true,
            ..ContourBuildConfig::default()
        },
    );

    let (dx, dy) = (x0 as f32, y0 as f32);
    Ok(graph
        .edges
        .iter()
        .enumerate()
        .map(|(i, e)| ContourOut {
            id: i,
            points: e.points.iter().flat_map(|p| [p.x + dx, p.y + dy]).collect(),
            closed: e.is_loop,
            length: e.length,
            mean_strength: e.score,
        })
        .collect())
}

pub fn teach_preview(
    state: &AppState,
    req: TeachPreviewRequest,
) -> AppResult<TeachPreviewResponse> {
    let image = state.decoded(&req.image_id)?;
    let contours = contours_in_roi(&image, req.roi, req.min_contrast)?;
    let total_points = contours.iter().map(|c| c.points.len() / 2).sum();
    Ok(TeachPreviewResponse {
        contours,
        total_points,
    })
}

/// Rasterise the chosen contours into a full-frame inclusion mask.
///
/// Every kept vertex paints a filled square of side `2 * MASK_DILATION + 1`.
/// Squares rather than discs, and per-vertex rather than along the segment,
/// because contour vertices are one pixel apart by construction — the squares
/// overlap into a band, and the difference from a proper swept disc is smaller
/// than the dilation itself.
pub fn mask_for_contours(
    width: usize,
    height: usize,
    contours: &[ContourOut],
    keep: &[usize],
) -> AppResult<Image<u8>> {
    let mut data = vec![0u8; width * height];
    let mut painted = 0usize;

    for id in keep {
        let Some(c) = contours.iter().find(|c| c.id == *id) else {
            return Err(format!("no contour with id {id} in this ROI").into());
        };
        for xy in c.points.chunks_exact(2) {
            let (cx, cy) = (xy[0].round() as i32, xy[1].round() as i32);
            for y in (cy - MASK_DILATION).max(0)..=(cy + MASK_DILATION).min(height as i32 - 1) {
                let row = y as usize * width;
                for x in (cx - MASK_DILATION).max(0)..=(cx + MASK_DILATION).min(width as i32 - 1) {
                    data[row + x as usize] = 255;
                }
            }
            painted += 1;
        }
    }

    if painted == 0 {
        return Err("no contours selected — a model needs at least one".into());
    }
    Ok(Image::from_vec(width, height, data)?)
}

/// `roi` as the library's own rectangle type.
pub fn to_rect(roi: [f32; 4]) -> Rect2f {
    Rect2f {
        x: roi[0],
        y: roi[1],
        width: roi[2],
        height: roi[3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(size: usize, cx: f32, cy: f32, r: f32) -> Image<u8> {
        let mut data = vec![20u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                if (dx * dx + dy * dy).sqrt() < r {
                    data[y * size + x] = 220;
                }
            }
        }
        Image::from_vec(size, size, data).expect("valid image")
    }

    #[test]
    fn a_preview_finds_the_disc_outline_and_reports_it_in_image_coordinates() {
        let img = disc(120, 60.0, 60.0, 30.0);
        let contours = contours_in_roi(&img, [10.0, 10.0, 100.0, 100.0], 0.1).expect("preview");
        assert!(!contours.is_empty(), "the disc has an outline");

        let longest = contours
            .iter()
            .max_by(|a, b| a.length.total_cmp(&b.length))
            .expect("non-empty");
        // Points are in image coordinates, so they sit on the circle of
        // radius 30 about (60, 60) — not about the crop's own origin.
        for xy in longest.points.chunks_exact(2) {
            let r = ((xy[0] - 60.0).powi(2) + (xy[1] - 60.0).powi(2)).sqrt();
            assert!((r - 30.0).abs() < 3.0, "point {xy:?} is not on the outline");
        }
    }

    #[test]
    fn the_preview_is_deterministic_so_the_build_can_recompute_it() {
        let img = disc(120, 60.0, 60.0, 30.0);
        let a = contours_in_roi(&img, [10.0, 10.0, 100.0, 100.0], 0.1).expect("preview");
        let b = contours_in_roi(&img, [10.0, 10.0, 100.0, 100.0], 0.1).expect("preview");
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.points, y.points);
        }
    }

    #[test]
    fn a_mask_covers_the_kept_contour_and_nothing_far_from_it() {
        let img = disc(120, 60.0, 60.0, 30.0);
        let contours = contours_in_roi(&img, [10.0, 10.0, 100.0, 100.0], 0.1).expect("preview");
        let longest = contours
            .iter()
            .max_by(|a, b| a.length.total_cmp(&b.length))
            .expect("non-empty");

        let mask = mask_for_contours(120, 120, &contours, &[longest.id]).expect("mask");
        for xy in longest.points.chunks_exact(2) {
            assert_ne!(
                mask.data()[xy[1].round() as usize * 120 + xy[0].round() as usize],
                0
            );
        }
        // The disc's own centre is far from its outline, so it stays out.
        assert_eq!(mask.data()[60 * 120 + 60], 0);
    }

    #[test]
    fn selecting_nothing_is_an_error_rather_than_an_empty_model() {
        let img = disc(120, 60.0, 60.0, 30.0);
        let contours = contours_in_roi(&img, [10.0, 10.0, 100.0, 100.0], 0.1).expect("preview");
        assert!(mask_for_contours(120, 120, &contours, &[]).is_err());
    }
}
