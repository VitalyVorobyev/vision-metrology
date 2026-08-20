//! `models_create` (teach), `models_list`, `model_geometry`, `model_crop` —
//! mirrors `lab/backend/src/vm_lab/routers/models.py`, plus the geometry and
//! crop commands a UI needs to *show* what was taught.

use vision_metrology::matching::{Contrast, CropSpec, ShapeModelBuilder, ShapeModelConfig};
use vision_metrology::warp::Interp;
use vm_primitives::{BorderMode, Image, Point2f, Rect2f};

use super::teach::{contours_in_roi, mask_for_contours, to_rect};
use crate::error::{AppResult, not_found};
use crate::state::AppState;
use crate::types::{ModelCreateRequest, ModelCropRequest, ModelGeometryOut, ModelOut};

pub fn models_create(state: &AppState, req: ModelCreateRequest) -> AppResult<ModelOut> {
    let image = state.decoded(&req.image_id)?;
    let view = image.as_view();

    let roi = to_rect(req.roi);
    let config = ShapeModelConfig {
        num_levels: req.num_levels.and_then(std::num::NonZeroUsize::new),
        min_contrast: Contrast::FractionOfRange(req.min_contrast),
        origin: req.origin.map(|o| Point2f::new(o[0], o[1])),
        reference_angle: req.reference_angle,
        ..ShapeModelConfig::default()
    };

    // Curated teach: recompute the same contours `teach_preview` offered (the
    // extraction is deterministic, so the ids still line up — see
    // `commands::teach`) and keep only what the caller selected.
    let mask = match req.keep_contours.as_deref() {
        None => None,
        Some(keep) => {
            let contours = contours_in_roi(&image, req.roi, req.min_contrast)?;
            Some(mask_for_contours(
                image.width(),
                image.height(),
                &contours,
                keep,
            )?)
        }
    };

    let model = ShapeModelBuilder::new()
        .build_with_mask(
            &view,
            roi,
            mask.as_ref().map(Image::as_view).as_ref(),
            &config,
        )
        .map_err(|e| crate::error::AppError(format!("teach failed: {e}")))?;

    state.add_model(
        req.image_id,
        req.roi,
        req.min_contrast,
        req.num_levels,
        model,
    )
}

pub fn models_list(state: &AppState) -> Vec<ModelOut> {
    let models = state.models.lock().expect("models mutex poisoned");
    let mut out: Vec<ModelOut> = models.iter().map(|(id, e)| to_out(id, e)).collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

pub fn to_out(id: &str, e: &crate::state::ModelEntry) -> ModelOut {
    ModelOut {
        id: id.to_string(),
        image_id: e.image_id.clone(),
        roi: e.roi,
        min_contrast: e.min_contrast,
        num_levels: e.num_levels,
        origin: {
            let o = e.model.origin();
            [o.x, o.y]
        },
        num_levels_built: e.model.num_levels(),
        point_counts: (0..e.model.num_levels())
            .map(|i| e.model.point_count(i))
            .collect(),
        reference_angle: e.model.reference_angle(),
    }
}

/// The model's own points, so the caller can draw what was learned.
///
/// `frame` picks which frame they come back in: `"reference"` to draw over the
/// image the model was taught from, `"model"` to transform by a match's pose
/// and draw over a scene. The two differ only for a model taught with a
/// canonical orientation.
pub fn model_geometry(
    state: &AppState,
    model_id: &str,
    level: usize,
    frame: &str,
) -> AppResult<ModelGeometryOut> {
    let models = state.models.lock().expect("models mutex poisoned");
    let entry = models
        .get(model_id)
        .ok_or_else(|| not_found("model", model_id))?;

    let pairs = match frame {
        "model" => entry.model.model_geometry(level),
        "reference" => entry.model.reference_geometry(level),
        other => {
            return Err(format!("unknown frame: {other} (want 'reference' or 'model')").into());
        }
    };
    let origin = entry.model.origin();

    Ok(ModelGeometryOut {
        model_id: model_id.to_string(),
        level,
        origin: [origin.x, origin.y],
        reference_angle: entry.model.reference_angle(),
        points: pairs
            .into_iter()
            .flat_map(|(p, t)| [p.x, p.y, t.x, t.y])
            .collect(),
        frame: frame.to_string(),
    })
}

/// The model's own reference image, rectified through `CropSpec` — the
/// untouched half of the model-versus-instance comparison whose other half is
/// `rectify`'s crop of a found match. Same spec both sides, so the two crops
/// are the same size, orientation and scale and can be composited directly.
pub fn model_crop(state: &AppState, req: &ModelCropRequest) -> AppResult<Vec<u8>> {
    let (model, image_id) = {
        let models = state.models.lock().expect("models mutex poisoned");
        let e = models
            .get(&req.model_id)
            .ok_or_else(|| not_found("model", &req.model_id))?;
        (e.model.clone(), e.image_id.clone())
    };
    let image = state.decoded(&image_id)?;

    let spec = CropSpec {
        rect: Rect2f {
            x: req.rect[0],
            y: req.rect[1],
            width: req.rect[2],
            height: req.rect[3],
        },
        px_per_unit: req.px_per_unit,
        normalize_scale: true,
    };
    let (w, h) = spec.output_size();
    let mut dst = vec![0u8; w * h];
    let mut mask = vec![0u8; w * h];
    model
        .reference_frame_map(&spec)
        .apply_with_mask(
            &image.as_view(),
            &mut dst,
            &mut mask,
            Interp::Bilinear,
            BorderMode::Constant(0),
        )
        .map_err(|e| crate::error::AppError(format!("model_crop: {e}")))?;

    let png = image::GrayImage::from_raw(w as u32, h as u32, dst)
        .ok_or_else(|| crate::error::AppError("could not build crop image".to_string()))?;
    let mut buf = Vec::new();
    png.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .map_err(crate::error::AppError::from)?;
    Ok(buf)
}
