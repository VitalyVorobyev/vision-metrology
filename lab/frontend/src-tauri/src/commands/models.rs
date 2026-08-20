//! `models_create` (teach), `models_list` — mirrors
//! `lab/backend/src/vm_lab/routers/models.py`.

use vision_metrology::matching::{Contrast, ShapeModelBuilder, ShapeModelConfig};
use vm_primitives::Rect2f;

use crate::error::{AppResult, not_found};
use crate::state::AppState;
use crate::types::{ModelCreateRequest, ModelOut};

pub fn models_create(state: &AppState, req: ModelCreateRequest) -> AppResult<ModelOut> {
    let images = state.images.lock().expect("images mutex poisoned");
    let entry = images
        .get(&req.image_id)
        .ok_or_else(|| not_found("image", &req.image_id))?;
    let view = entry.image.as_view();

    let roi = Rect2f {
        x: req.roi[0],
        y: req.roi[1],
        width: req.roi[2],
        height: req.roi[3],
    };
    let config = ShapeModelConfig {
        num_levels: req.num_levels.and_then(std::num::NonZeroUsize::new),
        min_contrast: Contrast::FractionOfRange(req.min_contrast),
        ..ShapeModelConfig::default()
    };

    let model = ShapeModelBuilder::new()
        .build(&view, roi, &config)
        .map_err(|e| crate::error::AppError(format!("teach failed: {e}")))?;
    drop(images);

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
    let mut out: Vec<ModelOut> = models
        .iter()
        .map(|(id, e)| ModelOut {
            id: id.clone(),
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
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}
