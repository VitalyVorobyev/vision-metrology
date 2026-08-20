//! `find` — mirrors `lab/backend/src/vm_lab/routers/find.py`'s `run_find`.

use vision_metrology::matching::{ShapeMatch, ShapeMatcher, ShapeSearchConfig};
use vm_primitives::Rect2f;

use crate::error::{AppResult, not_found};
use crate::state::AppState;
use crate::types::{FindRequest, FindResponse, MatchOut};

/// Returns the native `ShapeMatch`es — shared by the `find` and `measure`/`rectify`
/// commands, same reasoning as the Python backend's own `run_find`.
pub fn run_find(state: &AppState, req: &FindRequest) -> AppResult<Vec<ShapeMatch>> {
    let images = state.images.lock().expect("images mutex poisoned");
    let entry = images
        .get(&req.image_id)
        .ok_or_else(|| not_found("image", &req.image_id))?;
    let models = state.models.lock().expect("models mutex poisoned");
    let model_entry = models
        .get(&req.model_id)
        .ok_or_else(|| not_found("model", &req.model_id))?;

    let config = ShapeSearchConfig {
        min_score: req.min_score,
        max_matches: req.max_matches.and_then(std::num::NonZeroUsize::new),
        roi: req.roi.map(|r| Rect2f {
            x: r[0],
            y: r[1],
            width: r[2],
            height: r[3],
        }),
        angle_range: req.angle_range,
        ..ShapeSearchConfig::default()
    };

    let mut matcher = ShapeMatcher::new();
    Ok(matcher.find(&entry.image.as_view(), &model_entry.model, &config))
}

pub fn find(state: &AppState, req: FindRequest) -> AppResult<FindResponse> {
    let matches = run_find(state, &req)?;
    Ok(FindResponse {
        matches: matches
            .into_iter()
            .map(|m| MatchOut {
                x: m.position.x,
                y: m.position.y,
                angle: m.angle(),
                scale: m.scale(),
                score: m.score,
                support: m.support,
                level: m.level,
            })
            .collect(),
    })
}
