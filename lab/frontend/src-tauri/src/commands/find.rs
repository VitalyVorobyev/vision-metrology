//! `find` — mirrors `lab/backend/src/vm_lab/routers/find.py`'s `run_find`.

use std::num::NonZeroUsize;

use vision_metrology::matching::{
    Contrast, Refinement, ShapeMatch, ShapeMatcher, ShapeSearchConfig, ShapeSearchTuning,
};
use vm_primitives::Rect2f;

use crate::error::{AppResult, not_found};
use crate::state::AppState;
use crate::types::{FindRequest, FindResponse, MatchOut, SearchTuningIn};

/// Translate the wire request into the library's own search config.
///
/// The lab used to expose four of these fields and leave the rest at their
/// defaults, which made a slow search look like a slow *library*: with no way
/// to narrow the angle sweep, cap the match count, or stop the descent early,
/// the only search anyone could run was the most expensive one. Every knob the
/// library documents as a speed/recall trade is reachable from here now.
pub fn search_config(req: &FindRequest) -> ShapeSearchConfig {
    let d = ShapeSearchConfig::default();
    let tuning = req.tuning.as_ref().map_or_else(
        || d.tuning.clone(),
        |t: &SearchTuningIn| ShapeSearchTuning {
            greediness: t.greediness.unwrap_or(d.tuning.greediness),
            angle_step: t.angle_step,
            scale_step: t.scale_step,
            last_level: t.last_level.unwrap_or(d.tuning.last_level),
            max_candidates: t.max_candidates.unwrap_or(d.tuning.max_candidates),
            coarse_score_factor: t
                .coarse_score_factor
                .unwrap_or(d.tuning.coarse_score_factor),
        },
    );

    ShapeSearchConfig {
        min_score: req.min_score,
        max_matches: req.max_matches.and_then(NonZeroUsize::new),
        roi: req.roi.map(|r| Rect2f {
            x: r[0],
            y: r[1],
            width: r[2],
            height: r[3],
        }),
        angle_range: req.angle_range,
        scale_range: req.scale_range,
        refinement: match req.refinement.as_deref() {
            Some("none") => Refinement::None,
            Some("least_squares") => Refinement::LeastSquares,
            _ => Refinement::Interpolate,
        },
        min_contrast: req
            .min_contrast
            .map_or(d.min_contrast, Contrast::FractionOfRange),
        tuning,
        ..ShapeSearchConfig::default()
    }
}

/// Returns the native `ShapeMatch`es — shared by the `find` and
/// `measure`/`rectify` commands, same reasoning as the Python backend's own
/// `run_find`.
pub fn run_find(state: &AppState, req: &FindRequest) -> AppResult<Vec<ShapeMatch>> {
    // Resolve the model first and drop the lock: `decoded` takes the images
    // lock itself, and `std::sync::Mutex` is not reentrant.
    let model = {
        let models = state.models.lock().expect("models mutex poisoned");
        models
            .get(&req.model_id)
            .ok_or_else(|| not_found("model", &req.model_id))?
            .model
            .clone()
    };
    let image = state.decoded(&req.image_id)?;

    let config = search_config(req);
    let mut matcher = ShapeMatcher::new();
    Ok(matcher.find(&image.as_view(), &model, &config))
}

pub fn find(state: &AppState, req: FindRequest) -> AppResult<FindResponse> {
    let matches = run_find(state, &req)?;
    Ok(FindResponse {
        matches: matches.iter().map(to_match_out).collect(),
    })
}

pub fn to_match_out(m: &ShapeMatch) -> MatchOut {
    MatchOut {
        x: m.position.x,
        y: m.position.y,
        angle: m.angle(),
        scale: m.scale(),
        score: m.score,
        support: m.support,
        level: m.level,
    }
}
