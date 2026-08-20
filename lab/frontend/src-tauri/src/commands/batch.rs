//! `batch_find` — run one model over a whole set of frames.
//!
//! This is what opening a folder is *for*: a model is only as good as its
//! behaviour across a capture, and reading that off one frame at a time hides
//! exactly the tail — the handful of frames where the score falls off — that
//! decides whether the model is usable.
//!
//! One frame's failure is reported on that frame (`BatchFindItemOut::error`)
//! and the run continues. A batch over three thousand frames that abandons
//! everything because one file is unreadable has answered no question at all.

use std::time::Instant;

use crate::error::AppResult;
use crate::state::AppState;
use crate::types::{
    BatchFindItemOut, BatchFindRequest, BatchFindResponse, FindRequest, SearchTuningIn,
};

/// Per-image progress, emitted as `lab://batch` while the run is in flight.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchProgress {
    pub done: usize,
    pub total: usize,
    pub image_id: String,
    pub matches: usize,
    pub best_score: Option<f32>,
    pub elapsed_ms: f64,
}

fn find_request_for(req: &BatchFindRequest, image_id: &str) -> FindRequest {
    FindRequest {
        image_id: image_id.to_string(),
        model_id: req.model_id.clone(),
        min_score: req.search.min_score,
        max_matches: req.search.max_matches,
        roi: None,
        angle_range: req.search.angle_range,
        scale_range: req.search.scale_range,
        refinement: req.search.refinement.clone(),
        min_contrast: req.search.min_contrast,
        tuning: req.search.tuning.clone().or(None::<SearchTuningIn>),
    }
}

/// Run the model over every image, calling `on_item` as each finishes.
///
/// Sequential on purpose: `ShapeMatcher` already saturates a core, the frames
/// share one decode cache, and a progress stream a user can read beats a
/// shorter wall clock they cannot. Parallelism here is a change to make with a
/// measurement in hand, not on principle.
pub fn batch_find(
    state: &AppState,
    req: &BatchFindRequest,
    mut on_item: impl FnMut(BatchProgress),
) -> AppResult<BatchFindResponse> {
    let total = req.image_ids.len();
    let mut items = Vec::with_capacity(total);

    for (i, image_id) in req.image_ids.iter().enumerate() {
        let started = Instant::now();
        let find_req = find_request_for(req, image_id);
        let result = super::find::run_find(state, &find_req);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

        let item = match result {
            Ok(matches) => BatchFindItemOut {
                image_id: image_id.clone(),
                matches: matches.iter().map(super::find::to_match_out).collect(),
                elapsed_ms,
                error: None,
            },
            Err(e) => BatchFindItemOut {
                image_id: image_id.clone(),
                matches: Vec::new(),
                elapsed_ms,
                error: Some(e.0),
            },
        };

        on_item(BatchProgress {
            done: i + 1,
            total,
            image_id: image_id.clone(),
            matches: item.matches.len(),
            best_score: item.matches.iter().map(|m| m.score).max_by(f32::total_cmp),
            elapsed_ms,
        });
        items.push(item);
    }

    Ok(BatchFindResponse { items })
}
