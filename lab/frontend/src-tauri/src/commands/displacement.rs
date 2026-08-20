//! `displacement` — mirrors `lab/backend/src/vm_lab/routers/displacement.py`.

use vision_metrology::corr::{DisplacementConfig, Refine, displacement as native_displacement};
use vm_primitives::Rect2f;

use crate::error::{AppResult, not_found};
use crate::state::AppState;
use crate::types::{DisplacementPairOut, DisplacementRequest, DisplacementResponse};

fn refine_from(req: &DisplacementRequest) -> Refine {
    if req.refine == "none" {
        Refine::None
    } else {
        Refine::LucasKanade {
            iters: req.lk_iters,
        }
    }
}

pub fn displacement(state: &AppState, req: DisplacementRequest) -> AppResult<DisplacementResponse> {
    let images = state.images.lock().expect("images mutex poisoned");
    for id in &req.image_ids {
        if !images.contains_key(id) {
            return Err(not_found("image", id));
        }
    }

    let cfg = DisplacementConfig {
        window: Rect2f {
            x: req.window[0],
            y: req.window[1],
            width: req.window[2],
            height: req.window[3],
        },
        search: (req.search_x, req.search_y),
        refine: refine_from(&req),
        min_score: req.min_score,
    };

    let mut pairs = Vec::with_capacity(req.image_ids.len().saturating_sub(1));
    let mut cumulative_x = vec![0.0f32];
    let mut cumulative_y = vec![0.0f32];
    for w in req.image_ids.windows(2) {
        let (prev_id, curr_id) = (&w[0], &w[1]);
        let prev = &images[prev_id].image;
        let curr = &images[curr_id].image;
        let d = native_displacement(&prev.as_view(), &curr.as_view(), &cfg).map_err(|e| {
            crate::error::AppError(format!(
                "displacement failed between {prev_id} and {curr_id}: {e}"
            ))
        })?;
        pairs.push(DisplacementPairOut {
            from_image_id: prev_id.clone(),
            to_image_id: curr_id.clone(),
            dx: d.shift.x,
            dy: d.shift.y,
            score: d.score,
        });
        cumulative_x.push(cumulative_x.last().unwrap() + d.shift.x);
        cumulative_y.push(cumulative_y.last().unwrap() + d.shift.y);
    }

    Ok(DisplacementResponse {
        pairs,
        cumulative_x,
        cumulative_y,
    })
}
