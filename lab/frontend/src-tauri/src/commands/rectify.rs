//! `rectify`, `rectify_crop` — mirrors `lab/backend/src/vm_lab/routers/rectify.py`.
//!
//! No HTTP crop-cache URL here (there is no HTTP): `rectify` returns a `crop_key` per
//! match, cached in `AppState`, and `rectify_crop` returns the PNG bytes for one —
//! `tauriBackend.ts` turns that into an object URL, same as the browser backend's
//! `crop_url` becomes an `<img src>`.

use std::collections::HashMap;
use std::sync::Mutex;

use vision_metrology::matching::CropSpec;
use vision_metrology::warp::Interp;
use vm_primitives::{BorderMode, Rect2f};

use super::find::run_find;
use crate::error::{AppResult, not_found};
use crate::state::AppState;
use crate::types::{FindRequest, RectifyMatchOut, RectifyRequest, RectifyResponse};

/// `(image_id, model_id, index) -> PNG bytes`.
type CropKey = (String, String, usize);

/// In-memory only, overwritten wholesale by each `rectify` call for that
/// `(image_id, model_id)` pair — same spirit as the Python backend's own `_crop_cache`.
pub static CROP_CACHE: Mutex<Option<HashMap<CropKey, Vec<u8>>>> = Mutex::new(None);

fn crop_key(image_id: &str, model_id: &str, index: usize) -> String {
    format!("{image_id}/{model_id}/{index}")
}

pub fn rectify(state: &AppState, req: RectifyRequest) -> AppResult<RectifyResponse> {
    // `run_find` locks `state.images`/`state.models` itself, and so does
    // `AppState::decoded` below — neither may be called while this function
    // holds either lock (`std::sync::Mutex` is not reentrant, so the same
    // thread locking twice deadlocks rather than erroring).
    if !state
        .models
        .lock()
        .expect("models mutex poisoned")
        .contains_key(&req.model_id)
    {
        return Err(not_found("model", &req.model_id));
    }
    // The caller's whole search, not just its thresholds — see
    // `RectifyRequest`'s own docs for why anything less renumbers the matches.
    let find_req = FindRequest {
        image_id: req.image_id.clone(),
        model_id: req.model_id.clone(),
        min_score: req.min_score,
        max_matches: req.max_matches,
        angle_range: req.angle_range,
        scale_range: req.scale_range,
        refinement: req.refinement.clone(),
        min_contrast: req.min_contrast,
        tuning: req.tuning.clone(),
        roi: None,
    };
    let matches = run_find(state, &find_req)?;

    let image = state.decoded(&req.image_id)?;

    let spec = CropSpec {
        rect: Rect2f {
            x: req.crop.rect[0],
            y: req.crop.rect[1],
            width: req.crop.rect[2],
            height: req.crop.rect[3],
        },
        px_per_unit: req.crop.px_per_unit,
        normalize_scale: req.crop.normalize_scale,
    };
    let (width, height) = spec.output_size();

    let mut cache_guard = CROP_CACHE.lock().expect("crop cache mutex poisoned");
    let cache = cache_guard.get_or_insert_with(HashMap::new);
    cache.retain(|(iid, mid, _), _| !(iid == &req.image_id && mid == &req.model_id));

    let view = image.as_view();
    let mut out = Vec::with_capacity(matches.len());
    for (i, m) in matches.iter().enumerate() {
        let crop_map = m.model_frame_map(&spec);
        let mut dst = vec![0u8; width * height];
        let mut mask = vec![0u8; width * height];
        crop_map
            .apply_with_mask(
                &view,
                &mut dst,
                &mut mask,
                Interp::Bilinear,
                BorderMode::Constant(0),
            )
            .map_err(|e| crate::error::AppError(format!("rectify: {e}")))?;
        let validity = mask.iter().filter(|&&v| v == 255).count() as f32 / mask.len().max(1) as f32;

        let png = image::GrayImage::from_raw(width as u32, height as u32, dst)
            .ok_or_else(|| crate::error::AppError("could not build crop image".to_string()))?;
        let mut buf = Vec::new();
        png.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .map_err(crate::error::AppError::from)?;
        cache.insert((req.image_id.clone(), req.model_id.clone(), i), buf);

        out.push(RectifyMatchOut {
            index: i,
            x: m.position.x,
            y: m.position.y,
            angle: m.angle(),
            scale: m.scale(),
            score: m.score,
            support: m.support,
            level: m.level,
            validity,
            crop_key: crop_key(&req.image_id, &req.model_id, i),
            width,
            height,
        });
    }

    Ok(RectifyResponse {
        width,
        height,
        matches: out,
    })
}

pub fn rectify_crop(image_id: &str, model_id: &str, index: usize) -> AppResult<Vec<u8>> {
    let mut cache_guard = CROP_CACHE.lock().expect("crop cache mutex poisoned");
    let cache = cache_guard.get_or_insert_with(HashMap::new);
    cache
        .get(&(image_id.to_string(), model_id.to_string(), index))
        .cloned()
        .ok_or_else(|| {
            crate::error::AppError(
                "no cached crop for this image/model/index -- call rectify first".to_string(),
            )
        })
}
