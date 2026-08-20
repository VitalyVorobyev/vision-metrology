//! `images_upload`, `images_list`, `image_data` — mirrors
//! `lab/backend/src/vm_lab/routers/images.py`.
//!
//! **Deliberate simplification from the browser backend**: `image_data`'s tiers are all
//! PNG here (thumb/preview resized, full at native resolution), not WebP. The plan asks
//! for "binary via `tauri::ipc::Response` (PNG bytes, NO base64)" explicitly, and a
//! single codec keeps this command's one job (resize + encode) simple; no on-disk tier
//! cache either — the desktop shell renders on demand, since a resize is a few
//! milliseconds against images this size and a cache invalidation story is one more
//! thing to keep correct for a workbench, not a hot path.

use image::imageops::FilterType;

use crate::error::{AppResult, not_found};
use crate::state::AppState;
use crate::types::ImageOut;

fn to_out(id: &str, entry: &crate::state::ImageEntry) -> ImageOut {
    ImageOut {
        id: id.to_string(),
        filename: entry.filename.clone(),
        width: entry.image.width() as u32,
        height: entry.image.height() as u32,
        sha256: entry.sha256.clone(),
    }
}

pub fn images_upload(state: &AppState, filename: String, bytes: &[u8]) -> AppResult<ImageOut> {
    if bytes.is_empty() {
        return Err("empty upload".into());
    }
    state.add_image(filename, bytes)
}

pub fn images_list(state: &AppState) -> Vec<ImageOut> {
    let images = state.images.lock().expect("images mutex poisoned");
    let mut out: Vec<ImageOut> = images.iter().map(|(id, e)| to_out(id, e)).collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// `tier`: `"full"` (native resolution), `"preview"` (long edge 1024), `"thumb"` (long
/// edge 256) — same three tiers as the browser backend's `Tier` enum, PNG-encoded.
pub fn image_data(state: &AppState, image_id: &str, tier: &str) -> AppResult<Vec<u8>> {
    let images = state.images.lock().expect("images mutex poisoned");
    let entry = images
        .get(image_id)
        .ok_or_else(|| not_found("image", image_id))?;
    let (w, h) = (entry.image.width() as u32, entry.image.height() as u32);
    let buf = image::GrayImage::from_raw(w, h, entry.image.data().to_vec())
        .ok_or_else(|| crate::error::AppError("could not view image buffer".to_string()))?;

    let long_edge = match tier {
        "full" => None,
        "preview" => Some(1024u32),
        "thumb" => Some(256u32),
        other => return Err(format!("unknown tier: {other}").into()),
    };
    let resized = match long_edge {
        None => buf,
        Some(edge) if w.max(h) <= edge => buf,
        Some(edge) => {
            let (nw, nh) = if w >= h {
                (
                    edge,
                    (h as f32 * edge as f32 / w as f32).round().max(1.0) as u32,
                )
            } else {
                (
                    (w as f32 * edge as f32 / h as f32).round().max(1.0) as u32,
                    edge,
                )
            };
            image::imageops::resize(&buf, nw, nh, FilterType::Lanczos3)
        }
    };

    let mut out = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut out);
    resized
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(crate::error::AppError::from)?;
    Ok(out)
}
