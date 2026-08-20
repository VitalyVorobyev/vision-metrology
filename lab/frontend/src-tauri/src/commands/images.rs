//! `images_upload`, `images_open_paths`, `images_scan_dir`, `images_list`,
//! `image_tier_path`, `image_data` — mirrors
//! `lab/backend/src/vm_lab/routers/images.py`, plus the folder-oriented
//! commands the browser shell has no honest analogue for.
//!
//! ## Tiers are cached on disk, and served as files
//!
//! The three tiers (`thumb` long edge 256, `preview` 1024, `full` native) are
//! PNG-encoded **once** into `{app_cache}/tiers/{sha256}/{tier}.png` and from
//! then on handed to the webview as a path it loads through Tauri's asset
//! protocol. The previous design re-ran a Lanczos3 resize and a full PNG
//! encode on *every* request, with no cache anywhere, and shipped the bytes
//! back over IPC — which cost a full re-encode each time the canvas crossed
//! its zoom threshold, and forced the frontend into a placeholder-then-swap
//! dance that left the canvas blank until something else happened to
//! re-render (see `tauriBackend.ts`).
//!
//! Keying on the pixel content's `sha256` rather than on the image id is what
//! makes the cache survive re-opening the same file, and makes two entries
//! that are the same picture share one encode.
//!
//! `image_data` (bytes over IPC) is kept for callers that cannot take a path.

use std::path::{Path, PathBuf};

use image::imageops::FilterType;

use crate::error::{AppResult, not_found};
use crate::state::AppState;
use crate::types::{DirEntryOut, ImageOut};

/// Extensions `images_scan_dir` offers. Deliberately the set `image`'s
/// enabled codecs can actually decode — listing a `.tif` we cannot open is
/// worse than not listing it.
const SCANNED_EXTENSIONS: [&str; 3] = ["png", "bmp", "pgm"];

fn to_out(id: &str, entry: &crate::state::ImageEntry) -> ImageOut {
    ImageOut {
        id: id.to_string(),
        filename: entry.filename.clone(),
        width: entry.width as u32,
        height: entry.height as u32,
        sha256: entry.sha256.clone(),
        path: entry.path.as_ref().map(|p| p.display().to_string()),
    }
}

pub fn images_upload(state: &AppState, filename: String, bytes: &[u8]) -> AppResult<ImageOut> {
    if bytes.is_empty() {
        return Err("empty upload".into());
    }
    state.add_image_from_bytes(filename, bytes)
}

/// Register images already on disk, by path — no pixels cross the IPC
/// boundary at all.
///
/// Opening the same path twice returns the existing entry rather than a
/// second copy of it: a workbench where "open" silently duplicates is a
/// workbench whose image rail grows every time you click.
pub fn images_open_paths(state: &AppState, paths: Vec<String>) -> AppResult<Vec<ImageOut>> {
    let mut out = Vec::with_capacity(paths.len());
    for p in paths {
        out.push(state.add_image_from_path(Path::new(&p))?);
    }
    Ok(out)
}

/// List the image files in `dir` **without decoding any of them**.
///
/// Dimensions come from each file's header (`ImageReader::into_dimensions`),
/// which reads a few dozen bytes. That is what makes opening a folder of a
/// few thousand frames instant: nothing is decoded, nothing is copied, and
/// nothing enters the in-memory registry until a frame is actually selected.
pub fn images_scan_dir(dir: &str, recursive: bool) -> AppResult<Vec<DirEntryOut>> {
    let mut out = Vec::new();
    scan_into(Path::new(dir), recursive, &mut out)?;
    // Natural-ish order: a capture folder is `frame_1, frame_2, … frame_10`,
    // and plain lexicographic order puts `frame_10` second.
    out.sort_by(|a, b| natural_cmp(&a.name, &b.name));
    Ok(out)
}

fn scan_into(dir: &Path, recursive: bool, out: &mut Vec<DirEntryOut>) -> AppResult<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| crate::error::AppError(format!("reading {}: {e}", dir.display())))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if recursive {
                // A folder we cannot read is not a reason to abandon the scan.
                let _ = scan_into(&path, true, out);
            }
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        if !ext.is_some_and(|e| SCANNED_EXTENSIONS.contains(&e.as_str())) {
            continue;
        }
        let meta = entry.metadata().ok();
        let (width, height) = image::ImageReader::open(&path)
            .ok()
            .and_then(|r| r.into_dimensions().ok())
            .map_or((0, 0), |(w, h)| (w, h));
        out.push(DirEntryOut {
            path: path.display().to_string(),
            name: path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string(),
            bytes: meta.as_ref().map_or(0, std::fs::Metadata::len),
            width,
            height,
        });
    }
    Ok(())
}

/// Compare so that embedded digit runs order numerically — `frame_2` before
/// `frame_10`, which is the order the folder was captured in.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let (mut ai, mut bi) = (a.chars().peekable(), b.chars().peekable());
    loop {
        match (ai.peek().copied(), bi.peek().copied()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(x), Some(y)) if x.is_ascii_digit() && y.is_ascii_digit() => {
                let take_num = |it: &mut std::iter::Peekable<std::str::Chars<'_>>| {
                    let mut n = 0u128;
                    while let Some(d) = it.peek().and_then(|c| c.to_digit(10)) {
                        n = n.saturating_mul(10).saturating_add(u128::from(d));
                        it.next();
                    }
                    n
                };
                let (x, y) = (take_num(&mut ai), take_num(&mut bi));
                if x != y {
                    return x.cmp(&y);
                }
            }
            (Some(x), Some(y)) => {
                ai.next();
                bi.next();
                if x != y {
                    return x.to_ascii_lowercase().cmp(&y.to_ascii_lowercase());
                }
            }
        }
    }
}

pub fn images_list(state: &AppState) -> Vec<ImageOut> {
    let images = state.images.lock().expect("images mutex poisoned");
    let mut out: Vec<ImageOut> = images.iter().map(|(id, e)| to_out(id, e)).collect();
    out.sort_by(|a, b| natural_cmp(&a.id, &b.id));
    out
}

/// `tier`: `"full"` (native resolution), `"preview"` (long edge 1024),
/// `"thumb"` (long edge 256).
fn long_edge_for(tier: &str) -> AppResult<Option<u32>> {
    match tier {
        "full" => Ok(None),
        "preview" => Ok(Some(1024)),
        "thumb" => Ok(Some(256)),
        other => Err(format!("unknown tier: {other}").into()),
    }
}

/// Absolute path to the cached PNG for `(image_id, tier)`, rendering it first
/// if this is the first ask.
///
/// The frontend turns this into an `asset:` URL with `convertFileSrc`, so the
/// webview loads it the way it loads any other image — lazily, cached, and
/// with a URL that is available *synchronously*, which is what removes the
/// blank-canvas-until-you-click behaviour the blob-URL bridge had.
pub fn image_tier_path(state: &AppState, image_id: &str, tier: &str) -> AppResult<String> {
    let long_edge = long_edge_for(tier)?;
    let sha = {
        let images = state.images.lock().expect("images mutex poisoned");
        images
            .get(image_id)
            .ok_or_else(|| not_found("image", image_id))?
            .sha256
            .clone()
    };

    // A sidecar written by a run that died between registering an image and
    // hashing its pixels would carry an empty hash, and every such entry would
    // then share one cache directory. Falling back to the id keeps the key
    // unique in that case; it only costs a per-entry cache instead of a
    // per-content one.
    let key = if sha.is_empty() { image_id } else { &sha };
    let path = tier_path(&state.cache_dir, key, tier);
    if path.is_file() {
        return Ok(path.display().to_string());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let gray = state.decoded(image_id)?;
    let buf = image::GrayImage::from_raw(
        gray.width() as u32,
        gray.height() as u32,
        gray.data().to_vec(),
    )
    .ok_or_else(|| crate::error::AppError("could not view image buffer".to_string()))?;
    let resized = resize_to(buf, long_edge);

    // Written to a temporary neighbour and renamed, so a second command that
    // asks for the same tier while this one is still encoding never observes
    // a half-written PNG — `rename` within one directory is atomic.
    let tmp = path.with_extension("png.part");
    encode_png(&resized, &tmp)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path.display().to_string())
}

fn tier_path(cache_dir: &Path, sha: &str, tier: &str) -> PathBuf {
    cache_dir
        .join("tiers")
        .join(sha)
        .join(format!("{tier}.png"))
}

fn resize_to(buf: image::GrayImage, long_edge: Option<u32>) -> image::GrayImage {
    let (w, h) = (buf.width(), buf.height());
    match long_edge {
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
    }
}

/// PNG at `CompressionType::Fast`: these are cache entries a local webview
/// reads off the same disk, so spending the default encoder's time chasing a
/// smaller file buys nothing and costs the user the wait.
fn encode_png(img: &image::GrayImage, path: &Path) -> AppResult<()> {
    use image::ImageEncoder;
    use image::codecs::png::{CompressionType, FilterType as PngFilter, PngEncoder};

    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    PngEncoder::new_with_quality(writer, CompressionType::Fast, PngFilter::NoFilter)
        .write_image(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::L8,
        )
        .map_err(crate::error::AppError::from)?;
    Ok(())
}

/// Render the `thumb` tier for every image that does not have one yet.
///
/// The grid fetches thumbnails as they scroll into view, which keeps the cost
/// proportional to what is looked at — but it also means the first pass through
/// a folder is a decode per frame, arriving under the user's scroll. This warms
/// them ahead of that, off the UI thread, calling `on_done` after each one so
/// the caller can report progress and refresh the cards already on screen.
///
/// Sequential: each tier is a decode plus a resize plus an encode, all
/// CPU-bound, and a workbench that pins every core to fill a grid the user is
/// still reading is not a better workbench. Errors are per-image and do not
/// abandon the pass — one unreadable file in three thousand is a gap in the
/// grid, not a failed folder.
pub fn prewarm_thumbnails(
    state: &AppState,
    image_ids: &[String],
    mut on_done: impl FnMut(&str, usize, usize),
) {
    let total = image_ids.len();
    for (i, id) in image_ids.iter().enumerate() {
        let _ = image_tier_path(state, id, "thumb");
        on_done(id, i + 1, total);
    }
}

/// The same tiers as [`image_tier_path`], returned as bytes.
///
/// Kept for callers that cannot use a path — the contract-parity test, and
/// any future consumer without asset-protocol access.
pub fn image_data(state: &AppState, image_id: &str, tier: &str) -> AppResult<Vec<u8>> {
    let path = image_tier_path(state, image_id, tier)?;
    Ok(std::fs::read(path)?)
}

#[cfg(test)]
mod tests {
    use super::natural_cmp;

    #[test]
    fn digit_runs_order_numerically() {
        let mut names = vec!["frame_10.png", "frame_2.png", "frame_1.png"];
        names.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(names, ["frame_1.png", "frame_2.png", "frame_10.png"]);
    }

    #[test]
    fn ties_on_the_number_fall_through_to_the_rest_of_the_name() {
        assert_eq!(natural_cmp("a1b", "a1b"), std::cmp::Ordering::Equal);
        assert_eq!(natural_cmp("a1b", "a1c"), std::cmp::Ordering::Less);
        assert_eq!(natural_cmp("a01", "a1"), std::cmp::Ordering::Equal);
    }
}
