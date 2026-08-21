//! In-process registries backed by files on disk under the app data directory — a thin
//! Rust port of `lab/backend/src/vm_lab/store.py`'s essentials (images, models,
//! calibrations; counters; rehydrate-from-disk on startup).
//!
//! ## Images are registered eagerly and decoded lazily
//!
//! An [`ImageEntry`] is a *reference* to pixels — a path, a size, a content
//! hash — not the pixels. Decoding happens in [`AppState::decoded`], behind a
//! small LRU. That is what lets "open a folder" mean a directory listing plus
//! a few header reads instead of decoding several thousand frames into memory,
//! and it is why an entry can name a file the user still owns rather than a
//! copy of it under app-data.
//!
//! **Lock discipline**: `decoded` takes the `images` lock itself, briefly, to
//! resolve the entry. A caller must not already be holding it —
//! `std::sync::Mutex` is not reentrant, and the same thread locking twice is a
//! deadlock, not an error (`commands::rectify` has a comment about the one
//! place this bit before).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vision_metrology::matching::ShapeModel;
use vision_metrology::metric::{CameraModel, Pose3};
use vm_primitives::Image;

use crate::error::{AppError, AppResult};

pub struct ImageEntry {
    pub id: String,
    pub filename: String,
    pub sha256: String,
    pub width: usize,
    pub height: usize,
    /// The file the user opened, when this entry came from a path — `None`
    /// for a byte upload, whose pixels only ever existed inside the app.
    /// Reported to the frontend so a folder view can show provenance.
    pub path: Option<PathBuf>,
    /// The file [`AppState::decoded`] reads pixels from. Equal to `path` for
    /// an opened file; the app's own copy under `images/` for an upload.
    pub source: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct ImageSidecar {
    id: String,
    filename: String,
    width: usize,
    height: usize,
    sha256: String,
    /// The user's own file, for an entry opened by path. `#[serde(default)]`
    /// so a sidecar written before this field existed still loads — such an
    /// entry has its pixels in our own `images/{id}.png` copy.
    #[serde(default)]
    path: Option<String>,
}

pub struct ModelEntry {
    pub id: String,
    pub image_id: String,
    pub roi: [f32; 4],
    pub min_contrast: f32,
    pub num_levels: Option<usize>,
    pub model: ShapeModel,
}

#[derive(Serialize, Deserialize)]
struct ModelSidecar {
    id: String,
    image_id: String,
    roi: [f32; 4],
    min_contrast: f32,
    num_levels: Option<usize>,
}

pub struct CalibrationEntry {
    pub id: String,
    pub filename: String,
    pub format: String,
    pub cameras: Vec<(CameraModel, Pose3)>,
}

#[derive(Serialize, Deserialize)]
struct CalibrationSidecar {
    id: String,
    filename: String,
    format: String,
}

#[derive(Default)]
struct Counters {
    next_image: u64,
    next_model: u64,
    next_calibration: u64,
}

/// How many decoded frames stay resident. A workbench walks a folder one
/// frame at a time and occasionally compares two, so a handful covers the real
/// access pattern; the cost of being wrong is one decode, not a failure.
const DECODED_CACHE_CAPACITY: usize = 8;

pub struct AppState {
    pub data_dir: PathBuf,
    /// Where derived, reproducible artefacts live (image tiers). Separate from
    /// `data_dir` because everything here can be deleted and rebuilt, and
    /// should not be backed up with the user's own models.
    pub cache_dir: PathBuf,
    pub images: Mutex<HashMap<String, ImageEntry>>,
    pub models: Mutex<HashMap<String, ModelEntry>>,
    pub calibrations: Mutex<HashMap<String, CalibrationEntry>>,
    counters: Mutex<Counters>,
    /// Decoded pixels, most-recently-used last. `Arc` so a caller can hold a
    /// frame across a long operation without holding this lock.
    decoded: Mutex<Vec<(String, Arc<Image<u8>>)>>,
}

fn images_dir(root: &Path) -> PathBuf {
    root.join("images")
}
fn models_dir(root: &Path) -> PathBuf {
    root.join("models")
}
fn calibrations_dir(root: &Path) -> PathBuf {
    root.join("calibrations")
}

/// Read and parse a JSON sidecar. Errors are per-file and recoverable — see [`skip`].
fn read_sidecar<T: serde::de::DeserializeOwned>(path: &Path) -> AppResult<T> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

/// One unreadable file must not cost the whole session.
///
/// Rehydration used to be all-or-nothing: a single sidecar this build cannot
/// parse, or a model written in a format it no longer accepts, propagated out
/// of `setup` and took the *whole app* down — which, since the window is
/// already on screen by then, looked like a black window and nothing else. A
/// missing model is a missing row in a list the user can see; a dead startup
/// is not something they can act on at all.
fn skip(path: &Path, why: impl std::fmt::Display) {
    eprintln!("vm-lab: ignoring {}: {why}", path.display());
}

impl AppState {
    /// A fresh, empty state rooted at `data_dir` — directories are created but nothing
    /// is loaded. Use [`AppState::rehydrated`] to also load what a previous run left.
    pub fn new(data_dir: PathBuf) -> AppResult<Self> {
        let cache_dir = data_dir.join("cache");
        Self::with_cache_dir(data_dir, cache_dir)
    }

    /// [`AppState::new`] with the derived-artefact cache somewhere else — the
    /// running app puts it in the OS cache directory, tests in a tempdir.
    pub fn with_cache_dir(data_dir: PathBuf, cache_dir: PathBuf) -> AppResult<Self> {
        for dir in [
            images_dir(&data_dir),
            models_dir(&data_dir),
            calibrations_dir(&data_dir),
            cache_dir.clone(),
        ] {
            std::fs::create_dir_all(dir)?;
        }
        Ok(Self {
            data_dir,
            cache_dir,
            images: Mutex::new(HashMap::new()),
            models: Mutex::new(HashMap::new()),
            calibrations: Mutex::new(HashMap::new()),
            counters: Mutex::new(Counters::default()),
            decoded: Mutex::new(Vec::new()),
        })
    }

    /// Decoded pixels for `image_id`, from the LRU or from disk.
    ///
    /// See the module docs' lock-discipline note: do **not** call this while
    /// holding the `images` lock.
    pub fn decoded(&self, image_id: &str) -> AppResult<Arc<Image<u8>>> {
        let source = {
            let mut cache = self.decoded.lock().expect("decoded mutex poisoned");
            if let Some(i) = cache.iter().position(|(id, _)| id == image_id) {
                // Touch: move to the back, which is the most-recent end.
                let hit = cache.remove(i);
                let img = Arc::clone(&hit.1);
                cache.push(hit);
                return Ok(img);
            }
            let images = self.images.lock().expect("images mutex poisoned");
            images
                .get(image_id)
                .ok_or_else(|| crate::error::not_found("image", image_id))?
                .source
                .clone()
        };

        let decoded = image::open(&source)
            .map_err(|e| AppError(format!("decoding {}: {e}", source.display())))?
            .to_luma8();
        let (w, h) = (decoded.width() as usize, decoded.height() as usize);
        let img = Arc::new(Image::from_vec(w, h, decoded.into_raw())?);

        let mut cache = self.decoded.lock().expect("decoded mutex poisoned");
        cache.retain(|(id, _)| id != image_id);
        cache.push((image_id.to_string(), Arc::clone(&img)));
        while cache.len() > DECODED_CACHE_CAPACITY {
            cache.remove(0);
        }
        Ok(img)
    }

    /// [`AppState::new`], then load every image/model/calibration whose sidecar and
    /// payload files are both present — mirrors `store.py`'s `rehydrate`.
    pub fn rehydrated(data_dir: PathBuf) -> AppResult<Self> {
        let state = Self::new(data_dir)?;
        state.rehydrate()?;
        Ok(state)
    }

    /// Load what a previous run left behind. Separate from the constructor so
    /// a caller that built the state with its own cache directory can still
    /// rehydrate it.
    pub fn rehydrate(&self) -> AppResult<()> {
        self.rehydrate_images()?;
        self.rehydrate_models()?;
        self.rehydrate_calibrations()
    }

    fn rehydrate_images(&self) -> AppResult<()> {
        let dir = images_dir(&self.data_dir);
        let mut images = self.images.lock().expect("images mutex poisoned");
        let mut counters = self.counters.lock().expect("counters mutex poisoned");
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let sidecar: ImageSidecar = match read_sidecar(&path) {
                Ok(s) => s,
                Err(e) => {
                    skip(&path, e);
                    continue;
                }
            };
            // Prefer the file the user opened; fall back to our own copy for
            // an entry that came from a byte upload. Neither is decoded here —
            // rehydrating a session must not cost one decode per image.
            let own_copy = dir.join(format!("{}.png", sidecar.id));
            let source = match sidecar.path.as_ref().map(PathBuf::from) {
                Some(p) if p.is_file() => p,
                _ if own_copy.is_file() => own_copy,
                // Both gone: the user moved or deleted the file. Drop the
                // entry rather than keep one that cannot produce pixels.
                _ => continue,
            };
            counters.next_image = counters.next_image.max(next_counter(&sidecar.id));
            images.insert(
                sidecar.id.clone(),
                ImageEntry {
                    id: sidecar.id,
                    filename: sidecar.filename,
                    sha256: sidecar.sha256,
                    width: sidecar.width,
                    height: sidecar.height,
                    path: sidecar.path.map(PathBuf::from),
                    source,
                },
            );
        }
        Ok(())
    }

    fn rehydrate_models(&self) -> AppResult<()> {
        let dir = models_dir(&self.data_dir);
        let mut models = self.models.lock().expect("models mutex poisoned");
        let mut counters = self.counters.lock().expect("counters mutex poisoned");
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let sidecar: ModelSidecar = match read_sidecar(&path) {
                Ok(s) => s,
                Err(e) => {
                    skip(&path, e);
                    continue;
                }
            };
            let bin_path = dir.join(format!("{}.bin", sidecar.id));
            if !bin_path.is_file() {
                continue;
            }
            // A model written by a newer build, or one old enough that its
            // format is no longer supported, is exactly the case this must
            // survive: drop the model, keep the workbench.
            let model = match ShapeModel::load(&bin_path) {
                Ok(m) => m,
                Err(e) => {
                    skip(&bin_path, e);
                    continue;
                }
            };
            counters.next_model = counters.next_model.max(next_counter(&sidecar.id));
            models.insert(
                sidecar.id.clone(),
                ModelEntry {
                    id: sidecar.id,
                    image_id: sidecar.image_id,
                    roi: sidecar.roi,
                    min_contrast: sidecar.min_contrast,
                    num_levels: sidecar.num_levels,
                    model,
                },
            );
        }
        Ok(())
    }

    fn rehydrate_calibrations(&self) -> AppResult<()> {
        let dir = calibrations_dir(&self.data_dir);
        let mut calibrations = self
            .calibrations
            .lock()
            .expect("calibrations mutex poisoned");
        let mut counters = self.counters.lock().expect("counters mutex poisoned");
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("meta") {
                continue;
            }
            let sidecar: CalibrationSidecar = match read_sidecar(&path) {
                Ok(s) => s,
                Err(e) => {
                    skip(&path, e);
                    continue;
                }
            };
            let json_path = dir.join(format!("{}.json", sidecar.id));
            if !json_path.is_file() {
                continue;
            }
            let cameras = match std::fs::read(&json_path)
                .map_err(AppError::from)
                .and_then(|raw| load_calibration(&sidecar.format, &raw))
            {
                Ok(c) => c,
                Err(e) => {
                    skip(&json_path, e);
                    continue;
                }
            };
            counters.next_calibration = counters.next_calibration.max(next_counter(&sidecar.id));
            calibrations.insert(
                sidecar.id.clone(),
                CalibrationEntry {
                    id: sidecar.id,
                    filename: sidecar.filename,
                    format: sidecar.format,
                    cameras,
                },
            );
        }
        Ok(())
    }

    /// Register an image from bytes, keeping our own decoded PNG copy.
    ///
    /// This is the browser shell's shape of "upload", kept for parity and for
    /// drag-and-drop. The desktop path of choice is
    /// [`add_image_from_path`](Self::add_image_from_path), which moves no
    /// pixels at all.
    pub fn add_image_from_bytes(
        &self,
        filename: String,
        raw: &[u8],
    ) -> AppResult<crate::types::ImageOut> {
        let decoded = image::load_from_memory(raw)?.to_luma8();
        let (w, h) = (decoded.width() as usize, decoded.height() as usize);
        let sha256 = format!("{:x}", Sha256::digest(decoded.as_raw()));
        let id = self.next_image_id();

        let dir = images_dir(&self.data_dir);
        let own_copy = dir.join(format!("{id}.png"));
        decoded.save(&own_copy)?;
        self.register_image(id, filename, sha256, w, h, None, own_copy)
    }

    /// Register an image that already exists on disk, by path.
    ///
    /// Nothing is copied and nothing is decoded beyond the header: this is
    /// what "open a folder of 3000 frames" is made of. Re-opening a path
    /// already registered returns the existing entry instead of a duplicate.
    pub fn add_image_from_path(&self, path: &Path) -> AppResult<crate::types::ImageOut> {
        let path = path
            .canonicalize()
            .map_err(|e| AppError(format!("resolving {}: {e}", path.display())))?;

        if let Some(existing) = self
            .images
            .lock()
            .expect("images mutex poisoned")
            .values()
            .find(|e| e.path.as_deref() == Some(path.as_path()))
        {
            return Ok(crate::types::ImageOut {
                id: existing.id.clone(),
                filename: existing.filename.clone(),
                width: existing.width as u32,
                height: existing.height as u32,
                sha256: existing.sha256.clone(),
                path: Some(path.display().to_string()),
            });
        }

        let (w, h) = image::ImageReader::open(&path)
            .map_err(|e| AppError(format!("opening {}: {e}", path.display())))?
            .into_dimensions()
            .map_err(|e| AppError(format!("reading the header of {}: {e}", path.display())))?;

        // The content hash keys the tier cache, so it has to be of the pixels
        // rather than of the file — two files that decode to the same frame
        // should share one set of tiers. That costs one decode at open time,
        // which is also the decode the caller is about to want anyway, so it
        // is primed into the LRU rather than thrown away.
        let id = self.next_image_id();
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("image")
            .to_string();
        let out = self.register_image(
            id.clone(),
            filename,
            String::new(),
            w as usize,
            h as usize,
            Some(path.clone()),
            path,
        )?;
        let pixels = self.decoded(&id)?;
        let sha256 = format!("{:x}", Sha256::digest(pixels.data()));
        self.set_sha(&id, &sha256)?;
        Ok(crate::types::ImageOut { sha256, ..out })
    }

    fn next_image_id(&self) -> String {
        let mut counters = self.counters.lock().expect("counters mutex poisoned");
        counters.next_image += 1;
        format!("img-{}", counters.next_image)
    }

    #[allow(clippy::too_many_arguments)]
    fn register_image(
        &self,
        id: String,
        filename: String,
        sha256: String,
        width: usize,
        height: usize,
        path: Option<PathBuf>,
        source: PathBuf,
    ) -> AppResult<crate::types::ImageOut> {
        self.write_image_sidecar(&id, &filename, &sha256, width, height, path.as_deref())?;
        let out = crate::types::ImageOut {
            id: id.clone(),
            filename: filename.clone(),
            width: width as u32,
            height: height as u32,
            sha256: sha256.clone(),
            path: path.as_ref().map(|p| p.display().to_string()),
        };
        self.images.lock().expect("images mutex poisoned").insert(
            id.clone(),
            ImageEntry {
                id,
                filename,
                sha256,
                width,
                height,
                path,
                source,
            },
        );
        Ok(out)
    }

    fn write_image_sidecar(
        &self,
        id: &str,
        filename: &str,
        sha256: &str,
        width: usize,
        height: usize,
        path: Option<&Path>,
    ) -> AppResult<()> {
        let sidecar = ImageSidecar {
            id: id.to_string(),
            filename: filename.to_string(),
            width,
            height,
            sha256: sha256.to_string(),
            path: path.map(|p| p.display().to_string()),
        };
        std::fs::write(
            images_dir(&self.data_dir).join(format!("{id}.json")),
            serde_json::to_vec(&sidecar)?,
        )?;
        Ok(())
    }

    /// Fill in the content hash once the pixels have actually been read.
    fn set_sha(&self, id: &str, sha256: &str) -> AppResult<()> {
        let (filename, width, height, path) = {
            let mut images = self.images.lock().expect("images mutex poisoned");
            let entry = images
                .get_mut(id)
                .ok_or_else(|| crate::error::not_found("image", id))?;
            entry.sha256 = sha256.to_string();
            (
                entry.filename.clone(),
                entry.width,
                entry.height,
                entry.path.clone(),
            )
        };
        self.write_image_sidecar(id, &filename, sha256, width, height, path.as_deref())
    }

    pub fn add_model(
        &self,
        image_id: String,
        roi: [f32; 4],
        min_contrast: f32,
        num_levels: Option<usize>,
        model: ShapeModel,
    ) -> AppResult<crate::types::ModelOut> {
        let id = {
            let mut counters = self.counters.lock().expect("counters mutex poisoned");
            counters.next_model += 1;
            format!("model-{}", counters.next_model)
        };

        let dir = models_dir(&self.data_dir);
        model
            .save(dir.join(format!("{id}.bin")))
            .map_err(|e| AppError(format!("saving model: {e}")))?;
        let sidecar = ModelSidecar {
            id: id.clone(),
            image_id: image_id.clone(),
            roi,
            min_contrast,
            num_levels,
        };
        std::fs::write(
            dir.join(format!("{id}.json")),
            serde_json::to_vec(&sidecar)?,
        )?;

        let origin = model.origin();
        let out = crate::types::ModelOut {
            id: id.clone(),
            image_id: image_id.clone(),
            roi,
            min_contrast,
            num_levels,
            origin: [origin.x, origin.y],
            num_levels_built: model.num_levels(),
            point_counts: (0..model.num_levels())
                .map(|i| model.point_count(i))
                .collect(),
            reference_angle: model.reference_angle(),
        };
        self.models.lock().expect("models mutex poisoned").insert(
            id.clone(),
            ModelEntry {
                id,
                image_id,
                roi,
                min_contrast,
                num_levels,
                model,
            },
        );
        Ok(out)
    }

    pub fn add_calibration(
        &self,
        filename: String,
        raw: Vec<u8>,
    ) -> AppResult<crate::types::CalibrationOut> {
        let doc: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| AppError(format!("not valid JSON: {e}")))?;
        let format = detect_calibration_format(&doc)?;
        let cameras = load_calibration(format, &raw)?;

        let id = {
            let mut counters = self.counters.lock().expect("counters mutex poisoned");
            counters.next_calibration += 1;
            format!("cal-{}", counters.next_calibration)
        };
        let dir = calibrations_dir(&self.data_dir);
        std::fs::write(dir.join(format!("{id}.json")), &raw)?;
        let sidecar = CalibrationSidecar {
            id: id.clone(),
            filename: filename.clone(),
            format: format.to_string(),
        };
        std::fs::write(
            dir.join(format!("{id}.meta")),
            serde_json::to_vec(&sidecar)?,
        )?;

        let out = crate::types::CalibrationOut {
            id: id.clone(),
            filename: filename.clone(),
            format: format.to_string(),
            n_cameras: cameras.len(),
        };
        self.calibrations
            .lock()
            .expect("calibrations mutex poisoned")
            .insert(
                id.clone(),
                CalibrationEntry {
                    id,
                    filename,
                    format: format.to_string(),
                    cameras,
                },
            );
        Ok(out)
    }
}

fn next_counter(id: &str) -> u64 {
    id.rsplit('-')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn detect_calibration_format(doc: &serde_json::Value) -> AppResult<&'static str> {
    if doc.get("kind").and_then(|v| v.as_str()) == Some("rig_extrinsics") {
        return Ok("rig_extrinsics");
    }
    if doc.get("intrinsic").is_some() && doc.get("extrinsic").is_some() {
        return Ok("table_calibration");
    }
    Err(AppError(
        "unrecognized calibration format: expected a RigExtrinsicsExport document \
         (kind=\"rig_extrinsics\") or a table_calibration calibration.json \
         (top-level \"intrinsic\"/\"extrinsic\")"
            .to_string(),
    ))
}

fn load_calibration(format: &str, raw: &[u8]) -> AppResult<Vec<(CameraModel, Pose3)>> {
    let result = if format == "rig_extrinsics" {
        vision_metrology::metric::io::import_rig_extrinsics(raw)
    } else {
        vision_metrology::metric::io::import_table_calibration(raw)
    };
    result.map_err(|e| AppError(format!("loading calibration: {e}")))
}
