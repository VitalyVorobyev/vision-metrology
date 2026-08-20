//! In-process registries backed by files on disk under the app data directory — a thin
//! Rust port of `lab/backend/src/vm_lab/store.py`'s essentials (images, models,
//! calibrations; counters; rehydrate-from-disk on startup). Kept intentionally small:
//! no thumbnail/preview caching layer (see `commands::images` for why `image_data`
//! renders on demand instead), no watershed/segment registries (not exposed as
//! commands this wave).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

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
    pub image: Image<u8>,
}

#[derive(Serialize, Deserialize)]
struct ImageSidecar {
    id: String,
    filename: String,
    width: usize,
    height: usize,
    sha256: String,
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

pub struct AppState {
    pub data_dir: PathBuf,
    pub images: Mutex<HashMap<String, ImageEntry>>,
    pub models: Mutex<HashMap<String, ModelEntry>>,
    pub calibrations: Mutex<HashMap<String, CalibrationEntry>>,
    counters: Mutex<Counters>,
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

impl AppState {
    /// A fresh, empty state rooted at `data_dir` — directories are created but nothing
    /// is loaded. Use [`AppState::rehydrated`] to also load what a previous run left.
    pub fn new(data_dir: PathBuf) -> AppResult<Self> {
        for dir in [
            images_dir(&data_dir),
            models_dir(&data_dir),
            calibrations_dir(&data_dir),
        ] {
            std::fs::create_dir_all(dir)?;
        }
        Ok(Self {
            data_dir,
            images: Mutex::new(HashMap::new()),
            models: Mutex::new(HashMap::new()),
            calibrations: Mutex::new(HashMap::new()),
            counters: Mutex::new(Counters::default()),
        })
    }

    /// [`AppState::new`], then load every image/model/calibration whose sidecar and
    /// payload files are both present — mirrors `store.py`'s `rehydrate`.
    pub fn rehydrated(data_dir: PathBuf) -> AppResult<Self> {
        let state = Self::new(data_dir)?;
        state.rehydrate_images()?;
        state.rehydrate_models()?;
        state.rehydrate_calibrations()?;
        Ok(state)
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
            let sidecar: ImageSidecar = serde_json::from_slice(&std::fs::read(&path)?)?;
            let png_path = dir.join(format!("{}.png", sidecar.id));
            if !png_path.is_file() {
                continue;
            }
            let img = image::open(&png_path)?.to_luma8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let image = Image::from_vec(w, h, img.into_raw())?;
            counters.next_image = counters.next_image.max(next_counter(&sidecar.id));
            images.insert(
                sidecar.id.clone(),
                ImageEntry {
                    id: sidecar.id,
                    filename: sidecar.filename,
                    sha256: sidecar.sha256,
                    image,
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
            let sidecar: ModelSidecar = serde_json::from_slice(&std::fs::read(&path)?)?;
            let bin_path = dir.join(format!("{}.bin", sidecar.id));
            if !bin_path.is_file() {
                continue;
            }
            let model = ShapeModel::load(&bin_path)
                .map_err(|e| AppError(format!("loading {}: {e}", bin_path.display())))?;
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
            let sidecar: CalibrationSidecar = serde_json::from_slice(&std::fs::read(&path)?)?;
            let json_path = dir.join(format!("{}.json", sidecar.id));
            if !json_path.is_file() {
                continue;
            }
            let raw = std::fs::read(&json_path)?;
            let cameras = load_calibration(&sidecar.format, &raw)?;
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

    pub fn add_image(&self, filename: String, raw: &[u8]) -> AppResult<crate::types::ImageOut> {
        let decoded = image::load_from_memory(raw)?.to_luma8();
        let (w, h) = (decoded.width() as usize, decoded.height() as usize);
        let sha256 = format!("{:x}", Sha256::digest(decoded.as_raw()));
        let id = {
            let mut counters = self.counters.lock().expect("counters mutex poisoned");
            counters.next_image += 1;
            format!("img-{}", counters.next_image)
        };

        let dir = images_dir(&self.data_dir);
        decoded.save(dir.join(format!("{id}.png")))?;
        let sidecar = ImageSidecar {
            id: id.clone(),
            filename: filename.clone(),
            width: w,
            height: h,
            sha256: sha256.clone(),
        };
        std::fs::write(
            dir.join(format!("{id}.json")),
            serde_json::to_vec(&sidecar)?,
        )?;

        let image = Image::from_vec(w, h, decoded.into_raw())?;
        let out = crate::types::ImageOut {
            id: id.clone(),
            filename: filename.clone(),
            width: w as u32,
            height: h as u32,
            sha256: sha256.clone(),
        };
        self.images.lock().expect("images mutex poisoned").insert(
            id.clone(),
            ImageEntry {
                id,
                filename,
                sha256,
                image,
            },
        );
        Ok(out)
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
