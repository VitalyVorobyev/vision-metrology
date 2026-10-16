//! The stored model document: private types that only this module reads and writes.
//!
//! The in-memory [`ShapeModel`] has no serde derives, so its layout is free to
//! change; the document types below are the stored format. Their field names,
//! order and defaults are the format, so a change to them is a format change and
//! bumps [`FORMAT_VERSION`]. `tests/model_format.rs` holds a frozen format-5
//! document and checks that it still loads and writes back byte for byte.

use std::path::Path;

use serde::{Deserialize, Serialize};
use vm_primitives::{Error, Point2f, PreSmooth, Rect2f, SmoothKind, Vec2f};

use super::{ModelPoint, ShapeModel, ShapeModelLevel, TeachPoint};
use crate::matching::Polarity;

/// Bumped whenever the stored document's layout changes.
///
/// Deliberately **not public**: the number is a compatibility check between
/// this crate's writer and its reader, and every way a caller could act on
/// it — "is this file loadable?" — is answered by [`ShapeModel::load`]
/// returning an error.
//
// Exposing it invited callers to build the envelope themselves, which is exactly
// what the opaque byte API removes.
//
// Format history; this table is its single home:
//
// | Version | Change |
// |---|---|
// | 1 | Initial format. `Point2f` / `Vec2f` as `{"x": …, "y": …}`. |
// | 2 | `Point2f` / `Vec2f` became nalgebra aliases, which serialize as flat `[x, y]` arrays. |
// | 3 | The document is an opaque byte string rather than documented JSON, its point and level fields are read-only, and the model carries the pyramid `PreSmooth` it was built with (invariant 3). |
// | 4 | The model additionally carries its level-0 `teach_points` (pre-decimation edge points), which `ShapeModel::resample_at` needs. A format-3 document still loads — `teach_points` defaults to `None` on a document that predates the field — but `resample_at` on the result returns `Error::InvalidConfig`. |
// | 5 | The model carries its `reference_angle` — the canonical orientation its frame is rotated onto. A format-4 document still loads and reads `0.0`, which is exactly what it meant: model frame and reference image coincide. |
pub(crate) const FORMAT_VERSION: u32 = 5;

/// The oldest format version [`ShapeModel::load`]/[`from_bytes`](ShapeModel::from_bytes)
/// still accept. `teach_points` is `#[serde(default)]`, so a version-3
/// document (which has no such field at all) deserializes cleanly with
/// `teach_points: None` — the only behavioral difference a version-3
/// reader sees is that [`ShapeModel::resample_at`] then reports a clear
/// error instead of silently inventing teach data. Below this floor a
/// document is old enough (pre-nalgebra-alias, `{"x":…,"y":…}` points)
/// that its *shape* differs, not just a missing field, so it is refused
/// rather than guessed at.
const MIN_SUPPORTED_FORMAT_VERSION: u32 = 3;

#[derive(Serialize, Deserialize)]
struct Envelope {
    format_version: u32,
    model: ModelDoc,
}

#[derive(Serialize, Deserialize)]
struct ModelDoc {
    levels: Vec<LevelDoc>,
    origin: Point2f,
    angle_range: (f32, f32),
    scale_range: (f32, f32),
    polarity: Polarity,
    smooth: SmoothKind,
    pre_smooth: PreSmooth,
    #[serde(default)]
    teach_points: Option<Vec<TeachPointDoc>>,
    #[serde(default)]
    reference_angle: f32,
}

#[derive(Serialize, Deserialize)]
struct LevelDoc {
    points: Vec<PointDoc>,
    angle_step: f32,
    scale_step: f32,
    radius: f32,
    bbox: Rect2f,
}

#[derive(Serialize, Deserialize)]
struct PointDoc {
    d: Vec2f,
    t: Vec2f,
}

#[derive(Serialize, Deserialize)]
struct TeachPointDoc {
    d: Vec2f,
    t: Vec2f,
    strength: f32,
}

impl From<&ShapeModel> for ModelDoc {
    fn from(m: &ShapeModel) -> Self {
        Self {
            levels: m
                .levels
                .iter()
                .map(|l| LevelDoc {
                    points: l
                        .points
                        .iter()
                        .map(|p| PointDoc { d: p.d, t: p.t })
                        .collect(),
                    angle_step: l.angle_step,
                    scale_step: l.scale_step,
                    radius: l.radius,
                    bbox: l.bbox,
                })
                .collect(),
            origin: m.origin,
            angle_range: m.angle_range,
            scale_range: m.scale_range,
            polarity: m.polarity,
            smooth: m.smooth,
            pre_smooth: m.pre_smooth,
            teach_points: m.teach_points.as_ref().map(|tp| {
                tp.iter()
                    .map(|p| TeachPointDoc {
                        d: p.d,
                        t: p.t,
                        strength: p.strength,
                    })
                    .collect()
            }),
            reference_angle: m.reference_angle,
        }
    }
}

impl From<ModelDoc> for ShapeModel {
    fn from(doc: ModelDoc) -> Self {
        Self {
            levels: doc
                .levels
                .into_iter()
                .map(|l| ShapeModelLevel {
                    points: l
                        .points
                        .into_iter()
                        .map(|p| ModelPoint { d: p.d, t: p.t })
                        .collect(),
                    angle_step: l.angle_step,
                    scale_step: l.scale_step,
                    radius: l.radius,
                    bbox: l.bbox,
                })
                .collect(),
            origin: doc.origin,
            angle_range: doc.angle_range,
            scale_range: doc.scale_range,
            polarity: doc.polarity,
            smooth: doc.smooth,
            pre_smooth: doc.pre_smooth,
            teach_points: doc.teach_points.map(|tp| {
                tp.into_iter()
                    .map(|p| TeachPoint {
                        d: p.d,
                        t: p.t,
                        strength: p.strength,
                    })
                    .collect()
            }),
            reference_angle: doc.reference_angle,
        }
    }
}

impl ShapeModel {
    /// Serialize the model to an opaque, versioned byte string.
    ///
    /// The encoding is **not** part of the API: it carries a format version
    /// and this crate is the only thing that may read it. A model built by
    /// one version and loaded by another fails loudly rather than silently
    /// mis-deserializing.
    ///
    /// # Errors
    /// [`Error::InvalidConfig`] if serialization fails (non-finite floats;
    /// models built by this crate contain none).
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let env = Envelope {
            format_version: FORMAT_VERSION,
            model: ModelDoc::from(self),
        };
        serde_json::to_vec(&env)
            .map_err(|_| Error::InvalidConfig("shape model serialization failed"))
    }

    /// Read back a model written by [`to_bytes`](Self::to_bytes).
    ///
    /// # Errors
    /// [`Error::InvalidConfig`] when the bytes are not a model document or
    /// carry an unsupported format version.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let env: Envelope = serde_json::from_slice(bytes)
            .map_err(|_| Error::InvalidConfig("not a valid shape model document"))?;
        if env.format_version < MIN_SUPPORTED_FORMAT_VERSION || env.format_version > FORMAT_VERSION
        {
            return Err(Error::InvalidConfig(
                "unsupported shape model format version",
            ));
        }
        Ok(env.model.into())
    }

    /// Write the model to `path`.
    ///
    /// # Errors
    /// [`Error::InvalidConfig`] on a serialization or I/O failure.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes)
            .map_err(|_| Error::InvalidConfig("could not write shape model file"))
    }

    /// Read a model written by [`save`](Self::save).
    ///
    /// # Errors
    /// [`Error::InvalidConfig`] on an I/O failure, a document that is not a
    /// model, or an unsupported format version.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes = std::fs::read(path)
            .map_err(|_| Error::InvalidConfig("could not read shape model file"))?;
        Self::from_bytes(&bytes)
    }
}
