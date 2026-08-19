//! The multi-level shape model produced by [`ShapeModelBuilder`](super::ShapeModelBuilder).

use vm_primitives::{Point2f, PreSmooth, Rect2f, SmoothKind, Vec2f};

use super::config::Polarity;

/// One oriented model point.
///
/// 16 bytes, array-of-structs: all four floats are consumed by a single
/// inner-loop iteration, so keeping them adjacent is what the cache wants.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelPoint {
    pub(crate) d: Vec2f,
    pub(crate) t: Vec2f,
}

impl ModelPoint {
    /// Offset from this level's reference point, in this level's pixel units.
    #[inline]
    pub fn d(&self) -> Vec2f {
        self.d
    }

    /// Unit gradient direction, dark-to-bright — the same convention as
    /// [`Edgel::n`](vm_primitives::Edgel).
    #[inline]
    pub fn t(&self) -> Vec2f {
        self.t
    }
}

/// One pyramid level of a [`ShapeModel`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShapeModelLevel {
    pub(crate) points: Vec<ModelPoint>,
    pub(crate) angle_step: f32,
    pub(crate) scale_step: f32,
    pub(crate) radius: f32,
    pub(crate) bbox: Rect2f,
}

impl ShapeModelLevel {
    /// Model points, in spatially stratified order.
    ///
    /// The order matters: greedy early termination evaluates a prefix of this
    /// list, so the prefix must be spread over the whole contour. In contour
    /// order a localised occlusion would sit entirely inside the prefix and
    /// abort a pose that is in fact correct. Read-only for that reason — a
    /// caller who reorders or extends the points invalidates both the greedy
    /// bound and `score ≈ 1 − occluded_fraction`.
    #[inline]
    pub fn points(&self) -> &[ModelPoint] {
        &self.points
    }

    /// Angle step this level can be searched with, in radians.
    #[inline]
    pub fn angle_step(&self) -> f32 {
        self.angle_step
    }

    /// Relative scale step this level can be searched with.
    #[inline]
    pub fn scale_step(&self) -> f32 {
        self.scale_step
    }

    /// Largest `|d|` over the level's points, in this level's pixel units.
    #[inline]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Bounding box of the points, relative to the level reference point.
    #[inline]
    pub fn bbox(&self) -> Rect2f {
        self.bbox
    }
}

/// A rotation- and scale-searchable model of an object's edge geometry.
///
/// Built by [`ShapeModelBuilder`](super::ShapeModelBuilder) from a reference
/// image, or by [`from_edgels`](Self::from_edgels) /
/// [`from_polylines`](Self::from_polylines) from geometry alone. Lifetime-free
/// and `Clone`: roughly 21 KB for a 1000-point 5-level model.
///
/// Level `l` holds points extracted from level `l` of the reference image's own
/// pyramid — **not** level-0 points divided by `2^l`. Model and scene then
/// suffer the same box-downsample aliasing and their coarse gradient directions
/// agree; decimated level-0 points would give the model a sharpness the scene
/// does not have, systematically depressing the coarse score.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShapeModel {
    levels: Vec<ShapeModelLevel>,
    origin: Point2f,
    angle_range: (f32, f32),
    scale_range: (f32, f32),
    polarity: Polarity,
    smooth: SmoothKind,
    pre_smooth: PreSmooth,
}

impl ShapeModel {
    pub(crate) fn from_parts(
        levels: Vec<ShapeModelLevel>,
        origin: Point2f,
        angle_range: (f32, f32),
        scale_range: (f32, f32),
        polarity: Polarity,
        smooth: SmoothKind,
        pre_smooth: PreSmooth,
    ) -> Self {
        Self {
            levels,
            origin,
            angle_range,
            scale_range,
            polarity,
            smooth,
            pre_smooth,
        }
    }

    /// Pre-smoothing the model's own gradients were computed with.
    ///
    /// The scene must be smoothed the same way, or model and scene directions
    /// are computed from differently band-limited images and disagree for
    /// reasons that have nothing to do with the object.
    #[inline]
    pub fn smooth(&self) -> SmoothKind {
        self.smooth
    }

    /// Pyramid pre-filter the model's own levels were built with.
    ///
    /// Invariant 3: the model build and the scene search must use the **same**
    /// downsample kernel, or model and scene suffer different aliasing and
    /// their coarse gradient directions disagree for reasons that have nothing
    /// to do with the object. The matcher reads this off the model rather than
    /// from its own config, so a stored model cannot be searched with the
    /// wrong kernel.
    #[inline]
    pub fn pre_smooth(&self) -> PreSmooth {
        self.pre_smooth
    }

    /// All pyramid levels, finest first.
    #[inline]
    pub fn levels(&self) -> &[ShapeModelLevel] {
        &self.levels
    }

    /// Level `i`, or `None` when `i >= num_levels()`.
    #[inline]
    pub fn level(&self, i: usize) -> Option<&ShapeModelLevel> {
        self.levels.get(i)
    }

    /// Number of pyramid levels; always at least 1.
    #[inline]
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Reference point in reference-image (level-0) coordinates.
    ///
    /// A reported [`ShapeMatch::position`](super::ShapeMatch::position) is where
    /// this point landed in the scene.
    #[inline]
    pub fn origin(&self) -> Point2f {
        self.origin
    }

    /// Rotation range, in radians, the model was built for.
    #[inline]
    pub fn angle_range(&self) -> (f32, f32) {
        self.angle_range
    }

    /// Uniform scale range the model was built for.
    #[inline]
    pub fn scale_range(&self) -> (f32, f32) {
        self.scale_range
    }

    /// Polarity semantics baked into the model.
    #[inline]
    pub fn polarity(&self) -> Polarity {
        self.polarity
    }

    /// Number of points at level `i`, or 0 when the level does not exist.
    #[inline]
    pub fn point_count(&self, i: usize) -> usize {
        self.levels.get(i).map_or(0, |l| l.points.len())
    }

    /// Model points at level 0 mapped back into reference-image coordinates.
    ///
    /// Convenience for drawing the model over its own reference image; the
    /// search never needs this.
    pub fn reference_points(&self) -> Vec<Point2f> {
        self.levels.first().map_or_else(Vec::new, |l| {
            l.points.iter().map(|p| self.origin + p.d).collect()
        })
    }
}

/// Versioned persistence for [`ShapeModel`] (`serde` feature).
#[cfg(feature = "serde")]
mod persist {
    use std::path::Path;

    use vm_primitives::Error;

    use super::ShapeModel;

    /// Bumped whenever the serialized layout of [`ShapeModel`] changes.
    ///
    /// Deliberately **not public**: the number is a compatibility check between
    /// this crate's writer and its reader, and every way a caller could act on
    /// it — "is this file loadable?" — is answered by [`ShapeModel::load`]
    /// returning an error. Exposing it invited callers to build the envelope
    /// themselves, which is exactly what the opaque byte API removes.
    ///
    /// | Version | Change |
    /// |---|---|
    /// | 1 | Initial format. `Point2f` / `Vec2f` as `{"x": …, "y": …}`. |
    /// | 2 | `Point2f` / `Vec2f` became nalgebra aliases, which serialize as flat `[x, y]` arrays. |
    /// | 3 | The v0.3 API reset, batched: the document is an opaque byte string rather than documented JSON, `ModelPoint` / `ShapeModelLevel` fields became read-only, and the model carries the pyramid `PreSmooth` it was built with (invariant 3). |
    pub(crate) const FORMAT_VERSION: u32 = 3;

    #[derive(serde::Serialize, serde::Deserialize)]
    struct Envelope {
        format_version: u32,
        model: ShapeModel,
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
                model: self.clone(),
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
            if env.format_version != FORMAT_VERSION {
                return Err(Error::InvalidConfig(
                    "unsupported shape model format version",
                ));
            }
            Ok(env.model)
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
}
