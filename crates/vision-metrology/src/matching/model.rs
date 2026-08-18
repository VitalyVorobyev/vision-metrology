//! The multi-level shape model produced by [`ShapeModelBuilder`](super::ShapeModelBuilder).

use vm_primitives::{Point2f, Rect2f, SmoothKind, Vec2f};

use super::config::Polarity;

/// One oriented model point.
///
/// 16 bytes, array-of-structs: all four floats are consumed by a single
/// inner-loop iteration, so keeping them adjacent is what the cache wants.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelPoint {
    /// Offset from this level's reference point, in this level's pixel units.
    pub d: Vec2f,
    /// Unit gradient direction, dark-to-bright — the same convention as
    /// [`Edgel::n`](vm_primitives::Edgel).
    pub t: Vec2f,
}

/// One pyramid level of a [`ShapeModel`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShapeModelLevel {
    /// Model points, in spatially stratified order.
    ///
    /// The order matters: greedy early termination evaluates a prefix of this
    /// list, so the prefix must be spread over the whole contour. In contour
    /// order a localised occlusion would sit entirely inside the prefix and
    /// abort a pose that is in fact correct.
    pub points: Vec<ModelPoint>,
    /// Angle step this level can be searched with, in radians.
    pub angle_step: f32,
    /// Relative scale step this level can be searched with.
    pub scale_step: f32,
    /// Largest `|d|` over the level's points, in this level's pixel units.
    pub radius: f32,
    /// Bounding box of the points, relative to the level reference point.
    pub bbox: Rect2f,
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
}

impl ShapeModel {
    pub(crate) fn from_parts(
        levels: Vec<ShapeModelLevel>,
        origin: Point2f,
        angle_range: (f32, f32),
        scale_range: (f32, f32),
        polarity: Polarity,
        smooth: SmoothKind,
    ) -> Self {
        Self {
            levels,
            origin,
            angle_range,
            scale_range,
            polarity,
            smooth,
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
///
/// The JSON document carries an explicit `format_version` so a model built
/// offline and shipped to a machine fails loudly — not by silently
/// mis-deserializing — when the format evolves.
#[cfg(feature = "serde")]
mod persist {
    use vm_primitives::Error;

    use super::ShapeModel;

    /// Bumped whenever the serialized layout of [`ShapeModel`] changes.
    pub const SHAPE_MODEL_FORMAT_VERSION: u32 = 1;

    #[derive(serde::Serialize, serde::Deserialize)]
    struct Envelope {
        format_version: u32,
        model: ShapeModel,
    }

    impl ShapeModel {
        /// Serialize the model to a versioned JSON string.
        ///
        /// # Errors
        /// Returns [`Error::InvalidConfig`] if serialization fails (which
        /// only happens for non-finite floats with some serializers; models
        /// built by this crate contain none).
        pub fn to_json(&self) -> Result<String, Error> {
            let env = Envelope {
                format_version: SHAPE_MODEL_FORMAT_VERSION,
                model: self.clone(),
            };
            serde_json::to_string(&env)
                .map_err(|_| Error::InvalidConfig("shape model serialization failed"))
        }

        /// Deserialize a model from [`Self::to_json`] output.
        ///
        /// # Errors
        /// Returns [`Error::InvalidConfig`] when the document is not valid
        /// model JSON or its `format_version` is not supported.
        pub fn from_json(json: &str) -> Result<Self, Error> {
            let env: Envelope = serde_json::from_str(json)
                .map_err(|_| Error::InvalidConfig("not a valid shape model document"))?;
            if env.format_version != SHAPE_MODEL_FORMAT_VERSION {
                return Err(Error::InvalidConfig(
                    "unsupported shape model format version",
                ));
            }
            Ok(env.model)
        }
    }
}

#[cfg(feature = "serde")]
pub use persist::SHAPE_MODEL_FORMAT_VERSION;
