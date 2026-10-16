//! The multi-level shape model produced by [`ShapeModelBuilder`](super::ShapeModelBuilder).

use vm_primitives::{Point2f, PreSmooth, Rect2f, SmoothKind, Vec2f};

use super::config::Polarity;

/// One oriented model point.
///
/// 16 bytes, array-of-structs: all four floats are consumed by a single
/// inner-loop iteration, so keeping them adjacent is what the cache wants.
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// One teach-time edge point, stored **before** grid decimation, at level 0.
///
/// [`resample_at`](ShapeModel::resample_at) is what this exists for: rebuild
/// a model's pyramid levels at an arbitrary uniform scale without the
/// original reference image. It is the same information `RawPoint`
/// (`matching::build`) carries internally, minus the decimation and
/// per-level assembly a *fixed*-scale model applies once at build time —
/// `resample_at` reruns that assembly at a caller-chosen scale instead.
///
/// `strength` is the edge detector's own gradient-magnitude units (or `1.0`
/// for a geometry-only model, which has no such unit) — `decimate` in
/// `matching::build` uses it only to pick a cell's surviving point, never as
/// a score input, so its scale never matters outside that tie-break.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TeachPoint {
    /// Offset from the model's origin, level-0 pixel units.
    pub(crate) d: Vec2f,
    /// Unit gradient direction, dark-to-bright.
    pub(crate) t: Vec2f,
    /// Gradient magnitude (or `1.0` for a direction-only source).
    pub(crate) strength: f32,
}

/// One pyramid level of a [`ShapeModel`].
#[derive(Debug, Clone, PartialEq)]
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
pub struct ShapeModel {
    levels: Vec<ShapeModelLevel>,
    origin: Point2f,
    angle_range: (f32, f32),
    scale_range: (f32, f32),
    polarity: Polarity,
    smooth: SmoothKind,
    pre_smooth: PreSmooth,
    /// Level-0 edge points before grid decimation — `None` for a model
    /// loaded from a format-3 document, which predates this field. See
    /// [`resample_at`](Self::resample_at).
    teach_points: Option<Vec<TeachPoint>>,
    /// Angle, in radians, the model frame is rotated by relative to the
    /// reference image — the caller's chosen canonical orientation. `0.0`
    /// (the default, and what a format-4 document deserializes to) means the
    /// two frames coincide. See [`reference_angle`](Self::reference_angle).
    reference_angle: f32,
}

impl ShapeModel {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        levels: Vec<ShapeModelLevel>,
        origin: Point2f,
        angle_range: (f32, f32),
        scale_range: (f32, f32),
        polarity: Polarity,
        smooth: SmoothKind,
        pre_smooth: PreSmooth,
        teach_points: Option<Vec<TeachPoint>>,
        reference_angle: f32,
    ) -> Self {
        Self {
            levels,
            origin,
            angle_range,
            scale_range,
            polarity,
            smooth,
            pre_smooth,
            teach_points,
            reference_angle,
        }
    }

    /// Stamp the canonical orientation onto an already-assembled model.
    ///
    /// Only [`resample_at`](Self::resample_at) needs this: it rebuilds levels
    /// from `teach_points`, which are **already** in the rotated model frame,
    /// so its own assembly must not rotate a second time — it assembles with
    /// `reference_angle: 0.0` and restores the original value here.
    pub(super) fn set_reference_angle(&mut self, angle: f32) {
        self.reference_angle = angle;
    }

    /// The level-0 edge points recorded before grid decimation, or `None`
    /// for a model loaded from a format-3 document (see
    /// [`resample_at`](Self::resample_at)).
    pub(crate) fn teach_points(&self) -> Option<&[TeachPoint]> {
        self.teach_points.as_deref()
    }

    /// Number of teach-time points [`resample_at`](Self::resample_at) has to
    /// work with, or `0` when the model carries none (a format-3 document).
    ///
    /// Diagnostic only — not the same as [`point_count`](Self::point_count),
    /// which counts the *decimated* points a level was actually built from.
    #[inline]
    pub fn teach_point_count(&self) -> usize {
        self.teach_points.as_ref().map_or(0, Vec::len)
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
    /// The model build and the scene search must use the **same**
    /// downsample kernel, or model and scene suffer different aliasing and
    /// their coarse gradient directions disagree for reasons that have nothing
    /// to do with the object. The matcher reads this off the model rather than
    /// from its own config, so a stored model cannot be searched with the
    /// wrong kernel.
    // Invariant 3.
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

    /// Canonical orientation, in radians: how far the **model frame** is
    /// rotated from the reference image's own axes.
    ///
    /// `0.0` — the default, and what every model built before this field
    /// existed reports — means the two frames coincide, and everything below
    /// reduces to the identity. A non-zero value is the caller saying "this
    /// direction in the reference image is the part's natural 0°", so that a
    /// [`ShapeMatch::angle`](super::ShapeMatch::angle) reads as the natural
    /// axis's orientation *in the scene* rather than as rotation away from
    /// however the part happened to sit when it was taught.
    ///
    /// Which frame a point is in matters once this is non-zero:
    ///
    /// | Frame | What is in it | Read it with |
    /// |---|---|---|
    /// | model | [`ModelPoint::d`], [`CropSpec::rect`](super::CropSpec), what [`ShapeMatch::pose`](super::ShapeMatch::pose) consumes | [`model_geometry`](Self::model_geometry) |
    /// | reference image | the pixels the model was taught from | [`reference_geometry`](Self::reference_geometry), [`reference_points`](Self::reference_points) |
    #[inline]
    pub fn reference_angle(&self) -> f32 {
        self.reference_angle
    }

    /// Model points at level `level`, in **model-frame** coordinates
    /// (`origin + d`), paired with their unit gradient directions.
    ///
    /// This is the frame [`ShapeMatch::pose`](super::ShapeMatch::pose)
    /// consumes: `pose * p` places `p` into the scene, which is how a found
    /// instance is drawn over a scene image. Level `l`'s stored offsets are in
    /// level-`l` pixel units; they are scaled back to level-0 units here, so
    /// every level comes out in one comparable frame.
    ///
    /// Empty when `level >= num_levels()`.
    pub fn model_geometry(&self, level: usize) -> Vec<(Point2f, Vec2f)> {
        let scale = (1usize << level) as f32;
        self.levels.get(level).map_or_else(Vec::new, |l| {
            l.points
                .iter()
                .map(|p| (self.origin + p.d * scale, p.t))
                .collect()
        })
    }

    /// Model points at level `level`, in **reference-image** coordinates,
    /// paired with their unit gradient directions.
    ///
    /// [`model_geometry`](Self::model_geometry) rotated back by
    /// [`reference_angle`](Self::reference_angle) about the origin — what to
    /// draw over the image the model was taught from. Identical to
    /// `model_geometry` for the usual `reference_angle == 0.0` model.
    ///
    /// Empty when `level >= num_levels()`.
    pub fn reference_geometry(&self, level: usize) -> Vec<(Point2f, Vec2f)> {
        let (sin, cos) = self.reference_angle.sin_cos();
        let rot = |v: Vec2f| Vec2f::new(cos * v.x - sin * v.y, sin * v.x + cos * v.y);
        self.model_geometry(level)
            .into_iter()
            .map(|(p, t)| (self.origin + rot(p - self.origin), rot(t)))
            .collect()
    }

    /// Model points at level 0 mapped back into reference-image coordinates.
    ///
    /// Convenience for drawing the model over its own reference image; the
    /// search never needs this. Positions only — use
    /// [`reference_geometry`](Self::reference_geometry) when the gradient
    /// directions are wanted too.
    pub fn reference_points(&self) -> Vec<Point2f> {
        self.reference_geometry(0)
            .into_iter()
            .map(|(p, _)| p)
            .collect()
    }
}

/// Versioned persistence for [`ShapeModel`] (`serde` feature).
#[cfg(feature = "serde")]
mod persist;
