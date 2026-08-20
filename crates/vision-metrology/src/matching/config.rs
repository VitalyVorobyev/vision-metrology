//! Configuration for shape model creation and search.

use core::num::NonZeroUsize;

use vm_primitives::{Edge2DConfig, ImageView, Pixel, Point2f, PreSmooth, Rect2f};

/// How the sign of the gradient direction is treated when scoring.
///
/// The similarity measure is a mean of dot products between model directions
/// and scene directions. What differs between the modes is *where* the absolute
/// value is taken, and that choice decides which contrast reversals are
/// tolerated.
///
/// | Mode | Formula | Accepts |
/// |------|---------|---------|
/// | [`Match`](Self::Match) | `(1/n) Σ cᵢ` | only the polarity the model was built with |
/// | [`IgnoreGlobal`](Self::IgnoreGlobal) | `abs((1/n) Σ cᵢ)` | the object **or** its full contrast inversion |
/// | [`IgnoreLocal`](Self::IgnoreLocal) | `(1/n) Σ abs(cᵢ)` | any per-edge mix of the two |
///
/// `IgnoreLocal` is the most permissive and therefore the most prone to false
/// positives on cluttered scenes: it scores a contour highly even when every
/// second edge is inverted, which no real object does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Polarity {
    /// Model and scene must agree on the dark-to-bright direction of every edge.
    #[default]
    Match,
    /// The whole object may be contrast-inverted (bright-field vs dark-field).
    IgnoreGlobal,
    /// Each edge may be inverted independently.
    IgnoreLocal,
}

/// Pose refinement applied to the final, level-0 candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Refinement {
    /// Report the level-0 grid maximum: ±0.5 px, ±half an angle step.
    None,
    /// Fit a quadratic to the score surface. ~8 extra score evaluations,
    /// accuracy 0.1–0.3 px depending on edge blur.
    #[default]
    Interpolate,
    /// Locate each model edge to subpixel along its normal and solve a weighted
    /// similarity least-squares problem. Accuracy 0.02–0.05 px on a focused
    /// edge; degrades gracefully to fewer degrees of freedom on symmetric
    /// shapes, where rotation or scale is genuinely unobservable.
    LeastSquares,
}

/// A gradient-magnitude floor, and the unit it is expressed in.
///
/// Both `min_contrast` fields used to be a bare `f32` in **Scharr response
/// units on the input pixel scale** — a number that silently changes meaning
/// with the pixel type. A threshold of 10 sits just above 8-bit sensor noise
/// and admits essentially everything in a `u16` frame, where the same physical
/// contrast produces a response 257× larger. Every `u16` or `f32` user had to
/// re-derive a value that the `u8` user got for free, and nothing in the type
/// said so.
///
/// [`FractionOfRange`](Self::FractionOfRange) states the threshold relative to
/// the image itself and therefore transfers across pixel types unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Contrast {
    /// Absolute Scharr response units on the input pixel scale.
    ///
    /// What the field always meant. Exact and cheap — nothing is measured from
    /// the image — but tied to the pixel type it was tuned on.
    Raw(f32),
    /// A fraction of the response an ideal step across the image's full
    /// dynamic range would produce.
    ///
    /// The unsmoothed Scharr operator answers such a step with
    /// `16 · (max − min)` of the image, so `FractionOfRange(f)` resolves to
    /// `f · 16 · (max − min)`. The default binomial pre-smooth spreads a real
    /// step over two pixels and so reaches roughly half of that, which makes
    /// `f ≈ 0.1` the practical "clearly a real edge" setting and
    /// `f ≈ 0.0025` the equivalent of the historical `Raw(10.0)` on `u8`.
    ///
    /// Costs one min/max pass over the image per call.
    FractionOfRange(f32),
}

impl Default for Contrast {
    /// `Raw(0.0)` — accept everything the edge detector produced.
    fn default() -> Self {
        Contrast::Raw(0.0)
    }
}

/// Scharr's response to an ideal unit-amplitude step: `3 + 10 + 3` on each side.
const SCHARR_FULL_STEP_GAIN: f32 = 16.0;

impl Contrast {
    /// Resolve to Scharr response units, measuring `img` only if needed.
    ///
    /// Public because reproducing the model builder's own gradient floor is
    /// the only way to preview *exactly* the points a build will keep: a tool
    /// that shows a caller which edges are about to become model points has to
    /// threshold them the same way, and `FractionOfRange` is measured against
    /// the ROI crop rather than the whole frame, so the caller cannot derive
    /// the number without this.
    ///
    /// [`ShapeModelBuilder::build`](super::ShapeModelBuilder::build) resolves
    /// against the ROI crop; pass the same crop here.
    pub fn resolve_for<P: Pixel>(self, img: &ImageView<'_, P>) -> f32 {
        self.resolve(img)
    }

    /// Resolve to Scharr response units, measuring `img` only if needed.
    pub(crate) fn resolve<P: Pixel>(self, img: &ImageView<'_, P>) -> f32 {
        match self {
            Contrast::Raw(v) => v,
            Contrast::FractionOfRange(f) => f * SCHARR_FULL_STEP_GAIN * dynamic_range(img),
        }
    }

    /// Resolve without an image. `None` when the variant needs one.
    pub(crate) fn resolve_raw(self) -> Option<f32> {
        match self {
            Contrast::Raw(v) => Some(v),
            Contrast::FractionOfRange(_) => None,
        }
    }
}

/// `max − min` over the whole view, in the pixel type's own `f32` units.
fn dynamic_range<P: Pixel>(img: &ImageView<'_, P>) -> f32 {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for y in 0..img.height() {
        for &v in img.row(y) {
            let v = v.to_f32();
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if hi > lo { hi - lo } else { 0.0 }
}

/// Parameters for building a [`ShapeModel`](super::ShapeModel).
///
/// The intended `angle_range` and `scale_range` are stored in the model because
/// they, together with the model's radius, determine the angle and scale steps
/// the search sweeps with. A search may narrow them but not widen them.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeModelConfig {
    /// Number of pyramid levels; `None` chooses automatically.
    ///
    /// Automatic selection keeps adding levels while the level still has at
    /// least `min_points_per_level` points and a radius of at least 6 px, up to
    /// a hard ceiling of 5 levels.
    pub num_levels: Option<NonZeroUsize>,
    /// Edge detector configuration used at every pyramid level.
    pub edge: Edge2DConfig,
    /// Pre-filter applied before each pyramid decimation.
    ///
    /// Stored in the finished model, and the search reads it from there —
    /// invariant 3 requires model and scene to share the downsample kernel.
    /// [`PreSmooth::Binomial121`] is what keeps a fine-toothed contour alive at
    /// levels 3–4, where a plain box mean aliases it away.
    pub pre_smooth: PreSmooth,
    /// Additional gradient-magnitude floor for a point to enter the model.
    ///
    /// The single most consequential setting on low-relief parts: because the
    /// score divides by the full point count, a model point on faint
    /// non-repeating surface shading *dilutes* every later score rather than
    /// merely failing to help.
    pub min_contrast: Contrast,
    /// Maximum points per level; the excess is grid-decimated. `None` = unlimited.
    ///
    /// Decimation is always spatially uniform. Keeping the *strongest* `n`
    /// points instead would bias the score upward and break the
    /// `score ≈ 1 − occluded_fraction` property the `min_score` threshold rests
    /// on.
    pub max_points: Option<NonZeroUsize>,
    /// Reference point in reference-image coordinates. `None` = centroid of the
    /// level-0 points, which minimises the model radius and therefore the
    /// number of angle steps the search needs.
    pub origin: Option<Point2f>,
    /// The part's natural 0° direction in the reference image, in radians.
    ///
    /// `0.0` (the default) keeps the model frame and the reference image's
    /// axes identical, which is what every model built before this field
    /// existed does. Set it to say "*this* direction in the reference image is
    /// the part's natural orientation": the model's points are rotated by
    /// `-reference_angle` about the origin at build time, so a found
    /// [`ShapeMatch::angle`](super::ShapeMatch::angle) then reads as the
    /// natural axis's orientation **in the scene**, and a
    /// [`model_frame_map`](super::ShapeMatch::model_frame_map) crop comes out
    /// canonically oriented rather than however the part happened to sit when
    /// it was taught.
    ///
    /// Nothing about *which* points enter the model changes — this rotates the
    /// frame they are expressed in, not the extraction. Read it back off the
    /// finished model with
    /// [`ShapeModel::reference_angle`](super::ShapeModel::reference_angle),
    /// and draw over the teach image with
    /// [`reference_geometry`](super::ShapeModel::reference_geometry), which
    /// undoes it.
    pub reference_angle: f32,
    /// Rotation range the model is intended to be found over, in radians.
    ///
    /// Given as `(min, max)` with `max > min`; values are **not** wrapped, so a
    /// range straddling ±π is expressed as e.g. `(3.0, 3.4)`.
    pub angle_range: (f32, f32),
    /// Uniform scale range the model is intended to be found over.
    pub scale_range: (f32, f32),
    /// Polarity semantics baked into the model.
    pub polarity: Polarity,
    /// Automatic-level stopping threshold: a level with fewer points than this
    /// is not built. Below ~8 points a mean of dot products is noise.
    pub min_points_per_level: usize,
}

impl Default for ShapeModelConfig {
    fn default() -> Self {
        Self {
            num_levels: None,
            edge: Edge2DConfig::default(),
            pre_smooth: PreSmooth::None,
            min_contrast: Contrast::Raw(0.0),
            max_points: NonZeroUsize::new(512),
            origin: None,
            reference_angle: 0.0,
            angle_range: (-core::f32::consts::PI, core::f32::consts::PI),
            scale_range: (1.0, 1.0),
            polarity: Polarity::default(),
            min_points_per_level: 8,
        }
    }
}

/// Parameters for a [`ShapeMatcher`](super::ShapeMatcher) search.
///
/// These eight fields describe **what you are looking for**. The knobs that
/// describe *how hard the search works* — step sizes, candidate budgets, the
/// greedy abort — live in [`ShapeSearchTuning`] under `tuning`, so the common
/// case stays `ShapeSearchConfig { min_score: 0.6, ..Default::default() }` and
/// nobody reaches for `max_candidates` before they have measured anything.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeSearchConfig {
    /// Minimum score in `[0, 1]` for a match to be reported.
    pub min_score: f32,
    /// Maximum number of instances to report. `None` = unlimited.
    pub max_matches: Option<NonZeroUsize>,
    /// Maximum fraction of a candidate's model points that may fall on an
    /// already-accepted instance before the candidate is suppressed.
    pub max_overlap: f32,
    /// Region, in level-0 image coordinates, the model's reference point may
    /// lie in. `None` searches the whole image.
    pub roi: Option<Rect2f>,
    /// Rotation range to sweep; `None` uses the model's own range. A range
    /// wider than the model's is clamped to the model's.
    pub angle_range: Option<(f32, f32)>,
    /// Scale range to sweep; `None` uses the model's own range.
    pub scale_range: Option<(f32, f32)>,
    /// Scene gradients weaker than this contribute nothing to the score.
    pub min_contrast: Contrast,
    /// Subpixel pose refinement mode.
    pub refinement: Refinement,
    /// Search effort: the knobs a working setup rarely touches.
    pub tuning: ShapeSearchTuning,
}

/// Advanced search knobs for [`ShapeSearchConfig`].
///
/// Every field here trades run time against the chance of missing a match that
/// `min_score` says should be reported. Defaults are what the benches and the
/// canend dataset were tuned on.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeSearchTuning {
    /// Greedy early-termination strength.
    ///
    /// `0.0` uses a provably safe bound — it never rejects a pose that would
    /// have scored at least `min_score` — and is the exhaustive reference used
    /// in tests. `1.0` requires the running mean to stay above `min_score` at
    /// every step: fastest, but it will miss a match whose first-evaluated
    /// points happen to be the occluded ones.
    pub greediness: f32,
    /// Level-0 angle step in radians; `None` derives it from the model radius.
    pub angle_step: Option<f32>,
    /// Level-0 relative scale step; `None` derives it from the model radius.
    pub scale_step: Option<f32>,
    /// Finest pyramid level to descend to. `0` is full resolution.
    pub last_level: usize,
    /// Maximum candidates carried between pyramid levels.
    pub max_candidates: usize,
    /// Multiplier applied to `min_score` at levels above `last_level`.
    ///
    /// Coarse levels are box-downsampled without pre-smoothing, so a real match
    /// scores slightly lower there; this keeps it alive long enough to descend.
    pub coarse_score_factor: f32,
}

impl Default for ShapeSearchConfig {
    fn default() -> Self {
        Self {
            min_score: 0.5,
            max_matches: NonZeroUsize::new(1),
            max_overlap: 0.5,
            roi: None,
            angle_range: None,
            scale_range: None,
            min_contrast: Contrast::Raw(10.0),
            refinement: Refinement::default(),
            tuning: ShapeSearchTuning::default(),
        }
    }
}

impl Default for ShapeSearchTuning {
    fn default() -> Self {
        Self {
            greediness: 0.9,
            angle_step: None,
            scale_step: None,
            last_level: 0,
            max_candidates: 128,
            coarse_score_factor: 0.9,
        }
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroUsize;

    use super::{Contrast, Polarity, Refinement, ShapeModelConfig, ShapeSearchConfig};
    use vm_primitives::{Image, ImageView};

    #[test]
    fn defaults_are_the_documented_ones() {
        let m = ShapeModelConfig::default();
        assert_eq!(m.num_levels, None);
        assert_eq!(m.max_points, NonZeroUsize::new(512));
        assert_eq!(m.polarity, Polarity::Match);
        assert_eq!(m.scale_range, (1.0, 1.0));
        assert_eq!(m.min_points_per_level, 8);
        assert_eq!(m.min_contrast, Contrast::Raw(0.0));

        let s = ShapeSearchConfig::default();
        assert_eq!(s.min_score, 0.5);
        assert_eq!(s.tuning.greediness, 0.9);
        assert_eq!(s.max_matches, NonZeroUsize::new(1));
        assert_eq!(s.refinement, Refinement::Interpolate);
        assert_eq!(s.tuning.last_level, 0);
        assert_eq!(s.min_contrast, Contrast::Raw(10.0));
    }

    /// The whole point of `FractionOfRange`: the same number on `u8` and `u16`
    /// data of the same physical contrast resolves proportionally, so a value
    /// tuned on one transfers to the other.
    #[test]
    fn fraction_of_range_scales_with_the_pixel_type() {
        let img8 = Image::from_vec(2, 1, vec![0u8, 200]).expect("valid image");
        let img16 = Image::from_vec(2, 1, vec![0u16, 200 * 257]).expect("valid image");

        let c = Contrast::FractionOfRange(0.1);
        assert_eq!(c.resolve(&img8.as_view()), 0.1 * 16.0 * 200.0);
        assert_eq!(c.resolve(&img16.as_view()), 0.1 * 16.0 * 200.0 * 257.0);

        // Raw is untouched by the image, which is what makes it the default.
        let r = Contrast::Raw(10.0);
        assert_eq!(r.resolve(&img8.as_view()), 10.0);
        assert_eq!(r.resolve(&img16.as_view()), 10.0);
    }

    /// A flat image has no range, so no threshold can be derived from it.
    #[test]
    fn a_flat_image_has_zero_range() {
        let flat = Image::new_fill(4, 4, 128u8);
        let v: ImageView<'_, u8> = flat.as_view();
        assert_eq!(Contrast::FractionOfRange(0.5).resolve(&v), 0.0);
    }
}
