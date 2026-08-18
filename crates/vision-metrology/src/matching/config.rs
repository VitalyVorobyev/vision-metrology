//! Configuration for shape model creation and search.

use vm_primitives::{Edge2DConfig, Point2f, Rect2f};

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

/// Parameters for building a [`ShapeModel`](super::ShapeModel).
///
/// The intended `angle_range` and `scale_range` are stored in the model because
/// they, together with the model's radius, determine the angle and scale steps
/// the search sweeps with. A search may narrow them but not widen them.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeModelConfig {
    /// Number of pyramid levels, or `0` to choose automatically.
    ///
    /// Automatic selection keeps adding levels while the level still has at
    /// least `min_points_per_level` points and a radius of at least 6 px, up to
    /// a hard ceiling of 5 levels.
    pub num_levels: usize,
    /// Edge detector configuration used at every pyramid level.
    pub edge: Edge2DConfig,
    /// Additional gradient-magnitude floor for a point to enter the model.
    /// `0.0` keeps everything the edge detector produced.
    pub min_contrast: f32,
    /// Maximum points per level; the excess is grid-decimated. `0` = unlimited.
    ///
    /// Decimation is always spatially uniform. Keeping the *strongest* `n`
    /// points instead would bias the score upward and break the
    /// `score ≈ 1 − occluded_fraction` property the `min_score` threshold rests
    /// on.
    pub max_points: usize,
    /// Reference point in reference-image coordinates. `None` = centroid of the
    /// level-0 points, which minimises the model radius and therefore the
    /// number of angle steps the search needs.
    pub origin: Option<Point2f>,
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
            num_levels: 0,
            edge: Edge2DConfig::default(),
            min_contrast: 0.0,
            max_points: 512,
            origin: None,
            angle_range: (-core::f32::consts::PI, core::f32::consts::PI),
            scale_range: (1.0, 1.0),
            polarity: Polarity::default(),
            min_points_per_level: 8,
        }
    }
}

/// Parameters for a [`ShapeMatcher`](super::ShapeMatcher) search.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeSearchConfig {
    /// Minimum score in `[0, 1]` for a match to be reported.
    pub min_score: f32,
    /// Greedy early-termination strength.
    ///
    /// `0.0` uses a provably safe bound — it never rejects a pose that would
    /// have scored at least `min_score` — and is the exhaustive reference used
    /// in tests. `1.0` requires the running mean to stay above `min_score` at
    /// every step: fastest, but it will miss a match whose first-evaluated
    /// points happen to be the occluded ones.
    pub greediness: f32,
    /// Maximum number of instances to report. `0` = unlimited.
    pub max_matches: usize,
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
    /// Level-0 angle step in radians; `0.0` derives it from the model radius.
    pub angle_step: f32,
    /// Level-0 relative scale step; `0.0` derives it from the model radius.
    pub scale_step: f32,
    /// Scene gradients weaker than this contribute nothing to the score.
    ///
    /// Expressed in Scharr response units on the input pixel scale: a clean
    /// black/white `u8` step gives a gradient magnitude of about 2000, so the
    /// default of 10 sits safely above 8-bit sensor noise. **Re-tune for `u16`
    /// and `f32` inputs**, whose pixel scales differ by orders of magnitude.
    pub min_contrast: f32,
    /// Subpixel pose refinement mode.
    pub refinement: Refinement,
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
            greediness: 0.9,
            max_matches: 1,
            max_overlap: 0.5,
            roi: None,
            angle_range: None,
            scale_range: None,
            angle_step: 0.0,
            scale_step: 0.0,
            min_contrast: 10.0,
            refinement: Refinement::default(),
            last_level: 0,
            max_candidates: 128,
            coarse_score_factor: 0.9,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Polarity, Refinement, ShapeModelConfig, ShapeSearchConfig};

    #[test]
    fn defaults_are_the_documented_ones() {
        let m = ShapeModelConfig::default();
        assert_eq!(m.num_levels, 0);
        assert_eq!(m.max_points, 512);
        assert_eq!(m.polarity, Polarity::Match);
        assert_eq!(m.scale_range, (1.0, 1.0));
        assert_eq!(m.min_points_per_level, 8);

        let s = ShapeSearchConfig::default();
        assert_eq!(s.min_score, 0.5);
        assert_eq!(s.greediness, 0.9);
        assert_eq!(s.max_matches, 1);
        assert_eq!(s.refinement, Refinement::Interpolate);
        assert_eq!(s.last_level, 0);
    }
}
