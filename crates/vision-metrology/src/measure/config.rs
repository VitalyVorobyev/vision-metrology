//! What a caliper looks for, and how it builds the profile it looks in.

use std::num::NonZeroUsize;

use vm_primitives::{BorderMode, EdgePolarity, SubpixRefine};

/// Which edges to keep from a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EdgeSelect {
    /// Every edge that passes the threshold, in profile order.
    #[default]
    All,
    /// The first along the scan direction.
    First,
    /// The last along the scan direction.
    Last,
    /// The one with the largest amplitude.
    Strongest,
    /// One edge per entry of the sequence, in scan order: for each entry, the strongest
    /// edge of that polarity strictly after the edge chosen for the previous entry.
    ///
    /// "After" compares subpixel positions on the profile, not `t`, which runs
    /// backwards along an arc with a negative extent. Equal amplitudes go to the earlier
    /// edge. When the first entry finds nothing the caliper reports
    /// [`RejectReason::WrongPolarity`]; when a later one does, it reports
    /// [`RejectReason::IncompleteSequence`].
    StrongestInOrder(EdgeSequence),
}

/// The ordered polarities [`EdgeSelect::StrongestInOrder`] looks for — one edge, or two
/// in scan order (a bar, a gap, a step and its return).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeSequence {
    /// The polarity of the first edge.
    pub first: PolaritySelect,
    /// The polarity of the edge after it, if there is one.
    pub second: Option<PolaritySelect>,
}

impl PolaritySelect {
    /// Whether an edge of `polarity` counts.
    pub(crate) fn admits(self, polarity: EdgePolarity) -> bool {
        match self {
            PolaritySelect::Any => true,
            PolaritySelect::Rising => polarity == EdgePolarity::Rising,
            PolaritySelect::Falling => polarity == EdgePolarity::Falling,
        }
    }
}

/// Which transitions count as edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolaritySelect {
    /// Both dark→bright and bright→dark.
    #[default]
    Any,
    /// Dark→bright along the scan direction only.
    Rising,
    /// Bright→dark along the scan direction only.
    Falling,
}

/// How the profile is differentiated.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Derivative {
    /// Convolve with the analytic derivative of a Gaussian of σ
    /// [`ProfileConfig::sigma`], radius `ceil(3σ)`.
    #[default]
    DerivativeOfGaussian,
    /// Smooth with a normalised Gaussian of σ [`ProfileConfig::sigma`] and half-width
    /// `radius_px`, then take central differences (one-sided at the two ends) — the
    /// textbook "Gaussian, then finite differences" operator.
    SmoothThenCentral {
        /// Half-width of the smoothing kernel, in pixels; rounded to whole samples
        /// (at least one).
        radius_px: f32,
    },
}

/// How an edge position is located on the profile.
///
/// [`GradientPeak`](Self::GradientPeak) finds where the intensity changes fastest; the two
/// level methods find where the smoothed profile crosses the level halfway between the
/// intensities on either side. On a symmetric edge they agree. On an asymmetric one (a
/// shadow, a bevel, a long tail) they do not, and the level crossing is the definition a
/// width or a datum usually means. Both level methods read the profile smoothed with the
/// Gaussian of [`ProfileConfig::sigma`], at the half-width [`ProfileConfig::derivative`]
/// sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Locate {
    /// A local extremum of the derivative, refined to subpixel position.
    GradientPeak {
        /// Subpixel refinement of the extremum.
        refine: SubpixRefine,
    },
    /// One edge where the profile crosses the mean of its two end levels, the crossing
    /// nearest the middle of the profile.
    ///
    /// The checks run in this order:
    /// 1. the end levels, the medians of the first and last `endpoint_samples` samples,
    ///    must differ by at least `min_contrast`, or the caliper reports
    ///    [`RejectReason::LowContrast`];
    /// 2. their order gives the polarity (rising when the last level is the higher), which
    ///    [`MeasureConfig::polarity`] and the first entry of an
    ///    [`EdgeSelect::StrongestInOrder`] sequence must admit, or the caliper reports
    ///    [`RejectReason::WrongPolarity`];
    /// 3. the crossing of that polarity nearest sample `(n − 1) / 2` is the edge (equal
    ///    distances: the earlier one), or the caliper reports
    ///    [`RejectReason::NoCrossing`].
    ///
    /// `threshold` is not used. The edge's amplitude is the contrast between the levels.
    MidpointCrossing {
        /// Samples at each end whose median gives that end's level.
        endpoint_samples: NonZeroUsize,
        /// Minimum difference between the two end levels, on the input pixel scale.
        min_contrast: f32,
    },
    /// Edges found as gradient peaks (refined with a three-point parabola, filtered by
    /// threshold and polarity, narrowed by [`MeasureConfig::select`]), each then moved
    /// to the crossing of its local half-contrast level.
    ///
    /// The local level is the mean of the median intensities `flank_near_px` to
    /// `flank_far_px` before and after the current position; the position moves to the
    /// crossing of that level nearest it, within `flank_near_px`, until it moves by
    /// `tol_px` or less or `max_iter` iterations have run. An edge that ends up more than
    /// `flank_near_px` from its gradient peak, or whose flanks fall off the profile,
    /// reports [`RejectReason::NoCrossing`]; flanks closer than `min_contrast` report
    /// [`RejectReason::LowContrast`]. When refined edges are no longer in strictly
    /// increasing order along the profile, the caliper reports
    /// [`RejectReason::IncompleteSequence`]. An edge's amplitude is the contrast between
    /// its flanks.
    HalfContrast {
        /// Inner edge of each flank window, in pixels from the current position.
        flank_near_px: f32,
        /// Outer edge of each flank window, in pixels from the current position.
        flank_far_px: f32,
        /// Convergence tolerance, in pixels.
        tol_px: f32,
        /// The most iterations per edge.
        max_iter: NonZeroUsize,
        /// Minimum difference between the two flank levels, on the input pixel scale.
        min_contrast: f32,
    },
}

impl Default for Locate {
    fn default() -> Self {
        Self::GradientPeak {
            refine: SubpixRefine::Parabolic3,
        }
    }
}

/// What a caliper does when its placement reaches outside the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffImage {
    /// Sample the outside with [`ProfileConfig::border`] and measure anyway;
    /// [`RejectReason::OffImage`] is reported only when no edge is found.
    #[default]
    Fill,
    /// Reject the measurement with [`RejectReason::OffImage`] before looking for
    /// edges whenever any sample lies outside `[0, w − 1] × [0, h − 1]`.
    Reject,
}

/// How a caliper turns the image under its placement into a 1-D profile and smooths it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileConfig {
    /// Gaussian σ of the smoothing, in pixels.
    ///
    /// Roughly the edge blur to expect. Too small and noise produces edges;
    /// too large and neighbouring edges merge.
    pub sigma: f32,
    /// How the profile is differentiated.
    pub derivative: Derivative,
    /// Profile sampling step along the scan axis, in pixels.
    ///
    /// `1.0` samples one profile entry per pixel. Oversampling (`0.5`) buys
    /// resolution on a sharp edge at proportional cost; `sigma` is in pixels,
    /// so the same `sigma` smooths the same physical distance at any step.
    pub step: f32,
    /// Border behaviour when the caliper overhangs the image.
    pub border: BorderMode<f32>,
    /// Whether a placement that overhangs the image is measured or rejected.
    pub off_image: OffImage,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            derivative: Derivative::default(),
            step: 1.0,
            border: BorderMode::Clamp,
            off_image: OffImage::default(),
        }
    }
}

/// How a caliper extracts edges from its profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureConfig {
    /// Minimum `|DoG response|` for an edge to be reported.
    ///
    /// On the input pixel scale, like every other threshold in this workspace:
    /// re-tune for `u16` and `f32` images.
    pub threshold: f32,
    /// Which transitions count.
    pub polarity: PolaritySelect,
    /// Which of the surviving edges to return.
    pub select: EdgeSelect,
    /// How each edge position is located on the profile.
    pub locate: Locate,
    /// Maximum angle, in degrees, between the scan direction and the image
    /// gradient at the found edge. `180.0` disables the check.
    ///
    /// A caliper that crosses an edge obliquely reports a position along its
    /// own axis, not the edge's normal, and the two differ by `1/cos θ`. At a
    /// corner or a cap there is no meaningful crossing at all. Rejecting those
    /// is what keeps a bad caliper out of the fit instead of merely
    /// down-weighted — the same gate rejects field-of-view cuts and bead
    /// end-caps.
    pub max_obliquity_deg: f32,
    /// How the profile is sampled and smoothed.
    pub profile: ProfileConfig,
}

impl Default for MeasureConfig {
    fn default() -> Self {
        Self {
            threshold: 5.0,
            polarity: PolaritySelect::default(),
            select: EdgeSelect::default(),
            locate: Locate::default(),
            max_obliquity_deg: 180.0,
            profile: ProfileConfig::default(),
        }
    }
}

/// Why a caliper returned nothing.
///
/// A caliper that finds no edge is not an error — it is a measurement result,
/// and *which gate* rejected it is the difference between "the part is missing"
/// and "the search window was too short". Tallied across a scan, the dominant
/// reason is the fastest route to a misconfigured recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// The profile was shorter than the detector needs (3 samples).
    ProfileTooShort,
    /// No response reached [`MeasureConfig::threshold`]. Either there is no
    /// edge in the window, or the window does not reach it.
    NoEdge,
    /// Edges were found, but none had the polarity
    /// [`MeasureConfig::polarity`] asked for.
    WrongPolarity,
    /// The best edge crossed at more than
    /// [`MeasureConfig::max_obliquity_deg`] from the scan direction — a corner,
    /// a cap, or a badly placed caliper.
    TooOblique,
    /// The caliper reached outside the image, so the profile is partly border
    /// fill rather than data.
    OffImage,
    /// [`EdgeSelect::StrongestInOrder`] found its first edge but no edge for a later
    /// entry after it, or [`Locate::HalfContrast`] moved edges out of order.
    IncompleteSequence,
    /// The levels a [`Locate::MidpointCrossing`] or [`Locate::HalfContrast`] edge sits
    /// between differ by less than its `min_contrast`.
    LowContrast,
    /// A level method found no crossing: the profile never crosses the midpoint level
    /// with the expected polarity, or a half-contrast edge has no crossing near it.
    NoCrossing,
}
