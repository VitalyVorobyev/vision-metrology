//! What a caliper looks for, and how it builds the profile it looks in.

use vm_primitives::BorderMode;

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

/// How a caliper turns the image under its placement into a 1-D profile and smooths it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileConfig {
    /// Gaussian σ of the 1-D derivative-of-Gaussian kernel, in pixels.
    ///
    /// Roughly the edge blur to expect. Too small and noise produces edges;
    /// too large and neighbouring edges merge.
    pub sigma: f32,
    /// Profile sampling step along the scan axis, in pixels.
    ///
    /// `1.0` samples one profile entry per pixel. Oversampling (`0.5`) buys
    /// resolution on a sharp edge at proportional cost; `sigma` is in pixels,
    /// so the same `sigma` smooths the same physical distance at any step.
    pub step: f32,
    /// Border behaviour when the caliper overhangs the image.
    pub border: BorderMode<f32>,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            step: 1.0,
            border: BorderMode::Clamp,
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
}
