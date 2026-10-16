//! Inputs, configuration and results of the planar rig pose estimator.

use std::num::NonZeroUsize;

use nalgebra::Matrix3;
use vm_primitives::{Error, Isometry2f, Point2f, Projective2f, Vec2f};

// ── inputs ──────────────────────────────────────────────────────────────

/// One view of the rig: a camera looking at the common plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigView {
    /// Maps plane coordinates (mm, in the common plane frame) to this view's pixel
    /// coordinates. With [`homography_plane_to_image`](crate::metric::homography_plane_to_image)
    /// and a unit [`PlaneGrid`](crate::metric::PlaneGrid) (origin `(0, 0)`, 1 mm per
    /// pixel) it maps millimetres to undistorted pixels, so the shifts must then be
    /// measured on undistorted (or rectified-back) images.
    pub plane_to_image: Projective2f,
}

/// One view's measured translation over one frame step.
///
/// [`corr::TranslationEstimator`](crate::corr::TranslationEstimator) produces every field
/// but the centroid, which is the centroid of the mask it was given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewShift {
    /// Content displacement in view pixels: `p_current = p_previous + d_px`.
    pub d_px: Vec2f,
    /// Covariance of `d_px`, `[xx, xy, yy]` in px². An isotropic variance `v` is
    /// `[v, 0, v]`. [`RigPoseTuning::shift_sigma_floor_px`] is added in quadrature.
    pub covariance: [f32; 3],
    /// Centroid of the region the translation was measured on, in previous-frame view
    /// pixels. The translation is lifted to the plane at this one point.
    pub centroid: Point2f,
    /// Whether the measurement passed its own acceptance checks. A shift that did not is
    /// lifted for diagnostics but never used.
    pub accepted: bool,
    /// The measurement's closure error in pixels (for example forward-backward), when
    /// the caller has one. Gated by [`RigPoseConfig::max_closure_px`].
    pub closure_px: Option<f32>,
}

// ── configuration ───────────────────────────────────────────────────────

/// The rotation a step holds when it cannot measure one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RotationHold {
    /// The previous step's rotation increment: constant angular velocity. Zero before
    /// any step has been estimated.
    #[default]
    Previous,
    /// No rotation.
    Zero,
}

/// Per-step motion noise: the 1-σ change of the plane-frame increment from one step to
/// the next, stated at the rig's lever centre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionNoise {
    /// Translation change per step, mm.
    pub translation_mm: f32,
    /// Rotation change per step, rad.
    pub rotation_rad: f32,
}

impl Default for MotionNoise {
    fn default() -> Self {
        Self {
            translation_mm: 0.1,
            rotation_rad: 1e-3,
        }
    }
}

/// Numerical effort of the per-step solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigPoseTuning {
    /// Most reweighting iterations per step.
    pub iterations: NonZeroUsize,
    /// Stop when the update moves the lever centre by less than this, mm.
    pub tolerance_mm: f32,
    /// Added in quadrature to every shift's standard deviation, px. Covariances from one
    /// registration are typically several times optimistic, and the robust weights and
    /// the biweight radius are in units of these σ, so the floor states the accuracy a
    /// registration can actually claim.
    pub shift_sigma_floor_px: f32,
}

impl Default for RigPoseTuning {
    fn default() -> Self {
        Self {
            iterations: NonZeroUsize::new(20).unwrap(),
            tolerance_mm: 1e-7,
            shift_sigma_floor_px: 0.05,
        }
    }
}

/// Configuration of [`RigPoseEstimator`](super::RigPoseEstimator).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigPoseConfig {
    /// The substrate→plane pose of frame 0.
    pub anchor: Isometry2f,
    /// Huber radius on each view's Mahalanobis residual, in σ.
    pub huber_k: f32,
    /// When set, Tukey's biweight with this radius (σ) refines the Huber solution,
    /// graduated from a wide radius, so a view beyond it gets zero weight and is
    /// reported [`ViewOutcome::Rejected`]. Kept only while two or more spread views keep
    /// weight; otherwise the Huber solution stands.
    pub tukey_c: Option<f32>,
    /// When set, a shift whose `closure_px` exceeds this is not used.
    pub max_closure_px: Option<f32>,
    /// The rotation a step with one usable view holds.
    pub rotation_hold: RotationHold,
    /// 1-σ of that held rotation, rad: the rotation variance a single-view step reports.
    pub rotation_prior_sigma_rad: f32,
    /// Process noise a bridged step adds to the increment it carries forward.
    pub motion_noise: MotionNoise,
    /// Numerical effort.
    pub tuning: RigPoseTuning,
}

impl Default for RigPoseConfig {
    fn default() -> Self {
        Self {
            anchor: Isometry2f::identity(),
            huber_k: 2.0,
            tukey_c: Some(6.0),
            max_closure_px: None,
            rotation_hold: RotationHold::Previous,
            rotation_prior_sigma_rad: 1e-3,
            motion_noise: MotionNoise::default(),
            tuning: RigPoseTuning::default(),
        }
    }
}

/// Configuration of [`RigPoseEstimator::smooth`](super::RigPoseEstimator::smooth).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigSmootherConfig {
    /// Random-walk noise of the increment from one step to the next.
    pub motion_noise: MotionNoise,
    /// When set, a step whose increment lies further than this many σ (Mahalanobis, three
    /// degrees of freedom) from the filter's prediction is treated as unmeasured.
    pub gate_sigma: Option<f32>,
}

impl Default for RigSmootherConfig {
    fn default() -> Self {
        Self {
            motion_noise: MotionNoise::default(),
            gate_sigma: Some(5.0),
        }
    }
}

// ── results ─────────────────────────────────────────────────────────────

/// How a frame's pose was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigStep {
    /// Frame 0: the configured anchor.
    Anchor,
    /// Translation and rotation from two or more views.
    Measured,
    /// Translation only, rotation held at the prior: one usable view, or views lifted at
    /// one point.
    SingleView,
    /// No usable view: the previous increment carried forward with growing covariance.
    Bridged,
}

impl RigStep {
    /// Stable name, also used by Python.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anchor => "anchor",
            Self::Measured => "measured",
            Self::SingleView => "single_view",
            Self::Bridged => "bridged",
        }
    }
}

/// What a step did with one view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewOutcome {
    /// Used in the solve, with [`ViewUse::weight`].
    Used,
    /// No shift for this view.
    Missing,
    /// The shift failed its own acceptance.
    NotAccepted,
    /// The shift's closure exceeded [`RigPoseConfig::max_closure_px`].
    ClosureFailed,
    /// The shift could not be lifted: non-finite input, or a point the homography sends
    /// to infinity.
    Unliftable,
    /// Given zero weight by [`RigPoseConfig::tukey_c`].
    Rejected,
}

impl ViewOutcome {
    /// Stable name, also used by Python.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Used => "used",
            Self::Missing => "missing",
            Self::NotAccepted => "not_accepted",
            Self::ClosureFailed => "closure_failed",
            Self::Unliftable => "unliftable",
            Self::Rejected => "rejected",
        }
    }
}

/// One view's provenance in one step.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ViewUse {
    /// What the step did with the view.
    pub outcome: ViewOutcome,
    /// Robust weight in `[0, 1]` relative to the view's own covariance; 0 unless used.
    pub weight: f32,
    /// The lift point: the region centroid on the plane, mm. Present whenever the shift
    /// was lifted, used or not.
    pub lever_mm: Option<Point2f>,
    /// The lifted plane displacement, mm.
    pub displacement_mm: Option<Vec2f>,
    /// Distance between the lifted displacement and the step's increment at the lift
    /// point, mm. Present when the shift was lifted and the step was not bridged.
    pub residual_mm: Option<f32>,
}

impl ViewUse {
    pub(super) const MISSING: Self = Self {
        outcome: ViewOutcome::Missing,
        weight: 0.0,
        lever_mm: None,
        displacement_mm: None,
        residual_mm: None,
    };
}

impl Default for ViewUse {
    fn default() -> Self {
        Self::MISSING
    }
}

/// The substrate's pose at one frame, with its provenance.
///
/// Poses map substrate coordinates to plane coordinates (mm). Covariances are over
/// `(tx, ty, θ)` of the plane-frame transform, in mm², mm·rad and rad².
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RigFrame {
    /// Frame index; frame 0 is the anchor.
    pub frame: usize,
    /// How the pose was obtained.
    pub step: RigStep,
    /// Substrate→plane.
    pub pose: Isometry2f,
    /// First-order covariance of `pose`, accumulated over the increments from the anchor.
    pub covariance: Matrix3<f32>,
    /// The plane-frame motion from the previous frame: `pose = increment * previous pose`.
    /// The identity at the anchor.
    pub increment: Isometry2f,
    /// Covariance of `increment`.
    pub increment_covariance: Matrix3<f32>,
    /// Views used in the step.
    pub views_used: usize,
    /// RMS over the used views of [`ViewUse::residual_mm`]. Absent without redundancy
    /// (fewer than two used views).
    pub residual_rms_mm: Option<f32>,
    /// Largest used-view residual, mm, under the same condition.
    pub residual_max_mm: Option<f32>,
    /// How far the used views' residuals exceed their own covariances: the median
    /// Mahalanobis residual, never below 1. The increment covariance is inflated by its
    /// square. 1 without redundancy.
    pub residual_scale: f32,
}

/// Sequence statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RigStats {
    /// Frames, the anchor included.
    pub frames: usize,
    /// Steps of each kind.
    pub measured: usize,
    /// Steps that held rotation at the prior.
    pub single_view: usize,
    /// Steps carried forward without a measurement.
    pub bridged: usize,
    /// Longest run of consecutive bridged steps.
    pub longest_bridge: usize,
    /// Median over the steps of [`RigFrame::residual_rms_mm`].
    pub residual_median_mm: Option<f32>,
    /// Largest [`RigFrame::residual_max_mm`].
    pub residual_max_mm: Option<f32>,
}

/// A whole sequence: one [`RigFrame`] per frame and one [`ViewUse`] per step and view.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RigTrajectory {
    /// One per frame, the anchor first.
    pub frames: Vec<RigFrame>,
    /// Sequence statistics.
    pub stats: RigStats,
    pub(super) views: usize,
    pub(super) uses: Vec<ViewUse>,
}

impl RigTrajectory {
    /// Number of views.
    pub fn view_count(&self) -> usize {
        self.views
    }

    /// The per-view provenance of the step that produced `frame`; empty for the anchor
    /// and out of range.
    pub fn views(&self, frame: usize) -> &[ViewUse] {
        if frame == 0 || frame >= self.frames.len() {
            return &[];
        }
        let start = (frame - 1) * self.views;
        &self.uses[start..start + self.views]
    }

    /// The path-length ratio of the substrate point `point` (substrate coordinates, mm)
    /// against an external path with one plane position per frame. A scale diagnostic,
    /// see the [module docs](super#scale-is-not-checked-by-agreement). `Ok(None)` when the
    /// external path has zero length.
    pub fn path_length_ratio(
        &self,
        point: Point2f,
        external: &[Point2f],
    ) -> Result<Option<f32>, Error> {
        super::estimator::path_length_ratio(self.frames.iter().map(|f| f.pose), point, external)
    }
}

/// One frame of a smoothed sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SmoothedFrame {
    /// Frame index.
    pub frame: usize,
    /// Substrate→plane, composed from the smoothed increments.
    pub pose: Isometry2f,
    /// The smoothed plane-frame increment from the previous frame.
    pub increment: Isometry2f,
    /// Covariance of `increment`, over `(tx, ty, θ)`.
    pub increment_covariance: Matrix3<f32>,
    /// The step's own increment was rejected by [`RigSmootherConfig::gate_sigma`].
    pub gated: bool,
}

/// A smoothed sequence.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RigSmoothed {
    /// One per frame, the anchor first.
    pub frames: Vec<SmoothedFrame>,
    /// Steps rejected by the gate.
    pub gated: usize,
}

impl RigSmoothed {
    /// [`RigTrajectory::path_length_ratio`] over the smoothed poses.
    pub fn path_length_ratio(
        &self,
        point: Point2f,
        external: &[Point2f],
    ) -> Result<Option<f32>, Error> {
        super::estimator::path_length_ratio(self.frames.iter().map(|f| f.pose), point, external)
    }
}
