//! Planar rig substrate pose: one rigid SE(2) pose per frame for a planar substrate
//! moving under several fixed views, from one measured translation per view and step.
//!
//! Each [`RigView`] has a plane→image homography. Each step gives each view at most one
//! [`ViewShift`]: a translation in native pixels with its covariance, its acceptance, the
//! centroid of the region it was measured on and an optional closure error. The result
//! is a [`RigTrajectory`]: per frame, the substrate→plane pose, how it was obtained
//! ([`RigStep`]), the views used with their weights and residuals ([`ViewUse`]), the
//! residual in millimetres and the covariance.
//!
//! ```
//! use vision_metrology::metric::rig::{RigPoseConfig, RigPoseEstimator, RigStep, RigView, ViewShift};
//! use vision_metrology::{Point2f, Projective2f, Vec2f};
//! use nalgebra::Matrix3;
//!
//! // Three views at 20 px/mm, centred 30 mm from the plane origin at 120° spacing.
//! let views: Vec<RigView> = [0.0f32, 120.0, 240.0]
//!     .iter()
//!     .map(|deg| {
//!         let (s, c) = deg.to_radians().sin_cos();
//!         let (x0, y0) = (30.0 * c, 30.0 * s);
//!         RigView {
//!             plane_to_image: Projective2f::from_matrix_unchecked(Matrix3::new(
//!                 20.0, 0.0, 320.0 - 20.0 * x0,
//!                 0.0, 20.0, 240.0 - 20.0 * y0,
//!                 0.0, 0.0, 1.0,
//!             )),
//!         }
//!     })
//!     .collect();
//! // The substrate moves 0.5 mm along x: 10 px in every view.
//! let shift = Some(ViewShift {
//!     d_px: Vec2f::new(10.0, 0.0),
//!     covariance: [0.01, 0.0, 0.01],
//!     centroid: Point2f::new(320.0, 240.0),
//!     accepted: true,
//!     closure_px: None,
//! });
//! let shifts = vec![shift; 3 * 2]; // two steps, three views
//! let mut estimator = RigPoseEstimator::new();
//! let trajectory = estimator.estimate(&views, &shifts, &RigPoseConfig::default())?;
//! let last = trajectory.frames.last().unwrap();
//! assert_eq!(last.step, RigStep::Measured);
//! assert!((last.pose.translation.x - 1.0).abs() < 1e-5);
//! assert!(last.pose.rotation.angle().abs() < 1e-6);
//! # Ok::<(), vision_metrology::Error>(())
//! ```
//!
//! ## Method
//!
//! - **Lift at one point.** A view's translation is lifted to the plane at its region
//!   centroid `c` only: the plane displacement from `H⁻¹(c)` to `H⁻¹(c + d)`, with the
//!   pixel covariance carried through the local Jacobian. Lifting one translation at
//!   several points of a wide view would assert that the substrate does not rotate inside
//!   that view, and bias the rotation toward zero. Rotation comes from the lever arms
//!   between views.
//! - **Solve the step.** The increment `(tx, ty, θ)` is the weighted least-squares rigid
//!   motion of the lift points, solved about their mean (small-angle form first, then
//!   Gauss–Newton on the exact rotation), weighted by each view's lifted information.
//! - **Robust start.** With one view grossly wrong, a plain least-squares start inflates
//!   every view's residual and Huber weights computed from it cannot tell the wrong view
//!   apart. The weights therefore start from each view's distance to the cross-view median
//!   displacement, then iterate Huber on the Mahalanobis residuals
//!   ([`RigPoseConfig::huber_k`]).
//! - **Then a redescending loss.** With few views Huber only bounds a gross view's pull:
//!   the good views' residuals grow to balance it. Tukey's biweight
//!   ([`RigPoseConfig::tukey_c`]) refines, its radius shrinking from the largest residual,
//!   and gives the gross view zero weight. It runs from two starts, the Huber solution and
//!   the constant-velocity prediction, because with three views a gross error can be
//!   traded for a rotation that makes a good view look gross at the same cost; the
//!   prediction's Mahalanobis distance chooses between the two minima and never moves
//!   either.
//! - **Residual scale.** Registration covariances are usually optimistic.
//!   [`RigPoseTuning::shift_sigma_floor_px`] states the accuracy a shift can claim, and
//!   each step's covariance is inflated by its residual scale
//!   ([`RigFrame::residual_scale`]), the median Mahalanobis residual when above 1.
//! - **One view constrains translation only.** The step holds the rotation at the prior
//!   ([`RotationHold`]) and says [`RigStep::SingleView`]. It never invents a rotation.
//!   Two views constrain the rotation through their baseline, but cannot say which of them
//!   is wrong when they disagree.
//! - **Bridge, don't split.** A step with no usable view carries the previous increment
//!   forward (constant velocity), adds [`RigPoseConfig::motion_noise`] to its covariance
//!   for every bridged step, and says [`RigStep::Bridged`]. The chain is never split.
//! - **Compose.** `pose_k = increment_k · pose_{k−1}` from [`RigPoseConfig::anchor`], with
//!   first-order covariance propagation.
//! - **Smooth, optionally.** [`RigPoseEstimator::smooth`] runs a Rauch–Tung–Striebel
//!   smoother over the increments with a Mahalanobis gate ([`RigSmootherConfig`]).
//!
//! ## Scale is not checked by agreement
//!
//! Views that agree with each other do not prove the scale right. A plane-height error
//! common to the calibration scales every view's lift by about the same factor, and that
//! factor cancels in the comparison between views: the residuals stay small while every
//! increment is too long or too short. [`RigTrajectory::path_length_ratio`] compares the
//! estimated path with an external one (an encoder, a stage), which is the scale check;
//! it is a separate diagnostic.
//!
//! ## Units
//!
//! Plane quantities are in millimetres and radians, view quantities in pixels.
//! Covariances are over `(tx, ty, θ)`. The `d_px` sign follows
//! [`corr::Translation::shift`](crate::corr::Translation::shift): `p_current = p_previous + d_px`.

// Invariant 10: config plus a reusable estimator. Invariant 6: `step` allocates nothing
// once `begin` has sized the scratch.

mod estimator;
mod lift;
mod smooth;
mod solve;
mod types;

pub use estimator::RigPoseEstimator;
pub use types::{
    MotionNoise, RigFrame, RigPoseConfig, RigPoseTuning, RigSmoothed, RigSmootherConfig, RigStats,
    RigStep, RigTrajectory, RigView, RotationHold, SmoothedFrame, ViewOutcome, ViewShift, ViewUse,
};
