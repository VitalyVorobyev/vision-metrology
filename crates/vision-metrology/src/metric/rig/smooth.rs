//! Fixed-interval (Rauch–Tung–Striebel) smoothing of the increments.
//!
//! The state is one step's increment at the lever centre `c`, `x = (u, θ)` with
//! `t = c − R(θ)c + u`, following a random walk with the configured motion noise. Each
//! measured or single-view step observes it with the step's own covariance; a bridged
//! step observes nothing, so the smoother interpolates it from both sides. A step whose
//! innovation exceeds the gate is treated as unobserved and flagged.

use nalgebra::{Matrix3, Vector2, Vector3};
use vm_primitives::Error;

use super::RigPoseEstimator;
use super::estimator::{compose, noise_cov, to_f32_pose, validate_noise};
use super::solve::{Increment, pivot_jacobian, rot_minus_identity};
use super::types::{
    RigSmoothed, RigSmootherConfig, RigStep, RigTrajectory, SmoothedFrame, ViewOutcome,
};

// Invariant 20: the filter runs in f64.

/// The filter's record of one step.
#[derive(Debug, Clone, Copy)]
pub(super) struct FilterStep {
    /// Filtered state and covariance.
    x: Vector3<f64>,
    p: Matrix3<f64>,
    /// Predicted covariance (the predicted state is the previous filtered one).
    p_pred: Matrix3<f64>,
    gated: bool,
}

/// A diffuse start: far wider than any plausible increment.
fn diffuse() -> Matrix3<f64> {
    Matrix3::from_diagonal(&Vector3::new(1e6, 1e6, 1e2))
}

/// The increment `(t, θ)` about the plane origin as `(u, θ)` about `c`, with covariance.
fn to_pivot(t: Vector2<f64>, theta: f64, cov: &Matrix3<f64>, c: Vector2<f64>) -> Increment {
    let j = pivot_jacobian(theta, -c);
    Increment {
        t: t + rot_minus_identity(theta, c),
        theta,
        cov: j * cov * j.transpose(),
    }
}

impl RigPoseEstimator {
    /// Smooth a trajectory's increments over the whole sequence and recompose its poses
    /// from the anchor.
    ///
    /// The lever centre is the mean lift point of every used view in the sequence. Each
    /// smoothed increment carries its own covariance; the smoothed poses carry none,
    /// because smoothed increments are correlated across steps and a pose covariance
    /// composed as if they were not would be optimistic.
    pub fn smooth(
        &mut self,
        trajectory: &RigTrajectory,
        cfg: &RigSmootherConfig,
    ) -> Result<RigSmoothed, Error> {
        validate_noise(&cfg.motion_noise)?;
        if cfg.gate_sigma.is_some_and(|g| !(g.is_finite() && g > 0.0)) {
            return Err(Error::InvalidConfig("rig gate_sigma must be positive"));
        }
        let frames = &trajectory.frames;
        let Some(anchor) = frames.first() else {
            return Ok(RigSmoothed {
                frames: Vec::new(),
                gated: 0,
            });
        };

        let (mut sum, mut count) = (Vector2::zeros(), 0usize);
        for u in trajectory
            .uses
            .iter()
            .filter(|u| u.outcome == ViewOutcome::Used)
        {
            if let Some(l) = u.lever_mm {
                sum += Vector2::new(l.x as f64, l.y as f64);
                count += 1;
            }
        }
        let c = if count > 0 {
            sum / count as f64
        } else {
            Vector2::zeros()
        };
        let q = noise_cov(&cfg.motion_noise);
        let gate2 = cfg.gate_sigma.map(|g| (g as f64).powi(2));

        // Forward filter.
        let steps = frames.len() - 1;
        let mut filt: Vec<FilterStep> = std::mem::take(&mut self.filter);
        filt.clear();
        filt.reserve(steps);
        let (mut x, mut p) = (Vector3::zeros(), diffuse());
        for (k, f) in frames[1..].iter().enumerate() {
            let p_pred = if k == 0 { p } else { p + q };
            let mut gated = false;
            p = p_pred;
            if matches!(f.step, RigStep::Measured | RigStep::SingleView) {
                let t = Vector2::new(f.increment.translation.x, f.increment.translation.y);
                let z = to_pivot(
                    t.cast::<f64>(),
                    f.increment.rotation.angle() as f64,
                    &f.increment_covariance.cast::<f64>(),
                    c,
                );
                let zv = Vector3::new(z.t.x, z.t.y, z.theta);
                let y = zv - x;
                let s = p_pred + z.cov;
                if let Some(s_inv) = s.try_inverse() {
                    let d2 = y.dot(&(s_inv * y));
                    if gate2.is_some_and(|g2| d2 > g2) {
                        gated = true;
                    } else {
                        let gain = p_pred * s_inv;
                        let ik = Matrix3::identity() - gain;
                        x += gain * y;
                        p = ik * p_pred * ik.transpose() + gain * z.cov * gain.transpose();
                    }
                }
            }
            filt.push(FilterStep {
                x,
                p,
                p_pred,
                gated,
            });
        }

        // Backward pass, in place: `filt[k].x/p` become the smoothed state.
        for k in (0..steps.saturating_sub(1)).rev() {
            let next = filt[k + 1];
            let cur = filt[k];
            let Some(pred_inv) = next.p_pred.try_inverse() else {
                continue;
            };
            let gain = cur.p * pred_inv;
            filt[k].x = cur.x + gain * (next.x - cur.x);
            filt[k].p = cur.p + gain * (next.p - next.p_pred) * gain.transpose();
        }

        // Back to the plane origin, and recompose.
        let mut out = Vec::with_capacity(frames.len());
        out.push(SmoothedFrame {
            frame: 0,
            pose: anchor.pose,
            increment: anchor.increment,
            increment_covariance: Matrix3::zeros(),
            gated: false,
        });
        let mut pos = Vector2::new(
            anchor.pose.translation.x as f64,
            anchor.pose.translation.y as f64,
        );
        let mut phi = anchor.pose.rotation.angle() as f64;
        let zero = Matrix3::zeros();
        let mut gated = 0;
        for (k, s) in filt.iter().enumerate() {
            let theta = s.x.z;
            let j = pivot_jacobian(theta, c);
            let inc = Increment {
                t: Vector2::new(s.x.x, s.x.y) - rot_minus_identity(theta, c),
                theta,
                cov: j * s.p * j.transpose(),
            };
            (pos, phi, _) = compose(pos, phi, &zero, &inc);
            gated += s.gated as usize;
            out.push(SmoothedFrame {
                frame: k + 1,
                pose: to_f32_pose(pos, phi),
                increment: to_f32_pose(inc.t, inc.theta),
                increment_covariance: inc.cov.cast::<f32>(),
                gated: s.gated,
            });
        }
        self.filter = filt;
        Ok(RigSmoothed { frames: out, gated })
    }
}
