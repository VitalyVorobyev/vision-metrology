//! The reusable estimator: per-step lift, solve, bridge and composition.

use nalgebra::{Matrix3, Vector2};
use vm_primitives::{Error, Isometry2f, Point2f, Vec2f};

use super::lift::{Lift, image_to_plane, lift};
use super::solve::{Increment, Obs, SolveParams, d_rot, increment_residual, pivot_jacobian, solve};
use super::types::{
    RigFrame, RigPoseConfig, RigStats, RigStep, RigTrajectory, RigView, RotationHold, ViewOutcome,
    ViewShift, ViewUse,
};

// Invariant 20: the state and every sum are f64; results are stored as f32.

/// Where a sequence stands after its last step.
#[derive(Debug, Clone, Copy)]
struct State {
    frame: usize,
    p: Vector2<f64>,
    phi: f64,
    cov: Matrix3<f64>,
    /// The last increment, carried forward by a bridge.
    last: Option<Increment>,
    /// The last measured lever centre, where the motion noise is stated.
    pivot: Vector2<f64>,
}

/// Planar rig substrate pose: one rigid SE(2) pose per frame from per-view translations.
///
/// Reusable: [`begin`](Self::begin) starts a sequence and sizes the scratch for its views,
/// after which [`step`](Self::step) allocates nothing. [`estimate`](Self::estimate) runs a
/// whole sequence. The [module docs](super) give the method.
#[derive(Debug, Clone, Default)]
pub struct RigPoseEstimator {
    image_to_plane: Vec<Matrix3<f64>>,
    lifts: Vec<Option<Lift>>,
    obs: Vec<Obs>,
    /// Which view each entry of `obs` came from.
    obs_view: Vec<usize>,
    scratch: Vec<f64>,
    cfg: RigPoseConfig,
    state: Option<State>,
    /// The smoother's per-step record.
    pub(super) filter: Vec<super::smooth::FilterStep>,
}

fn finite_non_negative(v: f32) -> bool {
    v.is_finite() && v >= 0.0
}

fn finite_positive(v: f32) -> bool {
    v.is_finite() && v > 0.0
}

fn validate(cfg: &RigPoseConfig) -> Result<(), Error> {
    let a = cfg.anchor;
    if !(a.translation.x.is_finite()
        && a.translation.y.is_finite()
        && a.rotation.re.is_finite()
        && a.rotation.im.is_finite())
    {
        return Err(Error::InvalidConfig("rig anchor must be finite"));
    }
    if !finite_positive(cfg.huber_k) {
        return Err(Error::InvalidConfig("rig huber_k must be positive"));
    }
    if cfg.tukey_c.is_some_and(|v| !finite_positive(v)) {
        return Err(Error::InvalidConfig("rig tukey_c must be positive"));
    }
    if cfg.max_closure_px.is_some_and(|v| !finite_non_negative(v)) {
        return Err(Error::InvalidConfig(
            "rig max_closure_px must be non-negative",
        ));
    }
    if !finite_positive(cfg.rotation_prior_sigma_rad) {
        return Err(Error::InvalidConfig(
            "rig rotation_prior_sigma_rad must be positive",
        ));
    }
    validate_noise(&cfg.motion_noise)?;
    if !finite_non_negative(cfg.tuning.tolerance_mm) {
        return Err(Error::InvalidConfig(
            "rig tolerance_mm must be non-negative",
        ));
    }
    if !finite_positive(cfg.tuning.shift_sigma_floor_px) {
        return Err(Error::InvalidConfig(
            "rig shift_sigma_floor_px must be positive",
        ));
    }
    Ok(())
}

pub(super) fn validate_noise(n: &super::types::MotionNoise) -> Result<(), Error> {
    if finite_non_negative(n.translation_mm) && finite_non_negative(n.rotation_rad) {
        Ok(())
    } else {
        Err(Error::InvalidConfig(
            "rig motion noise must be non-negative",
        ))
    }
}

/// The motion noise as a covariance over `(u, θ)` at the lever centre.
pub(super) fn noise_cov(n: &super::types::MotionNoise) -> Matrix3<f64> {
    let t = (n.translation_mm as f64).powi(2);
    Matrix3::from_diagonal(&nalgebra::Vector3::new(
        t,
        t,
        (n.rotation_rad as f64).powi(2),
    ))
}

pub(super) fn to_f32_pose(p: Vector2<f64>, phi: f64) -> Isometry2f {
    Isometry2f::new(Vec2f::new(p.x as f32, p.y as f32), phi as f32)
}

/// Compose `inc` onto the pose `(p, φ, cov)`: `p' = R p + t`, `φ' = φ + θ`, with
/// first-order covariance propagation.
pub(super) fn compose(
    p: Vector2<f64>,
    phi: f64,
    cov: &Matrix3<f64>,
    inc: &Increment,
) -> (Vector2<f64>, f64, Matrix3<f64>) {
    let (s, c) = inc.theta.sin_cos();
    let p2 = Vector2::new(c * p.x - s * p.y, s * p.x + c * p.y) + inc.t;
    let a = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
    let dp = d_rot(inc.theta, p);
    let b = Matrix3::new(1.0, 0.0, dp.x, 0.0, 1.0, dp.y, 0.0, 0.0, 1.0);
    let cov2 = a * cov * a.transpose() + b * inc.cov * b.transpose();
    (p2, phi + inc.theta, cov2)
}

impl RigPoseEstimator {
    /// An estimator with no sequence started.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a sequence over `views` and return its anchor frame (frame 0).
    ///
    /// Errors when there is no view, a homography is not invertible, or `cfg` is out of
    /// range.
    pub fn begin(&mut self, views: &[RigView], cfg: &RigPoseConfig) -> Result<RigFrame, Error> {
        validate(cfg)?;
        if views.is_empty() {
            return Err(Error::InsufficientData { need: 1, got: 0 });
        }
        self.image_to_plane.clear();
        for v in views {
            let g = image_to_plane(&v.plane_to_image)
                .ok_or(Error::Degenerate("rig view homography is not invertible"))?;
            self.image_to_plane.push(g);
        }
        let n = views.len();
        self.lifts.clear();
        self.lifts.resize(n, None);
        self.obs.clear();
        self.obs.reserve(n);
        self.obs_view.clear();
        self.obs_view.reserve(n);
        self.scratch.clear();
        self.scratch.resize(n, 0.0);
        self.cfg = *cfg;
        let anchor = cfg.anchor;
        let state = State {
            frame: 0,
            p: Vector2::new(anchor.translation.x as f64, anchor.translation.y as f64),
            phi: anchor.rotation.angle() as f64,
            cov: Matrix3::zeros(),
            last: None,
            pivot: Vector2::zeros(),
        };
        self.state = Some(state);
        Ok(RigFrame {
            frame: 0,
            step: RigStep::Anchor,
            pose: anchor,
            covariance: Matrix3::zeros(),
            increment: Isometry2f::identity(),
            increment_covariance: Matrix3::zeros(),
            views_used: 0,
            residual_rms_mm: None,
            residual_max_mm: None,
            residual_scale: 1.0,
        })
    }

    /// Estimate the next frame from one shift per view (`None` where a view measured
    /// nothing), writing each view's provenance into `uses`.
    ///
    /// Both slices have one entry per view, in [`begin`](Self::begin)'s order. Allocates
    /// nothing. Errors when no sequence was started or a slice has the wrong length.
    pub fn step(
        &mut self,
        shifts: &[Option<ViewShift>],
        uses: &mut [ViewUse],
    ) -> Result<RigFrame, Error> {
        let Some(state) = self.state else {
            return Err(Error::InvalidConfig(
                "rig: begin a sequence before stepping",
            ));
        };
        let n = self.image_to_plane.len();
        for len in [shifts.len(), uses.len()] {
            if len != n {
                return Err(Error::SizeMismatch {
                    expected: n,
                    actual: len,
                });
            }
        }
        let cfg = self.cfg;
        let floor = cfg.tuning.shift_sigma_floor_px as f64;

        // Lift every shift that can be lifted, and collect the usable ones.
        self.obs.clear();
        self.obs_view.clear();
        for (v, (shift, out)) in shifts.iter().zip(uses.iter_mut()).enumerate() {
            *out = ViewUse::MISSING;
            self.lifts[v] = None;
            let Some(shift) = shift else { continue };
            let lifted = lift(&self.image_to_plane[v], shift, floor);
            self.lifts[v] = lifted;
            out.lever_mm = lifted.map(|l| Point2f::new(l.lever.x as f32, l.lever.y as f32));
            out.displacement_mm = lifted.map(|l| Vec2f::new(l.d.x as f32, l.d.y as f32));
            out.outcome = if !shift.accepted {
                ViewOutcome::NotAccepted
            } else if let (Some(limit), Some(c)) = (cfg.max_closure_px, shift.closure_px)
                && (c.is_nan() || c > limit)
            {
                ViewOutcome::ClosureFailed
            } else if let Some(l) = lifted {
                self.obs.push(Obs {
                    lift: l,
                    weight: 1.0,
                    active: true,
                    rejected: false,
                });
                self.obs_view.push(v);
                ViewOutcome::Used
            } else {
                ViewOutcome::Unliftable
            };
        }

        let prior_theta = match (cfg.rotation_hold, state.last) {
            (RotationHold::Previous, Some(last)) => last.theta,
            _ => 0.0,
        };
        // Constant velocity: the last increment, its covariance grown by the motion noise
        // at the last lever centre.
        let predicted = state.last.map(|last| {
            let j = pivot_jacobian(last.theta, state.pivot);
            Increment {
                cov: last.cov + j * noise_cov(&cfg.motion_noise) * j.transpose(),
                ..last
            }
        });
        let params = SolveParams {
            huber_k: cfg.huber_k as f64,
            tukey_c: cfg.tukey_c.map(f64::from),
            iterations: cfg.tuning.iterations.get(),
            tolerance_mm: cfg.tuning.tolerance_mm as f64,
            prior_theta,
            prior_var: (cfg.rotation_prior_sigma_rad as f64).powi(2),
            predicted,
        };
        let solved = solve(&mut self.obs, &params, &mut self.scratch);

        let (inc, step, pivot, scale) = match solved {
            Some(s) => {
                let step = if s.held {
                    RigStep::SingleView
                } else {
                    RigStep::Measured
                };
                (s.inc, step, s.pivot, s.scale)
            }
            None => {
                // Constant velocity, with the motion noise added for every bridged step.
                let inc = predicted.unwrap_or_else(|| Increment {
                    t: Vector2::zeros(),
                    theta: 0.0,
                    cov: noise_cov(&cfg.motion_noise),
                });
                (inc, RigStep::Bridged, state.pivot, 1.0)
            }
        };

        // Provenance: weights, rejections and residuals.
        for (o, &v) in self.obs.iter().zip(&self.obs_view) {
            if o.rejected {
                uses[v].outcome = ViewOutcome::Rejected;
            } else {
                uses[v].weight = o.weight as f32;
            }
        }
        let (mut sum2, mut max, mut used) = (0.0f64, 0.0f64, 0usize);
        if step != RigStep::Bridged {
            for (v, out) in uses.iter_mut().enumerate() {
                if let Some(l) = &self.lifts[v] {
                    let r = increment_residual(l, &inc);
                    out.residual_mm = Some(r as f32);
                    if out.outcome == ViewOutcome::Used {
                        sum2 += r * r;
                        max = max.max(r);
                        used += 1;
                    }
                }
            }
        }
        let redundant = used >= 2;

        let (p, phi, cov) = compose(state.p, state.phi, &state.cov, &inc);
        let frame = state.frame + 1;
        self.state = Some(State {
            frame,
            p,
            phi,
            cov,
            last: Some(inc),
            pivot,
        });
        Ok(RigFrame {
            frame,
            step,
            pose: to_f32_pose(p, phi),
            covariance: cov.cast::<f32>(),
            increment: to_f32_pose(inc.t, inc.theta),
            increment_covariance: inc.cov.cast::<f32>(),
            views_used: used,
            residual_rms_mm: redundant.then(|| (sum2 / used as f64).sqrt() as f32),
            residual_max_mm: redundant.then_some(max as f32),
            residual_scale: scale as f32,
        })
    }

    /// Run a whole sequence: `shifts` holds one entry per step and view, step-major
    /// (`shifts[k * views.len() + v]` is view `v`'s shift from frame `k` to `k + 1`).
    pub fn estimate(
        &mut self,
        views: &[RigView],
        shifts: &[Option<ViewShift>],
        cfg: &RigPoseConfig,
    ) -> Result<RigTrajectory, Error> {
        let anchor = self.begin(views, cfg)?;
        let n = views.len();
        if !shifts.len().is_multiple_of(n) {
            return Err(Error::SizeMismatch {
                expected: shifts.len().div_ceil(n) * n,
                actual: shifts.len(),
            });
        }
        let steps = shifts.len() / n;
        let mut frames = Vec::with_capacity(steps + 1);
        let mut uses = vec![ViewUse::MISSING; steps * n];
        frames.push(anchor);
        for (chunk, out) in shifts.chunks_exact(n).zip(uses.chunks_exact_mut(n)) {
            frames.push(self.step(chunk, out)?);
        }
        let stats = self.stats(&frames);
        Ok(RigTrajectory {
            frames,
            stats,
            views: n,
            uses,
        })
    }

    fn stats(&mut self, frames: &[RigFrame]) -> RigStats {
        let mut s = RigStats {
            frames: frames.len(),
            measured: 0,
            single_view: 0,
            bridged: 0,
            longest_bridge: 0,
            residual_median_mm: None,
            residual_max_mm: None,
        };
        let mut run = 0;
        self.scratch.clear();
        for f in frames {
            match f.step {
                RigStep::Anchor => {}
                RigStep::Measured => s.measured += 1,
                RigStep::SingleView => s.single_view += 1,
                RigStep::Bridged => s.bridged += 1,
            }
            run = if f.step == RigStep::Bridged {
                run + 1
            } else {
                0
            };
            s.longest_bridge = s.longest_bridge.max(run);
            if let Some(r) = f.residual_rms_mm {
                self.scratch.push(r as f64);
            }
            if let Some(m) = f.residual_max_mm {
                s.residual_max_mm = Some(s.residual_max_mm.map_or(m, |x: f32| x.max(m)));
            }
        }
        if !self.scratch.is_empty() {
            self.scratch.sort_unstable_by(f64::total_cmp);
            let k = self.scratch.len();
            let med = if k % 2 == 1 {
                self.scratch[k / 2]
            } else {
                0.5 * (self.scratch[k / 2 - 1] + self.scratch[k / 2])
            };
            s.residual_median_mm = Some(med as f32);
        }
        // A later `step` must not depend on the scratch's length.
        self.scratch.resize(self.image_to_plane.len(), 0.0);
        s
    }
}

/// The path length of `point` carried by `poses`, over that of `external`.
pub(super) fn path_length_ratio(
    poses: impl ExactSizeIterator<Item = Isometry2f>,
    point: Point2f,
    external: &[Point2f],
) -> Result<Option<f32>, Error> {
    if poses.len() != external.len() {
        return Err(Error::SizeMismatch {
            expected: poses.len(),
            actual: external.len(),
        });
    }
    let mut own = 0.0f64;
    let mut prev: Option<Point2f> = None;
    for pose in poses {
        let q = pose * point;
        if let Some(p) = prev {
            own += ((q.x - p.x) as f64).hypot((q.y - p.y) as f64);
        }
        prev = Some(q);
    }
    let ext: f64 = external
        .windows(2)
        .map(|w| ((w[1].x - w[0].x) as f64).hypot((w[1].y - w[0].y) as f64))
        .sum();
    Ok((ext > 0.0).then(|| (own / ext) as f32))
}
