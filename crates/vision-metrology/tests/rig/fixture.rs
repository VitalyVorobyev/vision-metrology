//! A synthetic planar rig with known truth, shared by the rig tests, the accuracy row and
//! the bench.
//!
//! Views are affine, centred `RADIUS_MM` from the plane origin at 360°/n spacing, each
//! rotated by its own angle and given its own small shear and aspect, at `SCALE` px/mm.
//! The truth is a per-step plane-frame increment `p' = R(θ)p + t`: a steady drift along
//! x with a slow sine across it, and a slow rotation of about 0.01° per step. Each view's
//! shift is the exact image displacement of its region centroid, plus seeded Gaussian
//! noise.
#![allow(dead_code)]

use nalgebra::Matrix3;
use vision_metrology::metric::rig::{RigView, ViewShift};
use vision_metrology::{Point2f, Projective2f, Vec2f};

pub const RADIUS_MM: f64 = 30.0;
pub const SCALE: f64 = 20.0;
const CENTRE_PX: (f64, f64) = (640.0, 512.0);

/// A plane-frame increment: `p' = R(theta) p + t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Inc {
    pub tx: f64,
    pub ty: f64,
    pub theta: f64,
}

impl Inc {
    pub fn apply(&self, p: (f64, f64)) -> (f64, f64) {
        let (s, c) = self.theta.sin_cos();
        (c * p.0 - s * p.1 + self.tx, s * p.0 + c * p.1 + self.ty)
    }
}

/// Seeded xorshift64* with Box–Muller normals (invariant 12).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// The affine plane→image map of view `v` of `n`, as a 3×3 f64 matrix.
pub fn view_matrix(v: usize, n: usize) -> Matrix3<f64> {
    let alpha = std::f64::consts::TAU * v as f64 / n as f64;
    let (cx, cy) = (RADIUS_MM * alpha.cos(), RADIUS_MM * alpha.sin());
    let (s, c) = alpha.sin_cos();
    let shear = 0.04 * (v as f64 + 1.0);
    let aspect = 1.0 - 0.02 * v as f64;
    // A = SCALE · R(alpha) · [[1, shear], [0, aspect]].
    let a = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
        * Matrix3::new(1.0, shear, 0.0, 0.0, aspect, 0.0, 0.0, 0.0, 1.0)
        * SCALE;
    let mut h = a;
    h[(2, 2)] = 1.0;
    h[(0, 2)] = CENTRE_PX.0 - (a[(0, 0)] * cx + a[(0, 1)] * cy);
    h[(1, 2)] = CENTRE_PX.1 - (a[(1, 0)] * cx + a[(1, 1)] * cy);
    h
}

/// `n` views; `scale` multiplies the plane→image map as a plane-height error would.
pub fn views_scaled(n: usize, scale: f64) -> Vec<RigView> {
    (0..n)
        .map(|v| {
            let mut h = view_matrix(v, n);
            // Scaling the plane coordinates about the origin: H · diag(s, s, 1).
            for r in 0..3 {
                h[(r, 0)] *= scale;
                h[(r, 1)] *= scale;
            }
            RigView {
                plane_to_image: Projective2f::from_matrix_unchecked(h.cast::<f32>()),
            }
        })
        .collect()
}

pub fn views(n: usize) -> Vec<RigView> {
    views_scaled(n, 1.0)
}

/// The true per-step increments.
pub fn increments(steps: usize) -> Vec<Inc> {
    (0..steps)
        .map(|k| {
            let k = k as f64;
            Inc {
                tx: 0.4 + 0.05 * (k / 15.0).sin(),
                ty: 0.08 * (k / 23.0).cos(),
                theta: 2e-4 + 1e-4 * (k / 31.0).sin(),
            }
        })
        .collect()
}

/// The true substrate→plane poses `(x, y, φ)` from the identity anchor.
pub fn poses(incs: &[Inc]) -> Vec<(f64, f64, f64)> {
    let mut out = vec![(0.0, 0.0, 0.0)];
    let (mut p, mut phi) = ((0.0, 0.0), 0.0);
    for inc in incs {
        p = inc.apply(p);
        phi += inc.theta;
        out.push((p.0, p.1, phi));
    }
    out
}

fn map(h: &Matrix3<f64>, p: (f64, f64)) -> (f64, f64) {
    let q = h * nalgebra::Vector3::new(p.0, p.1, 1.0);
    (q.x / q.z, q.y / q.z)
}

/// The region centroid of view `v` at step `k`: near the image centre, moving a little.
fn centroid(v: usize, k: usize) -> (f64, f64) {
    let a = 0.3 * k as f64 + v as f64;
    (CENTRE_PX.0 + 25.0 * a.cos(), CENTRE_PX.1 + 18.0 * a.sin())
}

/// One shift per step and view, step-major, from the true increments seen through the
/// true (unscaled) views. Noise `sigma_px` per axis, reported as the covariance.
pub fn shifts(n: usize, incs: &[Inc], sigma_px: f64, seed: u64) -> Vec<Option<ViewShift>> {
    let hs: Vec<Matrix3<f64>> = (0..n).map(|v| view_matrix(v, n)).collect();
    let inv: Vec<Matrix3<f64>> = hs.iter().map(|h| h.try_inverse().unwrap()).collect();
    let mut rng = Rng::new(seed);
    let var = (sigma_px * sigma_px).max(1e-6) as f32;
    let mut out = Vec::with_capacity(incs.len() * n);
    for (k, inc) in incs.iter().enumerate() {
        for v in 0..n {
            let c = centroid(v, k);
            let p0 = map(&inv[v], c);
            let c1 = map(&hs[v], inc.apply(p0));
            let d = (
                c1.0 - c.0 + sigma_px * rng.normal(),
                c1.1 - c.1 + sigma_px * rng.normal(),
            );
            out.push(Some(ViewShift {
                d_px: Vec2f::new(d.0 as f32, d.1 as f32),
                covariance: [var, 0.0, var],
                centroid: Point2f::new(c.0 as f32, c.1 as f32),
                accepted: true,
                closure_px: None,
            }));
        }
    }
    out
}
