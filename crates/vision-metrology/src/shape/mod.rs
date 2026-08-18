//! Shape detection for the `vision-metrology` workspace.
//!
//! Provides two detection families:
//!
//! ## Line Segment Detection (LSD)
//! [`LsdDetector`] implements gradient-coherence region growing with NFA
//! (Number of False Alarms) validation. Segments are returned as
//! [`LineSegment2f`] with subpixel endpoints, support-region width, and NFA
//! score.
//!
//! ```rust
//! use vision_metrology::{LsdDetector, LsdConfig};
//! use vm_primitives::{Image, Point2f};
//!
//! // Build a 64×64 image with a horizontal step edge at y=32.
//! let w = 64usize;
//! let h = 64usize;
//! let data: Vec<u8> = (0..h).flat_map(|y| {
//!     let v: u8 = if y >= 32 { 255 } else { 0 };
//!     vec![v; w]
//! }).collect();
//! let img = Image::from_vec(w, h, data).expect("valid image");
//!
//! let mut det = LsdDetector::new();
//! let cfg = LsdConfig { scale: 1.0, ..LsdConfig::default() };
//! let segs = det.detect(&img.as_view(), &cfg);
//! // At least one segment should be detected on a clean step edge.
//! assert!(!segs.is_empty());
//! ```
//!
//! ## Conic and Ellipse Fitting
//! [`ConicFitter`] fits algebraic conics (`Ax²+Bxy+Cy²+Dx+Ey+F=0`) to a set
//! of [`vm_primitives::Point2f`] samples, with optional RANSAC outlier rejection.
//! Ellipses can be converted to geometric form ([`Ellipse2f`]).
//!
//! ```rust
//! use vision_metrology::{ConicFitter, ConicFitConfig, Ellipse2f};
//! use vm_primitives::{Point2f, Vec2f};
//! use core::f32::consts::PI;
//!
//! // Generate 20 points on a known ellipse.
//! let ell = Ellipse2f {
//!     center: Point2f { x: 50.0, y: 40.0 },
//!     semi_axes: Vec2f { x: 20.0, y: 10.0 },
//!     angle: 0.0,
//! };
//! let pts: Vec<Point2f> = (0..20).map(|i| {
//!     ell.point_at(2.0 * PI * i as f32 / 20.0)
//! }).collect();
//!
//! let mut fitter = ConicFitter::new();
//! let conic = fitter.fit(&pts, &ConicFitConfig::default()).expect("fit");
//! assert!(conic.is_ellipse());
//! let recovered = conic.to_ellipse().expect("to_ellipse");
//! assert!((recovered.center.x - 50.0).abs() < 0.5);
//! assert!((recovered.semi_major() - 20.0).abs() < 0.5);
//! ```
//!
//! ## Coordinate conventions
//! All types follow the pixel-center convention: integer coordinate `i` refers
//! to the **center** of pixel `i`. Subpixel positions are `i as f32 + delta`
//! where `delta ∈ [-0.5, 0.5]`.
//!
//! ## Border mode
//! Gradient computation in LSD uses `Clamp` border replication (consistent
//! with `vm_primitives::edge` and `vm_primitives::core`).
//!
//! ## Allocation policy
//! All per-frame scratch allocations (gradient images, region buffers, bucket
//! arrays, RANSAC inlier indices) are owned by the detector/fitter struct and
//! reused across `detect` / `fit` calls. Only the output `Vec<LineSegment2f>`
//! or `Result<Ellipse2f>` is allocated per call.

mod conic;
mod fit_conic;
mod fitter;
mod lsd;
mod nfa;
mod ransac;

pub use conic::{Conic2f, Ellipse2f};
pub use fitter::{ConicFitConfig, ConicFitter};
pub use lsd::{LineSegment2f, LsdConfig, LsdDetector};
