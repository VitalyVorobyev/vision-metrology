//! Line-segment detection.
//!
//! ## Line Segment Detection (LSD)
//! [`LsdDetector`] implements gradient-coherence region growing with NFA
//! (Number of False Alarms) validation. Segments are returned as
//! [`LineSegment2f`] with subpixel endpoints, support-region width, and NFA
//! score.
//!
//! ```rust
//! use vision_metrology::shape::{LsdConfig, LsdDetector};
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
//! let cfg = LsdConfig { downscale_levels: 0, ..LsdConfig::default() };
//! let segs = det.detect(&img.as_view(), &cfg);
//! // At least one segment should be detected on a clean step edge.
//! assert!(!segs.is_empty());
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

mod lsd;
mod nfa;

pub use lsd::{LineSegment2f, LsdConfig, LsdDetector};
