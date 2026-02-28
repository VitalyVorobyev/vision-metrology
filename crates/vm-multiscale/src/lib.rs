//! Multi-scale 2-D edge detection for machine-vision metrology.
//!
//! This crate combines [`vm_pyr::PyramidF32`] and [`vm_edge::edge2d::Edge2DDetector`]
//! into a single convenient API that:
//!
//! 1. Builds a Gaussian pyramid from any supported input type (`u8`, `u16`, `f32`).
//! 2. Runs the 2-D edge detector at each pyramid level with automatically
//!    scaled thresholds.
//! 3. Maps all detected edgels back to **level-0 (original-resolution) pixel
//!    coordinates** so outputs from different scales are directly comparable.
//! 4. Optionally deduplicates edgels that land in the same level-0 pixel cell,
//!    keeping the finest-scale detection.
//!
//! ## Quick start
//!
//! ```
//! use vm_core::Image;
//! use vm_multiscale::{MultiScaleConfig, MultiScaleEdgeDetector};
//!
//! // A simple 64×64 vertical step edge (dark left, bright right).
//! let data: Vec<u8> = (0..64 * 64)
//!     .map(|i| if (i % 64) >= 32 { 200u8 } else { 0u8 })
//!     .collect();
//! let img = Image::from_vec(64, 64, data).unwrap();
//!
//! let mut det = MultiScaleEdgeDetector::new();
//! let cfg = MultiScaleConfig::default(); // 3 levels, auto-threshold, merge duplicates
//! let edgels = det.detect_u8(&img.as_view(), &cfg);
//!
//! // Each edgel carries its pyramid level and effective scale sigma.
//! for e in &edgels {
//!     let _ = (e.p, e.n, e.strength, e.scale, e.level);
//! }
//! ```
//!
//! ## Coordinate convention
//!
//! Pixel center convention is inherited from `vm-core`: integer coordinate `i`
//! refers to the center of pixel `i`. Edgel positions from coarser levels are
//! multiplied by `2^level` to bring them back to level-0 space.
//!
//! ## Threshold auto-scaling
//!
//! When [`MultiScaleConfig::edge`] has both thresholds set to `0.0` (the
//! default), the detector auto-thresholds at each level individually.
//! When explicit thresholds are given, they are divided by `2^level` to
//! compensate for the gradient magnitude reduction at coarser scales.

mod config;
mod detector;
mod edgel;

pub use config::MultiScaleConfig;
pub use detector::MultiScaleEdgeDetector;
pub use edgel::ScaleAnnotatedEdgel;
