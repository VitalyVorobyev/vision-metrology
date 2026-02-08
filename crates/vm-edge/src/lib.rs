//! 1D edge primitives for high-throughput metrology.
//!
//! Coordinates follow pixel-center convention: sample `signal[i]` is located at
//! position `x = i`.
//!
//! For laser line extraction we detect opposite-polarity edge pairs (stripe
//! boundaries) rather than fitting intensity peaks, which is typically more
//! robust for wide, flat, or saturated stripes.
//!
//! Thresholds in [`edge1d::Edge1DConfig`] default to zero. In production,
//! configure thresholds for your sensor/illumination or add auto-thresholding
//! on top.

pub mod conv1d;
pub mod edge1d;
pub mod kernels1d;
pub mod laser1d;

pub use edge1d::{Edge1DConfig, Edge1DDetector, EdgePeak, EdgePolarity, SubpixRefine};
pub use kernels1d::DoGKernel1D;
pub use laser1d::{EdgePair1D, EdgePairConfig, best_edge_pair, best_edge_pair_in_row_u8};
