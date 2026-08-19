//! 1D and 2D edge primitives for high-throughput metrology.
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

// Submodules are private: every name lives at `edge::…`, one canonical path
// (invariant 17). The split is a file-size concern, not an API one.
mod conv1d;
mod edge1d;
mod edge2d;
mod gradient;
mod kernels1d;
mod laser1d;

pub use conv1d::convolve_f32;
pub use edge1d::{Edge1DConfig, Edge1DDetector, EdgePeak, EdgePolarity, SubpixRefine};
pub use edge2d::{Edge2DConfig, Edge2DDetector, Edgel, GradientBuffers, SmoothKind, Subpix2D};
pub use gradient::DirectionField;
pub use kernels1d::DoGKernel1D;
pub use laser1d::{EdgePair1D, EdgePairConfig, best_edge_pair, best_edge_pair_in_row_u8};
