//! Fast laser stripe extraction for industrial machine vision.
//!
//! Core strategy:
//! - Run 1D DoG edge detection on each scan line.
//! - Select bright-on-dark opposite-polarity edge pairs.
//! - Use ROI + continuity prior for speed and robustness.
//!
//! Axis modes:
//! - [`ScanAxis::Rows`] is fastest (contiguous row slices).
//! - [`ScanAxis::Cols`] with [`ColAccess::Gather`] gathers column samples.
//! - [`ScanAxis::Cols`] with [`ColAccess::Transposed`] reuses row scanning
//!   when caller supplies a transposed image view.

mod extractor;

pub use extractor::{
    CoarseMethod, ColAccess, LaserExtractConfig, LaserExtractor, LaserLine, LaserSample, ScanAxis,
    best_pair_with_prior, coarse_center_f32, coarse_center_u8, coarse_center_u16,
};
