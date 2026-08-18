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
//!
//! Module layout: `types` holds the public data types, `extractor` the
//! reusable entry points, and the private `scan` / `pairing` / `coarse` /
//! `gather` / `postprocess` modules each own one stage of the pipeline. The
//! u8/u16/f32 paths share one generic implementation via the `ScanPixel`
//! trait; the typed `extract_line_*` wrappers are the stable public surface.

mod coarse;
mod extractor;
mod pairing;
mod postprocess;
mod scan;
mod types;

mod gather;
#[cfg(test)]
mod tests;

pub use coarse::{coarse_center_f32, coarse_center_u8, coarse_center_u16};
pub use extractor::LaserExtractor;
pub use pairing::best_pair_with_prior;
pub use types::{CoarseMethod, ColAccess, LaserExtractConfig, LaserLine, LaserSample, ScanAxis};
