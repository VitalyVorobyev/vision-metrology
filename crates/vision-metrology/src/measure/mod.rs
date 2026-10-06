//! Calipers and metrology models — measuring a located part.
//!
//! This is the module that turns detection into inspection. Matching answers
//! *where the part is*; this answers *what its dimensions are*.
//!
//! ```text
//! ShapeMatcher::find  ->  ShapeMatch::pose  ->  MetrologyModel::apply  ->  Fit + residuals
//!        where                 fixture              measure + fit            the measurement
//! ```
//!
//! ## The caliper
//!
//! [`Caliper`] places a [`MeasureRect`], [`MeasureArc`], [`MeasureRadial`] or
//! [`MeasureStrip`] on the image, averages intensity across its width into a 1-D profile,
//! and runs the subpixel [`Edge1DDetector`](vm_primitives::Edge1DDetector) along
//! it. The
//! cross-averaging is where the precision comes from: `n` interpolated samples
//! per profile entry drop noise by `√n` while leaving an edge perpendicular to
//! the scan exactly as sharp.
//!
//! The matching constraint: widen a caliper only while the edge stays parallel
//! to the averaging direction. On a curved edge a wide caliper smears the very
//! transition it is measuring.
//!
//! ## The metrology model
//!
//! [`MetrologyModel`] holds nominal primitives in the part's own frame,
//! distributes calipers along each, and fits the measured points robustly.
//! Applied at a fixture pose, a model taught once follows the part — including
//! through rotation and scale.
//!
//! Every result carries `rms`, `max_dev` and `n_used`. A roundness check reads
//! `max_dev`; a "did this measurement even work" gate reads `rms` and `n_used`.

// Invariant 21: residuals are always reported.

mod bead;
mod caliper;
mod config;
pub mod diagnostics;
mod model;
mod placement;
mod select;

pub use bead::{
    BeadCaliper, BeadConfig, BeadHit, BeadPass, BeadPolarity, BeadReject, BeadSample, BeadStats,
    BeadStop, BeadSummary, BeadTrack, BeadTracker, BeadTuning, TrackedBead,
};
pub use caliper::Caliper;
pub use config::{
    Derivative, EdgeSelect, EdgeSequence, Locate, MeasureConfig, OffImage, PolaritySelect,
    ProfileConfig, RejectReason,
};
pub use model::{MetrologyFit, MetrologyModel, MetrologyObject, MetrologyResult, MetrologyShape};
pub use placement::{MeasureArc, MeasureRadial, MeasureRect, MeasureStrip};
pub use select::{MeasureEdge, MeasurePair};
