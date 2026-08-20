//! Scale estimation for [`crate::matching`] — "estimate, then verify", not a
//! wider scan (roadmap W7).
//!
//! # Why estimate instead of scanning wider
//!
//! `matching::ShapeSearchConfig::scale_range` already searches scale: a
//! discrete scan over evenly-spaced steps. Widening that range to cover an
//! unknown scale is **linear in how wide the range is** — `tests/accuracy.rs`'s
//! `scale_estimate_vs_scan_cost` row measures exactly how much (see there
//! for numbers).
//! When the scale is not unknown but merely *unmeasured* — a segmentable
//! part whose rough size is visible from its own silhouette, or a coarse
//! correlation against the taught patch — the right move is to spend a
//! small, constant amount of work estimating it once, rebuild the model at
//! that estimate with [`ShapeModel::resample_at`](crate::matching::ShapeModel::resample_at),
//! and search a **narrow** verification band around it. The search cost is
//! then independent of how far the true scale is from 1.0.
//!
//! Two independent estimators, because they need different things from the
//! scene:
//!
//! * [`estimate_scale_moments`] — segments an isolated blob (Otsu + CCL,
//!   `crate::segment`) and compares its spatial spread to the taught
//!   model's own. Needs a part that segments cleanly against its
//!   background; cheap (`O(roi area)`); works on **any**
//!   [`ShapeModel`](crate::matching::ShapeModel) (format 3 or 4 — it reads
//!   `level(0).points()`, which every model has).
//! * [`estimate_scale_logpolar`] — no segmentation required, but needs an
//!   approximate center and format-4 teach data
//!   ([`ShapeModel::teach_point_count`](crate::matching::ShapeModel::teach_point_count)
//!   `> 0`). Builds a synthetic edge-density raster from the model's own
//!   teach points, log-polar-unwraps it and the scene around the hint
//!   center ([`crate::warp::Map::log_polar`]), and finds the scale (and,
//!   within a bounded margin, the rotation) as a translation via
//!   [`crate::corr`] ZNCC — the classic Fourier-Mellin trick, without an
//!   FFT.
//!
//! [`find_scale_invariant`] is the convenience that chains one estimator
//! (moments if a ROI hint is given, log-polar if only a center is), a
//! resample, and a narrow verify search.
//!
//! # `u8`-only, like `corr`
//!
//! Both estimators take `ImageView<'_, u8>`, matching this workspace's
//! established convention (`corr`, `segment::otsu_threshold_u8`) rather than
//! adding a third `Pixel`-generic threshold/correlation implementation for
//! this wave. `find_scale_invariant` is `u8`-only for the same reason, even
//! though the `ShapeMatcher::find` verify step it ends with is generic —
//! call `resample_at` and `ShapeMatcher::find` directly for a `u16`/`f32`
//! scene.

mod invariant;
mod logpolar;
mod moments;

pub use invariant::{ScaleHint, ScaleInvariantConfig, find_scale_invariant};
pub use logpolar::{LogPolarScaleConfig, estimate_scale_logpolar};
pub use moments::{BlobPolarity, MomentScaleConfig, estimate_scale_moments};

/// One estimator's answer: an approximate scale, optionally a rotation, and
/// a confidence-ish score whose meaning is **not shared** between
/// estimators (documented per-function) — do not compare a moments score
/// against a log-polar score, or gate on one with a threshold tuned for the
/// other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleEstimate {
    /// Estimated scale, relative to the model's own taught scale (1.0 =
    /// same size as taught).
    pub scale: f32,
    /// Estimated rotation, radians, when the estimator measures one.
    /// [`estimate_scale_moments`] never does (`None`);
    /// [`estimate_scale_logpolar`] does within its configured angle margin.
    pub angle: Option<f32>,
    /// Estimator-specific confidence; see each function's own docs.
    pub score: f32,
}
