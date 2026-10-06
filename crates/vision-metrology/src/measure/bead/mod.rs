//! The bead tracker: a curve refined from a prior by caliper evidence, then measured.
//!
//! ## Conventions
//!
//! - **Stations.** The prior is resampled to `N` stations uniform in arc length,
//!   `round(L / spacing) + 1` of them (at least 2) for a prior of length `L`. `N` is fixed
//!   for the call, so station `i` keeps its identity from pass to pass and in the result.
//!   Each pass moves every station, the two ends included, along its current normal: the
//!   tracker moves the curve sideways and does not find where the bead starts or ends.
//! - **Tangent and normal.** A station's tangent `t` is the chord over
//!   `±tangent_window_px` of arc length, pointing towards the curve's end. Near the ends
//!   the window shrinks so the chord stays centred on the station; the two end stations
//!   take the chord to their neighbour. Its normal is `n = t.perp() = (−t_y, t_x)`; with
//!   y down, that is to the right of travel as drawn on screen. Reversing the prior
//!   reverses `t` and `n`.
//! - **Offsets** are signed distances along `+n`, in pixels. A pair's `first` edge is on
//!   the `−n` side and its `second` on the `+n` side.
//! - **Polarity.** Each strip scans from `−n` to `+n`. A [`BeadPolarity::Light`] bead is a
//!   rising edge then a falling one along that scan; a [`BeadPolarity::Dark`] bead the
//!   reverse. Either way round the prior, the same bead gives the same pair.

// Invariants 6 (scratch reused), 12 (no RNG; explicit tie-breaks), 20 (f64 sums) and
// 21 (residuals reported) shape this module.

mod config;
mod curve;
mod pairing;
mod result;
mod run;
mod solve;
mod stats;
mod trace;

#[cfg(test)]
mod tests;

use vm_primitives::{Error, ImageView, Pixel, Point2f};

pub use config::{BeadCaliper, BeadConfig, BeadPolarity, BeadTuning};
pub use result::{
    BeadHit, BeadPass, BeadReject, BeadSample, BeadSolve, BeadStats, BeadStop, BeadSummary,
    BeadTrack, TrackedBead,
};
// Reached only through `measure::diagnostics` (invariant 17).
pub use trace::{BeadPassTrace, BeadStationTrace, BeadTrace, explain_bead};

use super::{Caliper, MeasureStrip};

/// Tracks an elongated bead or stripe along a prior curve, and measures its position and
/// width at every station of the refined curve.
///
/// Each tracking pass places a strip caliper along every station's normal, picks the
/// bead's edge pair on it, solves one regularised, robust problem for the normal
/// corrections, and moves and resamples the curve. A final stage then measures along the
/// refined curve with a stricter caliper, without moving it. A bead that is not there is
/// still a result: every station is rejected with a reason, and the tracking stops with
/// [`BeadStop::TooFewValid`].
///
/// The tracker owns both stages' calipers and every buffer, so after the first call at a
/// given size [`track`](Self::track) allocates only its result.
///
/// # Example
/// ```
/// use vision_metrology::measure::{BeadConfig, BeadTracker};
/// use vision_metrology::{Image, Point2f};
///
/// // A vertical light bead 40 px wide, centred on x = 100, anti-aliased.
/// let (w, h) = (200usize, 160usize);
/// let data: Vec<u8> = (0..w * h)
///     .map(|i| {
///         let x = (i % w) as f32;
///         let cover = ((x + 0.5 - 80.0).clamp(0.0, 1.0) - (x + 0.5 - 120.0).clamp(0.0, 1.0));
///         (40.0 + 160.0 * cover).round() as u8
///     })
///     .collect();
/// let img = Image::from_vec(w, h, data).unwrap();
///
/// // A prior 6 px off the bead's centreline.
/// let prior = [Point2f::new(94.0, 20.0), Point2f::new(94.0, 140.0)];
/// let mut tracker = BeadTracker::new(BeadConfig::default()).unwrap();
/// let bead = tracker.track(&img.as_view(), &prior).unwrap();
///
/// assert!(bead.centerline.iter().all(|p| (p.x - 100.0).abs() < 0.05));
/// let stats = bead.summary.stats.expect("found");
/// assert!((stats.width_mean - 40.0).abs() < 0.05, "width {}", stats.width_mean);
/// assert_eq!(bead.summary.support, 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct BeadTracker {
    cfg: BeadConfig,
    track: Caliper,
    measure: Caliper,
    scratch: run::Scratch,
}

/// A strip to create the calipers on; every station replaces it.
fn placeholder_strip() -> MeasureStrip {
    MeasureStrip {
        start: Point2f::origin(),
        end: Point2f::new(1.0, 0.0),
        half_width: 0.0,
        samples: None,
        across: None,
    }
}

impl BeadTracker {
    /// A tracker for `cfg`.
    ///
    /// Fails with [`Error::InvalidConfig`] when a setting is out of range: a width range
    /// with `min_width > max_width`, a non-positive spacing, step, σ or reach, a margin
    /// outside `(0, 1)`, a stage that locates edges with
    /// [`Locate::MidpointCrossing`](super::Locate::MidpointCrossing), or
    /// [`Locate::HalfContrast`](super::Locate::HalfContrast) flanks that would reach the
    /// opposite edge (`flank_far_px >= min_width`).
    pub fn new(cfg: BeadConfig) -> Result<Self, Error> {
        config::validate(&cfg)?;
        Ok(Self {
            track: Caliper::strip(placeholder_strip(), cfg.track.to_measure_config()),
            measure: Caliper::strip(placeholder_strip(), cfg.measure.to_measure_config()),
            cfg,
            scratch: run::Scratch::default(),
        })
    }

    /// The current config.
    pub fn config(&self) -> &BeadConfig {
        &self.cfg
    }

    /// Replace the config, keeping the buffers. Validated as [`new`](Self::new) does; on
    /// an error the tracker keeps its old config.
    pub fn set_config(&mut self, cfg: BeadConfig) -> Result<(), Error> {
        config::validate(&cfg)?;
        self.track.set_config(cfg.track.to_measure_config());
        self.measure.set_config(cfg.measure.to_measure_config());
        self.cfg = cfg;
        Ok(())
    }

    /// Track the bead on `img` from `prior`, an open polyline in image coordinates, and
    /// measure it along the refined curve.
    ///
    /// A bead that is missing, partly or wholly, is an `Ok` result with those stations
    /// rejected. An error means a prior that cannot be tracked: fewer than two points
    /// ([`Error::InsufficientData`]), a non-finite point or zero length
    /// ([`Error::Degenerate`]), or one so long it needs more than 2²² stations at the
    /// configured spacing ([`Error::InvalidConfig`]).
    pub fn track<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        prior: &[Point2f],
    ) -> Result<TrackedBead, Error> {
        self.run(img, prior, &mut run::Quiet)
    }

    /// The one run behind [`track`](Self::track) and
    /// [`explain_bead`](super::diagnostics::explain_bead): they differ only in `probe`.
    fn run<P: Pixel, R: run::Probe>(
        &mut self,
        img: &ImageView<'_, P>,
        prior: &[Point2f],
        probe: &mut R,
    ) -> Result<TrackedBead, Error> {
        run::run(
            &self.cfg,
            &mut self.track,
            &mut self.measure,
            &mut self.scratch,
            img,
            prior,
            probe,
        )
    }
}
