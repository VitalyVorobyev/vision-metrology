//! Seeing why a bead tracker measured what it did: [`explain_bead`] runs the tracker once
//! through a probe that keeps every station's evidence.

use vm_primitives::{Error, ImageView, Pixel, Point2f, Vec2f};

use super::BeadTracker;
use super::result::{BeadHit, BeadReject, TrackedBead};
use super::run::{Pose, Probe};
use crate::measure::diagnostics::{CaliperTrace, explain};
use crate::measure::{Caliper, MeasureEdge, MeasureStrip, RejectReason};

/// A bead tracker's run, explained: its result, and each station's evidence in every
/// tracking pass and in the final stage.
///
/// Built by [`explain_bead`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[non_exhaustive]
pub struct BeadTrace {
    /// What [`BeadTracker::track`] returns for the same config, image and prior, to the
    /// bit.
    pub result: TrackedBead,
    /// One entry per tracking pass, parallel to `result.track.passes`.
    pub passes: Vec<BeadPassTrace>,
    /// The final stage's stations, parallel to `result.samples`: the same points, normals
    /// and hits, with each station's window, strip and caliper trace beside them.
    pub measure: Vec<BeadStationTrace>,
}

/// One tracking pass, explained: what each station saw, and what the solve made of it.
///
/// Every vector has one entry per station, in station order. The pass's support,
/// rejections and [`BeadSolve`](super::BeadSolve) are the same entry of
/// `result.track.passes`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[non_exhaustive]
pub struct BeadPassTrace {
    /// Each station where the pass measured it, before the pass moved it. A hit's
    /// `offset` is the station's observation.
    pub stations: Vec<BeadStationTrace>,
    /// The weight each observation carried in the pass's last solve, after the robust
    /// loss: 0 at a rejected station, and at every station when the pass found too few
    /// pairs to solve.
    pub weights: Vec<f32>,
    /// The correction the pass applied to each station, in pixels along its normal; 0
    /// everywhere when it did not solve. It is the solved correction times
    /// [`BeadSolve::step_scale`](super::BeadSolve::step_scale), so its largest magnitude
    /// is [`BeadSolve::correction_max`](super::BeadSolve::correction_max).
    pub corrections: Vec<f32>,
}

/// One station's measurement, explained: where its strip lay, everything the strip's
/// caliper computed, and the pair or the gate that rejected every pair.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[non_exhaustive]
pub struct BeadStationTrace {
    /// The station.
    pub point: Point2f,
    /// The unit tangent there, towards the curve's end.
    pub tangent: Vec2f,
    /// The unit normal, `tangent.perp()`: the strip scans along it, and offsets are signed
    /// along it.
    pub normal: Vec2f,
    /// The offsets `(lo, hi)`, in pixels along the normal, that a pair's midpoint had to
    /// fall in: the stage's `±max_offset`, stopped short of the centre of curvature on a
    /// bend's concave side.
    pub window: (f32, f32),
    /// The strip the caliper measured: through the station, along its normal, from `−n`
    /// to `+n`.
    pub strip: MeasureStrip,
    /// Everything the strip's caliper computed. Its `edges` are the edges, of either
    /// polarity, that the gates paired, together with any in border fill, which the
    /// tracker drops before pairing; its `reject` is the reason in
    /// [`BeadReject::Caliper`].
    pub caliper: CaliperTrace,
    /// The bead's pair, or the gate that rejected every pair.
    pub hit: Result<BeadHit, BeadReject>,
}

/// Track the bead on `img` from `prior` as [`BeadTracker::track`] does, and keep each
/// station's evidence: its strip, its caliper's trace and its pair or rejection, in every
/// tracking pass and in the final stage, with each pass's robust weights and corrections.
///
/// It runs the tracker once, through the same code as `track`, and measures each strip
/// through [`explain`], whose edges and rejection are
/// [`Caliper::measure`]'s to the bit. Its `result` is therefore what `track` returns for
/// the same config, image and prior, and the tracker's later results are unchanged. It
/// fails as `track` does. It allocates a [`CaliperTrace`] per station and pass, so it
/// belongs in a tool that shows why a bead measured what it did, not in the inspection
/// loop.
///
/// # Example
/// ```
/// use vision_metrology::measure::diagnostics::explain_bead;
/// use vision_metrology::measure::{BeadConfig, BeadReject, BeadTracker, RejectReason};
/// use vision_metrology::{Image, Point2f};
///
/// // A vertical light bead 40 px wide, centred on x = 100, that ends at y = 100.
/// let (w, h) = (200usize, 160usize);
/// let data: Vec<u8> = (0..w * h)
///     .map(|i| {
///         let (x, y) = ((i % w) as f32, (i / w) as f32);
///         let across = (x + 0.5 - 80.0).clamp(0.0, 1.0) - (x + 0.5 - 120.0).clamp(0.0, 1.0);
///         let cover = if y < 100.0 { across } else { 0.0 };
///         (40.0 + 160.0 * cover).round() as u8
///     })
///     .collect();
/// let img = Image::from_vec(w, h, data).unwrap();
/// let prior = [Point2f::new(94.0, 20.0), Point2f::new(94.0, 140.0)];
///
/// let mut tracker = BeadTracker::new(BeadConfig::default()).unwrap();
/// let trace = explain_bead(&mut tracker, &img.as_view(), &prior).unwrap();
/// assert_eq!(trace.result, tracker.track(&img.as_view(), &prior).unwrap());
///
/// // Past the bead's end, the last station's strip found no edge at all...
/// let last = trace.measure.last().unwrap();
/// assert_eq!(last.hit, Err(BeadReject::Caliper(RejectReason::NoEdge)));
/// assert_eq!(last.caliper.reject, Some(RejectReason::NoEdge));
/// // ...and in the first pass it carried no weight in the solve.
/// assert_eq!(trace.passes[0].weights.last(), Some(&0.0));
/// ```
pub fn explain_bead<P: Pixel>(
    tracker: &mut BeadTracker,
    img: &ImageView<'_, P>,
    prior: &[Point2f],
) -> Result<BeadTrace, Error> {
    let mut tracer = Tracer::default();
    let result = tracker.run(img, prior, &mut tracer)?;
    Ok(BeadTrace {
        result,
        passes: tracer.passes,
        measure: tracer.measure,
    })
}

/// The probe [`explain_bead`] runs with: it measures through [`explain`] and keeps what
/// every station and pass did.
#[derive(Debug, Default)]
struct Tracer {
    passes: Vec<BeadPassTrace>,
    measure: Vec<BeadStationTrace>,
}

impl Probe for Tracer {
    type Seen = CaliperTrace;

    fn measure<P: Pixel>(
        &mut self,
        cal: &mut Caliper,
        img: &ImageView<'_, P>,
        edges: &mut Vec<MeasureEdge>,
    ) -> (Result<(), RejectReason>, CaliperTrace) {
        let trace = explain(cal, img);
        edges.clear();
        edges.extend_from_slice(&trace.edges);
        (trace.reject.map_or(Ok(()), Err), trace)
    }

    fn station(
        &mut self,
        pass: Option<usize>,
        _i: usize,
        pose: &Pose,
        strip: &MeasureStrip,
        seen: CaliperTrace,
        hit: &Result<BeadHit, BeadReject>,
    ) {
        let v = |[x, y]: [f64; 2]| Vec2f::new(x as f32, y as f32);
        let station = BeadStationTrace {
            point: Point2f::new(pose.point[0] as f32, pose.point[1] as f32),
            tangent: v(pose.tangent),
            normal: v(pose.normal),
            window: (pose.window.0 as f32, pose.window.1 as f32),
            strip: *strip,
            caliper: seen,
            hit: *hit,
        };
        let Some(pass) = pass else {
            self.measure.push(station);
            return;
        };
        // Passes run in order, and each reports its stations before it solves.
        if pass == self.passes.len() {
            self.passes.push(BeadPassTrace {
                stations: Vec::new(),
                weights: Vec::new(),
                corrections: Vec::new(),
            });
        }
        if let Some(p) = self.passes.get_mut(pass) {
            p.stations.push(station);
        }
    }

    fn solved(&mut self, pass: usize, weights: &[f64], corrections: &[f64]) {
        if let Some(p) = self.passes.get_mut(pass) {
            p.weights = weights.iter().map(|&w| w as f32).collect();
            p.corrections = corrections.iter().map(|&c| c as f32).collect();
        }
    }
}
