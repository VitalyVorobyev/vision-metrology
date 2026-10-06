//! The two-stage run: tracking passes that move the curve, then the final measurement.
//!
//! The algorithm is written once, over a [`Probe`] that performs each caliper measurement
//! and is told what each station and pass did. [`Quiet`] only measures.

use std::num::NonZeroUsize;

use vm_primitives::{Error, ImageView, Pixel, Point2f, Vec2f};

use super::config::{BeadCaliper, BeadConfig};
use super::curve::{
    V2, arc_lengths, chord_tangents, curvature, load_prior, offset_window, perp, resample,
    station_count, step_scale,
};
use super::pairing::{Gates, choose, sort_edges};
use super::result::{
    BeadHit, BeadPass, BeadReject, BeadSample, BeadStop, BeadSummary, BeadTrack, TrackedBead,
};
use super::solve::{Penalty, SolveScratch, solve_offsets};
use super::stats::{Tally, bead_stats, longest_run_missing};
use crate::measure::{Caliper, MeasureEdge, MeasureStrip, RejectReason};

/// Where one station measures: its point, its unit normal, and the window `[lo, hi]` its
/// pair's centre offset must fall in, in pixels along the normal.
#[derive(Debug, Clone, Copy)]
pub(super) struct Pose {
    pub point: V2,
    pub normal: V2,
    pub window: (f64, f64),
}

/// What the run reports to, and measures through.
pub(super) trait Probe {
    /// Measure with `cal` as placed, and leave its edges in `edges`.
    fn measure<P: Pixel>(
        &mut self,
        cal: &mut Caliper,
        img: &ImageView<'_, P>,
        edges: &mut Vec<MeasureEdge>,
    ) -> Result<(), RejectReason>;

    /// Station `i` measured: in tracking pass `pass`, or in the final stage when `None`.
    fn station(
        &mut self,
        _pass: Option<usize>,
        _i: usize,
        _pose: &Pose,
        _strip: &MeasureStrip,
        _hit: &Result<BeadHit, BeadReject>,
    ) {
    }

    /// Tracking pass `pass` solved: the last IRLS weights (0 at an invalid station) and the
    /// correction applied to each station, in pixels along its normal.
    fn solved(&mut self, _pass: usize, _weights: &[f64], _corrections: &[f64]) {}
}

/// The probe [`BeadTracker::track`](super::BeadTracker::track) runs with: it measures and
/// records nothing.
pub(super) struct Quiet;

impl Probe for Quiet {
    fn measure<P: Pixel>(
        &mut self,
        cal: &mut Caliper,
        img: &ImageView<'_, P>,
        edges: &mut Vec<MeasureEdge>,
    ) -> Result<(), RejectReason> {
        edges.clear();
        edges.extend_from_slice(cal.measure(img)?);
        Ok(())
    }
}

/// Reusable buffers: after the first call at a size, a run allocates only its result
/// (invariant 6).
#[derive(Debug, Clone, Default)]
pub(super) struct Scratch {
    /// The prior, then each pass's moved stations, before resampling.
    src: Vec<V2>,
    cum: Vec<f64>,
    /// The stations.
    pts: Vec<V2>,
    tan: Vec<V2>,
    normals: Vec<V2>,
    kappa: Vec<f64>,
    /// Each station's observed centre offset, read where `valid`.
    obs: Vec<f64>,
    valid: Vec<bool>,
    /// The correction applied to each station.
    applied: Vec<f64>,
    edges: Vec<MeasureEdge>,
    solve: SolveScratch,
}

/// One stage's strip geometry: every strip of the stage has the same length and counts.
#[derive(Debug, Clone, Copy)]
struct StripSpec {
    half_len: f64,
    half_width: f32,
    samples: NonZeroUsize,
    across: NonZeroUsize,
}

impl StripSpec {
    /// Long enough to hold a pair at the stage's offset reach, its clearance and the
    /// smoothing kernel: `max_offset + max_width/2 + clearance + 3σ + step`, rounded up so
    /// that the length is a whole number of steps.
    fn new(c: &BeadCaliper, cfg: &BeadConfig) -> Self {
        let step = f64::from(c.profile.step);
        let reach = f64::from(c.max_offset)
            + 0.5 * f64::from(cfg.max_width)
            + f64::from(cfg.clearance.unwrap_or(0.0))
            + 3.0 * f64::from(c.profile.sigma)
            + step;
        let steps = (2.0 * reach / step).ceil();
        let half_width = c.half_width;
        Self {
            half_len: 0.5 * steps * step,
            half_width,
            samples: NonZeroUsize::new(steps as usize + 1).unwrap_or(NonZeroUsize::MIN),
            across: NonZeroUsize::new((2.0 * half_width) as usize + 1).unwrap_or(NonZeroUsize::MIN),
        }
    }

    /// The strip through `pose`, scanning along its normal.
    fn place(&self, pose: &Pose) -> MeasureStrip {
        let ([x, y], [nx, ny]) = (pose.point, pose.normal);
        let h = self.half_len;
        MeasureStrip {
            start: Point2f::new((x - h * nx) as f32, (y - h * ny) as f32),
            end: Point2f::new((x + h * nx) as f32, (y + h * ny) as f32),
            half_width: self.half_width,
            samples: Some(self.samples),
            across: Some(self.across),
        }
    }
}

/// The gates of a stage whose caliper is `c`.
fn gates(cfg: &BeadConfig, c: &BeadCaliper) -> Gates {
    Gates {
        polarity: cfg.polarity,
        min_width: f64::from(cfg.min_width),
        max_width: f64::from(cfg.max_width),
        max_offset: f64::from(c.max_offset),
        clearance: cfg.clearance.map(f64::from),
        min_margin: cfg.min_margin.map(f64::from),
    }
}

/// Tangents, normals and curvature of the current stations, `h` apart.
fn frame(s: &mut Scratch, h: f64, window: f64) {
    chord_tangents(&s.pts, h, window, &mut s.tan);
    curvature(&s.tan, h, &mut s.kappa);
    s.normals.clear();
    s.normals.extend(s.tan.iter().map(|&t| perp(t)));
}

/// Place `cal` at `pose`, measure through `probe`, and choose the pair.
#[allow(clippy::too_many_arguments)]
fn measure_station<P: Pixel, R: Probe>(
    probe: &mut R,
    cal: &mut Caliper,
    img: &ImageView<'_, P>,
    edges: &mut Vec<MeasureEdge>,
    spec: &StripSpec,
    g: &Gates,
    pose: &Pose,
    station: (Option<usize>, usize),
) -> Result<BeadHit, BeadReject> {
    let strip = spec.place(pose);
    cal.set_strip(strip);
    let hit = match probe.measure(cal, img, edges) {
        Ok(()) => {
            // An edge located in border fill is not evidence: a strip off the image would
            // otherwise find whatever the border mode extends from the image's edge.
            let (w, h) = (
                img.width().saturating_sub(1) as f32,
                img.height().saturating_sub(1) as f32,
            );
            edges.retain(|e| e.p.x >= 0.0 && e.p.y >= 0.0 && e.p.x <= w && e.p.y <= h);
            if edges.is_empty() {
                Err(BeadReject::Caliper(RejectReason::OffImage))
            } else {
                sort_edges(edges);
                // The station's position along the strip, from the strip's own `f64`
                // geometry.
                let center_t = 0.5 * strip.geometry().length;
                choose(edges, center_t, pose.window, g)
            }
        }
        Err(reason) => Err(BeadReject::Caliper(reason)),
    };
    probe.station(station.0, station.1, pose, &strip, &hit);
    hit
}

/// The pose of station `i` with an offset reach of `reach` px.
fn pose_at(s: &Scratch, i: usize, reach: f64) -> Pose {
    Pose {
        point: s.pts[i],
        normal: s.normals[i],
        window: offset_window(s.kappa[i], reach),
    }
}

/// Track `prior` on `img` and measure the bead along the refined curve.
pub(super) fn run<P: Pixel, R: Probe>(
    cfg: &BeadConfig,
    track_cal: &mut Caliper,
    measure_cal: &mut Caliper,
    s: &mut Scratch,
    img: &ImageView<'_, P>,
    prior: &[Point2f],
    probe: &mut R,
) -> Result<TrackedBead, Error> {
    load_prior(prior, &mut s.src)?;
    let length = arc_lengths(&s.src, &mut s.cum);
    if length <= 0.0 {
        return Err(Error::Degenerate("bead prior has zero length"));
    }
    let n = station_count(length, f64::from(cfg.spacing))?;
    resample(&s.src, &s.cum, n, &mut s.pts);
    let mut h = length / (n - 1) as f64;

    let t = &cfg.tuning;
    let window = f64::from(t.tangent_window_px);
    let track_spec = StripSpec::new(&cfg.track, cfg);
    let track_gates = gates(cfg, &cfg.track);
    let reach = f64::from(cfg.track.max_offset);
    let mut passes = Vec::with_capacity(t.passes.get());
    let mut stop = BeadStop::PassLimit;
    for pass in 0..t.passes.get() {
        frame(s, h, window);
        let mut tally = Tally::default();
        s.obs.clear();
        s.valid.clear();
        for i in 0..n {
            let pose = pose_at(s, i, reach);
            let hit = measure_station(
                probe,
                track_cal,
                img,
                &mut s.edges,
                &track_spec,
                &track_gates,
                &pose,
                (Some(pass), i),
            );
            match hit {
                Ok(hit) => {
                    s.obs.push(f64::from(hit.offset));
                    s.valid.push(true);
                }
                Err(r) => {
                    tally.add(r);
                    s.obs.push(0.0);
                    s.valid.push(false);
                }
            }
        }
        let n_valid = s.valid.iter().filter(|&&v| v).count();
        let support = n_valid as f64 / n as f64;
        let gap = longest_run_missing(s.valid.iter().copied()) as f64 * h;
        let mut record = BeadPass {
            n_valid,
            support: support as f32,
            longest_gap: gap as f32,
            correction_rms: 0.0,
            correction_max: 0.0,
            residual_rms: 0.0,
            residual_max: 0.0,
            step_scale: 0.0,
            irls_iters: 0,
            rejects: tally.to_vec(),
        };
        if n_valid == 0 || support < f64::from(t.min_support) {
            s.applied.clear();
            s.applied.resize(n, 0.0);
            s.solve.weights.clear();
            s.solve.weights.resize(n, 0.0);
            probe.solved(pass, &s.solve.weights, &s.applied);
            passes.push(record);
            stop = BeadStop::TooFewValid;
            break;
        }

        let penalty = Penalty::new(
            f64::from(t.damping),
            f64::from(t.tension_px),
            f64::from(t.bending_px),
            h,
        );
        let solved = solve_offsets(
            &s.obs,
            &s.valid,
            penalty,
            t.loss,
            t.irls_iters.get(),
            &mut s.solve,
        )
        .ok_or(Error::Degenerate("bead solve lost positive definiteness"))?;
        let alpha = step_scale(&s.pts, &s.normals, &s.solve.d);
        s.applied.clear();
        s.applied.extend(s.solve.d.iter().map(|&d| alpha * d));
        let (mut sum2, mut max) = (0.0f64, 0.0f64);
        s.src.clear();
        for ((&p, &nrm), &c) in s.pts.iter().zip(&s.normals).zip(&s.applied) {
            sum2 += c * c;
            max = max.max(c.abs());
            s.src.push([p[0] + c * nrm[0], p[1] + c * nrm[1]]);
        }
        probe.solved(pass, &s.solve.weights, &s.applied);
        let moved = arc_lengths(&s.src, &mut s.cum);
        resample(&s.src, &s.cum, n, &mut s.pts);
        h = moved / (n - 1) as f64;

        record.correction_rms = (sum2 / n as f64).sqrt() as f32;
        record.correction_max = max as f32;
        record.residual_rms = solved.residual_rms as f32;
        record.residual_max = solved.residual_max as f32;
        record.step_scale = alpha as f32;
        record.irls_iters = solved.irls_iters;
        passes.push(record);
        if max < f64::from(t.tol) {
            stop = BeadStop::Converged;
            break;
        }
    }

    // The final stage: new poses on the refined stations, the stricter caliper, no
    // feedback into the curve.
    frame(s, h, window);
    let spec = StripSpec::new(&cfg.measure, cfg);
    let g = gates(cfg, &cfg.measure);
    let reach = f64::from(cfg.measure.max_offset);
    let mut tally = Tally::default();
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let pose = pose_at(s, i, reach);
        let hit = measure_station(
            probe,
            measure_cal,
            img,
            &mut s.edges,
            &spec,
            &g,
            &pose,
            (None, i),
        );
        if let Err(r) = hit {
            tally.add(r);
        }
        samples.push(BeadSample {
            point: Point2f::new(pose.point[0] as f32, pose.point[1] as f32),
            normal: Vec2f::new(pose.normal[0] as f32, pose.normal[1] as f32),
            hit,
        });
    }
    let hits = samples.iter().filter_map(|smp| smp.hit.as_ref().ok());
    let stats = bead_stats(hits);
    let n_used = stats.map_or(0, |st| st.n_used);
    let gap = longest_run_missing(samples.iter().map(|smp| smp.hit.is_ok()));
    Ok(TrackedBead {
        centerline: samples.iter().map(|smp| smp.point).collect(),
        spacing: h as f32,
        summary: BeadSummary {
            support: (n_used as f64 / n as f64) as f32,
            longest_gap: (gap as f64 * h) as f32,
            stats,
            rejects: tally.to_vec(),
        },
        samples,
        track: BeadTrack { passes, stop },
    })
}
