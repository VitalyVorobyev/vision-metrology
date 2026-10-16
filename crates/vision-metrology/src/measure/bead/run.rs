//! The two-stage run: tracking passes that move the curve, then the final measurement.
//!
//! The algorithm is written once, over a [`Probe`] that performs each caliper measurement
//! and is told what each station and pass did. [`Quiet`] only measures; the tracing probe
//! in `trace.rs` measures through `diagnostics::explain` and records everything. Each stage
//! chooses its stations' walls per station (`pairing.rs`) or along the whole curve
//! (`coupled_stage.rs`), as the config's pairing says.

use std::num::NonZeroUsize;

use vm_primitives::{Error, ImageView, Pixel, Point2f, Vec2f};

use super::config::{BeadCaliper, BeadConfig, strip_extent};
use super::coupled_stage::{self, CoupledScratch, Stage};
use super::coupling::BeadPairing;
use super::curve::{
    V2, arc_lengths, chord_tangents, curvature, load_prior, offset_window, perp, resample,
    station_count, step_scale,
};
use super::pairing::{Gates, choose, sort_edges};
use super::result::{
    BeadHit, BeadPass, BeadPathScore, BeadReject, BeadSample, BeadSolve, BeadStop, BeadSummary,
    BeadTrack, BeadWall, TrackedBead,
};
use super::solve::{Penalty, SolveScratch, solve_offsets};
use super::stats::{Tally, bead_stats, longest_run_missing};
use crate::measure::{Caliper, MeasureEdge, MeasureStrip, RejectReason};

/// Where one station measures: its point, its unit tangent and normal, and the window
/// `[lo, hi]` its pair's centre offset must fall in, in pixels along the normal.
#[derive(Debug, Clone, Copy)]
pub(super) struct Pose {
    pub point: V2,
    pub tangent: V2,
    pub normal: V2,
    pub window: (f64, f64),
}

/// What one stage made of one station.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Outcome {
    /// The pair, or why there is none.
    pub hit: Result<BeadHit, BeadReject>,
    /// The one wall of a [`BeadReject::OneSided`] station.
    pub wall: Option<BeadWall>,
    /// The coupled path's margin, at a station with a wall.
    pub margin: Option<f32>,
    /// The bead's centre offset for the tracking solve, where the stage saw the bead.
    pub obs: Option<f64>,
}

impl Outcome {
    /// A per-station pair or rejection.
    fn pair(hit: Result<BeadHit, BeadReject>) -> Self {
        Self {
            obs: hit.as_ref().ok().map(|h| f64::from(h.offset)),
            hit,
            wall: None,
            margin: None,
        }
    }
}

/// What the run reports to, and measures through.
pub(super) trait Probe {
    /// What a measurement leaves for its station's report.
    type Seen;

    /// Measure with `cal` as placed, and leave its edges in `edges`.
    fn measure<P: Pixel>(
        &mut self,
        cal: &mut Caliper,
        img: &ImageView<'_, P>,
        edges: &mut Vec<MeasureEdge>,
    ) -> (Result<(), RejectReason>, Self::Seen);

    /// Station `i` measured, in tracking pass `pass` or in the final stage when `None`:
    /// where, what `measure` saw there, and what the stage made of it.
    fn station(
        &mut self,
        _pass: Option<usize>,
        _i: usize,
        _pose: &Pose,
        _strip: &MeasureStrip,
        _seen: Self::Seen,
        _out: &Outcome,
    ) {
    }

    /// Tracking pass `pass` solved: the last IRLS weights (0 at an invalid station) and the
    /// correction applied to each station, in pixels along its normal. A pass with too
    /// few pairs to solve reports zeros for both.
    fn solved(&mut self, _pass: usize, _weights: &[f64], _corrections: &[f64]) {}
}

/// The probe [`BeadTracker::track`](super::BeadTracker::track) runs with: it measures and
/// records nothing.
pub(super) struct Quiet;

impl Probe for Quiet {
    type Seen = ();

    fn measure<P: Pixel>(
        &mut self,
        cal: &mut Caliper,
        img: &ImageView<'_, P>,
        edges: &mut Vec<MeasureEdge>,
    ) -> (Result<(), RejectReason>, ()) {
        edges.clear();
        let measured = cal.measure(img).map(|found| edges.extend_from_slice(found));
        (measured, ())
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
    /// One stage's poses and outcomes.
    poses: Vec<Pose>,
    outs: Vec<Outcome>,
    /// Each station's observed centre offset, read where `valid`.
    obs: Vec<f64>,
    valid: Vec<bool>,
    /// The correction applied to each station.
    applied: Vec<f64>,
    edges: Vec<MeasureEdge>,
    solve: SolveScratch,
    coupled: CoupledScratch,
}

/// One stage's strip geometry: every strip of the stage has the same length and counts.
#[derive(Debug, Clone, Copy)]
pub(super) struct StripSpec {
    half_len: f64,
    half_width: f32,
    pub samples: NonZeroUsize,
    across: NonZeroUsize,
}

impl StripSpec {
    /// The stage's strips, as [`strip_extent`] sizes them; the config's validation caps
    /// the counts.
    fn new(c: &BeadCaliper, cfg: &BeadConfig) -> Self {
        let e = strip_extent(c, cfg);
        let count = |v: f64| NonZeroUsize::new(v as usize).unwrap_or(NonZeroUsize::MIN);
        Self {
            half_len: e.half_len,
            half_width: c.half_width,
            samples: count(e.samples),
            across: count(e.across),
        }
    }

    /// The strip through `pose`, scanning along its normal.
    pub(super) fn place(&self, pose: &Pose) -> MeasureStrip {
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

/// Drop the edges located outside the image, in border fill: they are not evidence, and a
/// strip off the image would otherwise find whatever the border mode extends from the
/// image's edge.
pub(super) fn retain_inside<P: Pixel>(edges: &mut Vec<MeasureEdge>, img: &ImageView<'_, P>) {
    let (w, h) = (
        img.width().saturating_sub(1) as f32,
        img.height().saturating_sub(1) as f32,
    );
    edges.retain(|e| e.p.x >= 0.0 && e.p.y >= 0.0 && e.p.x <= w && e.p.y <= h);
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
) -> Outcome {
    let strip = spec.place(pose);
    cal.set_strip(strip);
    let (measured, seen) = probe.measure(cal, img, edges);
    let hit = match measured {
        Ok(()) => {
            retain_inside(edges, img);
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
    let out = Outcome::pair(hit);
    probe.station(station.0, station.1, pose, &strip, seen, &out);
    out
}

/// The pose of station `i` with an offset reach of `reach` px.
fn pose_at(s: &Scratch, i: usize, reach: f64) -> Pose {
    Pose {
        point: s.pts[i],
        tangent: s.tan[i],
        normal: s.normals[i],
        window: offset_window(s.kappa[i], reach),
    }
}

/// One stage over every station, as `cfg.pairing` says: each station's outcome into
/// `s.outs`, each reported to `probe`, and the coupled path's score. `h` is the station
/// spacing and `pass` the tracking pass, `None` in the final stage.
#[allow(clippy::too_many_arguments)]
fn measure_stage<P: Pixel, R: Probe>(
    probe: &mut R,
    cal: &mut Caliper,
    img: &ImageView<'_, P>,
    s: &mut Scratch,
    cfg: &BeadConfig,
    c: &BeadCaliper,
    h: f64,
    pass: Option<usize>,
) -> Option<BeadPathScore> {
    let spec = StripSpec::new(c, cfg);
    let reach = f64::from(c.max_offset);
    s.poses.clear();
    for i in 0..s.pts.len() {
        let pose = pose_at(s, i, reach);
        s.poses.push(pose);
    }
    s.outs.clear();
    match &cfg.pairing {
        BeadPairing::PerStation => {
            let g = gates(cfg, c);
            for (i, pose) in s.poses.iter().enumerate() {
                let out =
                    measure_station(probe, cal, img, &mut s.edges, &spec, &g, pose, (pass, i));
                s.outs.push(out);
            }
            None
        }
        BeadPairing::Coupled(coupling) => {
            let stage = Stage {
                spec,
                polarity: cfg.polarity,
                min_width: f64::from(cfg.min_width),
                max_width: f64::from(cfg.max_width),
                reach,
                h,
                clearance: cfg.clearance.map(f64::from),
                cfg: coupling,
            };
            Some(coupled_stage::measure(
                probe,
                cal,
                img,
                &s.poses,
                &mut s.edges,
                &mut s.outs,
                &mut s.coupled,
                &stage,
                pass,
            ))
        }
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
    let mut passes = Vec::with_capacity(t.passes.get());
    let mut stop = BeadStop::PassLimit;
    for pass in 0..t.passes.get() {
        frame(s, h, window);
        let path = measure_stage(probe, track_cal, img, s, cfg, &cfg.track, h, Some(pass));
        let mut tally = Tally::default();
        s.obs.clear();
        s.valid.clear();
        for out in &s.outs {
            if let Err(r) = out.hit {
                tally.add(r);
            }
            s.obs.push(out.obs.unwrap_or(0.0));
            s.valid.push(out.obs.is_some());
        }
        let n_valid = s.valid.iter().filter(|&&v| v).count();
        let support = n_valid as f64 / n as f64;
        let gap = longest_run_missing(s.valid.iter().copied()) as f64 * h;
        let mut record = BeadPass {
            n_valid,
            support: support as f32,
            longest_gap: gap as f32,
            solve: None,
            rejects: tally.to_vec(),
            path,
        };
        if n_valid == 0 || support < f64::from(t.min_support) {
            s.applied.clear();
            s.applied.resize(n, 0.0);
            s.solve.smooth.weights.clear();
            s.solve.smooth.weights.resize(n, 0.0);
            probe.solved(pass, s.solve.weights(), &s.applied);
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
        let alpha = step_scale(&s.pts, &s.normals, &s.kappa, s.solve.d());
        let solved_max = s.solve.d().iter().fold(0.0f64, |m, d| m.max(d.abs()));
        s.applied.clear();
        s.applied.extend(s.solve.d().iter().map(|&d| alpha * d));
        let (mut sum2, mut max) = (0.0f64, 0.0f64);
        s.src.clear();
        for ((&p, &nrm), &c) in s.pts.iter().zip(&s.normals).zip(&s.applied) {
            sum2 += c * c;
            max = max.max(c.abs());
            s.src.push([p[0] + c * nrm[0], p[1] + c * nrm[1]]);
        }
        probe.solved(pass, s.solve.weights(), &s.applied);
        let moved = arc_lengths(&s.src, &mut s.cum);
        resample(&s.src, &s.cum, n, &mut s.pts);
        h = moved / (n - 1) as f64;

        record.solve = Some(BeadSolve {
            correction_rms: (sum2 / n as f64).sqrt() as f32,
            correction_max: max as f32,
            residual_rms: solved.residual_rms as f32,
            residual_max: solved.residual_max as f32,
            step_scale: alpha as f32,
            irls_iters: solved.irls_iters,
        });
        passes.push(record);
        // Judged on the solved correction, so a step the fold guard shortened is never
        // taken for convergence.
        if alpha == 1.0 && solved_max < f64::from(t.tol) {
            stop = BeadStop::Converged;
            break;
        }
    }

    // The final stage: new poses on the refined stations, the stricter caliper, no
    // feedback into the curve.
    frame(s, h, window);
    let path = measure_stage(probe, measure_cal, img, s, cfg, &cfg.measure, h, None);
    let mut tally = Tally::default();
    let mut samples = Vec::with_capacity(n);
    for (pose, out) in s.poses.iter().zip(&s.outs) {
        if let Err(r) = out.hit {
            tally.add(r);
        }
        samples.push(BeadSample {
            point: Point2f::new(pose.point[0] as f32, pose.point[1] as f32),
            normal: Vec2f::new(pose.normal[0] as f32, pose.normal[1] as f32),
            hit: out.hit,
            wall: out.wall,
            margin: out.margin,
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
            path,
        },
        samples,
        track: BeadTrack { passes, stop },
    })
}
