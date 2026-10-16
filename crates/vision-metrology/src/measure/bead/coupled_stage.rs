//! One stage with coupled pairing: every station's strip is measured first, its caliper
//! response kept as the path's evidence and its edges for the walls; then the path
//! chooses, and each chosen wall moves to the nearest caliper edge of its polarity.
//!
//! A response sample is valid when the caliper differentiated its profile and the sample
//! lies on the image (for [`Derivative::Continuous`](crate::measure::Derivative), when its
//! whole support did). A wall with no caliper edge within the snap radius is not
//! measured: the station keeps the other wall alone, or none.

use vm_primitives::{ImageView, Pixel, Point2f};

use super::config::BeadPolarity;
use super::coupled::{
    ABSENT, BOTH, Choice, CoupledPath, Evidence, FIRST, SECOND, Setup, snap, wall_polarities,
};
use super::coupling::BeadCoupling;
use super::pairing::sort_edges;
use super::result::{BeadHit, BeadPathScore, BeadReject, BeadSide, BeadWall};
use super::run::{Outcome, Pose, Probe, StripSpec, retain_inside};
use crate::measure::{Caliper, MeasureEdge, MeasurePair, MeasureStrip, RejectReason};

/// One coupled stage's settings.
#[derive(Debug, Clone, Copy)]
pub(super) struct Stage<'a> {
    pub spec: StripSpec,
    pub polarity: BeadPolarity,
    pub min_width: f64,
    pub max_width: f64,
    pub reach: f64,
    pub h: f64,
    pub clearance: Option<f64>,
    pub cfg: &'a BeadCoupling,
}

/// The buffers of coupled stages.
#[derive(Debug, Clone, Default)]
pub(super) struct CoupledScratch {
    ev: Evidence,
    path: CoupledPath,
    windows: Vec<(f64, f64)>,
    /// Every station's edges on the image, sorted, one run per station.
    edges: Vec<MeasureEdge>,
    runs: Vec<(usize, usize)>,
    measured: Vec<Result<(), RejectReason>>,
    strips: Vec<MeasureStrip>,
    /// Whether the station had any valid response sample.
    seen_any: Vec<bool>,
}

/// Measure every pose's strip with `cal` through `probe`, choose the walls along the
/// curve, and push each station's outcome to `outs`, reporting it to `probe`.
#[allow(clippy::too_many_arguments)]
pub(super) fn measure<P: Pixel, R: Probe>(
    probe: &mut R,
    cal: &mut Caliper,
    img: &ImageView<'_, P>,
    poses: &[Pose],
    edges: &mut Vec<MeasureEdge>,
    outs: &mut Vec<Outcome>,
    cs: &mut CoupledScratch,
    st: &Stage<'_>,
    pass: Option<usize>,
) -> BeadPathScore {
    let samples = st.spec.samples.get();
    cs.windows.clear();
    cs.edges.clear();
    cs.runs.clear();
    cs.measured.clear();
    cs.strips.clear();
    cs.seen_any.clear();
    // Holds nothing for the quiet probe, so it does not allocate.
    let mut seen = Vec::with_capacity(poses.len());
    for (i, pose) in poses.iter().enumerate() {
        let strip = st.spec.place(pose);
        cal.set_strip(strip);
        let (measured, sn) = probe.measure(cal, img, edges);
        seen.push(sn);
        let geom = strip.geometry();
        let spacing = f64::from(cal.spacing());
        if i == 0 {
            cs.ev.reset(samples, spacing, -0.5 * geom.length);
        }
        let any = push_evidence(&mut cs.ev, cal, img, &strip, samples);
        cs.seen_any.push(any);
        retain_inside(edges, img);
        sort_edges(edges);
        let start = cs.edges.len();
        cs.edges.extend_from_slice(edges);
        cs.runs.push((start, cs.edges.len()));
        cs.measured.push(measured);
        cs.strips.push(strip);
        cs.windows.push(pose.window);
    }
    let setup = Setup {
        polarity: st.polarity,
        min_width: st.min_width,
        max_width: st.max_width,
        reach: st.reach,
        h: st.h,
        clearance: st.clearance,
        // The first pass's margins flag a competitor in the prior's window, and the final
        // stage's are reported per station; the passes between need only the path.
        margins: pass.is_none_or(|p| p == 0),
    };
    let out = cs.path.extract(&cs.ev, &cs.windows, &setup, st.cfg);
    let radius = cs.path.snap_radius();
    for (i, sn) in seen.into_iter().enumerate() {
        let (a, b) = cs.runs[i];
        let center_t = 0.5 * cs.strips[i].geometry().length;
        let o = outcome(
            &cs.path.choices[i],
            &cs.edges[a..b],
            center_t,
            radius,
            st.polarity,
            (cs.measured[i], cs.seen_any[i]),
        );
        probe.station(pass, i, &poses[i], &cs.strips[i], sn, &o);
        outs.push(o);
    }
    BeadPathScore {
        margin: out.margin.map(|m| m as f32),
        evidence: out.evidence as f32,
    }
}

/// Append one station's response, per pixel, and its validity to `ev`; whether any sample
/// was valid.
fn push_evidence<P: Pixel>(
    ev: &mut Evidence,
    cal: &Caliper,
    img: &ImageView<'_, P>,
    strip: &MeasureStrip,
    samples: usize,
) -> bool {
    let spacing = cal.spacing();
    let Some((resp, valid)) = cal.response().filter(|(r, _)| r.len() == samples) else {
        ev.resp.extend((0..samples).map(|_| 0.0));
        ev.valid.extend((0..samples).map(|_| false));
        return false;
    };
    let (w, h) = (
        img.width().saturating_sub(1) as f32,
        img.height().saturating_sub(1) as f32,
    );
    let (p0, p1) = (strip.start, strip.end);
    let denom = (samples - 1).max(1) as f32;
    let mut any = false;
    for (j, &g) in resp.iter().enumerate() {
        let f = j as f32 / denom;
        let p = Point2f::new(p0.x + f * (p1.x - p0.x), p0.y + f * (p1.y - p0.y));
        let on = p.x >= 0.0 && p.y >= 0.0 && p.x <= w && p.y <= h;
        let ok = on && g.is_finite() && valid.is_none_or(|v| v[j]);
        any |= ok;
        ev.resp.push(if ok { g / spacing } else { 0.0 });
        ev.valid.push(ok);
    }
    any
}

/// What the path's `choice` at one station means, with its sorted on-image `edges`.
fn outcome(
    choice: &Choice,
    edges: &[MeasureEdge],
    center_t: f64,
    radius: f64,
    polarity: BeadPolarity,
    (measured, any): (Result<(), RejectReason>, bool),
) -> Outcome {
    let (lead, trail) = wall_polarities(polarity);
    let near = |target: f64, pol| {
        snap(
            edges
                .iter()
                .map(|e| (f64::from(e.t) - center_t, e.polarity)),
            target,
            pol,
            radius,
        )
        .map(|k| edges[k])
    };
    let off = |e: &MeasureEdge| f64::from(e.t) - center_t;
    let half = 0.5 * (choice.second - choice.first);
    let margin = choice.margin.map(|m| m as f32);
    let none = || match measured {
        Err(r) => BeadReject::Caliper(r),
        Ok(()) => BeadReject::NoPair,
    };
    let one = |side, e: MeasureEdge, obs| Outcome {
        hit: Err(BeadReject::OneSided),
        wall: Some(BeadWall {
            side,
            edge: e,
            offset: off(&e) as f32,
        }),
        margin,
        obs: Some(obs),
    };
    let grid = |hit| Outcome {
        hit: Err(hit),
        wall: None,
        margin,
        obs: Some(0.5 * (choice.first + choice.second)),
    };
    match choice.plane {
        BOTH => match (near(choice.first, lead), near(choice.second, trail)) {
            (Some(a), Some(b)) if b.t > a.t => {
                let (lo, hi) = (a.amplitude.min(b.amplitude), a.amplitude.max(b.amplitude));
                let mid = 0.5 * (off(&a) + off(&b));
                Outcome {
                    hit: Ok(BeadHit {
                        pair: MeasurePair {
                            first: a,
                            second: b,
                            center: Point2f::new(0.5 * (a.p.x + b.p.x), 0.5 * (a.p.y + b.p.y)),
                            width: b.t - a.t,
                        },
                        offset: mid as f32,
                        confidence: if hi > 0.0 { lo / hi } else { 0.0 },
                    }),
                    wall: None,
                    margin,
                    obs: Some(mid),
                }
            }
            (Some(a), None) => one(BeadSide::First, a, off(&a) + half),
            (None, Some(b)) => one(BeadSide::Second, b, off(&b) - half),
            _ => grid(none()),
        },
        FIRST => match near(choice.first, lead) {
            Some(a) => one(BeadSide::First, a, off(&a) + half),
            None => grid(none()),
        },
        SECOND => match near(choice.second, trail) {
            Some(b) => one(BeadSide::Second, b, off(&b) - half),
            None => grid(none()),
        },
        _ => {
            debug_assert_eq!(choice.plane, ABSENT);
            let hit = match measured {
                _ if !any => BeadReject::Caliper(RejectReason::OffImage),
                Err(r) => BeadReject::Caliper(r),
                Ok(()) => BeadReject::Absent,
            };
            Outcome {
                hit: Err(hit),
                wall: None,
                margin: None,
                obs: None,
            }
        }
    }
}
