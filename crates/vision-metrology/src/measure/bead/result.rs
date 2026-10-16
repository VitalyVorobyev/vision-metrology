//! What a bead tracker returns: the refined curve, the final measurement at every station,
//! and the statistics that qualify both.

use vm_primitives::{Point2f, Vec2f};

use crate::measure::{MeasureEdge, MeasurePair, RejectReason};

/// A tracked bead: the refined centreline, the final measurement at each of its stations,
/// and the quality of both.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TrackedBead {
    /// The refined stations, uniform in arc length, in image coordinates. Passed back as
    /// the next call's prior, it tracks the bead from frame to frame.
    pub centerline: Vec<Point2f>,
    /// Arc length between consecutive stations of `centerline`, in pixels.
    pub spacing: f32,
    /// The final stage's measurement at each station, parallel to `centerline`.
    pub samples: Vec<BeadSample>,
    /// The final stage's quality.
    pub summary: BeadSummary,
    /// What each tracking pass did, and why the loop stopped.
    pub track: BeadTrack,
}

/// The final measurement at one station.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BeadSample {
    /// The station on the refined centreline.
    pub point: Point2f,
    /// The unit normal there, `t.perp() = (−t_y, t_x)` for the tangent `t` towards the
    /// curve's end. The strip scans along it, and offsets are signed along it.
    pub normal: Vec2f,
    /// The bead's edge pair, or the gate that rejected every pair.
    pub hit: Result<BeadHit, BeadReject>,
    /// The one wall measured where the other left the image: set exactly when `hit` is
    /// [`BeadReject::OneSided`].
    pub wall: Option<BeadWall>,
    /// [`BeadPairing::Coupled`](super::BeadPairing::Coupled) only, at a station with a
    /// wall: what the best path that moves a wall more than
    /// [`margin_tol_px`](super::BeadCoupling::margin_tol_px) here costs more than the
    /// chosen one, in the path's dimensionless cost; `+∞` when no such path exists. Low
    /// means another line was nearly as good.
    pub margin: Option<f32>,
}

/// Which wall of a bead: the one on the `−n` side of the station, or on the `+n` side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BeadSide {
    /// The `−n` side: a pair's `first` edge.
    First,
    /// The `+n` side: a pair's `second` edge.
    Second,
}

/// One wall of a bead, measured alone.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BeadWall {
    /// Which wall it is.
    pub side: BeadSide,
    /// The caliper's edge, with `t` along the station's strip.
    pub edge: MeasureEdge,
    /// The edge's signed distance from the station along the normal, in pixels.
    pub offset: f32,
}

/// What a coupled path ([`BeadPairing::Coupled`](super::BeadPairing::Coupled)) says about
/// one stage's stations as a whole.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BeadPathScore {
    /// The [`margin_quantile`](super::BeadCoupling::margin_quantile) of the station
    /// margins ([`BeadSample::margin`]) over the stations with a wall; `None` without
    /// any, and in the tracking passes after the first, which choose the path without
    /// its margins; `+∞` when no station has a different answer. A low value means another line
    /// in the window was nearly as good as the one chosen: it flags a competitor, and
    /// does not certify a lock.
    pub margin: Option<f32>,
    /// The path's mean gain per pixel of curve over calling every station absent, in the
    /// path's cost units: how much evidence the chosen walls have, smoothness included.
    pub evidence: f32,
}

/// The bead found at a station.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BeadHit {
    /// The two edges: `first` on the −n side, `second` on the +n side. `center` is their
    /// midpoint and `width` their distance along the normal, in pixels.
    pub pair: MeasurePair,
    /// The midpoint's signed distance from the station along the normal, in pixels.
    pub offset: f32,
    /// `margin · min(a₁, a₂) / max(a₁, a₂)`, in `[0, 1]`: the margin over the best other
    /// pair, `1 − s₂/s₁` in scores (1 when there is none), times the balance of the two
    /// edge amplitudes. For diagnostics; no gate reads it.
    pub confidence: f32,
}

/// Why a station found no bead: the gate that removed the last candidate pair.
///
/// The gates run in this order: the caliper itself, then [`NoPair`](Self::NoPair),
/// [`Width`](Self::Width), [`Offset`](Self::Offset), [`Clearance`](Self::Clearance) and,
/// on the chosen pair, [`Ambiguous`](Self::Ambiguous). With
/// [`BeadPairing::Coupled`](super::BeadPairing::Coupled) the path replaces the pair
/// gates: a station is [`Absent`](Self::Absent), [`OneSided`](Self::OneSided), or
/// [`NoPair`](Self::NoPair) when no caliper edge lies at a wall the path chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum BeadReject {
    /// The strip's caliper found no edge, and says why. [`RejectReason::OffImage`] also
    /// covers a strip whose every edge lies outside the image, in border fill: an edge
    /// there is not evidence, so the tracker drops it before pairing.
    Caliper(RejectReason),
    /// No edge of the bead's leading polarity is followed by one of the trailing polarity.
    /// With coupled pairing: no caliper edge of its polarity lies within a pixel of
    /// either wall the path chose.
    NoPair,
    /// Every ordered pair is narrower than `min_width` or wider than `max_width`.
    Width,
    /// Every pair of a valid width has its midpoint outside the stage's offset window.
    Offset,
    /// Every remaining pair has another edge within `clearance` outside it.
    Clearance,
    /// The best pair's margin over the next is below `min_margin`.
    Ambiguous,
    /// Coupled pairing: the path found no bead at the station, although its strip had
    /// valid evidence and edges.
    Absent,
    /// Coupled pairing: one wall was measured and the other left the image, so there is
    /// no width. [`BeadSample::wall`] holds the measured wall.
    OneSided,
}

impl BeadReject {
    /// Every reason in the order tallies list them: the caliper's, then the pair gates'.
    pub(super) const ALL: [Self; 15] = [
        Self::Caliper(RejectReason::ProfileTooShort),
        Self::Caliper(RejectReason::NoEdge),
        Self::Caliper(RejectReason::WrongPolarity),
        Self::Caliper(RejectReason::TooOblique),
        Self::Caliper(RejectReason::OffImage),
        Self::Caliper(RejectReason::IncompleteSequence),
        Self::Caliper(RejectReason::LowContrast),
        Self::Caliper(RejectReason::NoCrossing),
        Self::NoPair,
        Self::Width,
        Self::Offset,
        Self::Clearance,
        Self::Ambiguous,
        Self::Absent,
        Self::OneSided,
    ];

    /// The reason's position in [`ALL`](Self::ALL).
    pub(super) fn index(self) -> usize {
        match self {
            Self::Caliper(r) => match r {
                RejectReason::ProfileTooShort => 0,
                RejectReason::NoEdge => 1,
                RejectReason::WrongPolarity => 2,
                RejectReason::TooOblique => 3,
                RejectReason::OffImage => 4,
                RejectReason::IncompleteSequence => 5,
                RejectReason::LowContrast => 6,
                RejectReason::NoCrossing => 7,
            },
            Self::NoPair => 8,
            Self::Width => 9,
            Self::Offset => 10,
            Self::Clearance => 11,
            Self::Ambiguous => 12,
            Self::Absent => 13,
            Self::OneSided => 14,
        }
    }

    /// A stable snake_case name: the caliper's reason (`"no_edge"`, `"off_image"`, …) or
    /// the gate (`"no_pair"`, `"width"`, `"offset"`, `"clearance"`, `"ambiguous"`,
    /// `"absent"`, `"one_sided"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Caliper(r) => match r {
                RejectReason::ProfileTooShort => "profile_too_short",
                RejectReason::NoEdge => "no_edge",
                RejectReason::WrongPolarity => "wrong_polarity",
                RejectReason::TooOblique => "too_oblique",
                RejectReason::OffImage => "off_image",
                RejectReason::IncompleteSequence => "incomplete_sequence",
                RejectReason::LowContrast => "low_contrast",
                RejectReason::NoCrossing => "no_crossing",
            },
            Self::NoPair => "no_pair",
            Self::Width => "width",
            Self::Offset => "offset",
            Self::Clearance => "clearance",
            Self::Ambiguous => "ambiguous",
            Self::Absent => "absent",
            Self::OneSided => "one_sided",
        }
    }
}

/// The final stage's quality.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BeadSummary {
    /// The fraction of stations with a hit, in `[0, 1]`.
    pub support: f32,
    /// The longest run of consecutive stations without a hit, in pixels of arc length:
    /// the run's station count times the spacing.
    pub longest_gap: f32,
    /// Position and width statistics over the hits; `None` without any.
    pub stats: Option<BeadStats>,
    /// Stations rejected, counted by reason, in [`BeadReject`] declaration order; reasons
    /// that did not occur are left out.
    pub rejects: Vec<(BeadReject, usize)>,
    /// The final stage's path, with coupled pairing; `None` per station. The call's
    /// competitor flag is the first tracking pass's
    /// ([`BeadPass::path`]), whose window is the prior's.
    pub path: Option<BeadPathScore>,
}

/// Statistics over the final stage's hits.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BeadStats {
    /// The number of hits.
    pub n_used: usize,
    /// RMS of the hits' offsets, in pixels: how far the measured midpoints sit from the
    /// refined centreline.
    pub center_rms: f32,
    /// The largest `|offset|`, in pixels.
    pub center_max_dev: f32,
    /// Mean width, in pixels.
    pub width_mean: f32,
    /// Standard deviation of the width over the hits (divided by `n_used`), in pixels.
    pub width_std: f32,
    /// Narrowest width, in pixels.
    pub width_min: f32,
    /// Widest width, in pixels.
    pub width_max: f32,
}

/// What the tracking loop did.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BeadTrack {
    /// One entry per pass run, in order.
    pub passes: Vec<BeadPass>,
    /// Why the loop stopped.
    pub stop: BeadStop,
}

/// One tracking pass: its evidence and, when it had enough, its solve.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BeadPass {
    /// Stations whose tracking strip found a pair.
    pub n_valid: usize,
    /// `n_valid` as a fraction of the stations.
    pub support: f32,
    /// The longest run of stations without a pair, in pixels of arc length.
    pub longest_gap: f32,
    /// The solve and the correction it applied; `None` when the pass found too few pairs
    /// to solve and left the curve where it was.
    pub solve: Option<BeadSolve>,
    /// Stations rejected, counted by reason, as in [`BeadSummary::rejects`].
    pub rejects: Vec<(BeadReject, usize)>,
    /// The pass's path, with coupled pairing; `None` per station. The first pass's
    /// window is the prior's, so its margin is the call's flag for a competing line.
    pub path: Option<BeadPathScore>,
}

/// One pass's solve and the correction it applied to the stations.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BeadSolve {
    /// RMS of the correction applied to the stations, in pixels.
    pub correction_rms: f32,
    /// The largest correction applied, in pixels.
    pub correction_max: f32,
    /// RMS, over the valid stations, of the observed offset minus the solved one, in
    /// pixels: how far the smooth correction stays from this pass's evidence.
    pub residual_rms: f32,
    /// The largest such residual, in pixels.
    pub residual_max: f32,
    /// The fraction of the solved correction applied, in `(0, 1]`: less than 1 when the
    /// full step would fold the curve.
    pub step_scale: f32,
    /// Reweighted solves run after the first, least-squares one; 0 with
    /// [`RobustLoss::None`](crate::fit::RobustLoss::None).
    pub irls_iters: usize,
}

/// Why the tracking loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum BeadStop {
    /// The last pass's solved correction was below
    /// [`BeadTuning::tol`](super::BeadTuning::tol) at every station, and was applied in
    /// full. It says the loop stopped moving the curve, not that the curve sits on the
    /// bead: the final stage's [`BeadStats::center_rms`] and
    /// [`BeadStats::center_max_dev`] are the evidence of fit.
    Converged,
    /// Every pass ran.
    PassLimit,
    /// A pass found the bead at fewer stations than
    /// [`BeadTuning::min_support`](super::BeadTuning::min_support), or at none, and left
    /// the curve where it was.
    TooFewValid,
}

impl BeadStop {
    /// A stable snake_case name: `"converged"`, `"pass_limit"` or `"too_few_valid"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::PassLimit => "pass_limit",
            Self::TooFewValid => "too_few_valid",
        }
    }
}
