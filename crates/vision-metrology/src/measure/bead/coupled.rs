//! The coupled path (ADR-0019): both walls at every station, chosen together as one exact
//! min-path over (centre, width) bins, with one-wall and absent states and max-marginal
//! margins.
//!
//! The evidence is one stage's caliper responses, station-major, with a validity flag per
//! profile sample. Samples are pooled into bins of a whole number of samples: a bin's
//! evidence is the largest saturated response among its samples, and a bin is valid when
//! every one of its samples is. Per station the states are `PLANES` planes of a
//! `centres × widths` grid. Centre `ci` sits `(ci − kc)·b` along `+n` and half-width
//! `hw0 + wj` bins, so the walls fall on bins `ci − kc ∓ (hw0 + wj)`.
//!
//! Costs, per pixel of curve and times the spacing `h` (see `BeadCoupling`):
//! - [`BOTH`]: `−e_l − e_r + λ0 (c/reach)²`, both wall bins valid;
//! - [`FIRST`] or [`SECOND`]: `−e − e_a + λ0 (c/reach)²`, where the other wall's bin is
//!   not valid;
//! - [`ABSENT`]: `−2 e_a`.
//!
//! A state whose centre is outside its station's window does not exist. The step cost is
//! `2κb²/h` per centre bin squared and `(κ/2 + κ_w)(2b)²/h` per width bin squared, from
//! `Δl² + Δr² = 2Δc² + Δw²/2`.

// Invariants 6 (scratch reused), 12 (ties to the lower index, through the min-path) and
// 20 (costs accumulate in f64).

use vm_primitives::EdgePolarity;

use super::config::BeadPolarity;
use super::coupling::BeadCoupling;
use crate::measure::numeric::minpath::{Grid, LayeredPath, Steps};

/// Both walls measured.
pub(super) const BOTH: usize = 0;
/// The `−n` wall alone.
pub(super) const FIRST: usize = 1;
/// The `+n` wall alone.
pub(super) const SECOND: usize = 2;
/// No bead.
pub(super) const ABSENT: usize = 3;
const PLANES: usize = 4;

/// A chosen wall moves to a caliper edge at most this far, in pixels, or one and a half
/// bins if that is wider; the clearance looks beyond it.
const SNAP_PX: f64 = 1.0;

/// A bin's samples: none, all valid, or at least one invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bin {
    Empty,
    Valid,
    Invalid,
}

/// One stage's evidence, station-major.
#[derive(Debug, Clone, Default)]
pub(super) struct Evidence {
    /// Profile samples per station.
    pub samples: usize,
    /// Pixels between samples.
    pub spacing: f64,
    /// The offset of sample 0 from its station along `+n`, in pixels.
    pub origin: f64,
    /// The derivative along `+n`, in input units per pixel; read where `valid`.
    pub resp: Vec<f32>,
    pub valid: Vec<bool>,
}

impl Evidence {
    /// Empty, for stations of `samples` samples `spacing` px apart from `origin`.
    pub(super) fn reset(&mut self, samples: usize, spacing: f64, origin: f64) {
        self.samples = samples;
        self.spacing = spacing;
        self.origin = origin;
        self.resp.clear();
        self.valid.clear();
    }

    pub(super) fn stations(&self) -> usize {
        self.resp.len().checked_div(self.samples).unwrap_or(0)
    }
}

/// The geometry one stage's path searches.
#[derive(Debug, Clone, Copy)]
pub(super) struct Setup {
    pub polarity: BeadPolarity,
    pub min_width: f64,
    pub max_width: f64,
    /// The centre's reach either side of the station, in pixels.
    pub reach: f64,
    /// The station spacing, in pixels.
    pub h: f64,
    /// When set, a wall pays the evidence of the strongest other edge within this many
    /// pixels outside it, beyond the snap radius.
    pub clearance: Option<f64>,
    /// Whether to compute the margins, which take the min-path's backward pass.
    pub margins: bool,
}

/// The path's state at one station: its plane, its walls' offsets on the grid, and its
/// margin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Choice {
    /// [`BOTH`], [`FIRST`], [`SECOND`] or [`ABSENT`].
    pub plane: usize,
    /// The `−n` and `+n` walls' offsets along `+n`, in pixels; an unmeasured wall's is
    /// the path's geometry.
    pub first: f64,
    pub second: f64,
    /// Present stations, when the margins were asked for: the best path with a wall more
    /// than the tolerance away, less the best path's cost; `+∞` when there is none.
    pub margin: Option<f64>,
}

/// What one stage's path says as a whole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PathOut {
    pub margin: Option<f64>,
    pub evidence: f64,
}

/// The wall signs that make a bead's expected derivative positive: the `−n` wall, then the
/// `+n` one.
fn signs(p: BeadPolarity) -> (f32, f32) {
    match p {
        BeadPolarity::Light => (1.0, -1.0),
        BeadPolarity::Dark => (-1.0, 1.0),
    }
}

/// The polarity each wall has along `+n`: the `−n` wall, then the `+n` one.
pub(super) fn wall_polarities(p: BeadPolarity) -> (EdgePolarity, EdgePolarity) {
    match p {
        BeadPolarity::Light => (EdgePolarity::Rising, EdgePolarity::Falling),
        BeadPolarity::Dark => (EdgePolarity::Falling, EdgePolarity::Rising),
    }
}

/// `e(g) = (g/s)/(1 + |g/s|)`.
#[inline]
fn sat(g: f32, s: f32) -> f32 {
    let x = g / s;
    x / (1.0 + x.abs())
}

/// The coupled path's solver and buffers.
#[derive(Debug, Clone, Default)]
pub(super) struct CoupledPath {
    path: LayeredPath,
    steps: Steps,
    /// One station's bins: the `−n` and `+n` walls' evidence, and their state.
    ef: Vec<f32>,
    es: Vec<f32>,
    bins: Vec<Bin>,
    /// The strongest edge's evidence in each bin, for the clearance.
    peaks: Vec<f32>,
    /// Per bin, the cost of a wall that is not observable there: `−h·e_a` where the bin
    /// is not valid, `+∞` where it is.
    miss: Vec<f32>,
    /// The last call's choice per station.
    pub choices: Vec<Choice>,
    margins: Vec<f64>,
    /// The last call's bin, in pixels.
    bin: f64,
}

/// The grid of one call.
#[derive(Debug, Clone, Copy)]
struct Layout {
    /// The bin, in pixels: a whole number of samples.
    b: f64,
    /// The centre's reach in bins; centres run over `2·kc + 1` bins.
    kc: usize,
    /// The smallest half-width, in bins, and the width count.
    hw0: usize,
    widths: usize,
    /// The furthest wall bin from the station; bins run over `2·mb + 1`.
    mb: usize,
}

impl Layout {
    fn new(ev: &Evidence, s: &Setup, cfg: &BeadCoupling) -> Self {
        let kb = ((f64::from(cfg.bin_px) / ev.spacing).round() as usize).max(1);
        let b = kb as f64 * ev.spacing;
        let kc = (s.reach / b + 1e-9).floor() as usize;
        let hw0 = ((s.min_width / (2.0 * b) - 1e-9).ceil() as usize).max(1);
        let hw1 = ((s.max_width / (2.0 * b) + 1e-9).floor() as usize).max(hw0);
        Self {
            b,
            kc,
            hw0,
            widths: hw1 - hw0 + 1,
            mb: kc + hw1,
        }
    }

    fn grid(&self) -> Grid {
        Grid {
            planes: PLANES,
            dims: [2 * self.kc + 1, self.widths],
        }
    }

    /// The walls' bin offsets from the station, in bins, at cell `(ci, wj)`.
    fn walls(&self, ci: usize, wj: usize) -> (i64, i64) {
        let c = ci as i64 - self.kc as i64;
        let hw = (self.hw0 + wj) as i64;
        (c - hw, c + hw)
    }
}

impl CoupledPath {
    /// Solve one stage: `ev` holds its stations' evidence, `windows` each station's
    /// admissible centre offsets `(lo, hi)`. Fills [`choices`](Self::choices).
    pub(super) fn extract(
        &mut self,
        ev: &Evidence,
        windows: &[(f64, f64)],
        s: &Setup,
        cfg: &BeadCoupling,
    ) -> PathOut {
        let n = ev.stations();
        debug_assert_eq!(windows.len(), n);
        self.choices.clear();
        if n == 0 {
            return PathOut {
                margin: None,
                evidence: 0.0,
            };
        }
        let lay = Layout::new(ev, s, cfg);
        self.bin = lay.b;
        let grid = lay.grid();
        let (b, h) = (lay.b, s.h);
        let kappa = f64::from(cfg.wall_stiffness);
        let kappa_w = f64::from(cfg.width_stiffness);
        self.steps.coef = [
            (2.0 * kappa * b * b / h) as f32,
            ((0.5 * kappa + kappa_w) * 4.0 * b * b / h) as f32,
        ];
        let band =
            |slope: f32, bin: f64| ((f64::from(slope) * h / bin + 1e-9).floor() as usize).max(1);
        self.steps.band = [
            band(cfg.max_wall_slope, b),
            band(cfg.max_width_slope, 2.0 * b),
        ];
        let sw = cfg.switch_cost;
        self.steps.switch.clear();
        for p in 0..PLANES {
            for q in 0..PLANES {
                let free = p == q || ((p == BOTH) != (q == BOTH) && p != ABSENT && q != ABSENT);
                self.steps.switch.push(if free { 0.0 } else { sw });
            }
        }
        self.path.reset(n, grid, &self.steps);

        let ea = sat(cfg.absent_level, cfg.saturation);
        let absent = (h * f64::from(-2.0 * ea)) as f32;
        for (k, &(lo, hi)) in windows.iter().enumerate() {
            self.bin_station(ev, k, &lay, s, cfg);
            self.wall_costs(h, ea);
            let u = self.path.unary_mut(k);
            fill_unary(
                u,
                (&self.ef, &self.es, &self.miss),
                &lay,
                (lo, hi),
                s,
                cfg,
                absent,
            );
        }
        let Some(cost) = self.path.solve_with(s.margins) else {
            // Unreachable: the absent plane is finite inside every window, and every
            // window holds the station's own centre.
            self.choices.extend((0..n).map(|_| Choice {
                plane: ABSENT,
                first: 0.0,
                second: 0.0,
                margin: None,
            }));
            return PathOut {
                margin: None,
                evidence: 0.0,
            };
        };
        self.read_out(&lay, cfg, cost, s.margins);
        let total = h * n as f64;
        self.margins.clear();
        self.margins
            .extend(self.choices.iter().filter_map(|c| c.margin));
        self.margins.sort_unstable_by(f64::total_cmp);
        let margin = (!self.margins.is_empty()).then(|| {
            let q = f64::from(cfg.margin_quantile);
            let at = ((q * self.margins.len() as f64).ceil() as usize).clamp(1, self.margins.len());
            self.margins[at - 1]
        });
        PathOut {
            margin,
            evidence: (total * f64::from(-2.0 * ea) - cost) / total,
        }
    }

    /// How far, in pixels, a wall of the last call may move to its caliper edge: a pixel,
    /// or one and a half bins on a coarser grid, where a wall the smoothness holds back
    /// can sit a bin from its edge.
    pub(super) fn snap_radius(&self) -> f64 {
        SNAP_PX.max(1.5 * self.bin)
    }

    /// Pool station `k`'s samples into `self`'s bins, and charge each wall bin the
    /// clearance cost.
    fn bin_station(
        &mut self,
        ev: &Evidence,
        k: usize,
        lay: &Layout,
        s: &Setup,
        cfg: &BeadCoupling,
    ) {
        let nb = 2 * lay.mb + 1;
        let sat_s = cfg.saturation;
        for v in [&mut self.ef, &mut self.es] {
            v.clear();
            v.resize(nb, f32::NEG_INFINITY);
        }
        self.peaks.clear();
        self.peaks.resize(nb, 0.0);
        self.bins.clear();
        self.bins.resize(nb, Bin::Empty);
        let (sf, ss) = signs(s.polarity);
        let row = k * ev.samples..(k + 1) * ev.samples;
        let (resp, valid) = (&ev.resp[row.clone()], &ev.valid[row]);
        for (i, (&g, &ok)) in resp.iter().zip(valid).enumerate() {
            let o = ev.origin + i as f64 * ev.spacing;
            let m = (o / lay.b + 0.5).floor() as i64 + lay.mb as i64;
            if m < 0 || m >= nb as i64 {
                continue;
            }
            let m = m as usize;
            if !ok {
                self.bins[m] = Bin::Invalid;
                continue;
            }
            if self.bins[m] == Bin::Empty {
                self.bins[m] = Bin::Valid;
            }
            self.ef[m] = self.ef[m].max(sat(sf * g, sat_s));
            self.es[m] = self.es[m].max(sat(ss * g, sat_s));
            // An edge, for the clearance: a local maximum of |g| above the absent level.
            let peak = i > 0
                && i + 1 < resp.len()
                && valid[i - 1]
                && valid[i + 1]
                && g.abs() >= resp[i - 1].abs()
                && g.abs() > resp[i + 1].abs()
                && g.abs() >= cfg.absent_level;
            if peak {
                self.peaks[m] = self.peaks[m].max(sat(g.abs(), sat_s));
            }
        }
        let Some(c) = s.clearance else {
            return;
        };
        // The bins further than the snap radius from a wall, and within `c` outside it.
        let near = (SNAP_PX.max(1.5 * lay.b) / lay.b).floor() as usize + 1;
        let far = (c / lay.b + 1e-9).floor() as usize;
        if far < near {
            return;
        }
        for m in 0..nb {
            let below = m.saturating_sub(far)..m.saturating_sub(near - 1).min(m);
            let above = (m + near).min(nb)..(m + far + 1).min(nb);
            let worst =
                |r: std::ops::Range<usize>| self.peaks[r].iter().fold(0.0f32, |a, &v| a.max(v));
            let (out_first, out_second) = (worst(below), worst(above));
            self.ef[m] -= out_first;
            self.es[m] -= out_second;
        }
    }

    /// Turn the bins' evidence into wall costs per pixel of curve times `h`: `ef` and `es`
    /// become `−h·e` where valid and `+∞` elsewhere, and `miss` the cost of an
    /// unobservable wall.
    fn wall_costs(&mut self, h: f64, ea: f32) {
        let h = h as f32;
        self.miss.clear();
        for ((f, s), &b) in self.ef.iter_mut().zip(&mut self.es).zip(&self.bins) {
            let ok = b == Bin::Valid;
            *f = if ok { -h * *f } else { f32::INFINITY };
            *s = if ok { -h * *s } else { f32::INFINITY };
            self.miss.push(if ok { f32::INFINITY } else { -h * ea });
        }
    }

    /// Read the best path and each present station's margin.
    fn read_out(&mut self, lay: &Layout, cfg: &BeadCoupling, best: f64, margins: bool) {
        let g = self.path.grid();
        let (cells, widths) = (g.dims[0], g.dims[1]);
        // A wall is "elsewhere" when it moved by more than this many whole bins.
        let tol = (f64::from(cfg.margin_tol_px) / lay.b).floor() as i64;
        for (k, &state) in self.path.path().iter().enumerate() {
            let (plane, ci, wj) = g.coords(state);
            let (l, r) = lay.walls(ci, wj);
            let margin = (margins && plane != ABSENT).then(|| {
                let (off, alpha, beta) = self.path.marginal_parts(k);
                let mut m = f32::INFINITY;
                for p in [BOTH, FIRST, SECOND] {
                    for c2 in 0..cells {
                        let base = (p * cells + c2) * widths;
                        let (a, b) = (&alpha[base..base + widths], &beta[base..base + widths]);
                        let c = c2 as i64 - lay.kc as i64;
                        for (w2, (&x, &y)) in a.iter().zip(b).enumerate() {
                            let hw = (lay.hw0 + w2) as i64;
                            if (c - hw - l).abs() > tol || (c + hw - r).abs() > tol {
                                m = m.min(x + y);
                            }
                        }
                    }
                }
                off + f64::from(m) - best
            });
            self.choices.push(Choice {
                plane,
                first: l as f64 * lay.b,
                second: r as f64 * lay.b,
                margin,
            });
        }
    }
}

/// Station costs for one layer `u`, from its wall costs (`wall_costs`).
fn fill_unary(
    u: &mut [f32],
    (wf, ws, miss): (&[f32], &[f32], &[f32]),
    lay: &Layout,
    (lo, hi): (f64, f64),
    s: &Setup,
    cfg: &BeadCoupling,
    absent: f32,
) {
    let g = lay.grid();
    let (cells, widths) = (g.dims[0], g.dims[1]);
    let lambda = f64::from(cfg.offset_prior);
    for ci in 0..cells {
        let c_bins = ci as i64 - lay.kc as i64;
        let c = c_bins as f64 * lay.b;
        let row = |p: usize| (p * cells + ci) * widths..(p * cells + ci + 1) * widths;
        if !(c >= lo - 1e-9 && c <= hi + 1e-9) {
            for p in [BOTH, FIRST, SECOND, ABSENT] {
                u[row(p)].fill(f32::INFINITY);
            }
            continue;
        }
        let prior = (s.h * lambda * (c / s.reach) * (c / s.reach)) as f32;
        // The walls' bins at the narrowest width; each wider width moves them a bin out.
        let l0 = (c_bins - lay.hw0 as i64 + lay.mb as i64) as usize;
        let r0 = (c_bins + lay.hw0 as i64 + lay.mb as i64) as usize;
        for wj in 0..widths {
            let (l, r) = (l0 - wj, r0 + wj);
            let i = ci * widths + wj;
            let plane = cells * widths;
            u[BOTH * plane + i] = prior + wf[l] + ws[r];
            u[FIRST * plane + i] = prior + wf[l] + miss[r];
            u[SECOND * plane + i] = prior + miss[l] + ws[r];
        }
        u[row(ABSENT)].fill(absent);
    }
}

/// The index of the edge of `polarity` nearest `target` among `edges` (offset, polarity),
/// within `radius` px; ties to the earlier edge.
pub(super) fn snap(
    edges: impl Iterator<Item = (f64, EdgePolarity)>,
    target: f64,
    polarity: EdgePolarity,
    radius: f64,
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, (o, p)) in edges.enumerate() {
        let d = (o - target).abs();
        if p == polarity && d <= radius && best.is_none_or(|(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}
