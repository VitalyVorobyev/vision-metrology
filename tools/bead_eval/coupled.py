"""Coupled two-boundary extraction along a prior curve, from one image: a prototype.

Today's tracker picks each station's edge pair on its own (`pairing::choose`). Here both
walls are chosen together along the whole curve, as one shortest path, and the path's
max-marginals say how much a different choice at each station would cost. Two variants
share the stations, the profiles, the costs and the sub-pixel step:

- `sparse`: a Viterbi over the caliper's candidate edge pairs per station, plus absence;
- `dense`: a dynamic programme over every (centre, width) bin of the profile, plus
  absence.

**Stations and profiles.** The prior is resampled uniformly in arc length at `spacing`
(`h` px apart), with chord tangents over `±tangent_window` that shrink at the ends, and
`+n = (−t_y, t_x)`. Each station runs one strip caliper along its normal, with the
tracking stage's settings (DoG of `sigma`, `half_width`, every edge of either polarity
above `threshold`) and a profile step of `step` px. The caliper's signed derivative
response is the dense variant's evidence, and its sub-pixel candidate edges are the
sparse variant's pairs and both variants' final wall positions. No second edge detector.

**Costs.** Offsets are along `+n`. A bead's walls are `l < r`; for a dark bead the left
wall falls and the right one rises, for a light bead the reverse. With the soft
saturation `e(g) = (g/g0) / (1 + |g|/g0)` of the derivative signed so the expected
polarity is positive, a present station costs per px of curve

    −e_l(l) − e_r(r) + λ_w·huber(ln(w / w0) / s_w) + λ_0·(c / reach)²

and an absent one `−2·e(absent_level)`, both times `h`. Between stations `h` apart:

    κ·(Δl² + Δr²)/h + κ_w·Δw²/h,  |Δc| ≤ max_wall_slope·h,  |Δw| ≤ max_width_slope·h

plus `switch_cost` per change between present and absent. Per-px data and `1/h`
smoothness are the discretisation of a continuous energy, so the result does not depend
on `h`. `w0` is the call's own median width from a first solve without the width term;
the width term is off by default (`width_prior` 0), since a crack's width varies.

**Dense.** States are (centre bin, width bin): centres on the profile's `step` grid within
`±reach`, widths on a `2·step` grid, so both walls fall on profile samples; `(l, w)` with
`l = c − w/2` is the same grid. Since `Δl² + Δr² = 2Δc² + Δw²/2`, the transition is
separable in `(c, w)`, and the min over it is two banded passes. The absent layer keeps a
geometry, so a gap is bridged smoothly. Solved exactly by forward and backward passes.

**Sparse.** Nodes are the candidate pairs (polarity order, width range, centre within
`±reach`). A gap of up to `max_bridge` px is a skip edge between present nodes, whose
smoothness and step bounds scale with the gap; longer absences go through absent states
without geometry and pay `reacquire_cost` to resume. Leading and trailing absence cost
only the absent term.

**Max-marginals.** `M_i(s)` is the cost of the best path through state `s` at station
`i`, minus the best path's. At a present station the margin is the smallest `M_i` of a
present state with a wall more than `tol` px from the chosen one: what the second-best
geometrically different answer there costs. The call's ambiguity score is a low quantile
of the margins over its present stations (0.25 by default): low means another line was
almost as good.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import common


@dataclasses.dataclass
class CoupledConfig:
    polarity: str = "dark"
    min_width: float = 2.0
    max_width: float = 12.0
    #: px either side of the prior that the centre may move.
    reach: float = 8.0
    spacing: float = 2.0
    tangent_window: float = 10.0
    # The profile: the tracking stage's caliper, sampled finer.
    sigma: float = 1.0
    threshold: float = 3.0
    half_width: float = 2.0
    step: float = 0.25
    # Costs, per px of curve unless said otherwise.
    g0: float = 10.0
    absent_level: float = 4.0
    wall_stiffness: float = 0.2
    width_stiffness: float = 0.5
    max_wall_slope: float = 1.0
    max_width_slope: float = 0.5
    width_prior: float = 0.0
    width_prior_scale: float = 0.3
    offset_prior: float = 0.1
    #: Per change between present and absent.
    switch_cost: float = 2.0
    #: Sparse only: resuming after an absence longer than `max_bridge`.
    reacquire_cost: float = 10.0
    max_bridge: float = 20.0
    #: px: an alternative counts as different when a wall moves further than this. The
    #: margins are kept at `tol` and at each of `extra_tols`, for the evaluation.
    tol: float = 5.0
    extra_tols: tuple = (1.5, 3.0)
    #: Extractions per call: each after the first re-stations on the previous centreline.
    #: The ambiguity always comes from the first, whose window is the prior's.
    passes: int = 1


@dataclasses.dataclass
class Profiles:
    points: np.ndarray  # (N, 2) stations
    normals: np.ndarray  # (N, 2)
    h: float
    offsets: np.ndarray  # (T,) px along +n of each profile sample
    response: np.ndarray  # (N, T) signed derivative per px, 0 off the image
    edges: list  # per station, (k, 3): offset, amplitude, +1 rising / −1 falling


@dataclasses.dataclass
class Extraction:
    present: np.ndarray  # (N,) bool
    left: np.ndarray  # (N,) wall offsets along +n, NaN where absent
    right: np.ndarray
    center: np.ndarray  # (N,) centre offset, bridged across absence
    margins: dict  # tol -> (N,) geometric max-marginal margins, NaN where absent
    presence_margin: np.ndarray  # (N,) cost of flipping present/absent there
    cost: float
    snapped: np.ndarray  # (N,) both walls moved onto caliper edges

    @property
    def support(self) -> float:
        return float(np.mean(self.present)) if len(self.present) else 0.0

    @property
    def margin(self) -> np.ndarray:
        """The margins at the config's `tol`."""
        return next(iter(self.margins.values()))

    def ambiguity(self, q: float = 0.25, tol: float | None = None) -> float:
        """A low quantile of the station margins at `tol` (default: the config's): high
        is unambiguous; 0 without hits, `inf` when no station has a different alternative."""
        m = (self.margin if tol is None else self.margins[tol])[self.present]
        return float(np.quantile(m, q, method="inverted_cdf")) if len(m) else 0.0


def wall_signs(polarity: str) -> tuple[float, float]:
    """Signs that make each wall's expected derivative positive: a dark bead falls into
    its left wall and rises out of its right one."""
    if polarity == "dark":
        return -1.0, 1.0
    if polarity == "light":
        return 1.0, -1.0
    raise ValueError(f"polarity must be 'dark' or 'light', not {polarity!r}")


def sat(g: np.ndarray, g0: float) -> np.ndarray:
    x = np.asarray(g, dtype=np.float64) / g0
    return x / (1.0 + np.abs(x))


def huber(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    return np.where(a <= 1.0, 0.5 * x * x, a - 0.5)


# ---------------------------------------------------------------------------
# Stations and profiles
# ---------------------------------------------------------------------------
def stations(prior: np.ndarray, spacing: float, window: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Points uniform in arc length, their unit normals and the spacing `h`."""
    pts = common.resample(np.asarray(prior, dtype=np.float64), spacing)
    s = common.arc_length(pts)
    length = s[-1]
    h = length / (len(pts) - 1)
    r = np.minimum(window, np.minimum(s, length - s))
    r = np.maximum(r, h)  # at the very ends, the chord to the neighbour
    a = common.point_at(pts, np.clip(s - r, 0.0, length))
    b = common.point_at(pts, np.clip(s + r, 0.0, length))
    t = b - a
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-12)[:, None]
    return pts, np.stack([-t[:, 1], t[:, 0]], axis=1), h


def sample(img: np.ndarray, prior: np.ndarray, cfg: CoupledConfig) -> Profiles:
    """One strip caliper per station, `±(reach + max_width/2 + 3σ + 1)` along its normal."""
    import vision_metrology as vm

    pts, nrm, h = stations(prior, cfg.spacing, cfg.tangent_window)
    half = cfg.reach + 0.5 * cfg.max_width + 3.0 * cfg.sigma + 1.0
    n_half = int(np.ceil(half / cfg.step))
    offsets = cfg.step * np.arange(-n_half, n_half + 1, dtype=np.float64)
    extent = offsets[-1]
    mc = vm.MeasureConfig(
        sigma=cfg.sigma,
        # The caliper's response is a difference per profile sample, so its threshold
        # and amplitudes scale with the step; here both are per px.
        threshold=cfg.threshold * cfg.step,
        polarity="any",
        select="all",
        step=cfg.step,
        max_obliquity_deg=180.0,
    )
    height, width = img.shape
    resp = np.zeros((len(pts), len(offsets)))
    edges = []
    caliper = None
    for i, (p, n) in enumerate(zip(pts, nrm)):
        a, b = p - extent * n, p + extent * n
        args = ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])), cfg.half_width, len(offsets))
        if caliper is None:
            caliper = vm.Caliper.strip(*args, config=mc)
        else:
            caliper.move_to_strip(*args)
        tr = caliper.explain(img)
        xy = p[None, :] + offsets[:, None] * n[None, :]
        inside = (xy[:, 0] >= -0.5) & (xy[:, 0] <= width - 0.5) & (xy[:, 1] >= -0.5) & (xy[:, 1] <= height - 0.5)
        resp[i] = np.where(inside, np.asarray(tr.response, dtype=np.float64) / cfg.step, 0.0)
        e = [
            (c.t - extent, c.amplitude / cfg.step, 1.0 if c.polarity == "rising" else -1.0)
            for c in tr.candidates
            if -0.5 <= c.x <= width - 0.5 and -0.5 <= c.y <= height - 0.5
        ]
        edges.append(np.asarray(e, dtype=np.float64).reshape(-1, 3))
    return Profiles(pts, nrm, h, offsets, resp, edges)


def snap(pr: Profiles, i: int, off: float, sign: float, radius: float) -> tuple[float, bool]:
    """The caliper's sub-pixel edge of polarity `sign` nearest `off`, within `radius`."""
    e = pr.edges[i]
    e = e[e[:, 2] == sign] if len(e) else e
    if not len(e):
        return off, False
    k = int(np.argmin(np.abs(e[:, 0] - off)))
    return (float(e[k, 0]), True) if abs(e[k, 0] - off) <= radius else (off, False)


def width_term(w: np.ndarray, w0: float | None, cfg: CoupledConfig) -> np.ndarray:
    if w0 is None or cfg.width_prior <= 0.0:
        return np.zeros_like(w)
    return cfg.width_prior * huber(np.log(w / w0) / cfg.width_prior_scale)


def absent_cost(cfg: CoupledConfig) -> float:
    return -2.0 * float(sat(cfg.absent_level, cfg.g0))


def bridge(center: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Centre offsets interpolated across absent stations, held at the ends; 0 (the
    prior) when nothing is present."""
    idx = np.nonzero(present)[0]
    if not len(idx):
        return np.zeros(len(present))
    return np.interp(np.arange(len(present)), idx, center[idx])


def margins(m_states: np.ndarray, l_states: np.ndarray, r_states: np.ndarray, l: float, r: float, tols) -> list[float]:
    """Per tolerance, the smallest max-marginal among states with a wall more than it
    from `(l, r)`; `inf` when there is none."""
    dev = np.maximum(np.abs(l_states - l), np.abs(r_states - r))
    return [float(np.min(m_states[dev > t])) if np.any(dev > t) else np.inf for t in tols]


# ---------------------------------------------------------------------------
# Dense dynamic programme over (centre, width)
# ---------------------------------------------------------------------------
def band_min(f: np.ndarray, axis: int, k: int, coef: float) -> np.ndarray:
    """`g[x] = min_{|d| ≤ k} f[x + d] + coef·d²` along `axis`."""
    out = f.copy()
    f = np.moveaxis(f, axis, 0)
    o = np.moveaxis(out, axis, 0)
    for d in range(1, k + 1):
        pen = coef * d * d
        np.minimum(o[d:], f[:-d] + pen, out=o[d:])
        np.minimum(o[:-d], f[d:] + pen, out=o[:-d])
    return out


def dense_grid(pr: Profiles, cfg: CoupledConfig):
    """Centre and half-width indices into the profile, and their offsets in px."""
    step = cfg.step
    t0 = (len(pr.offsets) - 1) // 2
    kc = int(np.floor(cfg.reach / step + 1e-9))
    ci = t0 + np.arange(-kc, kc + 1)
    hw = np.arange(int(np.ceil(cfg.min_width / (2 * step) - 1e-9)), int(np.floor(cfg.max_width / (2 * step) + 1e-9)) + 1)
    hw = hw[hw >= 1]
    li = ci[:, None] - hw[None, :]
    ri = ci[:, None] + hw[None, :]
    return ci, hw, li, ri


def dense_solve(pr: Profiles, cfg: CoupledConfig, w0: float | None) -> Extraction:
    step, h, n = cfg.step, pr.h, len(pr.points)
    ci, hw, li, ri = dense_grid(pr, cfg)
    sl, sr = wall_signs(cfg.polarity)
    el, er = sat(sl * pr.response, cfg.g0), sat(sr * pr.response, cfg.g0)
    c_px = pr.offsets[ci][:, None] + 0.0 * hw[None, :]
    w_px = 2.0 * step * hw[None, :] + 0.0 * c_px
    l_px, r_px = c_px - 0.5 * w_px, c_px + 0.5 * w_px
    prior = width_term(w_px, w0, cfg) + cfg.offset_prior * (c_px / cfg.reach) ** 2
    u_p = h * (-el[:, li] - er[:, ri] + prior[None])  # (N, C, W)
    u_a = h * absent_cost(cfg)
    s = cfg.switch_cost
    kc = int(np.floor(cfg.max_wall_slope * h / step + 1e-9))
    kw = int(np.floor(cfg.max_width_slope * h / (2 * step) + 1e-9))
    ac = 2.0 * cfg.wall_stiffness * step**2 / h
    aw = (0.5 * cfg.wall_stiffness + cfg.width_stiffness) * (2 * step) ** 2 / h

    def dt(f: np.ndarray) -> np.ndarray:
        return band_min(band_min(f, 1, kw, aw), 0, kc, ac)

    shape = u_p.shape[1:]
    a_p = np.empty_like(u_p)
    a_a = np.empty_like(u_p)
    a_p[0], a_a[0] = u_p[0], np.full(shape, u_a)
    for i in range(1, n):
        a_p[i] = dt(np.minimum(a_p[i - 1], a_a[i - 1] + s)) + u_p[i]
        a_a[i] = dt(np.minimum(a_a[i - 1], a_p[i - 1] + s)) + u_a
    b_p = np.zeros_like(u_p)
    b_a = np.zeros_like(u_p)
    for i in range(n - 2, -1, -1):
        dp, da = dt(b_p[i + 1] + u_p[i + 1]), dt(b_a[i + 1] + u_a)
        b_p[i], b_a[i] = np.minimum(dp, da + s), np.minimum(da, dp + s)
    m_p, m_a = a_p + b_p, a_a + b_a
    best = float(min(m_p[-1].min(), m_a[-1].min()))

    # Backtrack: the cheapest predecessor of each chosen state, in its band.
    layer = np.zeros(n, dtype=int)  # 1 present, 0 absent
    cw = np.zeros((n, 2), dtype=int)
    last = int(np.argmin([m_a[-1].min(), m_p[-1].min()]))
    src = (a_a, a_p)[last][-1]
    layer[-1], cw[-1] = last, np.unravel_index(int(np.argmin(src)), shape)
    for i in range(n - 2, -1, -1):
        c1, w1 = cw[i + 1]
        cs = slice(max(0, c1 - kc), min(shape[0], c1 + kc + 1))
        ws = slice(max(0, w1 - kw), min(shape[1], w1 + kw + 1))
        dc = np.arange(cs.start, cs.stop)[:, None] - c1
        dw = np.arange(ws.start, ws.stop)[None, :] - w1
        trans = ac * dc**2 + aw * dw**2
        cand = [a_a[i][cs, ws] + trans + s * (layer[i + 1] != 0), a_p[i][cs, ws] + trans + s * (layer[i + 1] != 1)]
        mins = [c.min() for c in cand]
        layer[i] = int(np.argmin(mins))
        k = np.unravel_index(int(np.argmin(cand[layer[i]])), cand[layer[i]].shape)
        cw[i] = (cs.start + k[0], ws.start + k[1])

    present = layer == 1
    left = np.full(n, np.nan)
    right = np.full(n, np.nan)
    tols = (cfg.tol, *cfg.extra_tols)
    marg = np.full((len(tols), n), np.nan)
    pres = np.empty(n)
    snapped = np.zeros(n, dtype=bool)
    radius = 0.5 + 2.0 * step
    for i in range(n):
        c, w = cw[i]
        if present[i]:
            marg[:, i] = np.asarray(margins(m_p[i], l_px, r_px, l_px[c, w], r_px[c, w], tols)) - best
            pres[i] = m_a[i].min() - best
            left[i], ok_l = snap(pr, i, l_px[c, w], sl, radius)
            right[i], ok_r = snap(pr, i, r_px[c, w], sr, radius)
            snapped[i] = ok_l and ok_r
        else:
            pres[i] = m_p[i].min() - best
    center = np.where(present, 0.5 * (left + right), c_px[cw[:, 0], 0])
    if present.any():
        center = np.where(present, center, bridge(np.where(present, center, 0.0), present))
    return Extraction(present, left, right, center, dict(zip(tols, marg)), pres, best, snapped)


def dense(pr: Profiles, cfg: CoupledConfig) -> Extraction:
    """Two solves: the first without the width term sets the call's median width."""
    first = dense_solve(pr, cfg, None)
    if not first.present.any() or cfg.width_prior <= 0.0:
        return first
    return dense_solve(pr, cfg, float(np.median((first.right - first.left)[first.present])))


# ---------------------------------------------------------------------------
# Sparse Viterbi over candidate pairs
# ---------------------------------------------------------------------------
def pairs(pr: Profiles, cfg: CoupledConfig) -> list[np.ndarray]:
    """Per station, (k, 3) candidate pairs: left offset, right offset, data cost per px."""
    sl, sr = wall_signs(cfg.polarity)
    out = []
    for e in pr.edges:
        lead, trail = e[e[:, 2] == sl], e[e[:, 2] == sr]
        if not len(lead) or not len(trail):
            out.append(np.zeros((0, 3)))
            continue
        l, r = lead[:, None, 0], trail[None, :, 0]
        w = r - l
        c = 0.5 * (l + r)
        ok = (w >= cfg.min_width) & (w <= cfg.max_width) & (np.abs(c) <= cfg.reach)
        il, ir = np.nonzero(ok)
        cost = -sat(lead[il, 1], cfg.g0) - sat(trail[ir, 1], cfg.g0) + cfg.offset_prior * (c[il, ir] / cfg.reach) ** 2
        out.append(np.stack([lead[il, 0], trail[ir, 0], cost], axis=1))
    return out


def sparse_solve(pr: Profiles, cfg: CoupledConfig, w0: float | None) -> Extraction:
    h, n = pr.h, len(pr.points)
    cand = pairs(pr, cfg)
    u = [h * (c[:, 2] + width_term(c[:, 1] - c[:, 0], w0, cfg)) for c in cand]
    ab, s, rq = h * absent_cost(cfg), cfg.switch_cost, cfg.reacquire_cost
    kmax = max(1, int(np.floor(cfg.max_bridge / h)) + 1)

    def trans(i: int, j: int) -> np.ndarray:
        """Cost from every pair at `i` to every pair at `j > i`, across the gap."""
        d = (j - i) * h
        dl = cand[j][None, :, 0] - cand[i][:, None, 0]
        dr = cand[j][None, :, 1] - cand[i][:, None, 1]
        dw = dr - dl
        t = (cfg.wall_stiffness * (dl**2 + dr**2) + cfg.width_stiffness * dw**2) / d
        t += (j - i - 1) * ab + (2 * s if j - i > 1 else 0.0)
        bad = (np.abs(dl) > cfg.max_wall_slope * d) | (np.abs(dr) > cfg.max_wall_slope * d)
        bad |= np.abs(dw) > cfg.max_width_slope * d
        return np.where(bad, np.inf, t)

    tr = {(i, j): trans(i, j) for j in range(n) for i in range(max(0, j - kmax), j) if len(cand[i]) and len(cand[j])}
    # Forward, with back-pointers: ('L'|'M'|'P', station, index).
    a_l = np.cumsum(np.full(n, ab))
    a_m = np.full(n, np.inf)
    a_p = [np.full(len(c), np.inf) for c in cand]
    bp_p = [[None] * len(c) for c in cand]
    bp_m = [None] * n
    for j in range(n):
        if j > 0:
            mp = min((a_p[j - 1].min() if len(a_p[j - 1]) else np.inf), np.inf)
            if a_m[j - 1] <= mp + s:
                a_m[j], bp_m[j] = a_m[j - 1] + ab, ("M", j - 1, 0)
            else:
                a_m[j], bp_m[j] = mp + s + ab, ("P", j - 1, int(np.argmin(a_p[j - 1])))
        if not len(cand[j]):
            continue
        best = np.full(len(cand[j]), np.inf if j else 0.0)
        src = [("S", 0, 0)] * len(cand[j])
        if j > 0:
            opts = [(a_l[j - 1] + s, ("L", j - 1, 0)), (a_m[j - 1] + s + rq, ("M", j - 1, 0))]
            for v, tag in opts:
                better = v < best
                best = np.where(better, v, best)
                src = [tag if b else x for b, x in zip(better, src)]
            for i in range(max(0, j - kmax), j):
                if (i, j) not in tr:
                    continue
                tot = a_p[i][:, None] + tr[(i, j)]
                k = np.argmin(tot, axis=0)
                v = tot[k, np.arange(len(k))]
                better = v < best
                best = np.where(better, v, best)
                src = [("P", i, int(kk)) if b else x for b, x, kk in zip(better, src, k)]
        a_p[j] = best + u[j]
        bp_p[j] = src
    # Backward.
    b_l = np.zeros(n)
    b_m = np.zeros(n)
    b_p = [np.zeros(len(c)) for c in cand]
    for i in range(n - 2, -1, -1):
        nxt = u[i + 1] + b_p[i + 1] if len(cand[i + 1]) else np.zeros(0)
        enter = nxt.min() if len(nxt) else np.inf
        b_l[i] = min(b_l[i + 1] + ab, s + enter)
        b_m[i] = min(b_m[i + 1] + ab, s + rq + enter)
        if not len(cand[i]):
            continue
        v = np.full(len(cand[i]), s + ab + b_m[i + 1])
        for j in range(i + 1, min(n, i + kmax + 1)):
            if (i, j) in tr:
                v = np.minimum(v, (tr[(i, j)] + (u[j] + b_p[j])[None, :]).min(axis=1))
        b_p[i] = v
    ends = [(a_l[-1], ("L", n - 1, 0)), (a_m[-1], ("M", n - 1, 0))]
    if len(cand[-1]):
        ends.append((a_p[-1].min(), ("P", n - 1, int(np.argmin(a_p[-1])))))
    best_cost, node = min(ends, key=lambda x: x[0])

    present = np.zeros(n, dtype=bool)
    choice = np.full(n, -1)
    while node is not None and node[0] in ("P", "M"):
        kind, j, k = node
        if kind == "P":
            present[j], choice[j] = True, k
            node = bp_p[j][k]
        else:
            node = bp_m[j]
    left = np.full(n, np.nan)
    right = np.full(n, np.nan)
    tols = (cfg.tol, *cfg.extra_tols)
    marg = np.full((len(tols), n), np.nan)
    pres = np.full(n, np.nan)
    for i in np.nonzero(present)[0]:
        c = cand[i]
        left[i], right[i] = c[choice[i], 0], c[choice[i], 1]
        marg[:, i] = np.asarray(margins(a_p[i] + b_p[i], c[:, 0], c[:, 1], left[i], right[i], tols)) - best_cost
    center = bridge(0.5 * (left + right), present)
    return Extraction(present, left, right, center, dict(zip(tols, marg)), pres, float(best_cost), present.copy())


def sparse(pr: Profiles, cfg: CoupledConfig) -> Extraction:
    first = sparse_solve(pr, cfg, None)
    if not first.present.any() or cfg.width_prior <= 0.0:
        return first
    return sparse_solve(pr, cfg, float(np.median((first.right - first.left)[first.present])))


def centerline(pr: Profiles, ex: Extraction) -> np.ndarray:
    """The extracted centreline: each station moved along its normal."""
    return pr.points + ex.center[:, None] * pr.normals


def extract(img: np.ndarray, prior: np.ndarray, cfg: CoupledConfig, method: str) -> tuple[Profiles, Extraction, Extraction]:
    """`cfg.passes` extractions, each on the stations of the previous centreline: the
    last one's profiles and result, and the first one's result, whose margins are the
    ambiguity within the prior's window (a later pass re-centres the window on the line
    it chose, so a competitor can fall outside it)."""
    solve = sparse if method == "sparse" else dense
    curve = np.asarray(prior, dtype=np.float64)
    first = None
    for _ in range(max(1, cfg.passes)):
        pr = sample(img, curve, cfg)
        ex = solve(pr, cfg)
        first = first or ex
        curve = centerline(pr, ex)
    return pr, ex, first

