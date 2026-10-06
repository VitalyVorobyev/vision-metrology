# Run: python examples/python/bead_track.py
"""Track a bead along a perturbed prior, and see where and why stations were rejected.

A light S-shaped bead 40 px wide on a sloping background, with a gap, a brighter step
running close beside one edge at its start, and a highlight stripe inside. The prior is
the true centreline moved 5 px sideways with a 5 px bump in it. The tracker runs with
`clearance` on, so the stations beside the step are rejected rather than measured
against it."""

import numpy as np
import vision_metrology as vm

W, H = 640, 400
WIDTH = 40.0
GAP = (300.0, 335.0)  # px of arc length without bead


def s_bend(start, heading, radius, sweep):
    """Two tangent arcs turning opposite ways: (centre, radius, start angle, sweep) each.
    A positive sweep turns clockwise on screen."""
    turn = np.sign(sweep)
    a0 = heading - turn * np.pi / 2
    c1 = np.array(start) - radius * np.array([np.cos(a0), np.sin(a0)])
    a1 = a0 + sweep
    c2 = c1 + 2 * radius * np.array([np.cos(a1), np.sin(a1)])
    return [(c1, radius, a0, sweep), (c2, radius, a1 + turn * np.pi, -sweep)]


ARCS = s_bend((80.0, 300.0), 0.0, 260.0, -0.9)
LENGTH = sum(r * abs(sw) for _, r, _, sw in ARCS)


def frame(s):
    """The centreline point and unit normal (-t_y, t_x) at arc length s."""
    for c, r, a0, sw in ARCS:
        length = r * abs(sw)
        if s <= length:
            a = a0 + np.sign(sw) * s / r
            p = c + r * np.array([np.cos(a), np.sin(a)])
            t = np.sign(sw) * np.array([-np.sin(a), np.cos(a)])
            return p, np.array([-t[1], t[0]])
        s -= length
    raise ValueError("past the end")


def nearest(x, y):
    """Arc length and signed distance along the normal of points (x, y) from the
    centreline; s is NaN where a point projects past either end."""
    best_s = np.full(np.shape(x), np.nan)
    best_d = np.full(np.shape(x), np.inf)
    offset = 0.0
    for c, r, a0, sw in ARCS:
        rho = np.hypot(x - c[0], y - c[1])
        u = np.mod(np.sign(sw) * (np.arctan2(y - c[1], x - c[0]) - a0), 2 * np.pi)
        d = np.sign(sw) * (r - rho)
        take = (u <= abs(sw)) & (np.abs(d) < np.abs(best_d))
        best_s = np.where(take, offset + r * u, best_s)
        best_d = np.where(take, d, best_d)
        offset += r * abs(sw)
    return best_s, best_d


def render(seed=1):
    """The scene, averaged over 4 x 4 points per pixel, with 2 DN of noise, as uint8."""
    img = np.zeros((H, W))
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    # The step: 4 px outside the bead's +n edge 10 px along it, leaving it at 8 degrees.
    p0, n0 = frame(10.0)
    t0 = np.array([n0[1], -n0[0]])
    tilt = np.radians(8.0)
    step_n = -np.sin(tilt) * t0 + np.cos(tilt) * n0
    step_at = p0 + n0 * (0.5 * WIDTH + 4.0)
    for oy in (-0.375, -0.125, 0.125, 0.375):
        for ox in (-0.375, -0.125, 0.125, 0.375):
            x, y = xs + ox, ys + oy
            s, d = nearest(x, y)
            on = ~np.isnan(s) & ~((s >= GAP[0]) & (s <= GAP[1]))
            bead = on & (np.abs(d) <= 0.5 * WIDTH)
            stripe = on & (np.abs(d + 7.0) <= 3.0)
            step = (x - step_at[0]) * step_n[0] + (y - step_at[1]) * step_n[1] > 0
            img += 30.0 + 0.06 * x + 0.04 * y + 130.0 * bead + 45.0 * stripe + 60.0 * step
    img /= 16.0
    img += np.random.default_rng(seed).normal(0.0, 2.0, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def prior():
    """The truth from 10 px in from each end, a vertex every 8 px, moved 5 px along the
    normal, with a 5 px bump (sigma 15 px of arc) at 75% of its length."""
    ss = np.linspace(10.0, LENGTH - 10.0, int(np.ceil((LENGTH - 20.0) / 8.0)) + 1)
    pts = []
    for s in ss:
        p, n = frame(s)
        bump = 5.0 * np.exp(-((s - 0.75 * LENGTH) ** 2) / (2 * 15.0**2))
        pts.append(p + n * (5.0 + bump))
    return np.array(pts, dtype=np.float32)


def main() -> None:
    image = render()
    tracker = vm.BeadTracker(vm.BeadConfig(clearance=8.0))
    bead = tracker.track(image, prior())

    print(
        f"{len(bead.centerline)} stations {bead.spacing:.2f} px apart; "
        f"stopped {bead.stop} after {len(bead.passes)} passes"
    )
    for k, p in enumerate(bead.passes, 1):
        print(
            f"  pass {k}: {p.n_valid} stations found the bead, "
            f"correction up to {p.solve.correction_max:.3f} px"
        )
    print(
        f"support {bead.support:.3f}, longest gap {bead.longest_gap:.1f} px; "
        f"centre rms {bead.center_rms:.4f} px, max {bead.center_max_dev:.4f} px; "
        f"width {bead.width_mean:.3f} +- {bead.width_std:.3f} px"
    )
    print(f"rejected: {bead.rejects}")

    # Against the truth, away from the gap's square ends.
    s, d = nearest(bead.centerline[:, 0].astype(float), bead.centerline[:, 1].astype(float))
    hit = ~np.isnan(bead.width)
    away = (np.abs(s - GAP[0]) > 10.0) & (np.abs(s - GAP[1]) > 10.0)
    print(
        f"against the truth: refined curve within {np.abs(d[hit & away]).max():.3f} px, "
        f"widths within {np.abs(bead.width[hit & away] - WIDTH).max():.3f} px"
    )
    for i in np.flatnonzero(~hit)[:3]:
        print(f"  station {i} at s = {s[i]:.0f} px: {bead.reject[i]}")

    # The same run with every station's evidence: why the first rejected station failed.
    trace = tracker.explain(image, prior())
    i = int(np.flatnonzero(~hit)[0])
    st = trace.measure[i]
    print(
        f"explain, station {i}: {st.reject}, window {st.window}, "
        f"{len(st.caliper.edges)} edges on its strip"
    )


if __name__ == "__main__":
    main()
