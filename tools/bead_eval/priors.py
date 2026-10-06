"""Seeded, deterministic perturbations of the reference paths: the priors to track from.

Each reference path gets the unperturbed prior and a sweep of each kind:

    translate  the whole path moved by m px, perpendicular to its chord
    rotate     turned about its arc-length midpoint until its ends move by m px
    sine       displaced by m * sin(2 pi s / (L/2) + phase) along its smoothed normals
    bump       displaced by a Gaussian of height m px and sigma 10 px of arc, centred in
               its middle 40%, along its smoothed normals
    simplify   Douglas-Peucker, to the fewest vertices not above m (6 by default)
    truncate   a fraction m of its length cut off one end

The sign, phase, bump centre and truncated end are drawn from a generator seeded by the
path id, the kind and the magnitude, so a re-run gives the same priors. `priors.json`
stores these parameters, not the points: `build_prior` rebuilds a prior from its spec and
its reference path.

    python -I tools/bead_eval/priors.py --data-dir data/damsegment/extracted
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Isolated mode (-I) leaves the script's own directory off sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

import common  # noqa: E402

#: Magnitudes per kind: px for the displacements, vertices for `simplify`, a fraction of
#: the length for `truncate`.
SWEEP: dict[str, list[float]] = {
    "none": [0.0],
    "translate": [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    "rotate": [2.0, 4.0, 8.0, 12.0],
    "sine": [1.0, 2.0, 4.0, 6.0],
    "bump": [2.0, 4.0, 8.0, 12.0],
    "simplify": [10.0, 6.0],
    "truncate": [0.25],
}
UNITS = {
    "none": "",
    "translate": "px",
    "rotate": "px at the ends",
    "sine": "px amplitude",
    "bump": "px height",
    "simplify": "vertices",
    "truncate": "of the length",
}
BUMP_SIGMA_PX = 10.0
SMOOTH_NORMALS_PX = 10.0


def draw_spec(path_id: str, kind: str, magnitude: float) -> dict:
    """The random parameters of one perturbation."""
    rng = common.rng_for(path_id, kind, magnitude)
    spec: dict = {"kind": kind, "magnitude": magnitude}
    if kind in ("translate", "rotate", "bump"):
        spec["sign"] = int(rng.choice([-1, 1]))
    if kind == "sine":
        spec["phase"] = round(float(rng.uniform(0.0, 2.0 * np.pi)), 6)
    if kind == "bump":
        spec["center"] = round(float(rng.uniform(0.3, 0.7)), 6)
    if kind == "truncate":
        spec["end"] = int(rng.integers(0, 2))
    return spec


def smooth_normals(p: np.ndarray) -> np.ndarray:
    """Normals of the path smoothed over `SMOOTH_NORMALS_PX`, so that a displacement
    along them does not fold the curve at a tight wiggle."""
    q = ndimage.gaussian_filter1d(p, SMOOTH_NORMALS_PX, axis=0, mode="nearest")
    return common.normals(q)


def chord_normal(p: np.ndarray) -> np.ndarray:
    d = p[-1] - p[0]
    d = d / max(float(np.hypot(*d)), 1e-12)
    return np.array([-d[1], d[0]])


def douglas_peucker(p: np.ndarray, tol: float) -> np.ndarray:
    """Indices of the vertices Douglas-Peucker keeps at tolerance `tol`."""
    keep = np.zeros(len(p), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        ab = b - a
        n = float(np.hypot(*ab))
        seg = p[i + 1 : j] - a
        if n < 1e-12:
            d = np.hypot(seg[:, 0], seg[:, 1])
        else:
            d = np.abs(seg[:, 0] * ab[1] - seg[:, 1] * ab[0]) / n
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.extend([(i, m), (m, j)])
    return np.nonzero(keep)[0]


def simplify(p: np.ndarray, vertices: int) -> np.ndarray:
    """The Douglas-Peucker polygon with the fewest vertices not above `vertices`, found
    by bisection on the tolerance."""
    lo, hi = 0.0, float(np.max(np.hypot(*(p - p[0]).T))) + 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if len(douglas_peucker(p, mid)) <= vertices:
            hi = mid
        else:
            lo = mid
    return p[douglas_peucker(p, hi)]


def build_prior(ref: np.ndarray, spec: dict) -> np.ndarray:
    """The prior polyline for one spec, from the 1 px resampled reference `ref`."""
    kind, m = spec["kind"], float(spec["magnitude"])
    s = common.arc_length(ref)
    length = s[-1]
    if kind == "none":
        return ref.copy()
    if kind == "translate":
        return ref + spec["sign"] * m * chord_normal(ref)
    if kind == "rotate":
        mid = np.array([np.interp(0.5 * length, s, ref[:, 0]), np.interp(0.5 * length, s, ref[:, 1])])
        half = 0.5 * float(np.hypot(*(ref[-1] - ref[0])))
        theta = spec["sign"] * np.arcsin(min(1.0, m / max(half, 1e-9)))
        c, sn = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -sn], [sn, c]])
        return (ref - mid) @ rot.T + mid
    if kind == "sine":
        f = m * np.sin(2.0 * np.pi * s / (0.5 * length) + spec["phase"])
        return ref + f[:, None] * smooth_normals(ref)
    if kind == "bump":
        s0 = spec["center"] * length
        f = spec["sign"] * m * np.exp(-0.5 * ((s - s0) / BUMP_SIGMA_PX) ** 2)
        return ref + f[:, None] * smooth_normals(ref)
    if kind == "simplify":
        return simplify(ref, int(m))
    if kind == "truncate":
        keep = s <= (1.0 - m) * length if spec["end"] else s >= m * length
        return ref[keep]
    raise ValueError(f"unknown perturbation {kind!r}")


def main() -> None:
    parser = common.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    args = parser.parse_args()
    out_dir = common.resolve_out_dir(args)
    index = common.read_json(out_dir / "paths" / "index.json")
    priors = {}
    for entry in index["paths"]:
        pid = entry["id"]
        priors[pid] = [draw_spec(pid, k, m) for k, ms in SWEEP.items() for m in ms]
    common.write_json(
        out_dir / "priors.json",
        {
            "seed": common.SEED,
            "sweep": SWEEP,
            "units": UNITS,
            "bump_sigma_px": BUMP_SIGMA_PX,
            "smooth_normals_px": SMOOTH_NORMALS_PX,
            "priors": priors,
        },
        indent=None,
    )
    n = sum(len(v) for v in priors.values())
    print(f"{len(priors)} paths, {n} priors -> {out_dir / 'priors.json'}")


if __name__ == "__main__":
    main()
