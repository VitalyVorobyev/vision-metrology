"""Synthetic ribbons with a known centreline, for the acquisition evaluation.

The model is the one of the library's own test fixture: a ribbon of width `w` along a
centreline, with intensity

    I(p) = bg + c * g(s) * [Phi((w/2 - d) / sigma) - Phi((-w/2 - d) / sigma)]

for a point `p` whose closest centreline location is at arc length `s` and distance `d`:
a box across the ribbon blurred by a 1-D Gaussian of `sigma` px. `g(s)` is 0 in a gap and
1 elsewhere, and a point whose closest location is an end of the centreline is outside the
ribbon, so the ends are square. Pixel centres are sampled at integer coordinates (no area
integration), then seeded Gaussian noise is added and the image is rounded to `uint8`.

The blur is across the ribbon only, so the truth is exact by construction: the centre at
`d = 0` and the edges at `d = +-w/2`. The curves keep their radius of curvature well
above `w/2 + 6 sigma`, so every point has one closest location.

Scenes, 1280x1024: a line, an arc and a sine, each `w` in `WIDTHS` wide, under each of
`CONDITIONS`. A light ribbon of contrast 60 DN on a background of 80 DN.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

import common

SIZE = (1280, 1024)  # (width, height)
SHAPES = ("line", "arc", "sine")
WIDTHS = (3.0, 8.0, 30.0)
#: condition -> (noise sigma in DN, distractor step, gap)
CONDITIONS = {
    "clean": (2.0, False, False),
    "noisy": (8.0, False, False),
    "distractor": (2.0, True, False),
    "gap": (2.0, False, True),
}
BACKGROUND = 80.0
CONTRAST = 60.0
PSF_SIGMA = 1.0
STEP_DN = 40.0
#: The distractor step's boundary runs parallel to the ribbon's chord, this far beyond
#: the ribbon's outer edge.
STEP_CLEARANCE_PX = 10.0
GAP_PX = 30.0


def centreline(shape: str) -> np.ndarray:
    """The centreline of a scene, densely sampled, as (N, 2) `(x, y)`."""
    w, h = SIZE
    if shape == "line":
        a = np.radians(17.0)
        t = np.linspace(-400.0, 400.0, 3201)
        return np.stack([0.5 * w + t * np.cos(a), 0.5 * h + t * np.sin(a)], axis=1)
    if shape == "arc":
        # Radius 300 px over 140 degrees, bulging upwards.
        t = np.radians(np.linspace(200.0, 340.0, 4001))
        return np.stack([0.5 * w + 300.0 * np.cos(t), 850.0 + 300.0 * np.sin(t)], axis=1)
    if shape == "sine":
        # Amplitude 40 px, period 400 px: the tightest radius is about 100 px.
        x = np.linspace(240.0, 1040.0, 6401)
        return np.stack([x, 0.5 * h + 40.0 * np.sin(2.0 * np.pi * (x - 240.0) / 400.0)], axis=1)
    raise ValueError(f"unknown shape {shape!r}")


def scenes(seeds: int = 2) -> list[dict]:
    """Every scene of the grid, `seeds` noise draws each."""
    return [
        {"shape": sh, "width": w, "condition": cond, "seed": k}
        for sh in SHAPES
        for w in WIDTHS
        for cond in CONDITIONS
        for k in range(seeds)
    ]


def scene_id(scene: dict) -> str:
    return f"{scene['shape']}_w{scene['width']:g}_{scene['condition']}_{scene['seed']}"


def _step_line(curve: np.ndarray, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    """A point on the distractor step's boundary and the boundary's unit normal, which
    points away from the ribbon into the brighter half-plane."""
    chord = curve[-1] - curve[0]
    chord /= np.hypot(*chord)
    n = np.array([-chord[1], chord[0]])
    lateral = (curve - curve[0]) @ n
    off = lateral.max() + half_width + STEP_CLEARANCE_PX
    return curve[0] + off * n, n


def render(scene: dict) -> tuple[np.ndarray, dict]:
    """The scene's `uint8` image and its truth: the centreline resampled at 1 px, the
    width, and the gap as an arc-length interval or `None`."""
    noise, distractor, gap = CONDITIONS[scene["condition"]]
    w = float(scene["width"])
    curve = centreline(scene["shape"])
    length = float(common.arc_length(curve)[-1])
    gap_iv = (0.5 * (length - GAP_PX), 0.5 * (length + GAP_PX)) if gap else None

    width_px, height_px = SIZE
    yy, xx = np.mgrid[0:height_px, 0:width_px]
    pix = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    reach = 0.5 * w + 6.0 * PSF_SIGMA + 2.0
    d, _, s, at_end = common.PolylineIndex([curve], step=0.25).query(pix, upper=reach)
    inside = np.isfinite(d) & ~at_end
    if gap_iv is not None:
        inside &= (s < gap_iv[0]) | (s > gap_iv[1])
    img = np.full(len(pix), BACKGROUND)
    di = d[inside]
    img[inside] += CONTRAST * (ndtr((0.5 * w - di) / PSF_SIGMA) - ndtr((-0.5 * w - di) / PSF_SIGMA))
    if distractor:
        p0, n = _step_line(curve, 0.5 * w)
        img += STEP_DN * ndtr(((pix - p0) @ n) / PSF_SIGMA)
    img = img.reshape(height_px, width_px)
    img += common.rng_for("ribbon", scene_id(scene)).normal(0.0, noise, img.shape)
    out = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    truth = {
        "points": common.resample(curve, 1.0),
        "width": w,
        "gap": gap_iv,
        "length": length,
    }
    return out, truth
