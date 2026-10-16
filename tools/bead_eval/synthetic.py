"""Synthetic beads with known walls, for the coupled-extraction evaluation.

The model follows the library's ribbon fixture: a box across the normal of a centreline,
blurred by a Gaussian of `PSF_SIGMA` px across it and sampled at pixel centres. Here the
two walls are separate functions of arc length, `−wl(s)` and `+wr(s)` along `+n`, so a
bead can be asymmetric and vary in width. A pixel whose closest centreline location is at
arc length `s` and signed distance `d` along `+n` gets

    bg + c · [Φ((d + wl(s)) / σ) − Φ((d − wr(s)) / σ)]

inside the bead, and `bg` in a gap or past either end. The truth is exact by construction:
the walls at `d = −wl(s)` and `d = +wr(s)`.

Conditions, each the hard case named in the roadmap:

    clean       symmetric, 8 px, noise 2 DN
    asymmetric  wl from 1.5 to 3.5 px and wr from 3.5 to 6.5 px, at different periods
    distractor  a 3 px stripe of the bead's polarity 2 px outside its right wall
    parallel    an identical second bead 14 px away on +n, along its whole length
    gap         no bead over 30 px in the middle
    occlusion   a disc of radius 9 px, 30 DN above the background, over the bead
    highlight   a 1.5 px stripe 40 DN brighter inside the bead, off its centre

A light bead of contrast 60 DN on 80 DN, on a line, an arc (radius 250 px) and a sine
(amplitude 15 px, period 200 px), each in a 360×240 image.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

import common

SIZE = (360, 240)  # (width, height)
SHAPES = ("line", "arc", "sine")
CONDITIONS = ("clean", "asymmetric", "distractor", "parallel", "gap", "occlusion", "highlight")
BACKGROUND = 80.0
CONTRAST = 60.0
PSF_SIGMA = 1.0
NOISE = 2.0
HALF_WIDTH = 4.0
PARALLEL_PX = 14.0
PARALLEL_CONTRAST = 60.0
DISTRACTOR_GAP = 2.0
DISTRACTOR_WIDTH = 3.0
DISTRACTOR_CONTRAST = 40.0
GAP_PX = 30.0
OCCLUSION_RADIUS = 9.0
OCCLUSION_LEVEL = 30.0
#: Prior offsets along +n of the truth, px; +n points to the parallel bead.
PRIOR_OFFSETS = (0.0, 3.0, -3.0, 6.0, -6.0, 7.0)
SINE_PRIOR = 3.0


def centreline(shape: str) -> np.ndarray:
    w, h = SIZE
    if shape == "line":
        a = np.radians(10.0)
        t = np.linspace(-140.0, 140.0, 1121)
        return np.stack([0.5 * w + t * np.cos(a), 0.5 * h + t * np.sin(a)], axis=1)
    if shape == "arc":
        t = np.radians(np.linspace(237.0, 303.0, 2001))
        return np.stack([0.5 * w + 250.0 * np.cos(t), 330.0 + 250.0 * np.sin(t)], axis=1)
    if shape == "sine":
        x = np.linspace(40.0, 320.0, 2241)
        return np.stack([x, 0.5 * h + 15.0 * np.sin(2.0 * np.pi * (x - 40.0) / 200.0)], axis=1)
    raise ValueError(f"unknown shape {shape!r}")


def walls(condition: str, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`wl(s)` and `wr(s)`, px."""
    if condition == "asymmetric":
        return 2.5 + np.sin(2.0 * np.pi * s / 150.0), 5.0 + 1.5 * np.sin(2.0 * np.pi * s / 90.0 + 1.0)
    return np.full_like(s, HALF_WIDTH), np.full_like(s, HALF_WIDTH)


def scenes(seeds: int = 2) -> list[dict]:
    return [{"shape": sh, "condition": c, "seed": k} for sh in SHAPES for c in CONDITIONS for k in range(seeds)]


def scene_id(scene: dict) -> str:
    return f"{scene['shape']}_{scene['condition']}_{scene['seed']}"


class Truth:
    """A scene's centreline (1 px resampled), walls and presence, and signed offsets."""

    def __init__(self, scene: dict):
        self.curve = common.resample(centreline(scene["shape"]), 0.25)
        self.s = common.arc_length(self.curve)
        self.length = float(self.s[-1])
        self.normals = common.normals(self.curve)
        self.condition = scene["condition"]
        self.wl, self.wr = walls(self.condition, self.s)
        self.gap = (0.5 * (self.length - GAP_PX), 0.5 * (self.length + GAP_PX)) if self.condition == "gap" else None
        self.occluder = common.point_at(self.curve, np.array([0.62 * self.length]))[0] if self.condition == "occlusion" else None
        self.index = common.PolylineIndex([self.curve], step=0.25)

    def locate(self, pts: np.ndarray, upper: float = np.inf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arc length, signed offset along +n, and whether the location is interior."""
        d, _, s, at_end = self.index.query(pts, upper=upper)
        ok = np.isfinite(d)
        sc = np.where(ok, s, 0.0)
        foot = common.point_at(self.curve, sc)
        n = np.stack([np.interp(sc, self.s, self.normals[:, 0]), np.interp(sc, self.s, self.normals[:, 1])], axis=1)
        sign = np.sign(np.einsum("ij,ij->i", pts - foot, n))
        return sc, np.where(ok, np.where(sign == 0, 1.0, sign) * d, np.inf), ok & ~at_end

    def wall_at(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.interp(s, self.s, self.wl), np.interp(s, self.s, self.wr)

    def visible(self, s: np.ndarray) -> np.ndarray:
        """Where the bead is there and unoccluded: the stations its walls can be measured at."""
        ok = (s > 1.0) & (s < self.length - 1.0)
        if self.gap is not None:
            ok &= (s < self.gap[0] - 2.0) | (s > self.gap[1] + 2.0)
        if self.occluder is not None:
            p = common.point_at(self.curve, s)
            ok &= np.hypot(*(p - self.occluder).T) > OCCLUSION_RADIUS + 3.0
        return ok


def render(scene: dict) -> tuple[np.ndarray, Truth]:
    truth = Truth(scene)
    width_px, height_px = SIZE
    yy, xx = np.mgrid[0:height_px, 0:width_px]
    pix = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    s, d, inside = truth.locate(pix, upper=PARALLEL_PX + 12.0)
    wl, wr = truth.wall_at(s)
    present = inside & np.isfinite(d)
    if truth.gap is not None:
        present &= (s < truth.gap[0]) | (s > truth.gap[1])
    dd = np.where(present, d, 1e3)
    img = np.full(len(pix), BACKGROUND)
    img += CONTRAST * (ndtr((dd + wl) / PSF_SIGMA) - ndtr((dd - wr) / PSF_SIGMA))
    cond = truth.condition
    if cond == "highlight":
        img += 40.0 * (ndtr((dd - 0.5) / PSF_SIGMA) - ndtr((dd - 2.0) / PSF_SIGMA))
    if cond == "distractor":
        a = wr + DISTRACTOR_GAP
        img += DISTRACTOR_CONTRAST * (ndtr((dd - a) / PSF_SIGMA) - ndtr((dd - a - DISTRACTOR_WIDTH) / PSF_SIGMA))
    if cond == "parallel":
        dp = np.where(inside & np.isfinite(d), d - PARALLEL_PX, 1e3)
        img += PARALLEL_CONTRAST * (ndtr((dp + HALF_WIDTH) / PSF_SIGMA) - ndtr((dp - HALF_WIDTH) / PSF_SIGMA))
    if truth.occluder is not None:
        r = np.hypot(*(pix - truth.occluder).T)
        alpha = ndtr((OCCLUSION_RADIUS - r) / PSF_SIGMA)
        img = img * (1.0 - alpha) + (BACKGROUND + OCCLUSION_LEVEL) * alpha
    img = img.reshape(height_px, width_px)
    img += common.rng_for("coupled-synthetic", scene_id(scene)).normal(0.0, NOISE, img.shape)
    return np.clip(img, 0.0, 255.0).astype(np.float32), truth


def priors(truth: Truth) -> list[tuple[str, np.ndarray]]:
    """Named priors: the truth moved along its normals, and a sine of `SINE_PRIOR` px."""
    c = common.resample(truth.curve, 1.0)
    n = common.normals(c)
    out = [(f"offset {o:+g}", c + o * n) for o in PRIOR_OFFSETS]
    s = common.arc_length(c)
    out.append((f"sine {SINE_PRIOR:g}", c + (SINE_PRIOR * np.sin(2.0 * np.pi * s / 100.0))[:, None] * n))
    return out
