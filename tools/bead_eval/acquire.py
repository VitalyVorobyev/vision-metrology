"""Acquisition without a prior: candidate centrelines found from the image alone.

Each source returns open, non-branching paths as `(x, y)` polylines resampled at 1 px,
with a width per point, for each of its threshold settings:

- **A ridge filter, thresholded and skeletonised** (`sato` or `meijering` from
  scikit-image). The filter runs at the scales `scale_set` derives from the expected
  widths. Its response is thresholded with hysteresis, relative to the response's own
  spread: the high threshold is the larger of Otsu's and `median + k * s`, where `s` is
  the median absolute deviation scaled to a standard deviation, and the low threshold is
  halfway from the median to the high one. Each `k` of `--k` is one setting. The mask then
  goes through `paths.py`'s own steps: skeleton, spurs pruned, split at junctions, trimmed
  back from them, smoothed and resampled. The width is that step's `2 * EDT - 1` of the
  thresholded mask, so it says more about the threshold than about the line.
- **`ridge-detector`**, an optional pip package used as a black box: a multi-scale
  detector after Steger, with sub-pixel line points and a width from the edges on either
  side. It is not a dependency, and nothing of it is used beyond its public calls. Each
  `(low, high)` contrast pair of `--rd-contrast`, in grey levels, is one setting. A dark
  line is passed as a light line on the inverted image, where the contrasts mean what
  they say. It rounds its contrast thresholds down to a whole number of a unit that grows
  with the line width: about 19 grey levels at 8 px and 245 at 30 px. A wide line's
  thresholds therefore round to 0, and every noise ridge passes. It runs on the image
  mean-pooled by `k = ceil(max(widths) / 8)`, and its points and widths are scaled back.

Coordinates are `(x, y)`, pixel centres at integers, as in the rest of this folder.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from skimage.filters import apply_hysteresis_threshold, meijering, sato, threshold_otsu

import common
import paths as paths_mod

SOURCES = ("sato", "meijering", "ridge_detector")
#: The widest line `ridge-detector` sees after pooling.
RD_MAX_WIDTH_PX = 8.0


def add_source_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The settings the sources share, and each source's threshold settings."""
    parser.add_argument(
        "--min-length", type=float, default=20.0, help="shortest acquired path, px (default 20)"
    )
    parser.add_argument("--spur-px", type=float, default=12.0, help="prune shorter spurs (default 12)")
    parser.add_argument(
        "--junction-trim", type=float, default=6.0, help="px cut back from a junction (default 6)"
    )
    parser.add_argument(
        "--smooth-px", type=float, default=2.0, help="Gaussian sigma along a path (default 2)"
    )
    parser.add_argument(
        "--k",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 5.0, 8.0],
        help="filter thresholds, robust deviations above the median (default 2 3 5 8)",
    )
    parser.add_argument(
        "--default-k", type=float, default=3.0, help="the filters' setting for tracking (default 3)"
    )
    parser.add_argument(
        "--rd-contrast",
        nargs="+",
        default=["10,20", "20,40", "40,80"],
        help="ridge-detector's low,high contrasts, grey levels (default 10,20 20,40 40,80)",
    )
    parser.add_argument(
        "--default-rd", default="20,40", help="ridge-detector's setting for tracking (default 20,40)"
    )
    return parser


def settings(source: str, args: argparse.Namespace) -> list[str]:
    """The labels of a source's threshold settings, in sweep order."""
    if source == "ridge_detector":
        return [f"c{c.replace(',', '-')}" for c in args.rd_contrast]
    return [f"k{k:g}" for k in args.k]


def default_setting(source: str, args: argparse.Namespace) -> str:
    """The label of the setting whose paths are tracked; it must be one of the sweep's."""
    if source == "ridge_detector":
        label = f"c{args.default_rd.replace(',', '-')}"
    else:
        label = f"k{args.default_k:g}"
    if label not in settings(source, args):
        raise SystemExit(f"{source}: the default setting {label} is not in the sweep")
    return label


def available(source: str) -> bool:
    """Whether a source can run here: `ridge-detector` is an optional install."""
    if source != "ridge_detector":
        return True
    try:
        import ridge_detector  # noqa: F401
    except ImportError:
        return False
    return True


def scale_set(widths: list[float]) -> list[float]:
    """The filter scales for the expected line widths: `w / (2 sqrt 3)`, the smallest
    scale at which a bar of width `w` has a single second-derivative extremum at its
    centre (Steger 1998), and `w / 2`, where the scale-normalised response peaks. At
    least 1 px."""
    out = set()
    for w in widths:
        out.add(round(max(1.0, w / (2.0 * math.sqrt(3.0))), 3))
        out.add(round(max(1.0, 0.5 * w), 3))
    return sorted(out)


def _finish(points: np.ndarray, width: np.ndarray, min_length: float) -> dict | None:
    """A path resampled at 1 px, with its width interpolated along it, or `None` when it
    is shorter than `min_length`."""
    if len(points) < 2:
        return None
    s = common.arc_length(points)
    if s[-1] < min_length:
        return None
    p = common.resample(points, 1.0)
    return {"points": p, "width": np.interp(common.arc_length(p), s, width)}


def filter_paths(
    img: np.ndarray, method: str, widths: list[float], dark: bool, args: argparse.Namespace
) -> dict[str, tuple[list[dict], dict]]:
    """A scikit-image ridge filter, run once, then thresholded at each `k` and passed
    through `paths.py`'s skeleton graph."""
    fn = {"sato": sato, "meijering": meijering}[method]
    t = time.perf_counter()
    response = fn(img.astype(np.float32), sigmas=scale_set(widths), black_ridges=dark)
    filter_ms = 1e3 * (time.perf_counter() - t)
    med = float(np.median(response))
    spread = 1.4826 * float(np.median(np.abs(response - med)))
    otsu = float(threshold_otsu(response))
    graph = argparse.Namespace(
        min_length=args.min_length,
        spur_px=args.spur_px,
        junction_trim=args.junction_trim,
        smooth_px=args.smooth_px,
    )
    out = {}
    for k, label in zip(args.k, settings(method, args)):
        t = time.perf_counter()
        high = max(otsu, med + k * spread)
        mask = apply_hysteresis_threshold(response, med + 0.5 * (high - med), high)
        found, info = paths_mod.image_paths(mask, graph)
        found = [_finish(np.asarray(p["points"]), np.asarray(p["width"]), args.min_length) for p in found]
        info.update(
            {
                "threshold": high,
                "mask_pixels": int(mask.sum()),
                "filter_ms": filter_ms,
                "time_ms": filter_ms + 1e3 * (time.perf_counter() - t),
            }
        )
        out[label] = ([p for p in found if p is not None], info)
    return out


def _pool(img: np.ndarray, k: int) -> np.ndarray:
    """The mean of each `k x k` block, the ragged edge dropped, back to `uint8`."""
    if k == 1:
        return img
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    blocks = img[:h, :w].astype(np.float32).reshape(h // k, k, w // k, k)
    return np.clip(np.rint(blocks.mean(axis=(1, 3))), 0, 255).astype(np.uint8)


def ridge_detector_paths(
    img: np.ndarray, widths: list[float], dark: bool, args: argparse.Namespace
) -> dict[str, tuple[list[dict], dict]]:
    """`ridge-detector`'s lines at each contrast setting, on the image pooled so that no
    expected width exceeds `RD_MAX_WIDTH_PX`."""
    from ridge_detector import RidgeDetector

    k = max(1, math.ceil(max(widths) / RD_MAX_WIDTH_PX))
    pooled = _pool(255 - img if dark else img, k)
    line_widths = np.array(sorted({max(1.0, w / k) for w in widths}))
    out = {}
    for pair, label in zip(args.rd_contrast, settings("ridge_detector", args)):
        low, high = (float(v) for v in pair.split(","))
        t = time.perf_counter()
        det = RidgeDetector(
            line_widths=line_widths,
            low_contrast=low,
            high_contrast=high,
            min_len=max(2.0, args.min_length / k),
            dark_line=False,
            estimate_width=True,
            extend_line=False,
            correct_pos=False,
        )
        det.detect_lines(pooled)
        found = []
        for line in det.contours or []:
            # A pooled pixel i covers source pixels k*i .. k*i + k - 1, centred on k*i + (k-1)/2.
            xy = np.stack([np.asarray(line.col, float), np.asarray(line.row, float)], axis=1)
            w = np.asarray(line.width_l, float) + np.asarray(line.width_r, float)
            path = _finish(k * xy + 0.5 * (k - 1), k * w, args.min_length)
            if path is not None:
                found.append(path)
        dt = 1e3 * (time.perf_counter() - t)
        out[label] = (found, {"pool": k, "lines": len(det.contours or []), "time_ms": dt, "filter_ms": np.nan})
    return out


def acquire(
    img: np.ndarray, source: str, widths: list[float], dark: bool, args: argparse.Namespace
) -> dict[str, tuple[list[dict], dict]]:
    """One source's paths on one `uint8` image, per setting, with the bookkeeping: the wall
    time of the whole acquisition (`time_ms`) and of the filter alone (`filter_ms`)."""
    if source == "ridge_detector":
        return ridge_detector_paths(img, widths, dark, args)
    return filter_paths(img, source, widths, dark, args)
