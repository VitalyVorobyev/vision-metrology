"""Shared helpers for the 42781 glue-dispensing analysis scripts.

Each frame of the dataset is a grayscale BMP composed of three vertically
stacked camera strips of equal height. Strip A is the topmost.

These scripts are exploratory tooling built on numpy/scipy — they do **not**
exercise the `vision_metrology` bindings. Runnable library demos live in
`examples/python/`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

# Row ranges of the three stacked camera strips within one frame.
STRIP_ROWS: list[tuple[int, int]] = [(0, 97), (97, 194), (194, 291)]
STRIP_NAMES: list[str] = ["A", "B", "C"]
N_STRIPS: int = len(STRIP_ROWS)

#: Default median-filter width used to smooth per-frame shift estimates.
SMOOTH_WINDOW: int = 5


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the `--data-dir` / `--out-dir` options every script accepts."""
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing the Image_*.bmp frames.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Where to write PNG figures (default: <data-dir>/output).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=SMOOTH_WINDOW,
        help=f"Median-filter width for shift smoothing (default: {SMOOTH_WINDOW}).",
    )
    return parser


def resolve_out_dir(args: argparse.Namespace) -> pathlib.Path:
    """Return the output directory, creating it if needed."""
    out = args.out_dir if args.out_dir is not None else args.data_dir / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_images(data_dir: pathlib.Path) -> np.ndarray:
    """Load every `Image_*.bmp` frame as a float32 array of shape (N, H, W)."""
    paths = sorted(data_dir.glob("Image_*.bmp"))
    if not paths:
        sys.exit(f"No Image_*.bmp frames found in {data_dir}")
    frames = [np.array(Image.open(p), dtype=np.uint8) for p in paths]
    return np.stack(frames, axis=0).astype(np.float32)


def split_strips(imgs: np.ndarray) -> list[np.ndarray]:
    """Split frames of shape (N, H, W) into `N_STRIPS` arrays of (N, h, W)."""
    return [imgs[:, r0:r1, :] for r0, r1 in STRIP_ROWS]


def disk_se(radius: int) -> np.ndarray:
    """Binary disk structuring element of the given radius."""
    r = radius
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y <= r * r).astype(bool)


def phase_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return the (dx, dy) shift of `b` relative to `a` by phase correlation."""
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-8
    r = np.fft.fftshift(np.fft.ifft2(cross).real)
    peak = np.unravel_index(r.argmax(), r.shape)
    dy = float(peak[0] - r.shape[0] // 2)
    dx = float(peak[1] - r.shape[1] // 2)
    return dx, dy
