"""Shared helpers for the bead-tracker evaluation scripts.

Each script reads `--data-dir` (the DamSegment folder) and writes under `--out-dir`
(default `<data-dir>/output`). Steps hand over through files in the output directory,
so each can be re-run on its own:

    paths/       reference centrelines per image, JSON          (paths.py)
    priors.json  perturbed priors per reference path            (priors.py)
    runs/<name>/ one .npz per image of every call, and meta.json (run_tracker.py, baselines.py)
    report.*     the metrics, and overlays/ PNGs                (report.py)

Coordinates are `(x, y)` with pixel centres at integers: `x` is the column and `y` the
row of the numpy image, which is the library's convention.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zlib

import numpy as np
from PIL import Image

#: The dataset's own folder name, with its spelling.
DATASET_DIR = "Damage Segmentaion"
DIFFICULTIES = ("Easy", "Medium", "Hard")

#: Base seed for every random draw; each draw is seeded from it and a stable key.
SEED = 20261006


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the `--data-dir` / `--out-dir` options every script accepts."""
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
        help=f"The DamSegment folder: '{DATASET_DIR}' or the folder that contains it.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Where the steps write and read their outputs (default: <data-dir>/output).",
    )
    return parser


def resolve_out_dir(args: argparse.Namespace) -> pathlib.Path:
    """Return the output directory, creating it if needed."""
    out = args.out_dir if args.out_dir is not None else args.data_dir / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def dataset_root(data_dir: pathlib.Path) -> pathlib.Path:
    """The folder that holds `Easy/`, `Medium/` and `Hard/`."""
    for cand in (data_dir, data_dir / DATASET_DIR):
        if all((cand / d / "Images").is_dir() for d in DIFFICULTIES):
            return cand
    sys.exit(f"No DamSegment Easy/Medium/Hard folders found under {data_dir}")


def image_path(root: pathlib.Path, difficulty: str, stem: str) -> pathlib.Path:
    return root / difficulty / "Images" / f"{stem}.jpg"


def mask_path(root: pathlib.Path, difficulty: str, stem: str) -> pathlib.Path:
    return root / difficulty / "Labels" / "Mask" / f"{stem}_mask.png"


def image_stems(root: pathlib.Path, difficulty: str) -> list[str]:
    """Image names without extension, in natural order: `E (1)`, `E (2)`, ..."""

    def key(p: pathlib.Path) -> tuple[int, str]:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return (int(digits) if digits else 0, p.stem)

    return [p.stem for p in sorted((root / difficulty / "Images").glob("*.jpg"), key=key)]


def load_gray(path: pathlib.Path) -> np.ndarray:
    """An RGB image as float32 ITU-R BT.601 luma, `0.299 R + 0.587 G + 0.114 B`, on the
    8-bit scale (0-255) and not re-quantised."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.ascontiguousarray(gray, dtype=np.float32)


def load_crack_mask(path: pathlib.Path) -> np.ndarray:
    """The crack pixels of a DamSegment mask, as a boolean array. The masks paint cracks
    (category 0 in the Pascal VOC files) pure red and spalling (category 1) pure blue, on
    black."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return (rgb[..., 0] > 127) & (rgb[..., 1] < 128) & (rgb[..., 2] < 128)


def rng_for(*key: object) -> np.random.Generator:
    """A generator seeded from `SEED` and a stable key (Python's `hash` is salted)."""
    text = "|".join(str(k) for k in key)
    return np.random.default_rng([SEED, zlib.crc32(text.encode("utf-8"))])


def write_json(path: pathlib.Path, obj: object, indent: int | None = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent), encoding="utf-8")


def read_json(path: pathlib.Path) -> dict:
    if not path.is_file():
        sys.exit(f"Missing {path}: run the previous step first")
    return json.loads(path.read_text(encoding="utf-8"))


def select_paths(index: dict, max_paths: int | None) -> list[dict]:
    """The index's paths, or a seeded subset of at most `max_paths` per difficulty, in
    index order."""
    out = []
    for diff in DIFFICULTIES:
        entries = [e for e in index["paths"] if e["difficulty"] == diff]
        if max_paths is not None and len(entries) > max_paths:
            pick = np.sort(rng_for("subset", diff).choice(len(entries), max_paths, replace=False))
            entries = [entries[i] for i in pick]
        out.extend(entries)
    return out


def by_image(entries: list[dict]) -> dict[str, list[dict]]:
    """Index entries grouped by their per-image paths file, in order."""
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(e["file"], []).append(e)
    return groups


# ---------------------------------------------------------------------------
# The tracker's config
# ---------------------------------------------------------------------------
def add_tracker_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The tracker settings the scripts share. The defaults suit thin dark cracks a few
    pixels wide in rough concrete; every other setting is the library's default."""
    parser.add_argument("--reach", type=float, default=8.0, help="track.max_offset, px (default 8)")
    parser.add_argument("--spacing", type=float, default=2.0, help="px between stations (default 2)")
    parser.add_argument(
        "--threshold", type=float, default=3.0, help="edge threshold, 8-bit grey levels (default 3)"
    )
    parser.add_argument("--sigma", type=float, default=1.0, help="profile smoothing, px (default 1)")
    parser.add_argument(
        "--obliquity",
        type=float,
        default=180.0,
        help="the final stage's max_obliquity_deg (default 180: off; the library's is 30)",
    )
    return parser


def bead_config(polarity: str, min_width: float, max_width: float, args: argparse.Namespace):
    """A `vision_metrology.BeadConfig`: the library's defaults, with the tracking reach,
    spacing, threshold, profile sigma and final-stage obliquity gate from `args`."""
    import vision_metrology as vm

    base = vm.BeadConfig()
    track, measure = base.track, base.measure
    for stage in (track, measure):
        stage.threshold = args.threshold
        stage.sigma = args.sigma
    track.max_offset = args.reach
    measure.max_obliquity_deg = args.obliquity
    return vm.BeadConfig(
        polarity=polarity,
        min_width=float(min_width),
        max_width=float(max_width),
        spacing=args.spacing,
        track=track,
        measure=measure,
    )


# ---------------------------------------------------------------------------
# Run records: one compressed .npz per image and method
# ---------------------------------------------------------------------------
#: Per-call scalars; NaN where a method has no such value.
SCALARS = ("support", "longest_gap", "center_rms", "center_max_dev", "width_mean", "passes", "time_ms")
STOPS = ("converged", "pass_limit", "too_few_valid", "other")
REJECTS = (
    "profile_too_short",
    "no_edge",
    "wrong_polarity",
    "too_oblique",
    "off_image",
    "incomplete_sequence",
    "low_contrast",
    "no_crossing",
    "no_pair",
    "width",
    "offset",
    "clearance",
    "ambiguous",
)


def save_calls(path: pathlib.Path, calls: list[dict]) -> None:
    """Store calls: each has `path_id`, `spec` (its index in the path's prior list),
    `stop`, the `SCALARS`, and per station `centerline` (S, 2), `width` (S,) and
    `reject` (S,) codes into `REJECTS`, -1 for a hit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    off = np.cumsum([0] + [len(c["centerline"]) for c in calls]).astype(np.int64)

    def cat(key: str, shape: tuple[int, ...], dtype: type) -> np.ndarray:
        if not calls:
            return np.zeros(shape, dtype)
        return np.concatenate([np.asarray(c[key], dtype).reshape((-1,) + shape[1:]) for c in calls])

    np.savez_compressed(
        path,
        path_id=np.array([c["path_id"] for c in calls], dtype=str),
        spec=np.array([c["spec"] for c in calls], np.int16),
        stop=np.array([STOPS.index(c["stop"]) for c in calls], np.int8),
        offsets=off,
        centerline=cat("centerline", (0, 2), np.float32),
        width=cat("width", (0,), np.float32),
        reject=cat("reject", (0,), np.int8),
        **{k: np.array([c[k] for c in calls], np.float64) for k in SCALARS},
    )


def load_calls(path: pathlib.Path) -> list[dict]:
    """The calls `save_calls` stored, as dicts of numpy arrays and scalars."""
    z = np.load(path, allow_pickle=False)
    off = z["offsets"]
    calls = []
    for i in range(len(z["spec"])):
        a, b = int(off[i]), int(off[i + 1])
        c = {
            "path_id": str(z["path_id"][i]),
            "spec": int(z["spec"][i]),
            "stop": STOPS[int(z["stop"][i])],
            "centerline": z["centerline"][a:b].astype(np.float64),
            "width": z["width"][a:b].astype(np.float64),
            "reject": z["reject"][a:b],
        }
        c.update({k: float(z[k][i]) for k in SCALARS})
        calls.append(c)
    return calls


# ---------------------------------------------------------------------------
# Polyline geometry
# ---------------------------------------------------------------------------
def arc_length(p: np.ndarray) -> np.ndarray:
    """Cumulative chord length of an (N, 2) polyline, starting at 0."""
    seg = np.hypot(*np.diff(p, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample(p: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline uniformly in arc length; the last point is the end exactly."""
    s = arc_length(p)
    n = max(2, int(round(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def point_at(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Points of a polyline at arc lengths `s`, clamped to its ends."""
    sp = arc_length(p)
    return np.stack([np.interp(s, sp, p[:, 0]), np.interp(s, sp, p[:, 1])], axis=1)


def tangents(p: np.ndarray) -> np.ndarray:
    """Unit tangents by central differences (one-sided at the ends)."""
    t = np.gradient(p, axis=0)
    n = np.hypot(t[:, 0], t[:, 1])
    return t / np.maximum(n, 1e-12)[:, None]


def chord_tangents(p: np.ndarray, s: np.ndarray, half: float) -> np.ndarray:
    """Unit tangents of a polyline at arc lengths `s`, each the chord over `±half` px
    of arc length, clamped to its ends."""
    t = point_at(p, s + half) - point_at(p, s - half)
    return t / np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-12)[:, None]


def normals(p: np.ndarray) -> np.ndarray:
    """`(-t_y, t_x)`: the library's +n, to the right of travel with y down."""
    t = tangents(p)
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


def project(points: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest points on the polyline `ref` to each of `points`.

    Returns the distance, the arc length of the closest point along `ref`, and whether
    it falls strictly inside `ref` (not clamped to either end)."""
    a = ref[:-1]
    ab = ref[1:] - a
    len2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    ap = points[:, None, :] - a[None, :, :]
    u = np.einsum("nmk,mk->nm", ap, ab) / len2[None, :]
    uc = np.clip(u, 0.0, 1.0)
    d = np.hypot(*(ap - uc[..., None] * ab[None, :, :]).transpose(2, 0, 1))
    k = np.argmin(d, axis=1)
    rows = np.arange(len(points))
    s_ref = arc_length(ref)
    seg_len = np.sqrt(len2)
    s = s_ref[k] + uc[rows, k] * seg_len[k]
    inside = (s > 1e-6) & (s < s_ref[-1] - 1e-6)
    return d[rows, k], s, inside


def angle_error_deg(t_a: np.ndarray, t_b: np.ndarray) -> np.ndarray:
    """Unsigned angle between line directions, in degrees, in [0, 90]."""
    c = np.abs(np.einsum("ij,ij->i", t_a, t_b))
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))
