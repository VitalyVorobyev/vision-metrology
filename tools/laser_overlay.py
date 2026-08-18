#!/usr/bin/env python3
"""Plot laser line detection results overlaid on the source image.

Reads a horizontally-merged PNG and the JSON output from the `laserline`
Rust example, then draws one subplot per snap with detected laser centres
overlaid on the grayscale image.

Usage (from repo root):
    python tools/laser_overlay.py
    python tools/laser_overlay.py --show
    python tools/laser_overlay.py --out data/laser_0_overlay.png --show
    python tools/laser_overlay.py --input data/laser_0.png \\
        --results data/laser_0_results.json \\
        --out data/laser_0_overlay.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

ROOT = Path(__file__).resolve().parents[1]

# Cyan-ish accent — visible on both bright and dark backgrounds.
_POINT_COLOR = "#00d4ff"


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_snaps(img_array: np.ndarray, n: int) -> list[np.ndarray]:
    """Split a horizontally concatenated image into n equal-width snaps."""
    h, w = img_array.shape[:2]
    assert w % n == 0, f"width {w} not divisible by {n}"
    snap_w = w // n
    return [img_array[:, i * snap_w : (i + 1) * snap_w] for i in range(n)]


def plot_overlay(
    snaps: list[np.ndarray],
    results: list[dict],
    out: Path | None,
    show: bool,
) -> None:
    n = len(snaps)
    cols = min(n, 3)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 3.8))
    axes_flat = np.array(axes).ravel()

    total_ms = sum(r.get("elapsed_ms", 0.0) for r in results)
    total_valid = sum(
        sum(1 for s in r["samples"] if s["valid"]) for r in results
    )
    total_cols = sum(len(r["samples"]) for r in results)

    for ax, snap_arr, snap_res in zip(axes_flat, snaps, results):
        samples = snap_res["samples"]

        # cols mode: scan_i = column index (x), center = laser y-coordinate.
        xs = [s["scan_i"] for s in samples if s["valid"]]
        ys = [s["center"] for s in samples if s["valid"]]

        ax.imshow(snap_arr, cmap="gray", vmin=0, vmax=255, aspect="equal")

        if xs:
            ax.scatter(
                xs,
                ys,
                s=4,
                c=_POINT_COLOR,
                edgecolors="none",
                alpha=0.9,
            )

        ax.axis("off")

    # Hide any unused axes.
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Laser line detection  |  {total_valid}/{total_cols} valid  |  {total_ms:.1f} ms total",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay laser line results on source image")
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "laser_0.png",
        help="Path to the merged PNG (default: data/laser_0.png)",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=ROOT / "data" / "laser_0_results.json",
        help="Path to JSON output from the laserline example",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the overlay figure",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        sys.exit(f"input image not found: {args.input}\nRun the laserline example first.")
    if not args.results.exists():
        sys.exit(
            f"results JSON not found: {args.results}\n"
            "Run:  cargo run -p vision-metrology --example laserline"
        )

    img_array = np.asarray(Image.open(args.input).convert("L"), dtype=np.uint8)
    print(f"loaded image: {img_array.shape[1]}x{img_array.shape[0]}")

    results = load_results(args.results)
    n_snaps = len(results)
    print(f"loaded {n_snaps} snap results")

    snaps = split_snaps(img_array, n_snaps)

    if not args.show and args.out is None:
        args.out = args.results.with_name(
            args.results.stem.replace("_results", "_overlay") + ".png"
        )
        print(f"no --out or --show specified; saving to {args.out}")

    plot_overlay(snaps, results, out=args.out, show=args.show)


if __name__ == "__main__":
    main()
