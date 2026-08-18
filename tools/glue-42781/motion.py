"""Per-strip motion estimation for the 42781 dataset via phase correlation.

Each consecutive frame pair yields a (dx, dy) shift per strip. Shifts are
median-smoothed and accumulated to show the camera trajectory.

    python tools/glue-42781/motion.py --data-dir path/to/42781
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

import common
from common import (
    STRIP_NAMES,
    add_common_args,
    load_images,
    phase_corr,
    resolve_out_dir,
    split_strips,
)

# Set by main() once the command line has been parsed.
OUT_DIR: pathlib.Path = pathlib.Path()
SMOOTH_WINDOW: int = common.SMOOTH_WINDOW



# ---------------------------------------------------------------------------
# Phase correlation
# ---------------------------------------------------------------------------
def compute_shifts(strip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw and median-smoothed (dx, dy) time series for one strip."""
    n = strip.shape[0]
    raw_dx = np.zeros(n - 1)
    raw_dy = np.zeros(n - 1)
    for i in range(n - 1):
        raw_dx[i], raw_dy[i] = phase_corr(strip[i], strip[i + 1])

    smooth_dx = median_filter(raw_dx, size=SMOOTH_WINDOW)
    smooth_dy = median_filter(raw_dy, size=SMOOTH_WINDOW)
    return (smooth_dx, smooth_dy)


# ---------------------------------------------------------------------------
# Figure 1 — Shift time series
# ---------------------------------------------------------------------------
def fig_shift_series(
    all_dx: list[np.ndarray],
    all_dy: list[np.ndarray],
) -> None:
    n_pairs = len(all_dx[0])
    frames = np.arange(n_pairs)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig.suptitle("Per-frame shifts (median-smoothed) per strip", fontsize=13)

    for row, (dx, dy, name) in enumerate(zip(all_dx, all_dy, STRIP_NAMES)):
        axes[row, 0].plot(frames, dx, color="tab:blue")
        axes[row, 0].axhline(0, color="k", lw=0.5, ls="--")
        axes[row, 0].set_ylabel(f"Strip {name}\ndx (px)")

        axes[row, 1].plot(frames, dy, color="tab:orange")
        axes[row, 1].axhline(0, color="k", lw=0.5, ls="--")
        axes[row, 1].set_ylabel(f"dy (px)")

    axes[-1, 0].set_xlabel("Frame pair index")
    axes[-1, 1].set_xlabel("Frame pair index")
    fig.tight_layout()

    out = OUT_DIR / "motion_shifts.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Cumulative trajectory
# ---------------------------------------------------------------------------
def fig_cumulative(
    all_dx: list[np.ndarray],
    all_dy: list[np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Cumulative displacement trajectory per strip")

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for dx, dy, name, color in zip(all_dx, all_dy, STRIP_NAMES, colors):
        cx = np.concatenate([[0.0], np.cumsum(dx)])
        cy = np.concatenate([[0.0], np.cumsum(dy)])
        ax.plot(cx, cy, color=color, label=f"Strip {name}")
        ax.plot(cx[0], cy[0], "o", color=color, ms=6)
        ax.plot(cx[-1], cy[-1], "s", color=color, ms=6)

    ax.set_xlabel("Cumulative X displacement (px)")
    ax.set_ylabel("Cumulative Y displacement (px)")
    ax.legend()
    ax.set_aspect("equal", "datalim")
    ax.grid(True, lw=0.4)
    fig.tight_layout()

    out = OUT_DIR / "motion_trajectory.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Motion summary on frame 70
# ---------------------------------------------------------------------------
def fig_motion_summary(
    imgs: np.ndarray,
    all_dx: list[np.ndarray],
    all_dy: list[np.ndarray],
) -> None:
    mid_frame = imgs.shape[0] // 2
    strips = split_strips(imgs)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(f"Motion summary — frame {mid_frame}", fontsize=13)

    colors = ["lime", "yellow", "cyan"]
    for ax, strip, dx, dy, name, color in zip(
        axes, strips, all_dx, all_dy, STRIP_NAMES, colors
    ):
        ax.imshow(strip[mid_frame], cmap="gray", vmin=0, vmax=255)

        # Global shift: median of all per-frame shifts
        gdx = float(np.median(dx))
        gdy = float(np.median(dy))
        h, w = strip.shape[1:]
        cx, cy = w / 2, h / 2
        scale = 5.0
        ax.annotate(
            "",
            xy=(cx + gdx * scale, cy + gdy * scale),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )
        angle_deg = float(np.degrees(np.arctan2(gdy, gdx)))
        ax.set_title(
            f"Strip {name}\ndx={gdx:.2f}px  dy={gdy:.2f}px\nangle={angle_deg:.1f}°",
            fontsize=9,
        )
        ax.axis("off")

    fig.tight_layout()
    out = OUT_DIR / "motion_summary.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PCA-based dominant direction
# ---------------------------------------------------------------------------
def dominant_direction(dx: np.ndarray, dy: np.ndarray) -> float:
    """Return angle in degrees of first PCA component of shift vectors."""
    vecs = np.stack([dx, dy], axis=1)  # (N, 2)
    vecs -= vecs.mean(axis=0)
    cov = vecs.T @ vecs
    _, evecs = np.linalg.eigh(cov)
    dominant = evecs[:, -1]            # eigenvector with largest eigenvalue
    return float(np.degrees(np.arctan2(dominant[1], dominant[0])))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate per-strip motion in the 42781 dataset by phase correlation.")
    add_common_args(parser)
    args = parser.parse_args()

    global OUT_DIR, SMOOTH_WINDOW
    OUT_DIR = resolve_out_dir(args)
    SMOOTH_WINDOW = args.smooth_window

    imgs = load_images(args.data_dir)
    strips = split_strips(imgs)

    all_dx: list[np.ndarray] = []
    all_dy: list[np.ndarray] = []

    print("Computing phase-correlation shifts …")
    for strip, name in zip(strips, STRIP_NAMES):
        dx, dy = compute_shifts(strip)
        all_dx.append(dx)
        all_dy.append(dy)
        angle = dominant_direction(dx, dy)
        median_dx = float(np.median(dx))
        median_dy = float(np.median(dy))
        print(
            f"  Strip {name}: median shift = ({median_dx:+.2f}, {median_dy:+.2f}) px  "
            f"PCA angle = {angle:.1f}°"
        )

    # Overall direction from all strips combined
    all_dx_cat = np.concatenate(all_dx)
    all_dy_cat = np.concatenate(all_dy)
    overall_angle = dominant_direction(all_dx_cat, all_dy_cat)
    print(f"\nOverall dominant motion angle: {overall_angle:.1f}°")

    print("\nGenerating figures …")
    fig_shift_series(all_dx, all_dy)
    fig_cumulative(all_dx, all_dy)
    fig_motion_summary(imgs, all_dx, all_dy)

    print("\nDone. Figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
