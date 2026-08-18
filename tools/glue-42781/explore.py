"""Data survey for the 42781 glue-dispensing dataset.

Writes four figures: a frame gallery, mean brightness over time, per-strip
time-space maps, and per-strip temporal median/std images.

    python tools/glue-42781/explore.py --data-dir path/to/42781
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from common import (
    N_STRIPS,
    STRIP_NAMES,
    add_common_args,
    load_images,
    resolve_out_dir,
    split_strips,
)

# Set by main() once the command line has been parsed.
OUT_DIR: pathlib.Path = pathlib.Path()



# ---------------------------------------------------------------------------
# Figure 1 — Frame gallery (5 frames × 3 strips)
# ---------------------------------------------------------------------------
def fig_gallery(imgs: np.ndarray) -> None:
    n = imgs.shape[0]
    sample_idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    strips = split_strips(imgs)

    fig, axes = plt.subplots(len(sample_idx), N_STRIPS, figsize=(9, 12))
    fig.suptitle("Frame gallery (5 sampled frames × 3 strips)", fontsize=13)

    for row, fi in enumerate(sample_idx):
        for col, (strip, name) in enumerate(zip(strips, STRIP_NAMES)):
            ax = axes[row, col]
            ax.imshow(strip[fi], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"Strip {name}  frame {fi}", fontsize=8)
            ax.axis("off")

    fig.tight_layout()
    out = OUT_DIR / "explore_gallery.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Mean brightness over time
# ---------------------------------------------------------------------------
def fig_mean_brightness(strips: list[np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for strip, name in zip(strips, STRIP_NAMES):
        means = strip.mean(axis=(1, 2))
        ax.plot(means, label=f"Strip {name}")
        print(f"Strip {name}: mean={means.mean():.1f}  std={means.std():.2f}  "
              f"min={means.min():.1f}  max={means.max():.1f}")

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Mean pixel value")
    ax.set_title("Mean brightness per strip over time")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "explore_brightness.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Time-space maps (column-mean profile per frame)
# ---------------------------------------------------------------------------
def fig_timespace(strips: list[np.ndarray]) -> None:
    fig, axes = plt.subplots(1, N_STRIPS, figsize=(13, 5))
    fig.suptitle("Time-space maps (column-mean profile per frame)", fontsize=13)

    for ax, strip, name in zip(axes, strips, STRIP_NAMES):
        # profile[t, row] = mean over all columns
        profile = strip.mean(axis=2)          # (N, H_strip)
        im = ax.imshow(
            profile.T,
            aspect="auto",
            cmap="gray",
            origin="upper",
            extent=[0, profile.shape[0] - 1, profile.shape[1] - 1, 0],
        )
        ax.set_title(f"Strip {name}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = OUT_DIR / "explore_timespace.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Temporal statistics (median and std per strip)
# ---------------------------------------------------------------------------
def fig_temporal_stats(strips: list[np.ndarray]) -> None:
    fig, axes = plt.subplots(N_STRIPS, 2, figsize=(8, 10))
    fig.suptitle("Temporal statistics: median frame | std frame", fontsize=13)

    for row, (strip, name) in enumerate(zip(strips, STRIP_NAMES)):
        med = np.median(strip, axis=0)
        std = strip.std(axis=0)

        ax_med = axes[row, 0]
        ax_std = axes[row, 1]

        im_med = ax_med.imshow(med, cmap="gray", vmin=0, vmax=255)
        ax_med.set_title(f"Strip {name} — median")
        ax_med.axis("off")
        fig.colorbar(im_med, ax=ax_med, fraction=0.046, pad=0.04)

        im_std = ax_std.imshow(std, cmap="hot")
        ax_std.set_title(f"Strip {name} — std")
        ax_std.axis("off")
        fig.colorbar(im_std, ax=ax_std, fraction=0.046, pad=0.04)

        print(f"Strip {name}: median_mean={med.mean():.1f}  std_mean={std.mean():.2f}")

    fig.tight_layout()
    out = OUT_DIR / "explore_stats.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Survey the 42781 glue-dispensing dataset.")
    add_common_args(parser)
    args = parser.parse_args()

    global OUT_DIR
    OUT_DIR = resolve_out_dir(args)

    imgs = load_images(args.data_dir)
    strips = split_strips(imgs)

    print("\n--- Per-strip brightness stats ---")
    fig_mean_brightness(strips)

    print("\n--- Temporal statistics ---")
    fig_temporal_stats(strips)

    print("\n--- Frame gallery ---")
    fig_gallery(imgs)

    print("\n--- Time-space maps ---")
    fig_timespace(strips)

    print("\nDone. Figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
