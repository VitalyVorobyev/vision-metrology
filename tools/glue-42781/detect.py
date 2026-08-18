"""Nozzle localization, rear-camera identification, and glue detection (42781).

Steps:
  A. Nozzle segmentation via temporal median + brightness threshold.
  B. Rear-camera identification: the strip whose trailing half is consistently
     darker along the motion direction.
  C. Glue detection in the rear strip: background residual + morphology.

    python tools/glue-42781/detect.py --data-dir path/to/42781
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label, median_filter

import common
from common import (
    N_STRIPS,
    STRIP_NAMES,
    STRIP_ROWS,
    add_common_args,
    disk_se,
    load_images,
    phase_corr,
    resolve_out_dir,
    split_strips,
)

# Set by main() once the command line has been parsed.
OUT_DIR: pathlib.Path = pathlib.Path()
SMOOTH_WINDOW: int = common.SMOOTH_WINDOW



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def compute_motion(strips: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return median (vx, vy) per strip, shape (N_STRIPS,)."""
    vx = np.zeros(N_STRIPS)
    vy = np.zeros(N_STRIPS)
    for i, strip in enumerate(strips):
        dxs, dys = [], []
        for t in range(strip.shape[0] - 1):
            dx, dy = phase_corr(strip[t], strip[t + 1])
            dxs.append(dx)
            dys.append(dy)
        dxs_arr = median_filter(np.array(dxs, dtype=float), size=SMOOTH_WINDOW)
        dys_arr = median_filter(np.array(dys, dtype=float), size=SMOOTH_WINDOW)
        vx[i] = float(np.median(dxs_arr))
        vy[i] = float(np.median(dys_arr))
    return vx, vy


# ---------------------------------------------------------------------------
# Step A — Nozzle segmentation
# ---------------------------------------------------------------------------
def nozzle_bbox(median_frame: np.ndarray, se_radius: int = 5) -> dict:
    """Return bounding box and centroid of brightest stable blob."""
    thresh = float(np.percentile(median_frame, 90))
    mask = (median_frame >= thresh).astype(bool)
    se = disk_se(se_radius)
    mask = binary_closing(mask, structure=se)
    lbl, n_comp = label(mask)
    if n_comp == 0:
        return dict(r0=0, r1=0, c0=0, c1=0, cx=0.0, cy=0.0)
    sizes = [(lbl == k).sum() for k in range(1, n_comp + 1)]
    best = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lbl == best)
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    cx = float(xs.mean())
    cy = float(ys.mean())
    return dict(r0=r0, r1=r1, c0=c0, c1=c1, cx=cx, cy=cy)


def fig_nozzle(strips: list[np.ndarray]) -> list[dict]:
    fig, axes = plt.subplots(1, N_STRIPS, figsize=(12, 5))
    fig.suptitle("Nozzle localization (temporal median + bright blob)", fontsize=13)

    bboxes = []
    for ax, strip, name in zip(axes, strips, STRIP_NAMES):
        med = np.median(strip, axis=0)
        bbox = nozzle_bbox(med)
        bboxes.append(bbox)

        ax.imshow(med, cmap="gray", vmin=0, vmax=255)
        rect = mpatches.Rectangle(
            (bbox["c0"], bbox["r0"]),
            bbox["c1"] - bbox["c0"],
            bbox["r1"] - bbox["r0"],
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.plot(bbox["cx"], bbox["cy"], "r+", ms=10, mew=2)
        ax.set_title(
            f"Strip {name}\ncentroid=({bbox['cx']:.1f}, {bbox['cy']:.1f})", fontsize=9
        )
        ax.axis("off")
        print(f"  Strip {name}: nozzle centroid = ({bbox['cx']:.1f}, {bbox['cy']:.1f})")

    fig.tight_layout()
    out = OUT_DIR / "detect_nozzle.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close(fig)
    return bboxes


# ---------------------------------------------------------------------------
# Step B — Rear camera identification
# ---------------------------------------------------------------------------
def rear_camera_score(
    strips: list[np.ndarray],
    vx: np.ndarray,
    vy: np.ndarray,
) -> np.ndarray:
    """
    For each strip, compute (mean trailing intensity - mean leading intensity)
    averaged over all frames.  Most negative score = rear camera.

    "Trailing" = pixels whose projection onto the motion vector is below the
    median projection (i.e. behind the direction of travel).
    """
    # Global fallback direction: mean of non-zero per-strip velocities
    global_vx = float(np.mean(vx))
    global_vy = float(np.mean(vy))
    global_speed = (global_vx ** 2 + global_vy ** 2) ** 0.5

    scores = np.zeros(N_STRIPS)
    for si, strip in enumerate(strips):
        dvx, dvy = float(vx[si]), float(vy[si])
        speed = (dvx ** 2 + dvy ** 2) ** 0.5

        if speed < 0.5:
            # Per-strip motion too small; use global direction as fallback
            if global_speed < 0.5:
                # All strips nearly static — score stays 0 (ambiguous)
                scores[si] = 0.0
                continue
            dvx, dvy, speed = global_vx, global_vy, global_speed

        h, w = strip.shape[1], strip.shape[2]
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        proj = (xs * dvx + ys * dvy) / speed   # (H, W)
        mid_proj = float(np.median(proj))
        trailing_mask = proj < mid_proj         # behind the direction of travel
        leading_mask = ~trailing_mask

        if trailing_mask.sum() == 0 or leading_mask.sum() == 0:
            scores[si] = 0.0
            continue

        diffs = []
        for frame in strip:
            trailing_mean = float(frame[trailing_mask].mean())
            leading_mean = float(frame[leading_mask].mean())
            diffs.append(trailing_mean - leading_mean)
        scores[si] = float(np.mean(diffs))

    return scores


def fig_rear_camera(scores: np.ndarray) -> int:
    rear_idx = int(np.argmin(scores))

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["tab:blue"] * N_STRIPS
    colors[rear_idx] = "tab:red"
    bars = ax.bar(STRIP_NAMES, scores, color=colors)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Strip")
    ax.set_ylabel("Mean (trailing − leading) intensity")
    ax.set_title("Rear camera identification\n(most negative = rear)")
    bars[rear_idx].set_label(f"REAR → Strip {STRIP_NAMES[rear_idx]}")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "detect_rear_camera.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close(fig)
    return rear_idx


# ---------------------------------------------------------------------------
# Step C — Glue detection in rear camera
# ---------------------------------------------------------------------------
def glue_mask(frame: np.ndarray, background: np.ndarray, se_radius: int = 3) -> np.ndarray:
    """Return binary glue mask for one frame."""
    residual = background - frame          # positive = darker than background
    thr = residual.mean() + 2.0 * residual.std()
    mask = residual > thr
    se = disk_se(se_radius)
    mask = binary_opening(mask, structure=se)
    # keep blobs > 50 px
    lbl, n_comp = label(mask)
    final = np.zeros_like(mask)
    for k in range(1, n_comp + 1):
        comp = lbl == k
        if comp.sum() >= 50:
            final |= comp
    return final


def fig_glue_sequence(rear_strip: np.ndarray, rear_name: str) -> None:
    background = np.median(rear_strip, axis=0)
    n = rear_strip.shape[0]
    step = max(1, n // 8)
    sample_frames = list(range(0, n, step))[:8]

    cols = 4
    rows = (len(sample_frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, rows * 3.5))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(
        f"Glue sequence — strip {rear_name} (orange = glue mask)", fontsize=13
    )

    for ax, fi in zip(axes, sample_frames):
        frame = rear_strip[fi]
        mask = glue_mask(frame, background)
        rgb = np.stack([frame, frame, frame], axis=-1).astype(np.uint8)
        rgb[mask, 0] = 255
        rgb[mask, 1] = 140
        rgb[mask, 2] = 0
        ax.imshow(rgb)
        ax.set_title(f"Frame {fi}", fontsize=9)
        ax.axis("off")

    for ax in axes[len(sample_frames):]:
        ax.axis("off")

    fig.tight_layout()
    out = OUT_DIR / "detect_glue_sequence.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close(fig)


def fig_composite(
    imgs: np.ndarray,
    bboxes: list[dict],
    rear_idx: int,
) -> None:
    """Full 320×291 composite with strip labels, nozzle boxes, glue mask."""
    mid = imgs.shape[0] // 2
    frame_full = imgs[mid].astype(np.uint8)

    # Prepare glue mask in rear strip
    strips = split_strips(imgs)
    rear_strip = strips[rear_idx]
    background = np.median(rear_strip, axis=0)
    rear_frame = rear_strip[mid]
    gmask_local = glue_mask(rear_frame, background)

    # Full-resolution glue mask
    r0_rear, r1_rear = STRIP_ROWS[rear_idx]
    gmask_full = np.zeros((291, 320), dtype=bool)
    gmask_full[r0_rear:r1_rear, :] = gmask_local

    # Build RGB
    rgb = np.stack([frame_full, frame_full, frame_full], axis=-1)
    rgb[gmask_full, 0] = 255
    rgb[gmask_full, 1] = 140
    rgb[gmask_full, 2] = 0

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(rgb)
    ax.set_title(f"Composite — frame {mid}", fontsize=12)

    colors_box = ["lime", "yellow", "cyan"]
    for si, (bbox, name, color) in enumerate(zip(bboxes, STRIP_NAMES, colors_box)):
        r0_strip, _ = STRIP_ROWS[si]
        # Nozzle box in full-image coordinates
        rect = mpatches.Rectangle(
            (bbox["c0"], r0_strip + bbox["r0"]),
            bbox["c1"] - bbox["c0"],
            bbox["r1"] - bbox["r0"],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            label=f"Strip {name} nozzle",
        )
        ax.add_patch(rect)
        # Strip label
        ax.text(
            4,
            r0_strip + 10,
            f"Strip {name}" + (" [REAR]" if si == rear_idx else ""),
            color=color,
            fontsize=10,
            fontweight="bold",
        )
        # Strip separator line
        if si > 0:
            r0 = STRIP_ROWS[si][0]
            ax.axhline(r0, color="white", lw=0.8, ls="--")

    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    out = OUT_DIR / "detect_composite.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Locate the nozzle, identify the rear camera, and detect glue (42781).")
    add_common_args(parser)
    args = parser.parse_args()

    global OUT_DIR, SMOOTH_WINDOW
    OUT_DIR = resolve_out_dir(args)
    SMOOTH_WINDOW = args.smooth_window

    imgs = load_images(args.data_dir)
    strips = split_strips(imgs)

    # --- Step A: Nozzle ---
    print("\n--- Step A: Nozzle segmentation ---")
    bboxes = fig_nozzle(strips)

    # --- Motion for rear-camera ID ---
    print("\nComputing motion for rear-camera identification …")
    vx, vy = compute_motion(strips)
    for i, name in enumerate(STRIP_NAMES):
        print(f"  Strip {name}: median shift = ({vx[i]:+.2f}, {vy[i]:+.2f}) px")

    # --- Step B: Rear camera ---
    print("\n--- Step B: Rear camera identification ---")
    scores = rear_camera_score(strips, vx, vy)
    for name, score in zip(STRIP_NAMES, scores):
        print(f"  Strip {name}: score = {score:+.2f}")
    rear_idx = fig_rear_camera(scores)
    rear_name = STRIP_NAMES[rear_idx]
    print(f"  → REAR camera: Strip {rear_name}")

    # --- Step C: Glue detection ---
    print(f"\n--- Step C: Glue detection in strip {rear_name} ---")
    fig_glue_sequence(strips[rear_idx], rear_name)
    fig_composite(imgs, bboxes, rear_idx)

    print("\nDone. Figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
