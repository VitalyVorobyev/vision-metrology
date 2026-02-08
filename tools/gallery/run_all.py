#!/usr/bin/env python3
"""Run full external-fixture gallery pipeline.

1) Generate fixtures (Python)
2) Run Rust vm_gallery on each case
3) Build final figures from raw Rust outputs (Python plotting only)
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "fig"
FIX_DIR = FIG_DIR / "fixtures"
RAW_DIR = FIG_DIR / "raw"

CASES = [
    "morphology",
    "pyramid",
    "edge1d",
    "laser_rows",
    "laser_cols",
    "edgels2d",
    "contour_graph",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_values(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row["value"]))
    return np.asarray(values, dtype=np.float32)


def save_fig(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def generate_fixtures() -> None:
    run([sys.executable, str(ROOT / "tools" / "fixtures" / "gen_fixtures.py")], cwd=ROOT)


def run_rust_cases() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        inp = FIX_DIR / case / "input.png"
        truth = FIX_DIR / case / "truth.json"
        cmd = [
            "cargo",
            "run",
            "--release",
            "-p",
            "vm-gallery",
            "--bin",
            "vm_gallery",
            "--",
            case,
            "--input",
            str(inp),
            "--truth",
            str(truth),
            "--out",
            str(RAW_DIR),
        ]
        run(cmd, cwd=ROOT)


def fig_morphology() -> None:
    case = RAW_DIR / "morphology"
    inp = np.asarray(Image.open(case / "input.png"), dtype=np.uint8)
    opened = np.asarray(Image.open(case / "open.png"), dtype=np.uint8)
    closed = np.asarray(Image.open(case / "close.png"), dtype=np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, arr, title in zip(
        axes,
        [inp, opened, closed],
        ["input", "open (rust)", "close (rust)"],
    ):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("Morphology: open/close from vm-gallery", fontsize=12)
    save_fig(fig, "morphology_open_close.png")


def fig_pyramid() -> None:
    case = RAW_DIR / "pyramid"
    level_paths = sorted(case.glob("level_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    levels = [np.asarray(Image.open(p), dtype=np.uint8) for p in level_paths]

    fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 3.5))
    if len(levels) == 1:
        axes = [axes]

    for i, (ax, arr) in enumerate(zip(axes, levels)):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"L{i}: {arr.shape[1]}x{arr.shape[0]}")
        ax.axis("off")

    fig.suptitle("Pyramid levels from vm-pyr (via vm-gallery)", fontsize=12)
    save_fig(fig, "pyramid_levels.png")


def fig_edge1d() -> None:
    case = RAW_DIR / "edge1d"
    truth_env = load_json(case / "truth.json")
    truth = truth_env.get("truth", {})
    result = load_json(case / "result.json")

    signal = read_csv_values(case / "signal.csv")
    response = read_csv_values(case / "response.csv")
    x = np.arange(signal.size, dtype=np.float32)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax0.plot(x, signal, color="#2d2d2d", lw=2, label="signal")
    ax0.set_ylabel("intensity")
    ax0.grid(alpha=0.25)

    ax1.plot(x, response, color="#1f77b4", lw=2, label="DoG response")
    ax1.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax1.set_ylabel("response")
    ax1.set_xlabel("x (pixel-center index)")
    ax1.grid(alpha=0.25)

    def draw_marker(xv: float, color: str, label: str, ls: str) -> None:
        for ax in (ax0, ax1):
            ax.axvline(xv, color=color, ls=ls, lw=1.6, label=label)

    if truth:
        draw_marker(float(truth["x_left"]), "#00d4ff", "truth left", "--")
        draw_marker(float(truth["x_right"]), "#00d4ff", "truth right", "--")

    if result.get("left_x") is not None:
        draw_marker(float(result["left_x"]), "#ff4d4d", "detected left", ":")
    if result.get("right_x") is not None:
        draw_marker(float(result["right_x"]), "#ff4d4d", "detected right", ":")
    if result.get("center_x") is not None:
        draw_marker(float(result["center_x"]), "#f4a261", "detected center", "-.")

    ax0.set_title("1D DoG stripe: Rust signal/response/results")
    handles, labels = ax0.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        uniq[l] = h
    ax0.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

    save_fig(fig, "edge1d_dog_stripe.png")


def _truth_centers_to_array(true_center: list[float | None]) -> np.ndarray:
    arr = np.array([np.nan if v is None else float(v) for v in true_center], dtype=np.float32)
    return arr


def fig_laser(case_name: str, axis: str, out_name: str) -> None:
    case = RAW_DIR / case_name
    img = np.asarray(Image.open(case / "input.png"), dtype=np.uint8)
    truth_env = load_json(case / "truth.json")
    line = load_json(case / "line.json")

    centers_truth = _truth_centers_to_array(truth_env["truth"]["true_center"])
    samples = line["samples"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    if axis == "rows":
        y = np.arange(img.shape[0], dtype=np.float32)
        ax.plot(centers_truth, y, color="#00d4ff", lw=2.0, label="ground truth")

        xs = [float(s["center"]) for s in samples if bool(s["valid"]) and np.isfinite(centers_truth[int(s["scan_i"])])]
        ys = [float(s["scan_i"]) for s in samples if bool(s["valid"]) and np.isfinite(centers_truth[int(s["scan_i"])])]
        ax.scatter(xs, ys, s=12, c="#ff4d4d", edgecolors="white", linewidths=0.2, label="tracked center")
    else:
        x = np.arange(img.shape[1], dtype=np.float32)
        ax.plot(x, centers_truth, color="#00d4ff", lw=2.0, label="ground truth")

        xs = [float(s["scan_i"]) for s in samples if bool(s["valid"]) and np.isfinite(centers_truth[int(s["scan_i"])])]
        ys = [float(s["center"]) for s in samples if bool(s["valid"]) and np.isfinite(centers_truth[int(s["scan_i"])])]
        ax.scatter(xs, ys, s=12, c="#ff4d4d", edgecolors="white", linewidths=0.2, label="tracked center")

    ax.legend(loc="lower right")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    save_fig(fig, out_name)


def fig_edgels() -> None:
    case = RAW_DIR / "edgels2d"
    img = np.asarray(Image.open(case / "input.png"), dtype=np.uint8)
    edgels = load_json(case / "edgels.json")

    n = len(edgels)
    step = max(1, n // 3500)
    sample = edgels[::step]

    xs = np.array([e["x"] for e in sample], dtype=np.float32)
    ys = np.array([e["y"] for e in sample], dtype=np.float32)
    nx = np.array([e["nx"] for e in sample], dtype=np.float32)
    ny = np.array([e["ny"] for e in sample], dtype=np.float32)
    ixs = np.array([e["ix"] for e in sample], dtype=np.float32)
    iys = np.array([e["iy"] for e in sample], dtype=np.float32)

    h, w = img.shape
    fig_w = 9.0
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, origin="upper", interpolation="nearest")

    # Warm coral points + cool teal normals for a clean high-contrast palette.
    ax.scatter(
        xs,
        ys,
        s=4.5,
        c="#ff7b54",
        alpha=0.72,
        edgecolors="none",
    )

    max_arrows = 120
    if len(sample) > 0:
        idx = np.linspace(0, len(sample) - 1, min(max_arrows, len(sample)), dtype=int)
        ax.quiver(
            xs[idx],
            ys[idx],
            nx[idx],
            ny[idx],
            angles="xy",
            scale_units="xy",
            scale=0.22,
            color="#00b3a4",
            width=0.0022,
            headwidth=3.0,
            headlength=4.2,
            headaxislength=3.6,
            alpha=0.92,
        )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_axis_off()

    # Subpixel demonstration inset: pixel centers vs refined edgel points.
    x0, x1 = 86, 140
    y0, y1 = 58, 132
    inset = ax.inset_axes([0.60, 0.05, 0.36, 0.36])
    inset.imshow(
        img[y0 : y1 + 1, x0 : x1 + 1],
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper",
        extent=(x0 - 0.5, x1 + 0.5, y1 + 0.5, y0 - 0.5),
        interpolation="nearest",
    )

    in_roi = (
        (ixs >= x0)
        & (ixs <= x1)
        & (iys >= y0)
        & (iys <= y1)
    )
    roi_ix = ixs[in_roi]
    roi_iy = iys[in_roi]
    roi_x = xs[in_roi]
    roi_y = ys[in_roi]

    inset.scatter(
        roi_ix,
        roi_iy,
        s=11,
        c="#f8f9fa",
        alpha=0.55,
        marker="s",
        edgecolors="none",
    )
    inset.scatter(
        roi_x,
        roi_y,
        s=10,
        c="#ff7b54",
        alpha=0.9,
        edgecolors="none",
    )
    if roi_x.size > 0:
        link_sel = np.linspace(0, roi_x.size - 1, min(180, roi_x.size), dtype=int)
        for i in link_sel:
            inset.plot(
                [roi_ix[i], roi_x[i]],
                [roi_iy[i], roi_y[i]],
                color="#ffe8dc",
                lw=0.65,
                alpha=0.45,
            )

    inset.set_xlim(x0 - 0.5, x1 + 0.5)
    inset.set_ylim(y1 + 0.5, y0 - 0.5)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("#f8f9fa")
        spine.set_linewidth(0.8)

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    save_fig(fig, "edgels_overlay.png")


def fig_contour_graph() -> None:
    case = RAW_DIR / "contour_graph"
    img = np.asarray(Image.open(case / "input.png"), dtype=np.uint8)
    graph = load_json(case / "graph.json")

    h, w = img.shape
    fig_w = 9.0
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, alpha=0.58, origin="upper")

    for e in graph["edges"]:
        pts = np.asarray(e["points"], dtype=np.float32)
        if pts.shape[0] < 2:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color="#2a9dff", lw=1.5, alpha=0.9)

    kind_to_color = {
        "End": "#fcbf49",
        "Junction": "#e63946",
        "LoopAnchor": "#f77f00",
        "Isolated": "#9e9e9e",
    }

    for kind, color in kind_to_color.items():
        nodes = [n for n in graph["nodes"] if n["kind"] == kind]
        if not nodes:
            continue
        x = [n["x"] for n in nodes]
        y = [n["y"] for n in nodes]
        ax.scatter(x, y, s=24, c=color, edgecolors="black", linewidths=0.25)

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    save_fig(fig, "contour_graph.png")


def build_final_figures() -> None:
    fig_morphology()
    fig_pyramid()
    fig_edge1d()
    fig_laser("laser_rows", "rows", "laser_rows.png")
    fig_laser("laser_cols", "cols", "laser_cols.png")
    fig_edgels()
    fig_contour_graph()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    generate_fixtures()
    run_rust_cases()
    build_final_figures()


if __name__ == "__main__":
    main()
