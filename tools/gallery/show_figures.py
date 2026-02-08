#!/usr/bin/env python3
"""Interactive viewer for gallery figures."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN_ALL = ROOT / "tools" / "gallery" / "run_all.py"
FIG_DIR = ROOT / "docs" / "fig"

ORDERED_FIGS = [
    ("morphology", "Morphology", "morphology_open_close.png"),
    ("pyramid", "Pyramid", "pyramid_levels.png"),
    ("edge1d", "1D DoG", "edge1d_dog_stripe.png"),
    ("laser_rows", "Laser Rows", "laser_rows.png"),
    ("laser_cols", "Laser Cols", "laser_cols.png"),
    ("edgels2d", "2D Edgels", "edgels_overlay.png"),
    ("contour_graph", "Contour Graph", "contour_graph.png"),
]


def run_pipeline() -> None:
    cmd = [sys.executable, str(RUN_ALL)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def collect_available(case: str | None = None) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for case_key, title, name in ORDERED_FIGS:
        if case is not None and case_key != case:
            continue
        path = FIG_DIR / name
        if path.exists():
            out.append((title, path))
        else:
            print(f"warning: missing figure {path.relative_to(ROOT)}")
    return out


def render_grid(figs: list[tuple[str, Path]], export: Path | None, show: bool) -> None:
    if not figs:
        raise SystemExit("No figures found to display.")

    cols = 3
    rows = math.ceil(len(figs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 3.6))

    if rows == 1 and cols == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.ravel())

    for ax, (title, path) in zip(axes_list, figs):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    for ax in axes_list[len(figs) :]:
        ax.axis("off")

    fig.suptitle("vision-metrology Gallery", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if export is not None:
        if not export.is_absolute():
            export = (ROOT / export).resolve()
        export.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export, dpi=160, bbox_inches="tight")
        print(f"wrote {export.relative_to(ROOT)}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def render_browse(figs: list[tuple[str, Path]], show: bool) -> None:
    if not figs:
        raise SystemExit("No figures found to display.")

    fig, ax = plt.subplots(figsize=(11, 7))
    state = {"idx": 0}

    def draw() -> None:
        ax.clear()
        title, path = figs[state["idx"]]
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(f"{state['idx'] + 1}/{len(figs)}  {title}", fontsize=13)
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event) -> None:  # type: ignore[no-untyped-def]
        if event.key in ("right", "n", " "):
            state["idx"] = (state["idx"] + 1) % len(figs)
            draw()
        elif event.key in ("left", "p", "backspace"):
            state["idx"] = (state["idx"] - 1) % len(figs)
            draw()
        elif event.key in ("escape", "q"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()

    print("browse controls: right/n/space=next, left/p/backspace=prev, q/esc=quit")
    if show:
        plt.show()
    else:
        plt.close(fig)


def render_single(figs: list[tuple[str, Path]], export: Path | None, show: bool) -> None:
    if not figs:
        raise SystemExit("No figures found to display.")
    if len(figs) != 1:
        raise SystemExit("render_single requires exactly one figure.")

    title, path = figs[0]
    img = mpimg.imread(path)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(img)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()

    if export is not None:
        if not export.is_absolute():
            export = (ROOT / export).resolve()
        export.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export, dpi=160, bbox_inches="tight")
        print(f"wrote {export.relative_to(ROOT)}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and show gallery figures interactively")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip rebuilding; just display existing figures from docs/fig/",
    )
    parser.add_argument(
        "--mode",
        choices=["grid", "browse"],
        default="browse",
        help="Display mode: grid collage or keyboard browse",
    )
    parser.add_argument(
        "--case",
        choices=[c for c, _, _ in ORDERED_FIGS],
        default=None,
        help="Show one required use case only (e.g. --case edgels2d).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a window (useful with --export)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional output path for saving a grid contact sheet (e.g. docs/fig/gallery_sheet.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.no_build:
        run_pipeline()

    figs = collect_available(args.case)
    show = not args.no_show

    if args.case is not None:
        render_single(figs, args.export, show)
    elif args.mode == "grid":
        render_grid(figs, args.export, show)
    else:
        if args.export is not None:
            print("note: --export is ignored in browse mode")
        render_browse(figs, show)


if __name__ == "__main__":
    main()
