from __future__ import annotations

import numpy as np

from .common import save_fig
from .plot import plt


def _plot_graph_shape(
    ax: plt.Axes,
    polylines: list[list[tuple[float, float]]],
    junctions: list[tuple[float, float]],
    ends: list[tuple[float, float]],
    title: str,
) -> None:
    for pts in polylines:
        arr = np.asarray(pts, dtype=np.float32)
        ax.plot(arr[:, 0], arr[:, 1], color="#1f77b4", lw=2.6)

    if ends:
        ee = np.asarray(ends, dtype=np.float32)
        ax.scatter(ee[:, 0], ee[:, 1], s=50, c="#f4a261", edgecolors="black", label="End")

    if junctions:
        jj = np.asarray(junctions, dtype=np.float32)
        ax.scatter(
            jj[:, 0],
            jj[:, 1],
            s=70,
            c="#e63946",
            edgecolors="black",
            label="Junction",
        )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(8.5, 0.5)
    ax.grid(alpha=0.2)


def make_ty_figure() -> None:
    t_polylines = [[(4, 4), (4, 1)], [(4, 4), (4, 7)], [(4, 4), (7, 4)]]
    t_junctions = [(4, 4)]
    t_ends = [(4, 1), (4, 7), (7, 4)]

    y_polylines = [[(4, 4), (4, 1)], [(4, 4), (2, 6)], [(4, 4), (6, 6)]]
    y_junctions = [(4, 4)]
    y_ends = [(4, 1), (2, 6), (6, 6)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    _plot_graph_shape(axes[0], t_polylines, t_junctions, t_ends, "T-junction topology")
    _plot_graph_shape(axes[1], y_polylines, y_junctions, y_ends, "Y-junction topology")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Contour graph primitives from edgels", fontsize=12)

    save_fig(fig, "contour_graph_ty.png")


def make_loop_figure() -> None:
    loop = [(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
    arr = np.asarray(loop, dtype=np.float32)
    ax.plot(arr[:, 0], arr[:, 1], color="#1f77b4", lw=2.8, label="Loop edge")

    anchor = (2, 2)
    ax.scatter(
        [anchor[0]],
        [anchor[1]],
        s=85,
        c="#ff9f1c",
        edgecolors="black",
        label="LoopAnchor node",
    )

    ax.set_title("Closed contour component (loop)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(7.5, 0.5)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")

    save_fig(fig, "contour_graph_loop.png")
