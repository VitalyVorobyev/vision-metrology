from __future__ import annotations

import numpy as np

from .contour_fig import make_loop_figure, make_ty_figure
from .edge1d_fig import make_figure as make_edge1d_figure
from .edge2d_fig import make_figure as make_edge2d_figure
from .laser_fig import (
    make_cols_figure,
    make_laser_cols_image,
    make_laser_rows_image,
    make_rows_figure,
    track_stripe_centers,
)
from .morphology_fig import make_figure as make_morphology_figure
from .pyramid_fig import make_figure as make_pyramid_figure


def generate_all() -> None:
    make_morphology_figure()
    make_pyramid_figure()
    make_edge1d_figure()
    make_rows_figure()
    make_cols_figure()
    make_edge2d_figure()
    make_ty_figure()
    make_loop_figure()


def laser_tracking_report() -> tuple[float, float]:
    """Return median absolute center error for rows and cols fixtures."""

    img_r, gt_r = make_laser_rows_image()
    centers_r, _, _ = track_stripe_centers(img_r, axis="rows")
    mask_r = np.isfinite(gt_r) & np.isfinite(centers_r)
    rows_med = float(np.median(np.abs(centers_r[mask_r] - gt_r[mask_r])))

    img_c, gt_c = make_laser_cols_image()
    centers_c, _, _ = track_stripe_centers(img_c, axis="cols")
    mask_c = np.isfinite(gt_c) & np.isfinite(centers_c)
    cols_med = float(np.median(np.abs(centers_c[mask_c] - gt_c[mask_c])))

    return rows_med, cols_med
