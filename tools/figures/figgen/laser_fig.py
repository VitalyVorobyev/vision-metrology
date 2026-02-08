from __future__ import annotations

import math

import numpy as np

from .common import conv1d_clamp, dog_kernel_1d, parabolic_offset, save_fig
from .config import RNG
from .plot import plt


def coarse_center(line: np.ndarray, half_width: int = 8, threshold_frac: float = 0.5) -> float:
    idx = int(np.argmax(line))
    max_v = float(line[idx])
    if max_v <= 0.0:
        return float(idx)

    lo = max(0, idx - half_width)
    hi = min(len(line), idx + half_width + 1)
    xs = np.arange(lo, hi, dtype=np.float32)
    vals = line[lo:hi].astype(np.float32)

    thresh = threshold_frac * max_v
    w = np.where(vals >= thresh, vals, 0.0)
    den = float(np.sum(w))
    if den <= 1e-12:
        return float(idx)

    return float(np.sum(xs * w) / den)


def _collect_extrema(response: np.ndarray) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    maxima: list[tuple[int, float]] = []
    minima: list[tuple[int, float]] = []

    for i in range(1, len(response) - 1):
        v = float(response[i])
        if v >= response[i - 1] and v > response[i + 1]:
            maxima.append((i, v))
        if v <= response[i - 1] and v < response[i + 1]:
            minima.append((i, v))

    return maxima, minima


def _best_pair_for_order(
    response: np.ndarray,
    first_peaks: list[tuple[int, float]],
    second_peaks: list[tuple[int, float]],
    predicted_center: float | None,
    min_width: float,
    max_width: float,
    prior_weight: float,
) -> tuple[float, float, float] | None:
    best: tuple[float, float, float] | None = None
    best_score = -1.0e30

    for i1, v1 in first_peaks:
        if i1 <= 0 or i1 >= len(response) - 1:
            continue

        for i2, v2 in second_peaks:
            if i2 <= i1 or i2 <= 0 or i2 >= len(response) - 1:
                continue

            width = float(i2 - i1)
            if width < min_width or width > max_width:
                continue

            d1 = parabolic_offset(response[i1 - 1], response[i1], response[i1 + 1])
            d2 = parabolic_offset(response[i2 - 1], response[i2], response[i2 + 1])
            x1 = i1 + d1
            x2 = i2 + d2
            center = 0.5 * (x1 + x2)

            base = abs(v1) + abs(v2)
            prior = 0.0 if predicted_center is None else prior_weight * abs(center - predicted_center)
            total = base - prior

            if total > best_score:
                best_score = total
                best = (x1, x2, total)

    return best


def detect_best_pair(
    response: np.ndarray,
    predicted_center: float | None,
    min_width: float = 2.0,
    max_width: float = 10.0,
    prior_weight: float = 0.2,
    bright_on_dark: bool = True,
) -> tuple[float, float, float] | None:
    """Return (left_edge, right_edge, score) in response-index coordinates.

    For this script's DoG/correlation convention, bright-on-dark stripes map to
    trough->peak edge ordering.
    """

    if len(response) < 3:
        return None

    maxima, minima = _collect_extrema(response)
    if not maxima or not minima:
        return None

    if bright_on_dark:
        primary = _best_pair_for_order(
            response,
            minima,
            maxima,
            predicted_center,
            min_width,
            max_width,
            prior_weight,
        )
        if primary is not None:
            return primary

        return _best_pair_for_order(
            response,
            maxima,
            minima,
            predicted_center,
            min_width,
            max_width,
            prior_weight,
        )

    primary = _best_pair_for_order(
        response,
        maxima,
        minima,
        predicted_center,
        min_width,
        max_width,
        prior_weight,
    )
    if primary is not None:
        return primary

    return _best_pair_for_order(
        response,
        minima,
        maxima,
        predicted_center,
        min_width,
        max_width,
        prior_weight,
    )


def track_stripe_centers(
    img: np.ndarray,
    axis: str,
    sigma: float = 1.2,
    roi_half_width: int = 32,
    min_width: float = 2.0,
    max_width: float = 12.0,
    min_score: float = 50.0,
    max_jump_px: float = 8.0,
    max_gap_scans: int = 5,
    prior_weight: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, dg = dog_kernel_1d(sigma)

    if axis == "rows":
        num_scans = img.shape[0]

        def get_line(i: int) -> np.ndarray:
            return img[i, :]

    elif axis == "cols":
        num_scans = img.shape[1]

        def get_line(i: int) -> np.ndarray:
            return img[:, i]

    else:
        raise ValueError("axis must be 'rows' or 'cols'")

    centers = np.full(num_scans, np.nan, dtype=np.float32)
    widths = np.full(num_scans, np.nan, dtype=np.float32)
    scores = np.full(num_scans, np.nan, dtype=np.float32)

    last_valid: float | None = None
    gap_len = 0

    for i in range(num_scans):
        line = get_line(i).astype(np.float32)
        reacquire = last_valid is None or gap_len > max_gap_scans
        predicted = coarse_center(line) if reacquire else float(last_valid)

        center_i = int(round(predicted))
        roi0 = max(0, center_i - roi_half_width)
        roi1 = min(line.size, center_i + roi_half_width + 1)

        resp = conv1d_clamp(line[roi0:roi1], dg)
        pair = detect_best_pair(
            resp,
            predicted_center=(predicted - roi0),
            min_width=min_width,
            max_width=max_width,
            prior_weight=prior_weight,
            bright_on_dark=True,
        )

        if pair is None:
            gap_len += 1
            continue

        left, right, score = pair
        if score < min_score:
            gap_len += 1
            continue

        left += roi0
        right += roi0
        center = 0.5 * (left + right)

        if (not reacquire) and abs(center - predicted) > max_jump_px:
            gap_len += 1
            continue

        centers[i] = center
        widths[i] = right - left
        scores[i] = score
        last_valid = center
        gap_len = 0

    return centers, widths, scores


def make_laser_rows_image() -> tuple[np.ndarray, np.ndarray]:
    h, w = 200, 280
    yy, xx = np.mgrid[0:h, 0:w]
    img = 18.0 + 0.03 * xx + 0.01 * yy

    gt = np.full(h, np.nan, dtype=np.float32)
    for y in range(h):
        center = 120.0 + 0.1 * (y - h / 2.0) + 6.0 * math.sin(2.0 * math.pi * y / 90.0)
        width = 5.4

        if 76 <= y <= 102:
            continue

        gt[y] = center
        left = center - 0.5 * width
        right = center + 0.5 * width

        s_left = 1.0 / (1.0 + np.exp(-(xx[y] - left) * 2.8))
        s_right = 1.0 / (1.0 + np.exp((xx[y] - right) * 2.8))
        img[y] += 215.0 * (s_left * s_right)

        if 20 < y < 170 and y % 9 == 0:
            r_center = center + 28.0
            l2 = r_center - 2.5
            r2 = r_center + 2.5
            rs_left = 1.0 / (1.0 + np.exp(-(xx[y] - l2) * 2.8))
            rs_right = 1.0 / (1.0 + np.exp((xx[y] - r2) * 2.8))
            img[y] += 85.0 * (rs_left * rs_right)

    img += RNG.normal(0.0, 1.2, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), gt


def make_laser_cols_image() -> tuple[np.ndarray, np.ndarray]:
    h, w = 220, 260
    yy, xx = np.mgrid[0:h, 0:w]
    img = 16.0 + 0.02 * xx + 0.025 * yy

    gt = np.full(w, np.nan, dtype=np.float32)
    for x in range(w):
        center = 108.0 + 0.12 * (x - w / 2.0) + 5.5 * math.sin(2.0 * math.pi * x / 85.0)
        width = 5.1

        if 112 <= x <= 136:
            continue

        gt[x] = center
        top = center - 0.5 * width
        bottom = center + 0.5 * width

        s_top = 1.0 / (1.0 + np.exp(-(yy[:, x] - top) * 2.8))
        s_bot = 1.0 / (1.0 + np.exp((yy[:, x] - bottom) * 2.8))
        img[:, x] += 210.0 * (s_top * s_bot)

        if 30 < x < 225 and x % 10 == 0:
            refl = center - 25.0
            t2 = refl - 2.2
            b2 = refl + 2.2
            rs_top = 1.0 / (1.0 + np.exp(-(yy[:, x] - t2) * 2.8))
            rs_bot = 1.0 / (1.0 + np.exp((yy[:, x] - b2) * 2.8))
            img[:, x] += 75.0 * (rs_top * rs_bot)

    img += RNG.normal(0.0, 1.2, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), gt


def make_rows_figure() -> None:
    img, gt = make_laser_rows_image()
    centers, _, _ = track_stripe_centers(img, axis="rows")

    y = np.arange(img.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    # Keep NaNs to break the polyline at missing stripe segments.
    ax.plot(gt, y, color="#00d4ff", lw=2.0, label="ground truth")

    tracked_mask = np.isfinite(centers) & np.isfinite(gt)
    ax.scatter(
        centers[tracked_mask],
        y[tracked_mask],
        s=16,
        c="#ff4d4d",
        edgecolors="white",
        linewidths=0.3,
        alpha=1.0,
        zorder=5,
        label="tracked center",
    )
    ax.set_title("Laser extraction (Rows): coarse->ROI edge-pair tracking")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right")

    save_fig(fig, "laser_rows.png")


def make_cols_figure() -> None:
    img, gt = make_laser_cols_image()
    centers, _, _ = track_stripe_centers(img, axis="cols")

    x = np.arange(img.shape[1])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    # Keep NaNs to break the polyline at missing stripe segments.
    ax.plot(x, gt, color="#00d4ff", lw=2.0, label="ground truth")

    tracked_mask = np.isfinite(centers) & np.isfinite(gt)
    ax.scatter(
        x[tracked_mask],
        centers[tracked_mask],
        s=16,
        c="#ff4d4d",
        edgecolors="white",
        linewidths=0.3,
        alpha=1.0,
        zorder=5,
        label="tracked center",
    )
    ax.set_title("Laser extraction (Cols): gather + edge-pair tracking")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right")

    save_fig(fig, "laser_cols.png")
