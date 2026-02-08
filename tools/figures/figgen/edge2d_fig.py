from __future__ import annotations

import numpy as np

from .common import downsample2x2_mean, save_fig
from .plot import plt


def scharr_grad(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img.shape
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)

    for y in range(h):
        ym1 = max(0, y - 1)
        yp1 = min(h - 1, y + 1)
        for x in range(w):
            xm1 = max(0, x - 1)
            xp1 = min(w - 1, x + 1)

            p00 = img[ym1, xm1]
            p01 = img[ym1, x]
            p02 = img[ym1, xp1]
            p10 = img[y, xm1]
            p12 = img[y, xp1]
            p20 = img[yp1, xm1]
            p21 = img[yp1, x]
            p22 = img[yp1, xp1]

            gx[y, x] = (3 * p02 + 10 * p12 + 3 * p22) - (3 * p00 + 10 * p10 + 3 * p20)
            gy[y, x] = (3 * p20 + 10 * p21 + 3 * p22) - (3 * p00 + 10 * p01 + 3 * p02)

    mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, mag


def nms_quantized(gx: np.ndarray, gy: np.ndarray, mag: np.ndarray) -> np.ndarray:
    h, w = mag.shape
    out = np.zeros_like(mag, dtype=np.float32)
    tan22 = 0.41421357
    tan67 = 2.4142137

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            m = float(mag[y, x])
            if m <= 0.0:
                continue

            gxx = float(gx[y, x])
            gyy = float(gy[y, x])
            ax, ay = abs(gxx), abs(gyy)

            if ay <= ax * tan22:
                n1 = mag[y, x - 1]
                n2 = mag[y, x + 1]
            elif ay >= ax * tan67:
                n1 = mag[y - 1, x]
                n2 = mag[y + 1, x]
            elif gxx * gyy > 0.0:
                n1 = mag[y - 1, x - 1]
                n2 = mag[y + 1, x + 1]
            else:
                n1 = mag[y - 1, x + 1]
                n2 = mag[y + 1, x - 1]

            if m >= n1 and m >= n2:
                out[y, x] = m

    return out


def make_figure() -> None:
    h, w = 180, 260
    yy, xx = np.mgrid[0:h, 0:w]

    img = np.where(xx + 0.55 * yy > 165.0, 35.0, 165.0).astype(np.float32)

    img[40:45, 130:220] = 240
    img[95:100, 130:220] = 240
    img[40:100, 130:135] = 240
    img[40:100, 215:220] = 240

    img = downsample2x2_mean(np.repeat(np.repeat(img, 2, axis=0), 2, axis=1))
    gx, gy, mag = scharr_grad(img)
    nms = nms_quantized(gx, gy, mag)

    nz = nms[nms > 0]
    thresh = float(np.percentile(nz, 78)) if nz.size else 0.0
    ys, xs = np.nonzero(nms >= thresh)

    m = mag[ys, xs] + 1e-9
    nx = gx[ys, xs] / m
    ny = gy[ys, xs] / m

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.scatter(xs, ys, s=5, c="#ff2e63", alpha=0.8, label="edgels")

    sparse = np.arange(0, len(xs), 15)
    ax.quiver(
        xs[sparse],
        ys[sparse],
        nx[sparse],
        ny[sparse],
        color="#00f5d4",
        scale=35,
        width=0.003,
        headwidth=3,
        headlength=4,
        label="normals",
    )

    ax.set_title("2D edgels overlay (NMS + gradient normals)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right")

    save_fig(fig, "edgels_overlay.png")
