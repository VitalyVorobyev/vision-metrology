from __future__ import annotations

import math

import numpy as np

from .config import FIG_DIR, ROOT
from .plot import plt


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def conv1d_clamp(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = len(kernel) // 2
    n = len(signal)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        acc = 0.0
        for k, kv in enumerate(kernel):
            j = i + (k - radius)
            j = 0 if j < 0 else (n - 1 if j >= n else j)
            acc += kv * float(signal[j])
        out[i] = acc
    return out


def binomial_blur_1d(signal: np.ndarray, passes: int = 1) -> np.ndarray:
    out = signal.astype(np.float32, copy=True)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    for _ in range(passes):
        out = conv1d_clamp(out, kernel)
    return out


def dog_kernel_1d(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)

    g_unnorm = np.exp(-(x * x) / (2.0 * sigma * sigma))
    g = g_unnorm / np.sum(g_unnorm)
    dg = -(x / (sigma * sigma)) * g_unnorm

    return g.astype(np.float32), dg.astype(np.float32)


def parabolic_offset(y_m1: float, y0: float, y_p1: float) -> float:
    denom = y_m1 - 2.0 * y0 + y_p1
    if abs(denom) < 1e-12:
        return 0.0

    t = 0.5 * (y_m1 - y_p1) / denom
    if not math.isfinite(t):
        return 0.0

    return float(np.clip(t, -1.0, 1.0))


def downsample2x2_mean(img: np.ndarray) -> np.ndarray:
    h2, w2 = img.shape[0] // 2, img.shape[1] // 2
    if h2 == 0 or w2 == 0:
        return np.zeros((0, 0), dtype=np.float32)

    a = img[0 : 2 * h2 : 2, 0 : 2 * w2 : 2]
    b = img[1 : 2 * h2 : 2, 0 : 2 * w2 : 2]
    c = img[0 : 2 * h2 : 2, 1 : 2 * w2 : 2]
    d = img[1 : 2 * h2 : 2, 1 : 2 * w2 : 2]
    return (a + b + c + d) * 0.25
