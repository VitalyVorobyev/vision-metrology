#!/usr/bin/env python3
"""Deterministic external fixtures for vm-gallery."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SEED = 20260208
RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[2]
FIX_ROOT = ROOT / "docs" / "fig" / "fixtures"


def ensure_case_dir(case: str) -> Path:
    out = FIX_ROOT / case
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_luma(case_dir: Path, arr: np.ndarray) -> None:
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img.save(case_dir / "input.png")


def write_truth(case_dir: Path, case: str, width: int, height: int, truth: dict, notes: str = "") -> None:
    payload = {
        "case": case,
        "width": int(width),
        "height": int(height),
        "notes": notes,
        "truth": truth,
    }
    (case_dir / "truth.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def blur1d(signal: np.ndarray, passes: int = 1) -> np.ndarray:
    out = signal.astype(np.float32).copy()
    for _ in range(passes):
        pad = np.pad(out, (1, 1), mode="edge")
        out = (pad[:-2] + 2.0 * pad[1:-1] + pad[2:]) * 0.25
    return out


def blur2d_binomial(img: np.ndarray, passes: int = 2) -> np.ndarray:
    """Separable [1, 2, 1] / 4 smoothing; passes=2 is ~sigma 1 px."""
    out = img.astype(np.float32).copy()
    for _ in range(passes):
        pad_x = np.pad(out, ((0, 0), (1, 1)), mode="edge")
        out = (pad_x[:, :-2] + 2.0 * pad_x[:, 1:-1] + pad_x[:, 2:]) * 0.25
        pad_y = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        out = (pad_y[:-2, :] + 2.0 * pad_y[1:-1, :] + pad_y[2:, :]) * 0.25
    return out


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return img.astype(np.float32, copy=True)
    return img.astype(np.float32) + RNG.normal(0.0, sigma, size=img.shape)


def gen_morphology() -> None:
    case = "morphology"
    h, w = 160, 220
    yy, xx = np.mgrid[0:h, 0:w]

    mask = np.zeros((h, w), dtype=bool)
    mask[(xx >= 20) & (xx <= 90) & (yy >= 25) & (yy <= 120)] = True
    mask[((xx - 150) ** 2 + (yy - 85) ** 2) <= 35**2] = True
    mask[(xx >= 45) & (xx <= 170) & (yy >= 130) & (yy <= 150)] = True

    salt = RNG.random((h, w)) < 0.007
    holes = RNG.random((h, w)) < 0.006
    noisy = (mask | salt) & (~(mask & holes))

    out = noisy.astype(np.uint8) * 255
    case_dir = ensure_case_dir(case)
    save_luma(case_dir, out)
    write_truth(case_dir, case, w, h, {"binary": True}, notes="binary mask with holes and speckle")


def gen_pyramid() -> None:
    case = "pyramid"
    h, w = 200, 320
    yy, xx = np.mgrid[0:h, 0:w]

    img = np.where(xx + 0.7 * yy > 180.0, 35.0, 135.0).astype(np.float32)
    stripe_center = 130.0 + 0.08 * (yy - h / 2.0)
    img += 90.0 * np.exp(-((xx - stripe_center) ** 2) / (2.0 * 4.0**2))

    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(pil)
    draw.rectangle((28, 24, 38, 86), fill=210)
    draw.rectangle((58, 24, 68, 86), fill=210)
    draw.rectangle((38, 62, 58, 72), fill=210)
    draw.rectangle((92, 24, 104, 86), fill=225)
    draw.rectangle((118, 24, 130, 86), fill=225)

    out = np.asarray(pil, dtype=np.uint8)
    case_dir = ensure_case_dir(case)
    save_luma(case_dir, out)
    write_truth(case_dir, case, w, h, {}, notes="slanted edge + stripe + block glyphs")


def gen_edge1d() -> None:
    case = "edge1d"
    w = 192
    x = np.arange(w, dtype=np.float32)
    x_left = 54.3
    x_right = 98.7

    ideal = np.where((x >= x_left) & (x <= x_right), 1.0, 0.0).astype(np.float32)
    signal = np.minimum(255.0 * ideal, 255.0)
    signal = blur1d(signal, passes=3)
    signal = add_gaussian_noise(signal, sigma=0.8)

    row = np.clip(signal, 0, 255).astype(np.uint8)[None, :]
    case_dir = ensure_case_dir(case)
    save_luma(case_dir, row)
    write_truth(
        case_dir,
        case,
        w,
        1,
        {
            "x_left": float(x_left),
            "x_right": float(x_right),
            "center": float(0.5 * (x_left + x_right)),
            "width": float(x_right - x_left),
        },
        notes="single-row saturated stripe signal with blur (~1 px) and light noise",
    )


def gen_laser_rows() -> None:
    case = "laser_rows"
    h, w = 200, 280
    yy, xx = np.mgrid[0:h, 0:w]
    img = 18.0 + 0.03 * xx + 0.01 * yy

    centers: list[float | None] = [None] * h
    widths: list[float | None] = [None] * h

    for y in range(h):
        center = 120.0 + 0.1 * (y - h / 2.0) + 6.0 * math.sin(2.0 * math.pi * y / 90.0)
        width = 5.4
        if 76 <= y <= 102:
            continue

        centers[y] = float(center)
        widths[y] = float(width)

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

    img = blur2d_binomial(img, passes=2)
    img = add_gaussian_noise(img, sigma=1.4)
    out = np.clip(img, 0, 255).astype(np.uint8)

    case_dir = ensure_case_dir(case)
    save_luma(case_dir, out)
    write_truth(
        case_dir,
        case,
        w,
        h,
        {"axis": "rows", "true_center": centers, "true_width": widths},
        notes="vertical-ish bright stripe with gap/reflections, blur (~1 px), light noise",
    )


def gen_laser_cols() -> None:
    case = "laser_cols"
    h, w = 220, 260
    yy, xx = np.mgrid[0:h, 0:w]
    img = 16.0 + 0.02 * xx + 0.025 * yy

    centers: list[float | None] = [None] * w
    widths: list[float | None] = [None] * w

    for x in range(w):
        center = 108.0 + 0.12 * (x - w / 2.0) + 5.5 * math.sin(2.0 * math.pi * x / 85.0)
        width = 5.1
        if 112 <= x <= 136:
            continue

        centers[x] = float(center)
        widths[x] = float(width)

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

    img = blur2d_binomial(img, passes=2)
    img = add_gaussian_noise(img, sigma=1.4)
    out = np.clip(img, 0, 255).astype(np.uint8)

    case_dir = ensure_case_dir(case)
    save_luma(case_dir, out)
    write_truth(
        case_dir,
        case,
        w,
        h,
        {"axis": "cols", "true_center": centers, "true_width": widths},
        notes="horizontal-ish bright stripe with gap/reflections, blur (~1 px), light noise",
    )


def gen_edgels2d() -> None:
    case = "edgels2d"
    h, w = 180, 260
    yy, xx = np.mgrid[0:h, 0:w]

    img = np.where(xx + 0.55 * yy > 165.0, 35.0, 165.0).astype(np.float32)
    img[40:45, 130:220] = 240
    img[95:100, 130:220] = 240
    img[40:100, 130:135] = 240
    img[40:100, 215:220] = 240

    img = blur2d_binomial(img, passes=2)
    img = add_gaussian_noise(img, sigma=1.3)
    out = np.clip(img, 0, 255).astype(np.uint8)
    case_dir = ensure_case_dir(case)
    save_luma(case_dir, out)
    write_truth(
        case_dir,
        case,
        w,
        h,
        {},
        notes="slanted step plus rectangle with blur (~1 px) and light noise",
    )


def gen_contour_graph() -> None:
    case = "contour_graph"
    w, h = 240, 170
    img = Image.new("L", (w, h), color=28)
    draw = ImageDraw.Draw(img)

    # T-shape
    draw.line((40, 20, 40, 120), fill=255, width=7)
    draw.line((40, 65, 110, 65), fill=255, width=7)

    # Y-shape
    draw.line((170, 20, 170, 75), fill=255, width=7)
    draw.line((170, 75, 130, 130), fill=255, width=7)
    draw.line((170, 75, 210, 130), fill=255, width=7)

    img_f = np.asarray(img, dtype=np.float32)
    img_f = blur2d_binomial(img_f, passes=2)
    img_f = add_gaussian_noise(img_f, sigma=1.1)
    out = np.clip(img_f, 0, 255).astype(np.uint8)
    case_dir = ensure_case_dir(case)
    save_luma(case_dir, out)
    write_truth(
        case_dir,
        case,
        w,
        h,
        {},
        notes="thick T/Y structures with blur (~1 px) and light noise",
    )


def main() -> None:
    FIX_ROOT.mkdir(parents=True, exist_ok=True)

    gen_morphology()
    gen_pyramid()
    gen_edge1d()
    gen_laser_rows()
    gen_laser_cols()
    gen_edgels2d()
    gen_contour_graph()

    print(f"wrote fixtures under {FIX_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
