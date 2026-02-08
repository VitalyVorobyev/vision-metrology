from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .common import downsample2x2_mean, save_fig
from .plot import plt


def make_figure() -> None:
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
    draw.polygon([(104, 24), (118, 24), (118, 40), (104, 55)], fill=225)

    img = np.asarray(pil, dtype=np.float32)

    levels = [img]
    for _ in range(4):
        levels.append(downsample2x2_mean(levels[-1]))

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    for i, (ax, lv) in enumerate(zip(axes, levels)):
        ax.imshow(lv, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"L{i}: {lv.shape[1]}x{lv.shape[0]}")
        ax.axis("off")

    fig.suptitle("2x2 mean pyramid levels (drop-odd policy)", fontsize=12)
    save_fig(fig, "pyramid_levels.png")
