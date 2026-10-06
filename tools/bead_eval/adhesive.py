"""Track beads from priors you supply, on images you downloaded yourself, and draw the
result: for a local, qualitative look at real adhesive beads (GAN_Synth_Adhesive's
`adhesive1024` release, see the README) or any other images.

`--priors` is a JSON file mapping an image path, relative to `--data-dir`, to its priors:

    {"<image file>": [{"points": [[x, y], ...], "polarity": "light" | "dark",
                       "min_width": 20.0, "max_width": 60.0}]}

Points are pixel coordinates with pixel centres at integers. The image is converted to
BT.601 luma as float32 on the 8-bit scale. Every other setting is the library's default
unless given on the command line. Writes `<out-dir>/adhesive/<image stem>_<k>.png`
(prior, refined centreline, final edges, rejected stations) and `<out-dir>/adhesive.json`
(each bead's summary).

    python -I tools/bead_eval/adhesive.py --data-dir /path/to/adhesive1024 --priors priors.json
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

# Isolated mode (-I) leaves the script's own directory off sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import vision_metrology as vm  # noqa: E402

import common  # noqa: E402


def draw(path: pathlib.Path, img: np.ndarray, prior: np.ndarray, bead, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.plot(prior[:, 0], prior[:, 1], "--", color="white", lw=1, label="prior")
    cl = np.asarray(bead.centerline)
    ax.plot(cl[:, 0], cl[:, 1], "-", color="lime", lw=1.2, label="refined centreline")
    for key, color in (("first", "yellow"), ("second", "orange")):
        e = np.asarray(getattr(bead, key))
        ax.plot(e[:, 0], e[:, 1], ".", color=color, ms=2, label=f"{key} edge")
    rej = np.array([r is not None for r in bead.reject])
    ax.plot(cl[rej, 0], cl[rej, 1], "o", color="red", ms=3, label="rejected station")
    pad = 40
    lo, hi = cl.min(0) - pad, cl.max(0) + pad
    ax.set_xlim(max(lo[0], -0.5), min(hi[0], img.shape[1] - 0.5))
    ax.set_ylim(min(hi[1], img.shape[0] - 0.5), max(lo[1], -0.5))
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = common.add_tracker_args(
        common.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    )
    # The library's own defaults, for beads tens of pixels wide.
    parser.set_defaults(reach=15.0, spacing=4.0, threshold=5.0, obliquity=30.0)
    parser.add_argument("--priors", type=pathlib.Path, required=True, help="the priors JSON file")
    args = parser.parse_args()
    out_dir = common.resolve_out_dir(args)
    odir = out_dir / "adhesive"
    odir.mkdir(parents=True, exist_ok=True)
    priors = common.read_json(args.priors)

    summary = {}
    for name, beads in priors.items():
        img = common.load_gray(args.data_dir / name)
        for k, b in enumerate(beads):
            prior = np.asarray(b["points"], dtype=np.float64)
            cfg = common.bead_config(b.get("polarity", "light"), b["min_width"], b["max_width"], args)
            bead = vm.BeadTracker(cfg).track(img, prior)
            key = f"{name}#{k}"
            summary[key] = {
                "stations": len(bead.reject),
                "support": bead.support,
                "longest_gap": bead.longest_gap,
                "center_rms": bead.center_rms,
                "center_max_dev": bead.center_max_dev,
                "width_mean": bead.width_mean,
                "width_std": bead.width_std,
                "stop": bead.stop,
                "rejects": dict(collections.Counter(r for r in bead.reject if r)),
            }
            s = summary[key]
            width = "–" if s["width_mean"] is None else f"{s['width_mean']:.1f}"
            title = f"{key}: support {s['support']:.2f}, width {width} px, stop {s['stop']}"
            draw(odir / f"{pathlib.Path(name).stem}_{k}.png", img, prior, bead, title)
            print(f"{key}: {s}")
    common.write_json(out_dir / "adhesive.json", summary)
    print(f"wrote {out_dir / 'adhesive.json'} and {odir}")


if __name__ == "__main__":
    main()
