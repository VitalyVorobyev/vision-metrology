"""A baseline on the same priors: `skimage.segmentation.active_contour`, an open snake.

The image is the same BT.601 luma, scaled to [0, 1] and Gaussian-smoothed once per image
(sigma `--smooth`, 3 px by default): the snake's only pull towards the crack is the
gradient of that image, so the smoothing sets its capture range. The snake starts from the
prior resampled at the tracker's spacing, with `w_line = -1` (drawn to dark),
`w_edge = 0` and `boundary_condition="free"`, so neither end is held where the prior put
it. The other parameters are scikit-image's defaults (alpha 0.01, beta 0.1, gamma 0.01,
max_px_move 1, max_num_iter 2500, convergence 0.1). It is a baseline, not tuned per path.

It is much slower than the tracker, so by default it runs on a seeded subset of paths per
difficulty; `report.py` compares the two on exactly those calls. The time is the
`active_contour` call alone; the smoothing is per image and left out. `--workers` runs
images in parallel processes, each call still on one core.

    python -I tools/bead_eval/baselines.py --data-dir data/damsegment/extracted
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Isolated mode (-I) leaves the script's own directory off sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from skimage.filters import gaussian  # noqa: E402
from skimage.segmentation import active_contour  # noqa: E402

import common  # noqa: E402
import priors as priors_mod  # noqa: E402


def snake_image(job: tuple) -> list[dict]:
    """Every call on one image: `job` is (root, out_dir, rel, entries, specs, args)."""
    root, out_dir, rel, entries, specs, args = job
    doc = common.read_json(out_dir / "paths" / rel)
    refs = {p["id"]: p for p in doc["paths"]}
    img = gaussian(common.load_gray(root / doc["image"]) / 255.0, sigma=args.smooth)
    calls = []
    for e in entries:
        ref = np.asarray(refs[e["id"]]["points"], dtype=np.float64)
        for i, spec in enumerate(specs[e["id"]]):
            init = common.resample(priors_mod.build_prior(ref, spec), args.spacing)
            t = time.perf_counter()
            snake = active_contour(
                img,
                init[:, ::-1],  # (row, col)
                w_line=-1.0,
                w_edge=0.0,
                boundary_condition="free",
            )
            dt = time.perf_counter() - t
            n = len(snake)
            calls.append(
                {
                    "path_id": e["id"],
                    "spec": i,
                    "stop": "other",
                    "centerline": snake[:, ::-1].astype(np.float32),
                    "width": np.full(n, np.nan, np.float32),
                    "reject": np.full(n, -1, np.int8),
                    "support": np.nan,
                    "longest_gap": np.nan,
                    "center_rms": np.nan,
                    "center_max_dev": np.nan,
                    "width_mean": np.nan,
                    "passes": np.nan,
                    "time_ms": 1e3 * dt,
                }
            )
    return calls


def main() -> None:
    parser = common.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    parser.add_argument("--max-paths", type=int, default=40, help="seeded subset per difficulty (default 40)")
    parser.add_argument("--smooth", type=float, default=3.0, help="Gaussian sigma, px (default 3)")
    parser.add_argument("--spacing", type=float, default=2.0, help="snake point spacing, px (default 2)")
    parser.add_argument("--workers", type=int, default=1, help="images in parallel (default 1)")
    parser.add_argument("--name", default="active_contour", help="run name under runs/")
    args = parser.parse_args()
    root = common.dataset_root(args.data_dir)
    out_dir = common.resolve_out_dir(args)
    index = common.read_json(out_dir / "paths" / "index.json")
    specs = common.read_json(out_dir / "priors.json")["priors"]
    run_dir = out_dir / "runs" / args.name

    entries = common.select_paths(index, args.max_paths)
    groups = common.by_image(entries)
    jobs = [
        (root, out_dir, rel, group, {e["id"]: specs[e["id"]] for e in group}, args)
        for rel, group in groups.items()
    ]
    n_calls = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, (job, calls) in enumerate(zip(jobs, pool.map(snake_image, jobs))):
            common.save_calls(run_dir / job[2].replace(".json", ".npz"), calls)
            n_calls += len(calls)
            print(f"{k + 1}/{len(jobs)} images, {n_calls} calls, {time.perf_counter() - t0:.0f} s", flush=True)
    common.write_json(
        run_dir / "meta.json",
        {
            "method": "skimage.segmentation.active_contour",
            "label": "active_contour",
            "paths": len(entries),
            "calls": n_calls,
            "max_paths": args.max_paths,
            "workers": args.workers,
            "config": {
                "image": f"BT.601 luma / 255, Gaussian sigma {args.smooth} px",
                "spacing": args.spacing,
                "w_line": -1.0,
                "w_edge": 0.0,
                "boundary_condition": "free",
                "other": "scikit-image defaults",
            },
        },
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
