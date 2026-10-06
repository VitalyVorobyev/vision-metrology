"""Track every prior with `vision_metrology.BeadTracker`, through the Python bindings.

Per reference path: one tracker, `polarity="dark"` (a crack is darker than the concrete),
its width range from the path's mask width `w` (the median of `2 * EDT - 1`):
`min = max(floor, lo * w)` and `max = max(hi * w, min + 1)`, by default `max(2, w/4)` and
`2w`, since the masks are about twice as wide as the dark line. The reach, spacing,
threshold, sigma and final-stage obliquity gate are `common.add_tracker_args`'s; the rest
is the library's default. The image is BT.601 luma as float32 on the 8-bit scale. Each call records the refined centreline, the final stage's width and reject
reason per station, the summary, the stop reason and its wall time. The time is
`track()` alone, binding overhead included (the prior's conversion and the result's
numpy arrays); the tracker is built once per path and reused over its priors.

    python -I tools/bead_eval/run_tracker.py --data-dir data/damsegment/extracted
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# Isolated mode (-I) leaves the script's own directory off sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import vision_metrology as vm  # noqa: E402

import common  # noqa: E402
import priors as priors_mod  # noqa: E402


def width_range(w: float, args: argparse.Namespace) -> tuple[float, float]:
    lo = max(args.width_floor, args.width_lo * w)
    return lo, max(args.width_hi * w, lo + 1.0)


def record(path_id: str, spec: int, bead, seconds: float) -> dict:
    reject = np.array(
        [-1 if r is None else common.REJECTS.index(r) for r in bead.reject], dtype=np.int8
    )

    def opt(v):
        return np.nan if v is None else float(v)

    return {
        "path_id": path_id,
        "spec": spec,
        "stop": bead.stop if bead.stop in common.STOPS else "other",
        "centerline": np.asarray(bead.centerline, dtype=np.float32),
        "width": np.asarray(bead.width, dtype=np.float32),
        "reject": reject,
        "support": float(bead.support),
        "longest_gap": float(bead.longest_gap),
        "center_rms": opt(bead.center_rms),
        "center_max_dev": opt(bead.center_max_dev),
        "width_mean": opt(bead.width_mean),
        "passes": float(len(bead.passes)),
        "time_ms": 1e3 * seconds,
    }


def main() -> None:
    parser = common.add_tracker_args(
        common.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    )
    parser.add_argument("--width-lo", type=float, default=0.25, help="min_width / w (default 0.25)")
    parser.add_argument("--width-hi", type=float, default=2.0, help="max_width / w (default 2)")
    parser.add_argument("--width-floor", type=float, default=2.0, help="px (default 2)")
    parser.add_argument("--max-paths", type=int, default=None, help="seeded subset per difficulty")
    parser.add_argument("--name", default="tracker", help="run name under runs/ (default tracker)")
    args = parser.parse_args()
    root = common.dataset_root(args.data_dir)
    out_dir = common.resolve_out_dir(args)
    index = common.read_json(out_dir / "paths" / "index.json")
    specs = common.read_json(out_dir / "priors.json")["priors"]
    run_dir = out_dir / "runs" / args.name

    entries = common.select_paths(index, args.max_paths)
    groups = common.by_image(entries)
    n_calls = 0
    t0 = time.perf_counter()
    for k, (rel, group) in enumerate(groups.items()):
        doc = common.read_json(out_dir / "paths" / rel)
        refs = {p["id"]: p for p in doc["paths"]}
        img = common.load_gray(root / doc["image"])
        calls = []
        for e in group:
            ref = np.asarray(refs[e["id"]]["points"], dtype=np.float64)
            lo, hi = width_range(e["width_median"], args)
            tracker = vm.BeadTracker(common.bead_config("dark", lo, hi, args))
            for i, spec in enumerate(specs[e["id"]]):
                prior = priors_mod.build_prior(ref, spec)
                t = time.perf_counter()
                bead = tracker.track(img, prior)
                calls.append(record(e["id"], i, bead, time.perf_counter() - t))
        common.save_calls(run_dir / rel.replace(".json", ".npz"), calls)
        n_calls += len(calls)
        if (k + 1) % 100 == 0 or k + 1 == len(groups):
            print(f"{k + 1}/{len(groups)} images, {n_calls} calls, {time.perf_counter() - t0:.0f} s", flush=True)
    common.write_json(
        run_dir / "meta.json",
        {
            "method": "BeadTracker",
            "label": "BeadTracker",
            "paths": len(entries),
            "calls": n_calls,
            "max_paths": args.max_paths,
            "config": {
                "polarity": "dark",
                "width": f"[max({args.width_floor}, {args.width_lo} w), max({args.width_hi} w, min + 1)]",
                "reach": args.reach,
                "spacing": args.spacing,
                "threshold": args.threshold,
                "sigma": args.sigma,
                "measure_obliquity_deg": args.obliquity,
                "other": "library defaults",
            },
            "gray": "BT.601 luma, float32, 8-bit scale",
        },
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
