"""Acquisition without a prior, evaluated: can ridge-based candidate centrelines seed the
tracker?

Sources (`acquire.py`): scikit-image's `sato` and `meijering` filters, thresholded and
skeletonised through `paths.py`'s graph, and the optional `ridge-detector` package. Data:
synthetic ribbons with exact truth (`ribbons.py`, 1280x1024), and the DamSegment cracks
with `paths.py`'s reference paths (640x640). Per source and threshold setting:

- **recall**: the fraction of reference paths with at least `COVER` of their length within
  `max(2 px, w/2)` of an acquired path. **One-path recall** asks the same of a single
  acquired path, which is what a tracker prior needs;
- **precision**: the fraction of acquired length near a reference. On the synthetic
  ribbons that is within `max(2 px, w/2)` of the centreline; on DamSegment, within 1 px of
  the crack mask (`ON_MASK_PX` from mask pixel centres), so every annotated crack counts,
  and an unannotated dark line does not;
- **centre error**: the distance of each acquired point near a reference to it;
- **width**: the source's width there, against the truth (synthetic) or the mask width
  `2 * EDT - 1` (DamSegment, where the masks are about twice as wide as the dark line);
- **time** per image: the whole acquisition, and the filter alone.

**Acquire, then track.** Each reference path that one acquired path covers is tracked
three times with the same config: from the acquired path, cut to its longest run within
the tolerance of the reference; from the reference itself; and from the reference
translated by `TRANSLATE_PX` (the seeded sign of `priors.py`). The metrics are
`report.py`'s: the centre distance of the refined curve, support, locked, and converged
against the run from the reference itself. This uses each source's default setting.

    python -I tools/bead_eval/acquire_eval.py --data-dir data/damsegment/extracted --workers 6
    python -I tools/bead_eval/acquire_eval.py --out-dir /tmp/acq --datasets synthetic

DamSegment needs `paths.py`'s output in the output directory. Writes
`acquire_report.md`, `acquire_report.json` and, for three synthetic scenes,
`acquire_overlays/*.png` there. Times are per image inside each worker: run with
`--workers 1` when they matter.
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
from scipy import ndimage  # noqa: E402

import acquire  # noqa: E402
import acquire_report  # noqa: E402
import common  # noqa: E402
import priors as priors_mod  # noqa: E402
import report as report_mod  # noqa: E402
import ribbons  # noqa: E402
import run_tracker  # noqa: E402

COVER = 0.8
ON_MASK_PX = 1.5
TRANSLATE_PX = 4.0
#: The DamSegment cracks' expected dark-line widths, px.
CRACK_WIDTHS = [3.0, 5.0, 8.0]
OVERLAY_SCENES = ("sine_w8_distractor_0", "arc_w30_gap_0", "line_w3_noisy_0")


# ---------------------------------------------------------------------------
# Scoring one image
# ---------------------------------------------------------------------------
def tolerance(width: np.ndarray) -> np.ndarray:
    return np.maximum(2.0, 0.5 * np.asarray(width, dtype=np.float64))


def score(refs: list[dict], found: list[dict], on_mask=None) -> dict:
    """Recall, precision, centre and width errors of one source's paths on one image.
    `refs` have `points` (N, 2) at 1 px and `width` (N,); `on_mask(points)` decides
    precision when given, the reference tolerance otherwise."""
    acq = common.PolylineIndex([p["points"] for p in found])
    cover, cover_one, best = [], [], []
    for r in refs:
        d, which, _, _ = acq.query(r["points"])
        near = d <= tolerance(r["width"])
        cover.append(float(near.mean()))
        counts = np.bincount(which[near], minlength=max(1, len(found)))
        best.append(int(np.argmax(counts)) if near.any() else -1)
        cover_one.append(float(counts.max() / len(near)) if near.any() else 0.0)
    out = {
        "refs": len(refs),
        "cover": np.array(cover),
        "cover_one": np.array(cover_one),
        "best": best,
        "paths": len(found),
        "length": 0.0,
        "length_near": 0.0,
        "err": np.zeros(0, np.float32),
        "w_est": np.zeros(0, np.float32),
        "w_ref": np.zeros(0, np.float32),
    }
    if not found or not refs:
        return out
    pts = np.concatenate([p["points"] for p in found])
    w_est = np.concatenate([p["width"] for p in found])
    d, which, s, _ = common.PolylineIndex([r["points"] for r in refs]).query(pts)
    w_ref = np.full(len(pts), np.nan)
    for i, r in enumerate(refs):
        sel = which == i
        w_ref[sel] = np.interp(s[sel], common.arc_length(r["points"]), r["width"])
    near = d <= tolerance(np.nan_to_num(w_ref, nan=0.0))
    precise = on_mask(pts) if on_mask is not None else near
    out.update(
        {
            "length": float(len(pts)),
            "length_near": float(precise.sum()),
            "err": d[near].astype(np.float32),
            "w_est": w_est[near].astype(np.float32),
            "w_ref": w_ref[near].astype(np.float32),
        }
    )
    return out


def crop_to(path: np.ndarray, ref: dict) -> np.ndarray | None:
    """The longest run of `path` within the tolerance of the reference, if at least
    20 px long."""
    d, _, s, _ = common.PolylineIndex([ref["points"]]).query(path)
    w = np.interp(np.nan_to_num(s), common.arc_length(ref["points"]), ref["width"])
    near = np.concatenate([[0], (d <= tolerance(w)).astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(near))
    starts, ends = edges[::2], edges[1::2]
    if len(starts) == 0:
        return None
    k = int(np.argmax(ends - starts))
    run = path[starts[k] : ends[k]]
    return run if len(run) >= 2 and common.arc_length(run)[-1] >= 20.0 else None


def track_three(tracker, img, ref: dict, crops: dict[str, np.ndarray]) -> list[dict]:
    """The reference, its translation and each source's crop as priors, scored with
    `report.call_metrics` against the run from the reference."""
    spec = priors_mod.draw_spec(ref["id"], "translate", TRANSLATE_PX)
    priors = {"reference": ref["points"], "translated": priors_mod.build_prior(ref["points"], spec)}
    priors.update(crops)
    calls = {}
    for name, prior in priors.items():
        t = time.perf_counter()
        bead = tracker.track(img, np.asarray(prior, dtype=np.float32))
        calls[name] = run_tracker.record(ref["id"], 0, bead, time.perf_counter() - t)
    anchor = calls["reference"]["centerline"].astype(np.float64)
    rows = []
    for name, call in calls.items():
        call = dict(call, centerline=call["centerline"].astype(np.float64), width=call["width"].astype(np.float64))
        m = report_mod.call_metrics(ref["points"], ref["width"], np.asarray(priors[name]), call, anchor)
        rows.append(
            {
                "prior": name,
                "ref": ref["id"],
                "d": m["d"],
                "locked": m["locked"],
                "converged": m["converged"],
                "support": m["support"],
                "prior_on": m["prior_on"],
                "stop": m["stop"],
            }
        )
    return rows


def run_sources(img: np.ndarray, refs: list[dict], widths, dark, args, on_mask=None) -> tuple[dict, dict]:
    """Every source and setting on one image: the scores, and the default setting's paths."""
    scores, defaults = {}, {}
    for src in args.sources:
        per = acquire.acquire(img, src, widths, dark, args)
        scores[src] = {}
        for label, (found, info) in per.items():
            sc = score(refs, found, on_mask)
            sc.update({"time_ms": info["time_ms"], "filter_ms": info.get("filter_ms", np.nan)})
            scores[src][label] = sc
        defaults[src] = (per[args.default[src]][0], scores[src][args.default[src]])
    return scores, defaults


def crops_for(ref_i: int, ref: dict, defaults: dict) -> dict[str, np.ndarray]:
    out = {}
    for src, (found, sc) in defaults.items():
        if sc["cover_one"][ref_i] >= COVER:
            crop = crop_to(found[sc["best"][ref_i]]["points"], ref)
            if crop is not None:
                out[src] = crop
    return out


# ---------------------------------------------------------------------------
# The two datasets, one job per image
# ---------------------------------------------------------------------------
def warm_up(args: argparse.Namespace) -> None:
    """Compile `ridge-detector`'s kernels before anything is timed."""
    if "ridge_detector" in args.sources:
        img = np.full((64, 64), 80, np.uint8)
        img[30:34, :] = 140
        acquire.acquire(img, "ridge_detector", [4.0], False, args)


def synthetic_job(job: tuple) -> dict:
    import vision_metrology as vm

    scene, args = job
    warm_up(args)
    img, truth = ribbons.render(scene)
    sid = ribbons.scene_id(scene)
    w = truth["width"]
    ref = {"id": sid, "points": truth["points"], "width": np.full(len(truth["points"]), w)}
    scores, defaults = run_sources(img, [ref], [w], False, args)
    cfg = vm.BeadConfig(polarity="light", min_width=0.5 * w, max_width=1.5 * w)
    rows = track_three(vm.BeadTracker(cfg), img, ref, crops_for(0, ref, defaults))
    group = {"shape": scene["shape"], "width": w, "condition": scene["condition"]}
    overlay = {s: d[0] for s, d in defaults.items()} if sid in OVERLAY_SCENES else None
    return {"id": sid, "group": group, "scores": scores, "track": rows, "overlay": overlay}


def damsegment_job(job: tuple) -> dict:
    import vision_metrology as vm

    root, out_dir, rel, args = job
    warm_up(args)
    doc = common.read_json(out_dir / "paths" / rel)
    gray = common.load_gray(root / doc["image"])
    img = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
    mask = common.load_crack_mask(root / doc["mask"])
    edt = ndimage.distance_transform_edt(~mask)

    def on_mask(pts: np.ndarray) -> np.ndarray:
        return ndimage.map_coordinates(edt, [pts[:, 1], pts[:, 0]], order=1, mode="nearest") <= ON_MASK_PX

    refs = [
        {"id": p["id"], "points": np.asarray(p["points"], float), "width": np.asarray(p["width"], float)}
        for p in doc["paths"]
    ]
    scores, defaults = run_sources(img, refs, CRACK_WIDTHS, True, args)
    rows = []
    for i, (ref, p) in enumerate(zip(refs, doc["paths"])):
        lo, hi = run_tracker.width_range(p["width_median"], args)
        tracker = vm.BeadTracker(common.bead_config("dark", lo, hi, args))
        rows += track_three(tracker, gray, ref, crops_for(i, ref, defaults))
    return {"id": rel, "group": {"difficulty": doc["difficulty"]}, "scores": scores, "track": rows, "overlay": None}


def image_jobs(index: dict, max_images: int | None) -> list[str]:
    """The per-image paths files with at least one reference path, or a seeded subset of
    at most `max_images` per difficulty."""
    out = []
    for diff in common.DIFFICULTIES:
        files = sorted({e["file"] for e in index["paths"] if e["difficulty"] == diff})
        if max_images is not None and len(files) > max_images:
            pick = np.sort(common.rng_for("acquire", diff).choice(len(files), max_images, replace=False))
            files = [files[i] for i in pick]
        out += files
    return out


def run_jobs(fn, jobs: list, workers: int, label: str) -> list[dict]:
    out = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, res in enumerate(pool.map(fn, jobs)):
            out.append(res)
            if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                print(f"{label}: {k + 1}/{len(jobs)}, {time.perf_counter() - t0:.0f} s", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=None, help="the DamSegment folder")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None, help="default <data-dir>/output")
    parser.add_argument(
        "--datasets", nargs="+", default=["synthetic", "damsegment"], choices=["synthetic", "damsegment"]
    )
    parser.add_argument("--sources", nargs="+", default=list(acquire.SOURCES), choices=acquire.SOURCES)
    parser.add_argument("--seeds", type=int, default=2, help="noise draws per synthetic scene (default 2)")
    parser.add_argument("--max-images", type=int, default=None, help="seeded DamSegment subset per difficulty")
    parser.add_argument("--workers", type=int, default=1, help="images in parallel (default 1)")
    parser.add_argument("--width-lo", type=float, default=0.25, help="as run_tracker.py (default 0.25)")
    parser.add_argument("--width-hi", type=float, default=2.0, help="as run_tracker.py (default 2)")
    parser.add_argument("--width-floor", type=float, default=2.0, help="as run_tracker.py (default 2)")
    common.add_tracker_args(acquire.add_source_args(parser))
    args = parser.parse_args()
    if args.out_dir is None and args.data_dir is None:
        parser.error("give --data-dir or --out-dir")
    out_dir = args.out_dir if args.out_dir is not None else args.data_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped = [s for s in args.sources if not acquire.available(s)]
    for s in skipped:
        print(f"{s} is not installed: skipped (pip install -r requirements-baselines.txt)")
    args.sources = [s for s in args.sources if s not in skipped]
    labels = {s: acquire.settings(s, args) for s in args.sources}
    args.default = {s: acquire.default_setting(s, args) for s in args.sources}

    results = {}
    if "synthetic" in args.datasets:
        jobs = [(sc, args) for sc in ribbons.scenes(args.seeds)]
        results["synthetic"] = run_jobs(synthetic_job, jobs, args.workers, "synthetic")
    if "damsegment" in args.datasets:
        if args.data_dir is None:
            parser.error("DamSegment needs --data-dir")
        root = common.dataset_root(args.data_dir)
        index = common.read_json(out_dir / "paths" / "index.json")
        jobs = [(root, out_dir, rel, args) for rel in image_jobs(index, args.max_images)]
        results["damsegment"] = run_jobs(damsegment_job, jobs, args.workers, "damsegment")

    meta = {
        "sources": args.sources,
        "skipped": skipped,
        "settings": labels,
        "default": args.default,
        "cover": COVER,
        "on_mask_px": ON_MASK_PX,
        "translate_px": TRANSLATE_PX,
        "crack_widths": CRACK_WIDTHS,
        "source_args": {
            k: getattr(args, k)
            for k in ("min_length", "spur_px", "junction_trim", "smooth_px", "k", "rd_contrast")
        },
        "tracker_args": {
            k: getattr(args, k) for k in ("reach", "spacing", "threshold", "sigma", "obliquity")
        },
    }
    acquire_report.write(out_dir, results, meta)
    for res in results.get("synthetic", []):
        if res["overlay"] is not None:
            acquire_report.overlay(out_dir, res)
    print(f"wrote {out_dir / 'acquire_report.md'}")


if __name__ == "__main__":
    main()
