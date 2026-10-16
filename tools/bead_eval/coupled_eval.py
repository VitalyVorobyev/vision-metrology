"""Coupled two-boundary extraction against today's tracker: lock rate, wrong locks, and
whether a whole-call max-marginal ambiguity predicts the wrong locks better than support.

Four methods per call, on the same image and prior:
- `tracker`: `BeadTracker` with `run_tracker.py`'s settings (DamSegment) or the library's
  defaults with the scene's width range (synthetic);
- `sparse` and `dense`: `coupled.py`, from the same prior, with a reach equal to the
  tracker's and the same width range, threshold and profile sigma;
- `rust`: the same `BeadTracker` with coupled pairing (`BeadConfig.pairing`), the
  library's operator for what `dense` prototypes. `--rust-set` overrides its
  `BeadCoupling` fields and `track_step`, `measure_step`, `passes` and `clearance`; its
  ambiguity and evidence are its first pass's `path_margin` and `path_evidence`.

`--split dev|test` keeps the DamSegment paths of the registry split's images
(`tools/datasets/splits/damsegment.json`) before the seeded subset is drawn; `all`, the
default, is #76's set.

**DamSegment.** The calls of `signals.py`: a seeded subset of `--max-paths` paths per
difficulty and every prior within the reach. `--subset tune` takes a disjoint seeded
subset instead, for choosing the cost settings without touching the evaluation paths.
A call is locked by `report.py`'s test; a false lock is not locked with support of at
least 0.5.

**Synthetic.** `synthetic.py`'s scenes and priors. Locked is the same test against the
true centreline and widths. At every visible station (not in a gap or under the
occluder) a reported wall is compared with the true one: a wall more than 1 px off is a
wrong wall; bias and σ are over the others.

**Signals.** AUC of each per-call score for telling locked from not locked calls (as
`signals.py`), and the wrong-lock rate at equal coverage: each score's threshold is set so
that a given share of all calls is accepted and locked, and the share accepted but not
locked is read off.

    python -I tools/bead_eval/coupled_eval.py --data-dir data/damsegment/extracted --workers 6
    python -I tools/bead_eval/coupled_eval.py --out-dir /tmp/c --datasets synthetic

It writes `<name>.json` and `<name>.md` (default `coupled_report`) to the output directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Isolated mode (-I) leaves the script's own directory off sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import common  # noqa: E402
import coupled  # noqa: E402
import coupled_report  # noqa: E402

METHODS = coupled_report.METHODS
QUANTILES = (0.1, 0.25, 0.5)
MAIN_SCORE = coupled_report.MAIN_SCORE


def score_keys(cfg: coupled.CoupledConfig) -> list[str]:
    """The main score first."""
    keys = [f"amb_q{q:g}_t{t:g}" for t in (cfg.tol, *cfg.extra_tols) for q in QUANTILES]
    return [MAIN_SCORE, *(k for k in keys if k != MAIN_SCORE), "amb_rel", "evidence"]


def coupled_scores(ex: coupled.Extraction, cfg: coupled.CoupledConfig, h: float) -> dict:
    """The ambiguity at each tolerance and quantile; `evidence`, the best path's mean
    gain per px over absence; and `amb_rel`, the main ambiguity score over the path's
    total gain."""
    out = {}
    for t in (cfg.tol, *cfg.extra_tols):
        for q in QUANTILES:
            out[f"amb_q{q:g}_t{t:g}"] = ex.ambiguity(q, t)
    gain = h * len(ex.present) * coupled.absent_cost(cfg) - ex.cost
    out["evidence"] = gain / (h * len(ex.present))
    out["amb_rel"] = out[MAIN_SCORE] / max(gain, 1e-9)
    return out


def as_call(pr: coupled.Profiles, ex: coupled.Extraction, seconds: float) -> dict:
    """A coupled result in `run_tracker.record`'s shape, for `report.call_metrics`."""
    run = 0
    longest = 0
    for p in ex.present:
        run = 0 if p else run + 1
        longest = max(longest, run)
    return {
        "centerline": coupled.centerline(pr, ex),
        "width": np.where(ex.present, ex.right - ex.left, np.nan),
        "reject": np.where(ex.present, -1, common.REJECTS.index("no_pair")).astype(np.int8),
        "support": ex.support,
        "center_rms": np.nan,
        "longest_gap": longest * pr.h,
        "time_ms": 1e3 * seconds,
        "stop": "other",
    }


#: `--rust-set` keys that are not `BeadCoupling` fields.
RUST_KEYS = ("track_step", "measure_step", "passes", "clearance")
SPLIT_FILE = pathlib.Path(__file__).resolve().parents[1] / "datasets" / "splits" / "damsegment.json"


def rust_tracker(base, rust_kw: dict):
    """`base`, a `vm.BeadConfig`, with coupled pairing and `rust_kw`'s overrides. The
    stages' thresholds are per profile sample, so a stage whose step changes keeps its
    threshold per px."""
    import vision_metrology as vm

    kw = dict(rust_kw)
    stages = {"track": base.track, "measure": base.measure}
    for name, st in stages.items():
        step = kw.pop(f"{name}_step", None)
        if step is not None:
            st.threshold = st.threshold / st.step * step
            st.step = step
    tuning = base.tuning
    if "passes" in kw:
        tuning.passes = int(kw.pop("passes"))
    clearance = kw.pop("clearance", None)
    cfg = vm.BeadConfig(
        polarity=base.polarity, min_width=base.min_width, max_width=base.max_width, spacing=base.spacing,
        clearance=clearance if clearance and clearance > 0 else None,
        track=stages["track"], measure=stages["measure"], tuning=tuning, pairing=vm.BeadCoupling(**kw),
    )
    return vm.BeadTracker(cfg)


def rust_call(tracker, img, prior):
    """One coupled `track` call: the bead, its first pass's scores, and the time."""
    t = time.perf_counter()
    bead = tracker.track(img, prior)
    dt = time.perf_counter() - t
    p0 = bead.passes[0]
    nan = float("nan")
    scores = {
        MAIN_SCORE: nan if p0.path_margin is None else float(p0.path_margin),
        "evidence": nan if p0.path_evidence is None else float(p0.path_evidence),
    }
    return bead, dt, scores


def split_images(name: str) -> set[str] | None:
    """The image IDs of a registry split, or `None` for every image."""
    if name == "all":
        return None
    return set(common.read_json(SPLIT_FILE)[name])


def run_coupled(img: np.ndarray, prior: np.ndarray, cfg: coupled.CoupledConfig, methods) -> dict:
    """Each coupled method on one call. `sample_ms` is one pass's profile sampling, and
    `total_ms` the whole call, every pass's sampling included."""
    t0 = time.perf_counter()
    coupled.sample(img, prior, cfg)
    t_sample = time.perf_counter() - t0
    out = {}
    for m in methods:
        if m in ("tracker", "rust"):
            continue
        t = time.perf_counter()
        pr, ex, first = coupled.extract(img, prior, cfg, m)
        total = time.perf_counter() - t
        out[m] = (pr, ex, first, as_call(pr, ex, total), {"sample_ms": 1e3 * t_sample, "total_ms": 1e3 * total})
    return out


# ---------------------------------------------------------------------------
# DamSegment
# ---------------------------------------------------------------------------
def tune_subset(index: dict, n: int, exclude: set) -> list[dict]:
    out = []
    for diff in common.DIFFICULTIES:
        entries = [e for e in index["paths"] if e["difficulty"] == diff and e["id"] not in exclude]
        pick = np.sort(common.rng_for("coupled-tune", diff).choice(len(entries), min(n, len(entries)), replace=False))
        out.extend(entries[i] for i in pick)
    return out


def dam_image(job: tuple) -> list[dict]:
    import vision_metrology as vm

    import priors as priors_mod
    import report as report_mod
    import run_tracker

    root, out_dir, rel, group, specs, args, cfg_kw, rust_kw = job
    doc = common.read_json(out_dir / "paths" / rel)
    refs = {p["id"]: p for p in doc["paths"]}
    img = common.load_gray(root / doc["image"])
    rows = []
    for e in group:
        ref = np.asarray(refs[e["id"]]["points"], dtype=np.float64)
        ref_w = np.asarray(refs[e["id"]]["width"], dtype=np.float64)
        lo, hi = run_tracker.width_range(e["width_median"], args)
        cfg = coupled.CoupledConfig(**{
            "polarity": "dark", "min_width": lo, "max_width": hi, "reach": args.reach, "spacing": args.spacing,
            "sigma": args.sigma, "threshold": args.threshold, **cfg_kw,
        })
        tracker = vm.BeadTracker(common.bead_config("dark", lo, hi, args))
        rust = rust_tracker(common.bead_config("dark", lo, hi, args), rust_kw) if "rust" in args.methods else None
        anchors: dict = {}
        anchor_ok: dict = {}
        first_row = len(rows)
        for spec in [{"kind": "anchor", "magnitude": 0.0}, *specs[e["id"]]]:
            if spec["kind"] != "anchor" and not report_mod.in_reach(spec, args.reach):
                continue
            prior = ref if spec["kind"] == "anchor" else priors_mod.build_prior(ref, spec)
            calls = {}
            if "tracker" in args.methods:
                t = time.perf_counter()
                bead = tracker.track(img, prior)
                c = run_tracker.record(e["id"], 0, bead, time.perf_counter() - t)
                c = dict(c, centerline=c["centerline"].astype(np.float64), width=c["width"].astype(np.float64))
                calls["tracker"] = (c, {"longest_gap": c["longest_gap"]}, {"total_ms": c["time_ms"]})
            if rust is not None:
                bead, dt, scores = rust_call(rust, img, prior)
                c = run_tracker.record(e["id"], 0, bead, dt)
                c = dict(c, centerline=c["centerline"].astype(np.float64), width=c["width"].astype(np.float64))
                calls["rust"] = (c, scores, {"total_ms": c["time_ms"]})
            for m, (pr, ex, first, c, times) in run_coupled(img, prior, cfg, args.methods).items():
                calls[m] = (c, coupled_scores(first, cfg, pr.h), times)
            for m, (c, scores, times) in calls.items():
                if spec["kind"] == "anchor":
                    # As `signals.py`: the call from the reference itself, for `converged`;
                    # whether it locks says whether the reference path is lockable at all.
                    anchors[m] = c["centerline"]
                    anchor_ok[m] = bool(report_mod.call_metrics(ref, ref_w, ref, c, c["centerline"])["locked"])
                    continue
                met = report_mod.call_metrics(ref, ref_w, prior, c, anchors[m])
                rows.append({
                    "dataset": "damsegment", "difficulty": e["difficulty"], "path": e["id"],
                    "prior": f"{spec['kind']} {spec['magnitude']:g}", "method": m,
                    "locked": bool(met["locked"]), "converged": bool(met["converged"]), "on_frac": float(np.mean(met["on"])),
                    "support": float(c["support"]), **scores, **times,
                })
        for r in rows[first_row:]:
            r["lockable"] = all(anchor_ok.values())
    return rows


def damsegment(args: argparse.Namespace, cfg_kw: dict, rust_kw: dict) -> list[dict]:
    root = common.dataset_root(args.data_dir)
    out_dir = common.resolve_out_dir(args)
    index = common.read_json(out_dir / "paths" / "index.json")
    keep = split_images(args.split)
    if keep is not None:
        index = dict(index, paths=[e for e in index["paths"] if f"{e['difficulty']}/{e['stem']}" in keep])
    specs = common.read_json(out_dir / "priors.json")["priors"]
    entries = common.select_paths(index, args.max_paths)
    if args.subset == "tune":
        entries = tune_subset(index, args.tune_paths, {e["id"] for e in entries})
    jobs = [(root, out_dir, rel, g, {e["id"]: specs[e["id"]] for e in g}, args, cfg_kw, rust_kw)
            for rel, g in common.by_image(entries).items()]
    return run_jobs(dam_image, jobs, args.workers)


def run_jobs(fn, jobs: list, workers: int) -> list[dict]:
    rows: list[dict] = []
    t0 = time.perf_counter()
    if workers <= 1:
        results = map(fn, jobs)
    else:
        pool = ProcessPoolExecutor(workers)
        results = pool.map(fn, jobs, chunksize=4)
    for k, r in enumerate(results):
        rows.extend(r)
        if (k + 1) % 100 == 0 or k + 1 == len(jobs):
            print(f"{k + 1}/{len(jobs)} jobs, {len(rows)} rows, {time.perf_counter() - t0:.0f} s", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------
def wall_errors(truth, pts: np.ndarray, side: str) -> np.ndarray:
    s, d, _ = truth.locate(pts, upper=40.0)
    wl, wr = truth.wall_at(s)
    return d + wl if side == "left" else d - wr


def synth_scene(job: tuple) -> list[dict]:
    import vision_metrology as vm

    import synthetic

    scene, args, cfg_kw, rust_kw = job
    img, truth = synthetic.render(scene)
    lo, hi = 2.0, 16.0
    cfg = coupled.CoupledConfig(**{"polarity": "light", "min_width": lo, "max_width": hi, "reach": 8.0,
                                   "spacing": 2.0, "threshold": 5.0, **cfg_kw})
    bcfg = vm.BeadConfig(polarity="light", min_width=lo, max_width=hi, spacing=2.0)
    track = bcfg.track
    track.max_offset = 8.0
    bcfg.track = track
    tracker = vm.BeadTracker(bcfg)
    rust = rust_tracker(bcfg, rust_kw) if "rust" in args.methods else None
    rows = []
    for name, prior in synthetic.priors(truth):
        res = {}
        if "tracker" in args.methods:
            t = time.perf_counter()
            bead = tracker.track(img, prior)
            dt = time.perf_counter() - t
            hit = ~np.isnan(np.asarray(bead.width))
            res["tracker"] = (np.asarray(bead.centerline, np.float64), hit, np.asarray(bead.first, np.float64),
                              np.asarray(bead.second, np.float64), float(bead.support), {}, {"total_ms": 1e3 * dt})
        if rust is not None:
            bead, dt, scores = rust_call(rust, img, prior)
            hit = ~np.isnan(np.asarray(bead.width))
            res["rust"] = (np.asarray(bead.centerline, np.float64), hit, np.asarray(bead.first, np.float64),
                           np.asarray(bead.second, np.float64), float(bead.support), scores, {"total_ms": 1e3 * dt})
        for m, (pr, ex, first, c, times) in run_coupled(img, prior, cfg, args.methods).items():
            lp = pr.points + np.nan_to_num(ex.left)[:, None] * pr.normals
            rp = pr.points + np.nan_to_num(ex.right)[:, None] * pr.normals
            res[m] = (c["centerline"], ex.present, lp, rp, ex.support, coupled_scores(first, cfg, pr.h), times)
        for m, (cl, hit, lp, rp, support, scores, times) in res.items():
            s, d, _ = truth.locate(cl, upper=40.0)
            wl, wr = truth.wall_at(s)
            centre = 0.5 * (wr - wl)
            on = np.abs(d - centre) <= 0.5 * (wl + wr) + 1.0
            vis = truth.visible(s)
            meas = hit & vis
            el = wall_errors(truth, lp[meas], "left") if meas.any() else np.zeros(0)
            er = wall_errors(truth, rp[meas], "right") if meas.any() else np.zeros(0)
            rows.append({
                "dataset": "synthetic", "difficulty": scene["condition"], "shape": scene["shape"],
                "path": synthetic.scene_id(scene), "prior": name, "method": m,
                "locked": bool(np.mean(on) >= 0.9), "support": float(support),
                "coverage": float(meas.sum() / max(vis.sum(), 1)),
                "err_left": el.tolist(), "err_right": er.tolist(), **scores, **times,
            })
    return rows


def main() -> None:
    parser = common.add_tracker_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    parser.add_argument("--data-dir", type=pathlib.Path, default=None, help="the DamSegment folder")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None, help="default: <data-dir>/output")
    parser.add_argument("--datasets", nargs="+", default=["damsegment", "synthetic"], choices=["damsegment", "synthetic"])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--subset", default="eval", choices=["eval", "tune"])
    parser.add_argument("--split", default="all", choices=["all", "dev", "test"],
                        help="DamSegment images of the registry split (default all: #76's set)")
    parser.add_argument("--rust-set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="BeadCoupling field, or track_step, measure_step, passes, clearance")
    parser.add_argument("--max-paths", type=int, default=100, help="evaluation paths per difficulty (default 100)")
    parser.add_argument("--tune-paths", type=int, default=40, help="tuning paths per difficulty (default 40)")
    parser.add_argument("--seeds", type=int, default=2, help="synthetic noise seeds (default 2)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="CoupledConfig cost overrides")
    parser.add_argument("--name", default="coupled_report")
    parser.add_argument("--boot", type=int, default=500, help="bootstrap resamples of paths (default 500)")
    parser.add_argument("--width-lo", type=float, default=0.25)
    parser.add_argument("--width-hi", type=float, default=2.0)
    parser.add_argument("--width-floor", type=float, default=2.0)
    args = parser.parse_args()
    if args.out_dir is None and args.data_dir is None:
        sys.exit("give --data-dir or --out-dir")
    out_dir = common.resolve_out_dir(args)
    defaults = dataclasses.asdict(coupled.CoupledConfig())
    cfg_kw = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in defaults or not isinstance(defaults[k], (int, float)):
            sys.exit(f"not a numeric CoupledConfig field: {k!r}")
        cfg_kw[k] = type(defaults[k])(float(v))
    keys = score_keys(coupled.CoupledConfig(**cfg_kw))
    rust_kw = {}
    for kv in args.rust_set:
        k, v = kv.split("=", 1)
        rust_kw[k] = float(v)

    rows: list[dict] = []
    if "damsegment" in args.datasets:
        if args.data_dir is None:
            sys.exit("damsegment needs --data-dir")
        rows += damsegment(args, cfg_kw, rust_kw)
    if "synthetic" in args.datasets:
        import synthetic

        rows += run_jobs(synth_scene, [(s, args, cfg_kw, rust_kw) for s in synthetic.scenes(args.seeds)], args.workers)

    common.write_json(out_dir / f"{args.name}_rows.json", rows, indent=None)
    rep = coupled_report.build(rows, keys, args.datasets, args.boot)
    rep.update({"config": dataclasses.asdict(coupled.CoupledConfig(**cfg_kw)), "subset": args.subset,
                "split": args.split, "rust": rust_kw})
    common.write_json(out_dir / f"{args.name}.json", rep)
    (out_dir / f"{args.name}.md").write_text(coupled_report.markdown(rep), encoding="utf-8")
    print(f"wrote {out_dir / (args.name + '.md')}")


if __name__ == "__main__":
    main()
