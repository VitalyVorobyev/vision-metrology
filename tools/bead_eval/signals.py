"""The tracker's own signals on real cracks: does any of them tell a call that locked from
one that did not, and how far is a rough crack from the `Converged` test?

On a seeded subset of DamSegment paths (`--max-paths` per difficulty), every prior within
the reach is tracked with `run_tracker.py`'s config, and again with `min_margin` set to
each of `--margins`. A call is locked by `report.py`'s test. Per call, the signals:

- `support`, `center_rms` and `longest_gap`, as `report.py` reads them;
- `confidence`: the median of the final stage's `confidence` over its hits (0 without);
- `support, min_margin m`: the support of the run with `min_margin = m`, judged against
  that run's own lock.

**Separation.** For each signal, the AUC: the probability that a call that locked scores
better than one that did not (0.5 is no separation, 1 perfect). And the share of the
calls that did not lock which pass a threshold set to keep 90% of those that did.

**Convergence.** From the default run's passes: the last solved pass's correction (max and
rms) and its residual, and how often candidate stopping tests would hold at each pass,
each with the step applied in full as `Converged` requires; and the median correction and
residual at each pass. `--passes` runs more passes than the library's default.

    python -I tools/bead_eval/signals.py --data-dir data/damsegment/extracted
    python -I tools/bead_eval/signals.py --data-dir ... --passes 10 --name signals_10pass

It needs `paths.py`'s and `priors.py`'s outputs, and writes `<name>.md` and `<name>.json`
to the output directory (`signals_report` by default).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Isolated mode (-I) leaves the script's own directory off sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import vision_metrology as vm  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import common  # noqa: E402
import priors as priors_mod  # noqa: E402
import report as report_mod  # noqa: E402
import run_tracker  # noqa: E402

KEEP = 0.9
#: Candidate stopping tests on one pass's solve, with the step applied in full.
STOP_TESTS = {
    "largest correction < tol (today's)": lambda s, tol: s.correction_max < tol,
    "rms correction < tol": lambda s, tol: s.correction_rms < tol,
    "rms correction < 2 tol": lambda s, tol: s.correction_rms < 2.0 * tol,
    "largest correction < residual rms / 2": lambda s, tol: s.correction_max < 0.5 * s.residual_rms,
    "rms correction < residual rms / 10": lambda s, tol: s.correction_rms < 0.1 * s.residual_rms,
}


def auc(good: np.ndarray, bad: np.ndarray) -> float:
    """P(a good call scores higher than a bad one), ties counted half."""
    if not len(good) or not len(bad):
        return float("nan")
    r = rankdata(np.concatenate([good, bad]))
    return float((r[: len(good)].sum() - len(good) * (len(good) + 1) / 2) / (len(good) * len(bad)))


def separation(rows: list[dict], key: str, higher_better: bool, locked_key: str = "locked") -> dict:
    x = np.array([r[key] for r in rows], dtype=np.float64)
    x = np.nan_to_num(x, nan=-np.inf if higher_better else np.inf)
    if not higher_better:
        x = -x
    lk = np.array([r[locked_key] for r in rows], dtype=bool)
    good, bad = x[lk], x[~lk]
    t = float(np.quantile(good, 1.0 - KEEP)) if len(good) else float("nan")
    return {
        "auc": auc(good, bad),
        "bad_passing": float(np.mean(bad >= t)) if len(bad) else float("nan"),
        "locked": float(np.mean(lk)),
        "calls": int(len(x)),
    }


def config(lo: float, hi: float, margin: float | None, args: argparse.Namespace):
    """`run_tracker.py`'s config, with a `min_margin` and the pass count from `args`."""
    cfg = common.bead_config("dark", lo, hi, args)
    cfg.min_margin = margin
    if args.passes is not None:
        tuning = cfg.tuning
        tuning.passes = args.passes
        cfg.tuning = tuning
    return cfg


def call_row(bead, ref: np.ndarray, ref_w: np.ndarray, prior: np.ndarray, anchor: np.ndarray, pid: str) -> dict:
    call = run_tracker.record(pid, 0, bead, 0.0)
    call = dict(call, centerline=call["centerline"].astype(np.float64), width=call["width"].astype(np.float64))
    m = report_mod.call_metrics(ref, ref_w, prior, call, anchor)
    conf = np.asarray(bead.confidence, dtype=np.float64)
    conf = conf[np.isfinite(conf)]
    return {
        "locked": bool(m["locked"]),
        "support": float(bead.support),
        "center_rms": np.nan if bead.center_rms is None else float(bead.center_rms),
        "longest_gap": float(bead.longest_gap),
        "confidence": float(np.median(conf)) if len(conf) else 0.0,
        "stop": bead.stop,
    }


def main() -> None:
    parser = common.add_tracker_args(
        common.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    )
    parser.add_argument("--max-paths", type=int, default=100, help="seeded subset per difficulty (default 100)")
    parser.add_argument(
        "--margins", type=float, nargs="+", default=[0.1, 0.2, 0.3], help="min_margin values (default 0.1 0.2 0.3)"
    )
    parser.add_argument("--passes", type=int, default=None, help="tuning.passes (default: the library's)")
    parser.add_argument("--name", default="signals_report", help="report file name (default signals_report)")
    parser.add_argument("--width-lo", type=float, default=0.25, help="as run_tracker.py (default 0.25)")
    parser.add_argument("--width-hi", type=float, default=2.0, help="as run_tracker.py (default 2)")
    parser.add_argument("--width-floor", type=float, default=2.0, help="as run_tracker.py (default 2)")
    args = parser.parse_args()
    root = common.dataset_root(args.data_dir)
    out_dir = common.resolve_out_dir(args)
    index = common.read_json(out_dir / "paths" / "index.json")
    specs = common.read_json(out_dir / "priors.json")["priors"]
    entries = common.select_paths(index, args.max_paths)

    rows: list[dict] = []
    passes: list[list] = []
    for rel, group in common.by_image(entries).items():
        doc = common.read_json(out_dir / "paths" / rel)
        refs = {p["id"]: p for p in doc["paths"]}
        img = common.load_gray(root / doc["image"])
        for e in group:
            ref = np.asarray(refs[e["id"]]["points"], dtype=np.float64)
            ref_w = np.asarray(refs[e["id"]]["width"], dtype=np.float64)
            lo, hi = run_tracker.width_range(e["width_median"], args)
            trackers = {m: vm.BeadTracker(config(lo, hi, m, args)) for m in [None, *args.margins]}
            anchor = np.asarray(trackers[None].track(img, ref).centerline, dtype=np.float64)
            for spec in specs[e["id"]]:
                if not report_mod.in_reach(spec, args.reach):
                    continue
                prior = priors_mod.build_prior(ref, spec)
                row = {"difficulty": e["difficulty"]}
                for m, tracker in trackers.items():
                    bead = tracker.track(img, prior)
                    r = call_row(bead, ref, ref_w, prior, anchor, e["id"])
                    if m is None:
                        row.update(r)
                        passes.append([p.solve for p in bead.passes])
                    else:
                        row[f"support_m{m:g}"] = r["support"]
                        row[f"locked_m{m:g}"] = r["locked"]
                rows.append(row)

    tol = float(vm.BeadConfig().tuning.tol)
    sep = {
        "support": separation(rows, "support", True),
        "center_rms": separation(rows, "center_rms", False),
        "longest_gap": separation(rows, "longest_gap", False),
        "confidence": separation(rows, "confidence", True),
    }
    for m in args.margins:
        sep[f"support, min_margin {m:g}"] = separation(rows, f"support_m{m:g}", True, f"locked_m{m:g}")
    conv = convergence(passes, tol)
    rep = {
        "calls": len(rows),
        "paths": len(entries),
        "keep": KEEP,
        "separation": sep,
        "convergence": conv,
        "stops": {s: int(sum(r["stop"] == s for r in rows)) for s in common.STOPS},
        "passes": args.passes,
    }
    common.write_json(out_dir / f"{args.name}.json", rep)
    (out_dir / f"{args.name}.md").write_text(markdown(rep), encoding="utf-8")
    print(f"wrote {out_dir / f'{args.name}.md'}")


def convergence(passes: list[list], tol: float) -> dict:
    """Last-pass magnitudes, and how often each candidate test holds at each pass."""
    last = [next((s for s in reversed(p) if s is not None), None) for p in passes]
    last = [s for s in last if s is not None]

    def stats(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype=np.float64)
        return {"median": float(np.median(a)), "p90": float(np.quantile(a, 0.9))} if len(a) else {}

    def holds(solve, fn) -> bool:
        return solve is not None and solve.step_scale >= 1.0 and fn(solve, tol)

    def median_at(i: int, key: str) -> float:
        v = [getattr(p[i], key) for p in passes if len(p) > i and p[i] is not None]
        return float(np.median(v)) if v else float("nan")

    n_pass = max((len(p) for p in passes), default=0)
    tests = {
        name: [float(np.mean([holds(p[i], fn) for p in passes if len(p) > i])) for i in range(n_pass)]
        for name, fn in STOP_TESTS.items()
    }
    return {
        "tol": tol,
        "last_correction_max": stats([s.correction_max for s in last]),
        "last_correction_rms": stats([s.correction_rms for s in last]),
        "last_residual_rms": stats([s.residual_rms for s in last]),
        "tests_by_pass": tests,
        "median_by_pass": {
            key: [median_at(i, key) for i in range(n_pass)]
            for key in ("correction_rms", "correction_max", "residual_rms")
        },
    }


def markdown(rep: dict) -> str:
    def f(v: float, nd: int = 2) -> str:
        return "–" if not np.isfinite(v) else f"{v:.{nd}f}"

    out = [
        "# The tracker's own signals on DamSegment cracks",
        "",
        f"{rep['calls']} calls from the priors within the reach, on {rep['paths']} paths.",
        "",
        f"| Signal | Calls locked | AUC | Not locked, passing a threshold that keeps {rep['keep']:.0%} of the locked |",
        "|---|---:|---:|---:|",
    ]
    for name, s in rep["separation"].items():
        out.append(f"| {name} | {s['locked']:.0%} | {f(s['auc'])} | {s['bad_passing']:.0%} |")
    c = rep["convergence"]
    out += [
        "",
        f"## Convergence (tol {c['tol']:g} px)",
        "",
        f"Stop reasons: {rep['stops']}.",
        "",
        "Last solved pass, median / p90: correction max "
        f"{f(c['last_correction_max']['median'], 3)} / {f(c['last_correction_max']['p90'], 3)} px, rms "
        f"{f(c['last_correction_rms']['median'], 3)} / {f(c['last_correction_rms']['p90'], 3)} px; "
        f"residual rms {f(c['last_residual_rms']['median'], 3)} / {f(c['last_residual_rms']['p90'], 3)} px.",
        "",
    ]
    n_pass = len(next(iter(c["tests_by_pass"].values())))
    out += [
        "| Test, with the step applied in full | "
        + " | ".join(f"Holds at pass {i + 1}" for i in range(n_pass))
        + " |",
        "|---|" + "---:|" * n_pass,
    ]
    for name, fr in c["tests_by_pass"].items():
        out.append(f"| {name} | " + " | ".join(f"{v:.0%}" for v in fr) + " |")
    for key, vals in c["median_by_pass"].items():
        out.append(f"| median {key.replace('_', ' ')}, px | " + " | ".join(f(v, 3) for v in vals) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
