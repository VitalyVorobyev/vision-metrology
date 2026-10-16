"""The metrics and the markdown of `coupled_eval.py`'s report.

Calls are clustered: every path is tracked from many priors, and a path that fails tends
to fail from all of them. Intervals are therefore 90% bootstrap intervals over paths
(scenes, for the synthetic set), resampled with replacement; differences between methods
are paired within each resample.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

METHODS = ("tracker", "sparse", "dense", "rust")
#: The ambiguity score chosen on the tuning subset, before the evaluation run.
MAIN_SCORE = "amb_q0.25_t5"
SIGNALS = ("support", MAIN_SCORE, "evidence")
FALSE_LOCK_SUPPORT = 0.5
WRONG_WALL_PX = 1.0


def auc(score: np.ndarray, locked: np.ndarray) -> float:
    """P(a locked call scores higher than one that is not), ties counted half."""
    good, bad = score[locked], score[~locked]
    if not len(good) or not len(bad):
        return float("nan")
    r = rankdata(np.concatenate([good, bad]))
    return float((r[: len(good)].sum() - len(good) * (len(good) + 1) / 2) / (len(good) * len(bad)))


def wrong_at(score: np.ndarray, locked: np.ndarray, cov: float) -> float:
    """Accept the calls scoring at least a threshold, lowered until `cov` of all calls
    are accepted and locked: the share accepted but not locked. Within a block of equal
    scores acceptance is pro rata. NaN when even accepting every call falls short."""
    n = len(score)
    if not n or locked.mean() < cov:
        return float("nan")
    vals, inv = np.unique(-score, return_inverse=True)
    good = np.bincount(inv, weights=locked, minlength=len(vals)) / n
    bad = np.bincount(inv, weights=~locked, minlength=len(vals)) / n
    cg, cb = np.cumsum(good), np.cumsum(bad)
    k = int(np.searchsorted(cg, cov - 1e-12))
    g0, b0 = (cg[k - 1], cb[k - 1]) if k else (0.0, 0.0)
    frac = (cov - g0) / good[k] if good[k] > 0 else 1.0
    return float(b0 + frac * bad[k])


def clean(x: np.ndarray) -> np.ndarray:
    """NaN is the worst score, `inf` (no alternative anywhere) the best finite one."""
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=-1e12, posinf=1e12, neginf=-1e12)


def point(arr: dict, idx: np.ndarray, cov: float) -> dict:
    lk = arr["locked"][idx]
    sup = arr["support"][idx]
    out = {"locked": float(lk.mean()), "false_lock": float(np.mean(~lk & (sup >= FALSE_LOCK_SUPPORT)))}
    for k in SIGNALS:
        if k in arr:
            out[f"auc {k}"] = auc(arr[k][idx], lk)
            out[f"wrong {k}"] = wrong_at(arr[k][idx], lk, cov)
    return out


def method_arrays(rows: list[dict], keys: list[str]) -> tuple[dict, list[str]]:
    """Per method, its calls' arrays in one order shared by every method, and the path
    of each call."""
    order = sorted({(r["path"], r["prior"]) for r in rows})
    pos = {k: i for i, k in enumerate(order)}
    out = {}
    for m in METHODS:
        sel = [r for r in rows if r["method"] == m]
        if not sel:
            continue
        sel.sort(key=lambda r: pos[(r["path"], r["prior"])])
        assert len(sel) == len(order), f"{m}: not every call was run"
        a = {"locked": np.array([r["locked"] for r in sel]), "support": clean([r["support"] for r in sel])}
        for k in keys:
            if k in sel[0]:
                a[k] = clean([r[k] for r in sel])
        out[m] = a
    return out, [p for p, _ in order]


def compare(rows: list[dict], keys: list[str], boot: int, seed: int = 0) -> dict:
    """Point values and bootstrap intervals per method, the tracker's operating point,
    and every coupled method's paired difference from the tracker."""
    arrs, paths = method_arrays(rows, keys)
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(paths):
        groups.setdefault(p, []).append(i)
    members = [np.asarray(v) for v in groups.values()]

    def cov_of(idx: np.ndarray) -> float:
        """The tracker's operating point, or 80% when it was not run."""
        if "tracker" not in arrs:
            return 0.8
        a = arrs["tracker"]
        return float(np.mean(a["locked"][idx] & (a["support"][idx] >= FALSE_LOCK_SUPPORT)))

    full = np.arange(len(paths))
    cov = cov_of(full)
    res = {m: {"point": point(a, full, cov)} for m, a in arrs.items()}
    rng = np.random.default_rng(seed)
    samples: dict = {m: [] for m in arrs}
    for _ in range(boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        c = cov_of(idx)
        for m, a in arrs.items():
            samples[m].append(point(a, idx, c))
    for m in arrs:
        res[m]["ci"] = {k: [float(np.nanquantile([s[k] for s in samples[m]], q)) for q in (0.05, 0.95)] for k in res[m]["point"]}
        if m != "tracker" and "tracker" in arrs:
            res[m]["minus_tracker"] = {}
            for k in ("locked", "false_lock", "wrong support"):
                d = np.array([s[k] - t[k] for s, t in zip(samples[m], samples["tracker"])])
                res[m]["minus_tracker"][k] = {
                    "point": res[m]["point"][k] - res["tracker"]["point"][k],
                    "ci": [float(np.nanquantile(d, 0.05)), float(np.nanquantile(d, 0.95))],
                }
        if m == "rust" and "dense" in arrs:
            res[m]["minus_dense"] = {}
            for k in ("locked", "false_lock", "wrong support"):
                d = np.array([s[k] - t[k] for s, t in zip(samples[m], samples["dense"])])
                res[m]["minus_dense"][k] = {
                    "point": res[m]["point"][k] - res["dense"]["point"][k],
                    "ci": [float(np.nanquantile(d, 0.05)), float(np.nanquantile(d, 0.95))],
                }
        res[m]["auc_all"] = {k: auc(arrs[m][k], arrs[m]["locked"]) for k in keys if k in arrs[m]}
    return {"calls": len(paths), "paths": len(members), "coverage": cov, "methods": res}


def walls(rows: list[dict]) -> dict:
    out = {}
    for m in METHODS:
        sel = [r for r in rows if r["method"] == m]
        if not sel:
            continue
        err = np.concatenate([np.concatenate([r["err_left"], r["err_right"]]) for r in sel])
        ok = np.abs(err) <= WRONG_WALL_PX
        lk = np.array([r["locked"] for r in sel])
        sup = np.array([r["support"] for r in sel])
        out[m] = {
            "locked": float(lk.mean()),
            "false_lock": float(np.mean(~lk & (sup >= FALSE_LOCK_SUPPORT))),
            "coverage": float(np.mean([r["coverage"] for r in sel])),
            "wrong_walls": float(np.mean(~ok)) if len(err) else float("nan"),
            "bias": float(np.mean(err[ok])) if ok.any() else float("nan"),
            "sigma": float(np.std(err[ok])) if ok.any() else float("nan"),
            "time_ms": float(np.median([r["total_ms"] for r in sel])),
        }
    return out


def build(rows: list[dict], keys: list[str], datasets: list[str], boot: int) -> dict:
    rep: dict = {"results": {}}
    if "damsegment" in datasets:
        sel = [r for r in rows if r["dataset"] == "damsegment"]
        d = {"all": compare(sel, keys, boot), "lockable": compare([r for r in sel if r["lockable"]], keys, boot)}
        d["by_difficulty"] = {
            g: compare([r for r in sel if r["difficulty"] == g], keys, 0) for g in sorted({r["difficulty"] for r in sel})
        }
        d["time_ms"] = {
            m: {k: float(np.median([r[k] for r in sel if r["method"] == m and k in r]))
                for k in ("sample_ms", "total_ms") if any(k in r for r in sel if r["method"] == m)}
            for m in METHODS
        }
        rep["results"]["damsegment"] = d
    if "synthetic" in datasets:
        sel = [r for r in rows if r["dataset"] == "synthetic"]
        rep["results"]["synthetic"] = {
            "all": compare(sel, keys, boot),
            "by_condition": {g: walls([r for r in sel if r["difficulty"] == g]) for g in sorted({r["difficulty"] for r in sel})},
            "walls": walls(sel),
        }
    return rep


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _f(v, nd: int = 2) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _p(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{100 * v:.1f}%"


def _ci(c: dict, k: str, pct: bool) -> str:
    """A value and its interval; '–' when the value is undefined, such as a coverage
    the method cannot reach."""
    v, (lo, hi) = c["point"][k], c["ci"][k]
    fmt = _p if pct else _f
    if not np.isfinite(v):
        return "–"
    return f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}]" if np.isfinite(lo) else fmt(v)


def headline(cmp: dict) -> list[str]:
    out = [
        f"{cmp['calls']} calls on {cmp['paths']} paths. The tracker's operating point accepts "
        f"{_p(cmp['coverage'])} of the calls as locked (support ≥ {FALSE_LOCK_SUPPORT:g}); "
        "'wrong' is the share of all calls accepted but not locked when each method's signal "
        "accepts that same share as locked.",
        "",
        "| Method | Locked | False lock | AUC support | AUC ambiguity | AUC evidence | Wrong, by support "
        "| Wrong, by ambiguity | Wrong, by evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m, r in cmp["methods"].items():
        cells = [_ci(r, "locked", True), _ci(r, "false_lock", True)]
        cells += [_ci(r, f"auc {k}", False) if f"auc {k}" in r["point"] else "–" for k in SIGNALS]
        cells += [_ci(r, f"wrong {k}", True) if f"wrong {k}" in r["point"] else "–" for k in SIGNALS]
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    diffs = [(m, r["minus_tracker"]) for m, r in cmp["methods"].items() if "minus_tracker" in r]
    diffs += [("rust − dense", r["minus_dense"]) for m, r in cmp["methods"].items() if "minus_dense" in r]
    if diffs:
        out += ["", "| Paired difference (minus the tracker unless named) | Locked | False lock | Wrong, by support |",
                "|---|---:|---:|---:|"]
        for m, d in diffs:
            out.append(
                f"| {m} | " + " | ".join(
                    f"{100 * d[k]['point']:+.1f} [{100 * d[k]['ci'][0]:+.1f}, {100 * d[k]['ci'][1]:+.1f}] pp"
                    if np.isfinite(d[k]["point"]) else "–"
                    for k in ("locked", "false_lock", "wrong support")
                ) + " |"
            )
    return out


def markdown(rep: dict) -> str:
    cfg = rep["config"]
    out = [
        "# Coupled two-boundary extraction",
        "",
        f"Subset: {rep['subset']}, split: {rep.get('split', 'all')}. Coupled config: "
        + ", ".join(f"{k} {v}" for k, v in cfg.items()) + ". Rust overrides: " + (str(rep.get("rust")) or "none") + ".",
        "",
        f"Signals: `support`, the share of stations with a hit; ambiguity, `{MAIN_SCORE}`: the 0.25 "
        "quantile of the station max-marginal margins for alternatives with a wall more than 5 px "
        "away; `evidence`, the best path's mean gain per px over absence. Brackets: 90% bootstrap "
        "intervals over paths.",
    ]
    res = rep["results"]
    if "damsegment" in res:
        d = res["damsegment"]
        out += ["", "## DamSegment, all calls", ""] + headline(d["all"])
        out += ["", "## DamSegment, lockable paths only", "",
                "Paths whose reference itself locks for every method.", ""] + headline(d["lockable"])
        out += ["", "## DamSegment by difficulty", "", "| Difficulty | Method | Locked | False lock | AUC support | AUC ambiguity |",
                "|---|---|---:|---:|---:|---:|"]
        for g, c in d["by_difficulty"].items():
            for m, r in c["methods"].items():
                p = r["point"]
                out.append(f"| {g} | {m} | {_p(p['locked'])} | {_p(p['false_lock'])} | {_f(p['auc support'])} "
                           f"| {_f(p.get(f'auc {MAIN_SCORE}'))} |")
        out += ["", "Median time per call: " + "; ".join(
            f"{m} {_f(t.get('total_ms'), 1)} ms" + (f" (one profile sampling {_f(t['sample_ms'], 1)} ms)" if "sample_ms" in t else "")
            for m, t in d["time_ms"].items() if t) + "."]
        out += ["", "Every ambiguity score's AUC, all calls:", "", "| Score | sparse | dense |", "|---|---:|---:|"]
        ms = d["all"]["methods"]
        for k in ms.get("dense", ms.get("sparse", {})).get("auc_all", {}):
            out.append(f"| `{k}` | {_f(ms.get('sparse', {}).get('auc_all', {}).get(k))} | {_f(ms.get('dense', {}).get('auc_all', {}).get(k))} |")
    if "synthetic" in res:
        s = res["synthetic"]
        out += ["", "## Synthetic", ""] + headline(s["all"])
        out += ["", f"Walls at visible stations: wrong when more than {WRONG_WALL_PX:g} px off; bias and σ over the rest.", "",
                "| Condition | Method | Locked | False lock | Station coverage | Wrong walls | Bias | σ | Time |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for g, ms in [*s["by_condition"].items(), ("all", s["walls"])]:
            for m, w in ms.items():
                out.append(f"| {g} | {m} | {_p(w['locked'])} | {_p(w['false_lock'])} | {_p(w['coverage'])} | {_p(w['wrong_walls'])} "
                           f"| {_f(w['bias'], 3)} px | {_f(w['sigma'], 3)} px | {_f(w['time_ms'], 1)} ms |")
    return "\n".join(out) + "\n"
