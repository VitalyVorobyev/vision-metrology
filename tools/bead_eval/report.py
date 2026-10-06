"""Metrics over the runs, per difficulty and overall: `report.md`, `report.json` and a
few overlay PNGs under the output directory.

Per station of a refined curve, against its reference path:
- centre distance: the distance to the reference centreline;
- on the crack: within half the local mask width of the centreline, plus `ON_SLACK_PX`
  for a pixel-level annotation;
- tangent error: the angle between the refined curve and the reference there, each the
  chord over `±TANGENT_CHORD_PX` of arc length;
- width: the final stage's, against the mask width `2 * EDT - 1` there.

Per call:
- support: the fraction of stations the final stage measured (the tracker's own);
- locked: at least `FRACTION` of the stations on the crack. The same test on the prior
  says whether it already was;
- converged: at least `FRACTION` of the stations within `AGREE_PX` of the curve the same
  method returns from the unperturbed reference. It needs no annotation: it is the
  basin, whether the result depends on the prior;
- false lock: not locked, yet with a support of at least `FALSE_LOCK_SUPPORT`, so its
  own summary does not flag it.

The reference is pixel-level: a skeleton of a hand-drawn mask. These numbers measure
whether the tracker locks onto and follows a real crack from a perturbed prior, not
subpixel accuracy.

    python -I tools/bead_eval/report.py --data-dir data/damsegment/extracted
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Isolated mode (-I) leaves the script's own directory off sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import common  # noqa: E402
import priors as priors_mod  # noqa: E402

ON_SLACK_PX = 1.0
AGREE_PX = 1.0
TANGENT_CHORD_PX = 4.0
FRACTION = 0.9
FALSE_LOCK_SUPPORT = 0.5
DISPLACEMENTS = ("translate", "rotate", "sine", "bump")
POOLED = ("d", "on", "ang", "w_meas", "w_mask", "dxy", "prior_d")


def call_metrics(
    ref: np.ndarray, ref_w: np.ndarray, prior: np.ndarray, call: dict, anchor: np.ndarray
) -> dict:
    """One call's metrics, and its per-station arrays for pooling.

    Distances, tangents and widths use the stations that project inside the reference.
    The on-the-crack and convergence tests use every station: one past the reference's
    end is measured to that end, so a curve that slides off the crack's end fails them."""
    st = call["centerline"]
    d, s, inside = common.project(st, ref)
    s_ref = common.arc_length(ref)
    w_at = np.interp(s, s_ref, ref_w)
    on = d <= 0.5 * w_at + ON_SLACK_PX
    s_st = common.arc_length(st)
    ang = common.angle_error_deg(
        common.chord_tangents(st, s_st, TANGENT_CHORD_PX),
        common.chord_tangents(ref, s, TANGENT_CHORD_PX),
    )
    closest = common.point_at(ref, s)
    hit = ~np.isnan(call["width"])

    pd, ps, p_in = common.project(common.resample(prior, 1.0), ref)
    p_on = pd <= 0.5 * np.interp(ps, s_ref, ref_w) + ON_SLACK_PX
    ad, _, _ = common.project(st, anchor)

    def frac(mask: np.ndarray) -> float:
        return float(np.mean(mask)) if mask.size >= 3 else 0.0

    rejects = np.bincount(call["reject"][call["reject"] >= 0], minlength=len(common.REJECTS))
    return {
        "d": d[inside].astype(np.float32),
        "on": on,
        "ang": ang[inside].astype(np.float32),
        "w_meas": call["width"][inside & hit].astype(np.float32),
        "w_mask": w_at[inside & hit].astype(np.float32),
        "dxy": (st - closest)[inside & hit].astype(np.float32),
        "prior_d": pd[p_in].astype(np.float32),
        "locked": frac(on) >= FRACTION,
        "prior_on": frac(p_on) >= FRACTION,
        "converged": frac(ad <= AGREE_PX) >= FRACTION,
        "support": call["support"],
        "center_rms": call["center_rms"],
        "longest_gap": call["longest_gap"],
        "time_ms": call["time_ms"],
        "stations": len(st),
        "stop": call["stop"],
        "rejects": rejects,
    }


def q(a: np.ndarray, p: float) -> float:
    return float(np.quantile(a, p)) if a.size else float("nan")


def aggregate(ms: list[dict]) -> dict:
    """Pooled station statistics and per-call rates over a list of call metrics."""
    if not ms:
        return {"calls": 0}
    cat = {k: np.concatenate([m[k] for m in ms]) for k in POOLED}
    locked = np.array([m["locked"] for m in ms])
    support = np.array([m["support"] for m in ms])
    has_support = not np.all(np.isnan(support))
    times = np.array([m["time_ms"] for m in ms])
    stations = np.array([m["stations"] for m in ms])
    rejects = np.sum([m["rejects"] for m in ms], axis=0)
    false_lock = ~locked & (support >= FALSE_LOCK_SUPPORT) if has_support else ~locked
    dxy = cat["dxy"].reshape(-1, 2)
    return {
        "calls": len(ms),
        "paths": len({m["path_id"] for m in ms}),
        "stations": int(stations.sum()),
        "center_median": q(cat["d"], 0.5),
        "center_p95": q(cat["d"], 0.95),
        "on_crack": float(np.mean(cat["on"])) if cat["on"].size else float("nan"),
        "offset_median_xy": [q(dxy[:, 0], 0.5), q(dxy[:, 1], 0.5)],
        "tangent_median_deg": q(cat["ang"], 0.5),
        "tangent_p95_deg": q(cat["ang"], 0.95),
        "support_mean": float(np.nanmean(support)) if has_support else float("nan"),
        "width_measured_median": q(cat["w_meas"], 0.5),
        "width_mask_median": q(cat["w_mask"], 0.5),
        "width_error_median": q(cat["w_meas"] - cat["w_mask"], 0.5),
        "prior_median": q(cat["prior_d"], 0.5),
        "prior_on": float(np.mean([m["prior_on"] for m in ms])),
        "locked": float(np.mean(locked)),
        "converged": float(np.mean([m["converged"] for m in ms])),
        "false_lock": float(np.mean(false_lock)),
        "time_median_ms": float(np.median(times)),
        "time_p95_ms": float(np.quantile(times, 0.95)),
        "stations_median": float(np.median(stations)),
        "stops": {s: int(sum(m["stop"] == s for m in ms)) for s in common.STOPS},
        "rejects": {r: int(n) for r, n in zip(common.REJECTS, rejects) if n},
    }


def signals(rows: list[dict]) -> dict:
    """The tracker's own quality signals, for the calls that locked and those that did
    not: whether its summary tells the two apart."""
    out = {}
    for name, sel in (
        ("locked", [r for r in rows if r["locked"]]),
        ("not locked", [r for r in rows if not r["locked"]]),
    ):
        out[name] = {"calls": len(sel)}
        for key in ("support", "center_rms", "longest_gap"):
            v = np.array([r[key] for r in sel], dtype=np.float64)
            v = v[~np.isnan(v)]
            out[name][key] = {"p10": q(v, 0.1), "median": q(v, 0.5), "p90": q(v, 0.9)}
    return out


def in_reach(spec: dict, reach: float) -> bool:
    """A perturbation the tracker is configured to converge from."""
    if spec["kind"] in DISPLACEMENTS:
        return spec["magnitude"] <= reach
    return spec["kind"] in ("simplify", "truncate")


def load_method(
    out_dir: pathlib.Path, name: str, index: dict, specs: dict, keep_curve: bool = False
) -> tuple[dict, list]:
    """A run's meta and every call's metrics, tagged with difficulty and spec, and with
    the refined curve itself when `keep_curve` is set."""
    run_dir = out_dir / "runs" / name
    meta = common.read_json(run_dir / "meta.json")
    by_id = {e["id"]: e for e in index["paths"]}
    rows = []
    for npz in sorted(run_dir.rglob("*.npz")):
        calls = common.load_calls(npz)
        rel = str(npz.relative_to(run_dir)).replace(".npz", ".json")
        doc = common.read_json(out_dir / "paths" / rel)
        refs = {p["id"]: p for p in doc["paths"]}
        anchors = {
            c["path_id"]: c["centerline"]
            for c in calls
            if specs[c["path_id"]][c["spec"]]["kind"] == "none"
        }
        for c in calls:
            p = refs[c["path_id"]]
            ref = np.asarray(p["points"], dtype=np.float64)
            spec = specs[c["path_id"]][c["spec"]]
            prior = priors_mod.build_prior(ref, spec)
            m = call_metrics(ref, np.asarray(p["width"]), prior, c, anchors[c["path_id"]])
            m.update(
                {
                    "path_id": c["path_id"],
                    "difficulty": by_id[c["path_id"]]["difficulty"],
                    "kind": spec["kind"],
                    "magnitude": spec["magnitude"],
                    "spec": c["spec"],
                    "in_reach": in_reach(spec, float(meta.get("config", {}).get("reach", np.inf))),
                }
            )
            if keep_curve:
                m["centerline"] = c["centerline"]
            rows.append(m)
    return meta, rows


def per_difficulty(sel: list[dict]) -> dict:
    agg = {d: aggregate([r for r in sel if r["difficulty"] == d]) for d in common.DIFFICULTIES}
    agg["All"] = aggregate(sel)
    return agg


def basin_rows(rows: list[dict], units: dict) -> list[dict]:
    order = list(units)
    keys = sorted({(r["kind"], r["magnitude"]) for r in rows}, key=lambda k: (order.index(k[0]), k[1]))
    out = []
    for kind, mag in keys:
        sel = [r for r in rows if r["kind"] == kind and r["magnitude"] == mag]
        row = {"kind": kind, "magnitude": mag, "unit": units[kind], "All": aggregate(sel)}
        for d in common.DIFFICULTIES:
            row[d] = aggregate([r for r in sel if r["difficulty"] == d])
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def fmt(v, nd: int = 2) -> str:
    return "–" if v is None or np.isnan(v) else f"{v:.{nd}f}"


def pct(v) -> str:
    return "–" if v is None or np.isnan(v) else f"{100 * v:.0f}%"


def headline(agg: dict[str, dict], anchor: bool) -> list[str]:
    lines = [
        "| Set | Paths | Calls | Centre median / p95 | On the crack | Tangent median / p95 "
        "| Support | Width, measured / mask | Locked | Converged | False lock | Time median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, a in agg.items():
        if not a.get("calls"):
            continue
        lines.append(
            f"| {name} | {a['paths']} | {a['calls']} "
            f"| {fmt(a['center_median'])} / {fmt(a['center_p95'])} px | {pct(a['on_crack'])} "
            f"| {fmt(a['tangent_median_deg'], 1)} / {fmt(a['tangent_p95_deg'], 1)}° "
            f"| {pct(a['support_mean'])} "
            f"| {fmt(a['width_measured_median'], 1)} / {fmt(a['width_mask_median'], 1)} px "
            f"| {pct(a['locked'])} | {'–' if anchor else pct(a['converged'])} "
            f"| {pct(a['false_lock'])} | {fmt(a['time_median_ms'])} ms |"
        )
    return lines


def write_markdown(path: pathlib.Path, rep: dict) -> None:
    s = rep["summary"]
    cfg = rep["methods"]["tracker"]["config"]
    out = [
        "# Bead tracker on DamSegment cracks",
        "",
        f"{s['images']} images; {s['paths']} reference paths of at least 80 px; "
        f"{s['priors_per_path']} priors per path; {s['calls']} `track()` calls.",
        "",
        "Tracker: " + ", ".join(f"{k} {v}" for k, v in cfg.items()) + ".",
        "",
        f"On the crack: within half the mask width of its centreline, plus {ON_SLACK_PX:g} px. "
        f"Locked: at least {FRACTION:.0%} of the stations on the crack. Converged: at least "
        f"{FRACTION:.0%} within {AGREE_PX:g} px of the curve tracked from the unperturbed "
        f"reference. False lock: not locked, with support of at least {FALSE_LOCK_SUPPORT:g}. "
        "Station statistics are pooled over calls.",
        "",
        "## From the reference itself",
        "",
    ]
    out += headline(rep["tracker"]["none"], anchor=True)
    off = rep["tracker"]["none"]["All"]["offset_median_xy"]
    out += [
        "",
        f"Median offset of the measured stations from the mask centreline: "
        f"x {off[0]:+.2f} px, y {off[1]:+.2f} px.",
        "",
        f"## From priors within the reach ({rep['reach']:g} px)",
        "",
    ]
    out += headline(rep["tracker"]["in_reach"], anchor=False)
    diffs = list(common.DIFFICULTIES)
    out += [
        "",
        "## Basin",
        "",
        "| Perturbation | Magnitude | Prior median | Prior on the crack | Locked | Converged | "
        + " | ".join(f"Converged, {d}" for d in diffs)
        + " | False lock | Centre median |",
        "|---|---:|---:|---:|---:|---:|" + "---:|" * len(diffs) + "---:|---:|",
    ]
    for row in rep["tracker"]["basin"]:
        a = row["All"]
        out.append(
            f"| {row['kind']} | {row['magnitude']:g} {row['unit']} | {fmt(a['prior_median'], 1)} px "
            f"| {pct(a['prior_on'])} | {pct(a['locked'])} | {pct(a['converged'])} | "
            + " | ".join(pct(row[d]["converged"]) if row[d].get("calls") else "–" for d in diffs)
            + f" | {pct(a['false_lock'])} | {fmt(a['center_median'])} px |"
        )
    for name, c in rep.get("compare", {}).items():
        out += ["", f"## Sensitivity: `{name}`", ""]
        out.append(f"Config: {c['config']}.")
        out.append("")
        out += headline(c["in_reach"], anchor=False)
    if "baseline" in rep:
        b = rep["baseline"]
        out += [
            "",
            f"## Against `active_contour`, on {b['paths']} paths with the same priors",
            "",
            f"Config: {rep['methods']['baseline']['config']}.",
            "",
            "| Method | Set | Calls | Centre median / p95 | On the crack | Locked | Converged | Time median |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method, sets in b["sets"].items():
            for name, a in sets.items():
                if not a.get("calls"):
                    continue
                out.append(
                    f"| {method} | {name} | {a['calls']} | {fmt(a['center_median'])} / "
                    f"{fmt(a['center_p95'])} px | {pct(a['on_crack'])} | {pct(a['locked'])} "
                    f"| {'–' if name == 'reference' else pct(a['converged'])} "
                    f"| {fmt(a['time_median_ms'])} ms |"
                )
        out += [
            "",
            "| Perturbation | Magnitude | BeadTracker locked / converged | active_contour locked / converged |",
            "|---|---:|---:|---:|",
        ]
        for row in b["basin"]:
            out.append(
                f"| {row['kind']} | {row['magnitude']:g} {row['unit']} "
                f"| {pct(row['tracker']['locked'])} / {pct(row['tracker']['converged'])} "
                f"| {pct(row['baseline']['locked'])} / {pct(row['baseline']['converged'])} |"
            )
    f = rep["tracker"]["failures"]
    out += [
        "",
        "## Failure modes, within the reach",
        "",
        f"Calls that did not lock: {f['calls']}, of which {f['false_lock']} with support of at "
        f"least {FALSE_LOCK_SUPPORT:g}. Their stop reasons: {f['stops']}.",
        "",
        f"Final-stage rejections in them: {f['rejects']}.",
        "",
        f"Final-stage rejections over every call within the reach: {f['all_rejects']}.",
        "",
        f"Stop reasons over every call within the reach: {f['all_stops']}.",
        "",
        "Does the tracker's own summary tell a locked call from one that is not? p10 / median "
        "/ p90 over the calls within the reach:",
        "",
        "| Calls | Count | Support | `center_rms` | `longest_gap` |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, sig in rep["tracker"]["signals"].items():
        cells = [
            " / ".join(fmt(sig[k][p], nd) for p in ("p10", "median", "p90"))
            for k, nd in (("support", 2), ("center_rms", 2), ("longest_gap", 0))
        ]
        out.append(f"| {name} | {sig['calls']} | {cells[0]} | {cells[1]} px | {cells[2]} px |")
    out += ["", "## Runtime", ""]
    t = rep["tracker"]["runtime"]
    out.append(
        f"`track()` per call, binding overhead included: median {t['median_ms']:.2f} ms, "
        f"p95 {t['p95_ms']:.2f} ms, median {t['stations_median']:.0f} stations, "
        f"{t['us_per_station']:.1f} µs per station (median over calls)."
    )
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------
def overlays(
    out_dir: pathlib.Path, root: pathlib.Path, rows: list[dict], brows: list[dict], specs
) -> None:
    """Two overlays per difficulty, each a seeded pick within the reach: a call that
    locked from a prior that was off the crack, and one that did not lock. The
    baseline's curve is drawn when it ran on the same prior."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    odir = out_dir / "overlays"
    odir.mkdir(parents=True, exist_ok=True)
    base_by_key = {(m["path_id"], m["spec"]): m for m in brows}
    for diff in common.DIFFICULTIES:
        cands = [r for r in rows if r["difficulty"] == diff and r["in_reach"]]
        rng = common.rng_for("overlay", diff)
        for tag, pool in (
            ("locked", [r for r in cands if r["locked"] and not r["prior_on"]]),
            ("failed", [r for r in cands if not r["locked"]]),
        ):
            if not pool:
                continue
            r = pool[int(rng.integers(len(pool)))]
            _, stem, _ = r["path_id"].split("/")
            doc = common.read_json(out_dir / "paths" / diff / f"{stem}.json")
            p = next(x for x in doc["paths"] if x["id"] == r["path_id"])
            ref = np.asarray(p["points"])
            spec = specs[r["path_id"]][r["spec"]]
            prior = priors_mod.build_prior(ref, spec)
            call = next(
                c
                for c in common.load_calls(out_dir / "runs" / "tracker" / diff / f"{stem}.npz")
                if c["path_id"] == r["path_id"] and c["spec"] == r["spec"]
            )
            img = np.asarray(Image.open(common.image_path(root, diff, stem)).convert("RGB"))
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(img)
            ax.plot(ref[:, 0], ref[:, 1], "-", color="lime", lw=1, label="reference")
            ax.plot(prior[:, 0], prior[:, 1], "--", color="white", lw=1, label="prior")
            if (r["path_id"], r["spec"]) in base_by_key:
                bc = base_by_key[(r["path_id"], r["spec"])]["centerline"]
                ax.plot(bc[:, 0], bc[:, 1], "-", color="magenta", lw=1, label="active_contour")
            cl = call["centerline"]
            ax.plot(cl[:, 0], cl[:, 1], "-", color="cyan", lw=1.2, label="BeadTracker")
            rej = call["reject"] >= 0
            ax.plot(cl[rej, 0], cl[rej, 1], "o", color="red", ms=2.5, label="rejected station")
            pad = 30
            lo = np.minimum(ref.min(0), prior.min(0)) - pad
            hi = np.maximum(ref.max(0), prior.max(0)) + pad
            ax.set_xlim(max(lo[0], -0.5), min(hi[0], img.shape[1] - 0.5))
            ax.set_ylim(min(hi[1], img.shape[0] - 0.5), max(lo[1], -0.5))
            ax.set_title(
                f"{r['path_id']}: {spec['kind']} {spec['magnitude']:g}, support {r['support']:.2f}",
                fontsize=9,
            )
            ax.legend(loc="lower right", fontsize=7)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(odir / f"{diff.lower()}_{tag}.png", dpi=110)
            plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = common.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    parser.add_argument("--baseline", default="active_contour", help="run name, used if present")
    parser.add_argument(
        "--compare", nargs="*", default=[], help="other tracker runs to summarise (e.g. tracker_defaults)"
    )
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()
    root = common.dataset_root(args.data_dir)
    out_dir = common.resolve_out_dir(args)
    index = common.read_json(out_dir / "paths" / "index.json")
    pri = common.read_json(out_dir / "priors.json")
    specs, units = pri["priors"], pri["units"]

    meta, rows = load_method(out_dir, "tracker", index, specs)
    reach = float(meta["config"]["reach"])
    reach_rows = [r for r in rows if r["in_reach"]]
    failed = [r for r in reach_rows if not r["locked"]]
    fail_agg, reach_agg = aggregate(failed), aggregate(reach_rows)
    times = np.array([r["time_ms"] for r in rows])
    stations = np.array([r["stations"] for r in rows])
    rep: dict = {
        "summary": {
            "images": sum(v["images"] for v in index["difficulties"].values()),
            "paths": len({r["path_id"] for r in rows}),
            "priors_per_path": sum(len(v) for v in pri["sweep"].values()),
            "calls": len(rows),
            "difficulties": index["difficulties"],
        },
        "reach": reach,
        "definitions": {
            "on_slack_px": ON_SLACK_PX,
            "agree_px": AGREE_PX,
            "fraction": FRACTION,
            "false_lock_support": FALSE_LOCK_SUPPORT,
        },
        "methods": {"tracker": meta},
        "tracker": {
            "none": per_difficulty([r for r in rows if r["kind"] == "none"]),
            "in_reach": per_difficulty(reach_rows),
            "basin": basin_rows(rows, units),
            "signals": signals(reach_rows),
            "failures": {
                "calls": len(failed),
                "false_lock": int(sum(r["support"] >= FALSE_LOCK_SUPPORT for r in failed)),
                "stops": fail_agg.get("stops", {}),
                "rejects": fail_agg.get("rejects", {}),
                "all_rejects": reach_agg.get("rejects", {}),
                "all_stops": reach_agg.get("stops", {}),
            },
            "runtime": {
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.quantile(times, 0.95)),
                "stations_median": float(np.median(stations)),
                "us_per_station": float(1e3 * np.median(times / stations)),
                "per_difficulty_median_ms": {
                    d: float(np.median([r["time_ms"] for r in rows if r["difficulty"] == d]))
                    for d in common.DIFFICULTIES
                },
            },
        },
    }

    rep["compare"] = {}
    for name in args.compare:
        cmeta, crows = load_method(out_dir, name, index, specs)
        rep["compare"][name] = {
            "config": cmeta["config"],
            "in_reach": per_difficulty([r for r in crows if r["in_reach"]]),
        }

    brows: list[dict] = []
    if (out_dir / "runs" / args.baseline / "meta.json").is_file():
        bmeta, brows = load_method(out_dir, args.baseline, index, specs, keep_curve=True)
        for r in brows:
            r["in_reach"] = in_reach(specs[r["path_id"]][r["spec"]], reach)
        keys = {(r["path_id"], r["spec"]) for r in brows}
        trows = [r for r in rows if (r["path_id"], r["spec"]) in keys]
        tb = {(r["kind"], r["magnitude"]): r for r in basin_rows(trows, units)}
        rep["methods"]["baseline"] = bmeta
        rep["baseline"] = {
            "paths": len({r["path_id"] for r in brows}),
            "sets": {
                "BeadTracker": {
                    "reference": aggregate([r for r in trows if r["kind"] == "none"]),
                    "within the reach": aggregate([r for r in trows if r["in_reach"]]),
                },
                bmeta["label"]: {
                    "reference": aggregate([r for r in brows if r["kind"] == "none"]),
                    "within the reach": aggregate([r for r in brows if r["in_reach"]]),
                },
            },
            "basin": [
                {
                    "kind": row["kind"],
                    "magnitude": row["magnitude"],
                    "unit": row["unit"],
                    "tracker": tb[(row["kind"], row["magnitude"])]["All"],
                    "baseline": row["All"],
                }
                for row in basin_rows(brows, units)
            ],
        }

    common.write_json(out_dir / "report.json", rep)
    write_markdown(out_dir / "report.md", rep)
    if not args.no_overlays:
        overlays(out_dir, root, rows, brows, specs)
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
