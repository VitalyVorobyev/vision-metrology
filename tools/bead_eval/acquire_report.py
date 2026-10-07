"""The acquisition evaluation's report: `acquire_report.json`, `acquire_report.md` and a
few overlays, from the per-image results of `acquire_eval.py`."""

from __future__ import annotations

import pathlib

import numpy as np

import common
import ribbons

PRIORS = ("acquired", "reference", "translated")


def q(a: np.ndarray, p: float) -> float:
    return float(np.quantile(a, p)) if len(a) else float("nan")


def acq_agg(scores: list[dict], cover: float) -> dict:
    """Pooled acquisition metrics over images."""
    if not scores:
        return {"images": 0}
    cov = np.concatenate([s["cover"] for s in scores])
    one = np.concatenate([s["cover_one"] for s in scores])
    err = np.concatenate([s["err"] for s in scores])
    w_est = np.concatenate([s["w_est"] for s in scores])
    w_ref = np.concatenate([s["w_ref"] for s in scores])
    length = sum(s["length"] for s in scores)
    filt = np.array([s["filter_ms"] for s in scores], dtype=np.float64)
    return {
        "images": len(scores),
        "refs": int(len(cov)),
        "recall": float(np.mean(cov >= cover)) if len(cov) else float("nan"),
        "recall_one": float(np.mean(one >= cover)) if len(one) else float("nan"),
        "precision": sum(s["length_near"] for s in scores) / length if length else float("nan"),
        "paths_per_image": float(np.mean([s["paths"] for s in scores])),
        "length_per_image": float(length / len(scores)),
        "center_median": q(err, 0.5),
        "center_p95": q(err, 0.95),
        "width_est_median": q(w_est, 0.5),
        "width_ref_median": q(w_ref, 0.5),
        "width_err_median": q(w_est - w_ref, 0.5),
        "width_abs_err_median": q(np.abs(w_est - w_ref), 0.5),
        "time_median_ms": float(np.median([s["time_ms"] for s in scores])),
        "filter_median_ms": float(np.nanmedian(filt)) if np.isfinite(filt).any() else float("nan"),
    }


def track_agg(rows: list[dict], source: str) -> dict:
    """Tracking from `source`'s acquired paths, and from the reference and its
    translation over the same reference paths."""
    acq = [r for r in rows if r["prior"] == source]
    refs = {r["ref"] for r in acq}
    out = {}
    for name in PRIORS:
        sel = acq if name == "acquired" else [r for r in rows if r["prior"] == name and r["ref"] in refs]
        if not sel:
            out[name] = {"calls": 0}
            continue
        d = np.concatenate([r["d"] for r in sel])
        out[name] = {
            "calls": len(sel),
            "center_median": q(d, 0.5),
            "center_p95": q(d, 0.95),
            "support": float(np.mean([r["support"] for r in sel])),
            "prior_on": float(np.mean([r["prior_on"] for r in sel])),
            "locked": float(np.mean([r["locked"] for r in sel])),
            "converged": float(np.mean([r["converged"] for r in sel])),
        }
    out["refs_total"] = len({r["ref"] for r in rows})
    return out


def label(values: tuple) -> str:
    return ", ".join(f"{v:g} px" if isinstance(v, float) else str(v) for v in values)


def ordered(values: list[tuple]) -> list[tuple]:
    """Distinct values in order of first appearance, then by a leading number if any."""
    out = list(dict.fromkeys(values))
    return sorted(out, key=lambda g: g[0] if isinstance(g[0], float) else 0.0)


def summarise(results: list[dict], meta: dict, keys: list[str]) -> dict:
    """Acquisition per source, setting and group, and tracking per source and by the
    first group key."""

    def key(r: dict, ks: list[str]) -> tuple:
        return tuple(r["group"][k] for k in ks)

    groups = ordered([key(r, keys) for r in results])
    firsts = ordered([key(r, keys[:1]) for r in results])
    out: dict = {"acquire": {}, "track": {}}
    for src in meta["sources"]:
        out["acquire"][src] = {}
        for setting in meta["settings"][src]:
            per = {}
            for g in groups:
                sel = [r["scores"][src][setting] for r in results if key(r, keys) == g]
                per[label(g)] = acq_agg(sel, meta["cover"])
            per["all"] = acq_agg([r["scores"][src][setting] for r in results], meta["cover"])
            out["acquire"][src][setting] = per
        per_t = {}
        for g in firsts:
            sel = [row for r in results if key(r, keys[:1]) == g for row in r["track"]]
            per_t[label(g)] = track_agg(sel, src)
        per_t["all"] = track_agg([row for r in results for row in r["track"]], src)
        out["track"][src] = per_t
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def fmt(v, nd: int = 2) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def pct(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{100 * v:.0f}%"


ACQ_HEAD = (
    "| {first} | Images | Recall | One-path recall | Precision | Centre median / p95 "
    "| Width, source / truth | Paths per image | Time, all / filter |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
)


def acq_row(name: str, a: dict) -> str:
    return (
        f"| {name} | {a['images']} | {pct(a['recall'])} | {pct(a['recall_one'])} | {pct(a['precision'])} "
        f"| {fmt(a['center_median'])} / {fmt(a['center_p95'])} px "
        f"| {fmt(a['width_est_median'], 1)} / {fmt(a['width_ref_median'], 1)} px "
        f"| {fmt(a['paths_per_image'], 1)} | {fmt(a['time_median_ms'], 0)} / {fmt(a['filter_median_ms'], 0)} ms |"
    )


def track_table(per: dict, first: str, translate_px: float) -> list[str]:
    lines = [
        f"| {first} | Prior | Calls | Prior on the line | Centre median / p95 | Support | Locked | Converged |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g, t in per.items():
        for name in PRIORS:
            a = t[name]
            if not a.get("calls"):
                continue
            label = f"translated {translate_px:g} px" if name == "translated" else name
            lines.append(
                f"| {g} | {label} | {a['calls']} | {pct(a['prior_on'])} "
                f"| {fmt(a['center_median'], 3)} / {fmt(a['center_p95'], 3)} px | {pct(a['support'])} "
                f"| {pct(a['locked'])} | {pct(a['converged'])} |"
            )
    return lines



def markdown(rep: dict, meta: dict) -> str:
    out = [
        "# Acquisition without a prior",
        "",
        f"Sources: {', '.join(meta['sources'])}"
        + (f"; not installed: {', '.join(meta['skipped'])}" if meta["skipped"] else "")
        + ". Settings: "
        + "; ".join(f"{s} {', '.join(v)} (default {meta['default'][s]})" for s, v in meta["settings"].items())
        + ".",
        "",
        f"Recall: at least {meta['cover']:.0%} of a reference within max(2 px, w/2) of acquired paths; "
        "one-path recall: of one acquired path. Precision: acquired length near a reference "
        f"(DamSegment: within {meta['on_mask_px']:g} px of a crack-mask pixel centre). Centre and "
        "width: acquired points near a reference. Time per image: the whole acquisition, and the "
        "filter alone where it is separate.",
    ]
    syn = rep.get("synthetic")
    if syn:
        out += ["", "## Synthetic ribbons, 1280x1024", ""]
        for src in meta["sources"]:
            d = meta["default"][src]
            out += [f"### {src}, {d}", "", ACQ_HEAD[0].format(first="Width / condition"), ACQ_HEAD[1]]
            per = syn["acquire"][src][d]
            out += [acq_row(g, a) for g, a in per.items()] + [""]
        out += ["### Other settings, all scenes", "", ACQ_HEAD[0].format(first="Source, setting"), ACQ_HEAD[1]]
        for src in meta["sources"]:
            for setting, per in syn["acquire"][src].items():
                out.append(acq_row(f"{src}, {setting}", per["all"]))
        out += ["", "### Acquire, then track", ""]
        for src in meta["sources"]:
            out += [f"From {src}'s paths, by ribbon width:", ""]
            out += track_table(syn["track"][src], "Width", meta["translate_px"])
            out.append("")
    dam = rep.get("damsegment")
    if dam:
        out += ["", "## DamSegment cracks, 640x640", "", ACQ_HEAD[0].format(first="Source, setting"), ACQ_HEAD[1]]
        for src in meta["sources"]:
            for setting, per in dam["acquire"][src].items():
                out.append(acq_row(f"{src}, {setting}", per["all"]))
        out += ["", "By difficulty, default settings:", "", ACQ_HEAD[0].format(first="Source, set"), ACQ_HEAD[1]]
        for src in meta["sources"]:
            for g, a in dam["acquire"][src][meta["default"][src]].items():
                if g != "all":
                    out.append(acq_row(f"{src}, {g}", a))
        out += ["", "### Acquire, then track", ""]
        for src in meta["sources"]:
            t = dam["track"][src]["all"]
            out += [
                f"From {src}'s paths: {t['acquired'].get('calls', 0)} of {t['refs_total']} reference paths "
                "covered by one acquired path.",
                "",
            ]
            out += track_table(dam["track"][src], "Set", meta["translate_px"])
            out.append("")
    return "\n".join(out) + "\n"


def write(out_dir: pathlib.Path, results: dict, meta: dict) -> None:
    rep = {"meta": meta}
    if "synthetic" in results:
        rep["synthetic"] = summarise(results["synthetic"], meta, ["width", "condition"])
    if "damsegment" in results:
        rep["damsegment"] = summarise(results["damsegment"], meta, ["difficulty"])
    common.write_json(out_dir / "acquire_report.json", rep)
    (out_dir / "acquire_report.md").write_text(markdown(rep, meta), encoding="utf-8")


def overlay(out_dir: pathlib.Path, res: dict) -> None:
    """One synthetic scene: its truth, and each source's paths at its default setting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene = dict(res["group"], seed=int(res["id"].rsplit("_", 1)[1]))
    img, truth = ribbons.render(scene)
    fig, ax = plt.subplots(figsize=(9, 7.2))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.plot(truth["points"][:, 0], truth["points"][:, 1], "-", color="lime", lw=0.8, label="truth")
    colours = {"sato": "cyan", "meijering": "orange", "ridge_detector": "magenta"}
    for src, found in res["overlay"].items():
        for i, p in enumerate(found):
            ax.plot(p["points"][:, 0], p["points"][:, 1], "-", color=colours.get(src, "red"), lw=1,
                    label=src if i == 0 else None)
    ax.set_title(res["id"], fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    ax.axis("off")
    fig.tight_layout()
    odir = out_dir / "acquire_overlays"
    odir.mkdir(parents=True, exist_ok=True)
    fig.savefig(odir / f"{res['id']}.png", dpi=110)
    plt.close(fig)
