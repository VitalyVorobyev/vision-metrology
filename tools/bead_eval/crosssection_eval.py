"""Wall accuracy on the public two-boundary benchmark: the tracker's walls against
hand-annotated cross-sections, per pairing mode, with the annotator's σ beside them.

For each frame (a centreline segment) with annotated stations in the chosen split, it
runs `BeadTracker` along the segment's curve, once per mode:
- `tracker`: per-station pairing, with `run_tracker.py`'s settings;
- `coupled`: the same settings with coupled pairing (`BeadConfig.pairing`) and one
  tracking pass, the setting frozen on DamSegment's dev split.

Each annotated station's walls are where the station's normal crosses the tracker's two
wall polylines, between consecutive measured stations; a station in a gap has no wall.
`annotations_eval.evaluate` then scores them: wall bias and σ, width error, coverage and
the annotator's two-pass σ. 90% intervals come from a bootstrap over images, since the
stations of one image share its errors.

    python -I tools/bead_eval/crosssection_eval.py --manifest data/hrf/crosssections/sequence.json \\
        --annotations tools/datasets/annotations/hrf/annotations.json --split dev --out /tmp/hrf-dev.json

`--split test` needs `--held-out`: run it once, with the settings frozen on dev. Outputs
must lie outside the repository. The report holds aggregates only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "registration_eval"))

import numpy as np  # noqa: E402

import annotation  # noqa: E402
import annotations_eval  # noqa: E402
import common  # noqa: E402
from manifest import Sequence  # noqa: E402

REPO = HERE.parent.parent
MODES = ("tracker", "coupled")
#: Width range per dataset, as factors of a segment's median mask width `w`:
#: `(lo, hi, floor)` gives `min_width = max(floor, lo·w)` and `max_width = hi·w`.
WIDTHS = {"damsegment": (0.25, 2.0, 2.0), "hrf": (0.5, 2.0, 2.0)}
#: Provisional per-dataset settings, until they are frozen on dev annotations. DamSegment
#: keeps `run_tracker.py`'s and the coupled defaults. HRF's green channel has vessel edges
#: of a few grey levels per pixel, so its edge threshold and the coupled evidence scale
#: (`saturation`, `absent_level`, grey levels per px) are lower; chosen on dev only.
DEFAULTS = {
    "damsegment": {"threshold": 3.0, "coupling": {}},
    "hrf": {"threshold": 0.5, "coupling": {"absent_level": 1.0, "saturation": 3.0}},
}
BOOTSTRAP = 2000
CROP_MARGIN = 40


def tracker_for(mode: str, polarity: str, lo: float, hi: float, args):
    """A `BeadTracker` for one mode: `common.bead_config`'s settings, and for `coupled`
    the default `BeadCoupling` with `--coupled-passes` tracking passes."""
    import vision_metrology as vm

    base = common.bead_config(polarity, lo, hi, args)
    if mode == "tracker":
        return vm.BeadTracker(base)
    tuning = base.tuning
    tuning.passes = args.coupled_passes
    cfg = vm.BeadConfig(polarity=base.polarity, min_width=base.min_width, max_width=base.max_width,
                        spacing=base.spacing, track=base.track, measure=base.measure, tuning=tuning,
                        pairing=vm.BeadCoupling(**args.coupling))
    return vm.BeadTracker(cfg)


def side_walls(bead) -> tuple[np.ndarray, np.ndarray]:
    """The −n and +n wall points per station, NaN where none; a one-sided station's
    single wall counts on its side."""
    first, second = np.array(bead.first, dtype=float), np.array(bead.second, dtype=float)
    wall = np.asarray(getattr(bead, "wall", np.full_like(first, np.nan)), dtype=float)
    for i, side in enumerate(getattr(bead, "wall_side", [None] * len(first))):
        if side == "first":
            first[i] = wall[i]
        elif side == "second":
            second[i] = wall[i]
    return first, second


def crossing(points: np.ndarray, station, normal, reach: float):
    """Where the line `station + t·normal` crosses the polyline `points`, between
    consecutive finite points: the crossing nearest the station within `reach`, or None."""
    station, normal = np.asarray(station, float), np.asarray(normal, float)
    a, b = points[:-1], points[1:]
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    d = b - a
    # Solve station + t n = a + u d for (t, u).
    det = normal[0] * (-d[:, 1]) + normal[1] * d[:, 0]
    rel = a - station
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (rel[:, 0] * (-d[:, 1]) + rel[:, 1] * d[:, 0]) / det
        u = (normal[0] * rel[:, 1] - normal[1] * rel[:, 0]) / det
    hit = ok & (np.abs(det) > 1e-9) & (u >= 0) & (u <= 1) & (np.abs(t) <= reach)
    if not hit.any():
        return None
    k = np.flatnonzero(hit)[np.argmin(np.abs(t[hit]))]
    return (station + t[k] * normal).tolist()


def run_frame(image, curve: dict, sections: list[dict], mode: str, polarity: str, reach: float, args):
    """One mode on one frame: a results row per annotated section, and the call's time."""
    lo_f, hi_f, floor = WIDTHS[args.dataset]
    w = float(curve["width_median"])
    tracker = tracker_for(mode, polarity, max(floor, lo_f * w), hi_f * w, args)
    prior = np.asarray(curve["polyline"], dtype=np.float64)
    pad = CROP_MARGIN + hi_f * w + args.reach
    x0, y0 = (np.maximum(np.floor(prior.min(axis=0) - pad), 0)).astype(int)
    x1, y1 = np.minimum(np.ceil(prior.max(axis=0) + pad) + 1, image.shape[::-1]).astype(int)
    crop = np.ascontiguousarray(image[y0:y1, x0:x1], dtype=np.float32)
    origin = np.array([x0, y0], dtype=float)
    t = time.perf_counter()
    bead = tracker.track(crop, prior - origin)
    seconds = time.perf_counter() - t
    walls = [w_ + origin for w_ in side_walls(bead)]
    rows = []
    for sec in sections:
        found = [crossing(points, sec["station"], sec["normal"], reach) for points in walls]
        # The prior is the annotated curve itself, so the tracker's −n side is the
        # annotation's left; two walls that cross over are put back in order.
        offsets = [None if p is None else float(np.dot(np.subtract(p, sec["station"]), sec["normal"])) for p in found]
        if None not in offsets and offsets[0] > offsets[1]:
            found = found[::-1]
        rows.append({"frame": sec["frame"], "station": sec["station"], "normal": sec["normal"],
                     "left": found[0], "right": found[1]})
    return rows, seconds


def per_section(annotations, rows) -> list[dict]:
    """Per annotated section: its image, and per side the reference state and error."""
    report = annotations_eval.evaluate(annotations, rows, per_section=True)
    refs = annotations_eval.references(annotations)
    by_key = {(r["frame"], *map(float, r["station"])): r for r in report["per_section"]}
    out = []
    for k, ref in refs.items():
        row = by_key.get((ref["frame"], *map(float, ref["station"])))
        if row is not None:
            out.append({"frame": ref["frame"], "states": [ref["left_state"], ref["right_state"]],
                        "errors": [row.get("left_error"), row.get("right_error")],
                        "width_error": row.get("width_error")})
    return out


def pooled(records) -> dict:
    walls = [e for r in records for e, s in zip(r["errors"], r["states"]) if s == "visible" and e is not None]
    visible = sum(s == "visible" for r in records for s in r["states"])
    widths = [r["width_error"] for r in records if r["width_error"] is not None]
    return {"wall_bias": np.mean(walls) if walls else np.nan,
            "wall_sigma": np.std(walls, ddof=1) if len(walls) > 1 else np.nan,
            "width_bias": np.mean(widths) if widths else np.nan,
            "width_sigma": np.std(widths, ddof=1) if len(widths) > 1 else np.nan,
            "coverage": len(walls) / visible if visible else np.nan}


def bootstrap(records, image_of, seed=0) -> dict:
    """90% percentile intervals of `pooled`'s statistics, resampling images."""
    images = sorted({image_of[r["frame"]] for r in records})
    groups = {i: [r for r in records if image_of[r["frame"]] == i] for i in images}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTSTRAP if len(images) > 1 else 0):
        pick = rng.integers(len(images), size=len(images))
        draws.append(pooled([r for j in pick for r in groups[images[j]]]))
    out = {}
    for key in pooled(records):
        values = np.array([d[key] for d in draws], dtype=float)
        values = values[np.isfinite(values)]
        out[key] = [round(float(v), 4) for v in np.quantile(values, (0.05, 0.95))] if values.size else None
    return out


def evaluate(sequence, doc, curves, args) -> dict:
    """Every mode on the split's annotated frames: one aggregate report."""
    protocol = doc["protocol"]
    frames = protocol["frames"]
    keep = {i for i, f in enumerate(frames) if f["split"] == args.split}
    annotations = [a for a in doc["annotations"] if a["frame"] in keep
                   and (args.annotator is None or a["annotator"] == args.annotator)]
    if not annotations:
        raise SystemExit(f"no annotations in the {args.split} split")
    image_of = {i: f["image"] for i, f in enumerate(frames)}
    refs = annotations_eval.references(annotations)
    by_frame: dict[int, list] = {}
    for ref in refs.values():
        by_frame.setdefault(ref["frame"], []).append(ref)
    reach = float(protocol.get("half_length_px", annotation.HALF_LENGTH))
    polarity = protocol.get("selection", {}).get("config", {}).get("polarity", "dark")
    report = {"dataset": protocol["dataset"], "split": args.split, "sections": len(refs),
              "frames": len(by_frame), "images": len({image_of[f] for f in by_frame}),
              "synthetic": bool(protocol.get("synthetic")),
              "settings": {"reach": args.reach, "spacing": args.spacing, "threshold": args.threshold,
                           "sigma": args.sigma, "obliquity": args.obliquity, "widths": WIDTHS[args.dataset],
                           "coupled_passes": args.coupled_passes, "coupling": args.coupling},
              "annotator": annotations_eval.annotator_spread(annotations), "modes": {}}
    for mode in args.modes:
        rows, times = [], []
        for frame, sections in sorted(by_frame.items()):
            got, seconds = run_frame(sequence.image(frame), curves[frame], sections, mode, polarity, reach, args)
            rows += got
            times.append(seconds)
        result = annotations_eval.evaluate(annotations, rows)
        result.pop("annotator")
        result["intervals_90"] = bootstrap(per_section(annotations, rows), image_of)
        result["median_ms"] = round(1e3 * float(np.median(times)), 3)
        report["modes"][mode] = result
    return report


def summary(report) -> str:
    a = report["annotator"]
    lines = [f"{report['dataset']} {report['split']}{' (SYNTHETIC annotations)' if report['synthetic'] else ''}: "
             f"{report['sections']} sections on {report['images']} images; annotator σ {a['sigma']} px "
             f"(width {a['width_sigma']}) over {a['pairs']} repeated sections"]
    for mode, r in report["modes"].items():
        w, width, cov, ci = r["walls"]["both"], r["width"], r["coverage"], r["intervals_90"]
        lines.append(f"  {mode:8s} walls n={w['n']} bias {w['bias']} {ci['wall_bias']} σ {w['sigma']} "
                     f"{ci['wall_sigma']} px; width bias {width['bias']} σ {width['sigma']} px; "
                     f"coverage {cov['left']['rate']}/{cov['right']['rate']} {ci['coverage']}; {r['median_ms']} ms")
    return "\n".join(lines)


def coupling_item(text: str) -> tuple[str, float]:
    key, value = text.split("=")
    return key, float(value)


def main() -> None:
    parser = common.add_tracker_args(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--annotations", type=pathlib.Path, required=True)
    parser.add_argument("--curves", type=pathlib.Path, help="default: the protocol's, beside the manifest")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--held-out", action="store_true", help="allow --split test: once, frozen settings")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--annotator", help="only this annotator (default: all)")
    parser.add_argument("--coupled-passes", type=int, default=1, help="coupled tracking passes (default 1)")
    parser.add_argument("--coupling", type=coupling_item, action="append", default=[],
                        help="KEY=VALUE: a BeadCoupling field for the coupled mode (repeatable)")
    parser.add_argument("--out", type=pathlib.Path, help="write the report as JSON here")
    parser.set_defaults(threshold=None)
    args = parser.parse_args()
    if args.split == "test" and not args.held_out:
        parser.error("the test split needs --held-out; freeze the settings on dev first")
    if args.out is not None and args.out.resolve().is_relative_to(REPO):
        parser.error("outputs must be outside the repository")
    sequence = Sequence(args.manifest)
    doc = annotation.read(args.annotations, len(sequence))
    args.dataset = doc["protocol"]["dataset"]
    defaults = DEFAULTS[args.dataset]
    if args.threshold is None:
        args.threshold = defaults["threshold"]
    args.coupling = {**defaults["coupling"], **dict(args.coupling)}
    curves_path = args.curves or sequence.root / doc["protocol"].get("curves", "curves.json")
    curves = {c["frame"]: c for c in json.loads(curves_path.read_text())["curves"]}
    report = evaluate(sequence, doc, curves, args)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(summary(report))


if __name__ == "__main__":
    main()
