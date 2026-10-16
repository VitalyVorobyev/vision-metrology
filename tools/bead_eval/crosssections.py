"""Cross-section stations for the public two-boundary benchmark: HRF vessels, DamSegment cracks.

Per image of the registry split: the dataset's own CC BY mask -> `paths.py`'s skeleton
graph, split at junctions into non-branching centreline segments -> candidate stations
every `step` px along each segment, kept when they are away from the segment's ends and
from every junction, locally straight, within the width range, isolated across the normal and inside the
frame -> a deterministic, seeded selection stratified by width and contrast terciles,
separately in dev and test.

It writes two things:
- `data/<id>/crosssections/`: the neutral `sequence.json` that `annotate.py` opens, its
  `curves.json`, and the single-channel frames, visibility and masks. A frame is one
  centreline segment; the frames of one image share its file. Never committed.
- `tools/datasets/annotations/<id>/annotations.json`, once: the frozen stations, with no
  annotation. It is committed before anyone annotates; a rerun compares and never
  overwrites it, and `--check` writes nothing at all.

    python -I tools/bead_eval/crosssections.py hrf --data-root data
    python -I tools/bead_eval/crosssections.py damsegment --data-root data --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "registration_eval"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

import annotation  # noqa: E402
import common  # noqa: E402
import paths  # noqa: E402

REPO = HERE.parent.parent
DATASETS_DIR = REPO / "tools" / "datasets"
SEED = "vision-metrology-crosssections-v1"
DEV_FRACTION = 0.3
TERCILES = (1 / 3, 2 / 3)
RULE = ("candidates every `step` px of each centreline segment, at least max(10, 2w) px from "
        "its ends and max(15, 2w + 6) px from any skeleton junction, with a chord over "
        "+-max(8, 1.5w) px at least `straightness` of its arc, a width w in `width`, one mask "
        "run across the normal at the station and w/2 either side, the profile inside the "
        "frame and the visibility mask, and a contrast of at least `min_contrast`; per split, "
        "an equal quota per width x contrast tercile cell, candidates in sha256 rank order, "
        "at most ceil(1.5 count / images) per image and 40 px apart on one image")

#: Per dataset: the skeleton graph, the candidate rules and the profile the annotator sees.
CONFIGS = {
    "hrf": {
        "count": 120, "width": (3.0, 20.0), "half_length": 20.0, "step": 10.0,
        "min_length": 40.0, "spur_px": 12.0, "junction_trim": 8.0, "smooth_px": 3.0,
        "min_contrast": 2.0, "straightness": 0.95, "polarity": "dark",
        "image": "the green channel, 8-bit", "visibility": "the dataset's field-of-view mask",
    },
    "damsegment": {
        "count": 120, "width": (3.0, 14.0), "half_length": 12.0, "step": 10.0,
        "min_length": 80.0, "spur_px": 12.0, "junction_trim": 6.0, "smooth_px": 3.0,
        "min_contrast": 2.0, "straightness": 0.95, "polarity": "dark",
        "image": "BT.601 luma, float32 (tools/bead_eval/common.py load_gray)",
        "visibility": "the whole frame",
    },
}


# The datasets -------------------------------------------------------------------------

def hrf_load(folder: pathlib.Path, image_id: str):
    """(green channel uint8, vessel mask, field-of-view mask) of one HRF image."""
    root = folder / "extracted"
    image = next(p for p in (root / "images").iterdir() if p.stem == image_id)
    rgb = np.asarray(Image.open(image).convert("RGB"))
    vessels = np.asarray(Image.open(root / "manual1" / f"{image_id}.tif").convert("L")) > 127
    fov = np.asarray(Image.open(root / "mask" / f"{image_id}_mask.tif").convert("L")) > 127
    return np.ascontiguousarray(rgb[..., 1]), vessels, fov


def damsegment_load(folder: pathlib.Path, image_id: str):
    """(BT.601 luma float32, crack mask, None) of one DamSegment image `<difficulty>/<stem>`."""
    root = common.dataset_root(folder / "extracted")
    difficulty, stem = image_id.split("/")
    gray = common.load_gray(common.image_path(root, difficulty, stem))
    return gray, common.load_crack_mask(common.mask_path(root, difficulty, stem)), None


LOADERS = {"hrf": hrf_load, "damsegment": damsegment_load}


def registry_entry(dataset: str) -> dict:
    entries = json.loads((DATASETS_DIR / "registry.json").read_text(encoding="utf-8"))["datasets"]
    entry = next(e for e in entries if e["id"] == dataset)
    if entry["status"] != "verified":
        sys.exit(f"{dataset}: registered as '{entry['status']}', not 'verified'")
    return entry


def split_ids(entry: dict) -> dict[str, list[str]]:
    split = json.loads((DATASETS_DIR / entry["split"]).read_text(encoding="utf-8"))
    return {"dev": split["dev"], "test": split["test"]}


# Candidates ---------------------------------------------------------------------------

def sample(array, station, normal, t, order=1):
    points = station + t[:, None] * normal
    return ndimage.map_coordinates(array, [points[:, 1], points[:, 0]], order=order, mode="nearest")


def isolated(mask, station, normal, t, width):
    """True when the mask, across the normal, is one run through the station with nothing
    else in reach: no parallel structure or crossing inside the profile."""
    on = sample(mask, station, normal, t, order=0) > 0.5
    centre = len(t) // 2
    if not on[centre]:
        return False
    lo = hi = centre
    while lo > 0 and on[lo - 1]:
        lo -= 1
    while hi < len(t) - 1 and on[hi + 1]:
        hi += 1
    run = t[hi] - t[lo]
    return not on[:lo].any() and not on[hi + 1:].any() and run < 2.5 * width + 2


def contrast(image, station, normal, width, reach):
    """Background minus core, in grey levels, on the weaker side: the median at
    `w/2 + 2 … reach` px either side, against the mean within `max(0.5, w/4)` px."""
    t = np.arange(-reach, reach + 1e-9, 0.25)
    values = sample(image, station, normal, t)
    core = values[np.abs(t) <= max(0.5, width / 4)].mean()
    outer = np.abs(t) >= width / 2 + 2
    left, right = np.median(values[outer & (t < 0)]), np.median(values[outer & (t > 0)])
    return float(min(left, right) - core)


def candidates(dataset: str, image_id: str, split: str, image, mask, fov, cfg: dict) -> list[dict]:
    """Every eligible station of one image's centreline segments."""
    args = argparse.Namespace(**{k: cfg[k] for k in ("spur_px", "min_length", "junction_trim", "smooth_px")})
    segments, _ = paths.image_paths(mask, args)
    # The skeleton's junctions, crossings included: a station keeps clear of every one.
    adjacency = paths.adjacency(paths.prune_spurs(skeletonize(mask), cfg["spur_px"]))
    nodes = [(c, r) for (r, c), nb in adjacency.items() if len(nb) >= 3]
    junctions = cKDTree(np.asarray(nodes, dtype=float)) if nodes else None
    h, w = mask.shape
    # Converted once: `sample` reads them at a few points per call.
    image, mask = np.asarray(image, dtype=np.float32), mask.astype(np.float32)
    fov = None if fov is None else fov.astype(np.float32)
    lo_w, hi_w = cfg["width"]
    out = []
    for index, seg in enumerate(segments):
        polyline = np.round(np.asarray(seg["points"], dtype=float), 3)
        arc = annotation.arc_lengths(polyline)
        widths = np.asarray(seg["width"], dtype=float)
        for s in np.arange(cfg["step"] / 2, arc[-1], cfg["step"]):
            width = float(np.interp(s, arc, widths))
            margin = max(10.0, 2 * width)
            if not lo_w <= width <= hi_w or s < margin or s > arc[-1] - margin:
                continue
            half = max(8.0, 1.5 * width)
            chord = np.hypot(*(annotation.point_at(polyline, s + half) - annotation.point_at(polyline, s - half)))
            if chord < cfg["straightness"] * 2 * half:
                continue
            station, normal = annotation.point_at(polyline, s), annotation.normal_at(polyline, s)
            if junctions is not None and junctions.query(station)[0] < max(15.0, 2 * width + 6):
                continue
            reach = min(width / 2 + 8, cfg["half_length"])
            t = np.arange(-reach, reach + 1e-9, 0.25)
            ends = station + np.array([[-reach], [reach]]) * normal
            if (ends < 2).any() or (ends[:, 0] > w - 3).any() or (ends[:, 1] > h - 3).any():
                continue
            if fov is not None and not (sample(fov, station, normal, t, order=0) > 0.5).all():
                continue
            tangent = np.array([normal[1], -normal[0]])
            if not all(isolated(mask, station + d * tangent, normal, t, width) for d in (-width / 2, 0, width / 2)):
                continue
            c = contrast(image, station, normal, width, reach)
            if c < cfg["min_contrast"]:
                continue
            out.append({"image": image_id, "segment": index, "split": split, "s": round(float(s), 3),
                        "width": round(width, 3), "contrast": round(c, 3), "polyline": polyline,
                        "station": station, "width_median": seg["width_median"]})
    return out


# Selection ----------------------------------------------------------------------------

def rank_key(*parts) -> str:
    return hashlib.sha256("|".join([SEED, *map(str, parts)]).encode("utf-8")).hexdigest()


def cell(c: dict, edges: dict) -> tuple[int, int]:
    return (int(np.searchsorted(edges["width"], c["width"], side="right")),
            int(np.searchsorted(edges["contrast"], c["contrast"], side="right")))


def select(dataset: str, pool: list[dict], count: int, split: str, edges: dict) -> list[dict]:
    """`count` stations of one split: an equal quota per width × contrast cell, each cell's
    candidates in seeded rank order, at most `cap` stations per image and none within
    `40 px` of another on the same image. An under-filled cell's quota goes to the rest."""
    pool = sorted(pool, key=lambda c: rank_key(dataset, c["image"], c["segment"], c["s"]))
    images = {c["image"] for c in pool}
    cap = max(1, int(np.ceil(1.5 * count / max(len(images), 1))))
    cells = sorted({(i, j) for i in range(3) for j in range(3)}, key=lambda k: rank_key(dataset, split, k))
    quota = {k: count // len(cells) + (n < count % len(cells)) for n, k in enumerate(cells)}
    by_cell = {k: [c for c in pool if cell(c, edges) == k] for k in cells}
    picked: list[dict] = []
    taken: set[int] = set()

    def admissible(c):
        same = [p for p in picked if p["image"] == c["image"]]
        return len(same) < cap and all(np.hypot(*(p["station"] - c["station"])) >= 40 for p in same)

    def take(k):
        for c in by_cell[k]:
            if id(c) not in taken and admissible(c):
                taken.add(id(c))
                picked.append(c)
                return True
        return False

    for k in cells:
        while sum(cell(p, edges) == k for p in picked) < quota[k] and take(k):
            pass
    while len(picked) < count:
        open_cells = sorted(cells, key=lambda k: (sum(cell(p, edges) == k for p in picked), cells.index(k)))
        if not any(take(k) for k in open_cells):
            break
    return picked


# Output -------------------------------------------------------------------------------

def file_name(image_id: str) -> str:
    return image_id.replace("/", "__").replace(" ", "_")


def pools(dataset: str, data_root: pathlib.Path) -> dict[str, list[dict]]:
    """Every candidate of the registry split's images, per split."""
    cfg, entry = CONFIGS[dataset], registry_entry(dataset)
    out = {"dev": [], "test": []}
    for split, ids in split_ids(entry).items():
        for image_id in ids:
            image, mask, fov = LOADERS[dataset](data_root / dataset, image_id)
            out[split] += candidates(dataset, image_id, split, image, mask, fov, cfg)
        print(f"{dataset} {split}: {len(ids)} images, {len(out[split])} candidates")
    return out


def choose(dataset: str, pooled: dict[str, list[dict]], count: int):
    """(stations, tercile edges): the stratified stations of both splits."""
    every = pooled["dev"] + pooled["test"]
    edges = {k: [round(float(v), 3) for v in np.quantile([c[k] for c in every], TERCILES)]
             for k in ("width", "contrast")}
    n_dev = round(DEV_FRACTION * count)
    chosen = (select(dataset, pooled["dev"], n_dev, "dev", edges)
              + select(dataset, pooled["test"], count - n_dev, "test", edges))
    return chosen, edges


def assemble(dataset: str, chosen: list[dict], edges: dict, cfg: dict, citation: str, seed: int = 0):
    """(sections, protocol, curves, frame keys): one frame per (image, segment) with a station."""
    keys = sorted({(c["image"], c["segment"]) for c in chosen}, key=lambda k: (paths_natural(k[0]), k[1]))
    frame_of = {k: i for i, k in enumerate(keys)}
    chosen = sorted(chosen, key=lambda c: (frame_of[(c["image"], c["segment"])], c["s"]))
    sections = [annotation.section(frame_of[(c["image"], c["segment"])], c["polyline"], c["s"], "stratified")
                for c in chosen]
    for entry_, c in zip(sections, chosen):
        entry_["cell"] = list(cell(c, edges))
    first = {k: next(c for c in chosen if (c["image"], c["segment"]) == k) for k in keys}
    counts = {split: {"stations": sum(c["split"] == split for c in chosen),
                      "images": len({c["image"] for c in chosen if c["split"] == split}),
                      "cells": {f"{i}{j}": sum(c["split"] == split and cell(c, edges) == (i, j) for c in chosen)
                                for i in range(3) for j in range(3)}}
              for split in ("dev", "test")}
    protocol = {
        "dataset": dataset, "licence": "CC-BY-4.0", "source": citation,
        "seed": seed, "count": len(sections), "curves": "curves.json", "profile": "cubic",
        "step_px": annotation.STEP, "half_length_px": cfg["half_length"],
        "selection": {"rule": RULE, "generator": "tools/bead_eval/crosssections.py",
                      "seed": SEED, "config": {k: v for k, v in cfg.items() if k not in ("image", "visibility")},
                      "edges": edges, "cells": "cell [i, j]: width tercile i, contrast tercile j (0 lowest)",
                      "counts": counts},
        "frames": [{"image": k[0], "segment": k[1], "split": first[k]["split"]} for k in keys],
    }
    curves = [{"frame": frame_of[k], "polyline": first[k]["polyline"].tolist(), "image": k[0], "segment": k[1],
               "split": first[k]["split"], "width_median": first[k]["width_median"],
               "mask": f"masks/{file_name(k[0])}.png"} for k in keys]
    return sections, protocol, curves, keys


def paths_natural(text: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def write_data(out: pathlib.Path, dataset, load, curves, keys, cfg, citation):
    """The manifest, curves and frames under `out`; `load(image_id)` reads one image."""
    for sub in ("frames", "visibility", "masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    frames, kinds = [], {}
    for image_id in sorted({k[0] for k in keys}, key=paths_natural):
        image, mask, fov = load(image_id)
        kinds[image_id] = ("png" if image.dtype == np.uint8 else "npy", fov is not None)
        name = file_name(image_id)
        if image.dtype == np.uint8:
            Image.fromarray(image).save(out / "frames" / f"{name}.png")
        else:
            np.save(out / "frames" / f"{name}.npy", image.astype(np.float32))
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / "masks" / f"{name}.png")
        if fov is not None:
            Image.fromarray(fov.astype(np.uint8) * 255).save(out / "visibility" / f"{name}.png")
    for image_id, _ in keys:
        name = file_name(image_id)
        ext, has_fov = kinds[image_id]
        frame = {"image": f"frames/{name}.{ext}", "group": 0}
        if has_fov:
            frame["visibility"] = f"visibility/{name}.png"
        frames.append(frame)
    manifest = {
        "version": 1, "name": f"{dataset}-crosssections", "convention": "pixel_centres",
        "provenance": {
            "dataset": dataset, "citation": citation, "licence": "CC-BY-4.0",
            "frames": f"one per centreline segment; the frames of one image share its file. {cfg['image']}",
            "visibility": cfg["visibility"], "generator": "tools/bead_eval/crosssections.py",
            "annotations": f"tools/datasets/annotations/{dataset}/annotations.json",
        },
        "frames": frames,
    }
    (out / "sequence.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "curves.json").write_text(json.dumps({"curves": curves}) + "\n")


def same_sections(a: list[dict], b: list[dict]) -> bool:
    if len(a) != len(b):
        return False
    return all(x["frame"] == y["frame"] and np.allclose(x["station"], y["station"], atol=2e-3)
               and np.allclose(x["normal"], y["normal"], atol=1e-4) for x, y in zip(a, b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(CONFIGS))
    parser.add_argument("--data-root", type=pathlib.Path, default=REPO / "data",
                        help="the folder that holds data/<id>/ (default: data/ in this checkout)")
    parser.add_argument("--check", action="store_true", help="recompute and compare; write nothing")
    args = parser.parse_args()
    cfg, entry = CONFIGS[args.dataset], registry_entry(args.dataset)
    chosen, edges = choose(args.dataset, pools(args.dataset, args.data_root), cfg["count"])
    sections, protocol, curves, keys = assemble(args.dataset, chosen, edges, cfg, entry["citation"])
    committed = DATASETS_DIR / "annotations" / args.dataset / "annotations.json"
    counts = protocol["selection"]["counts"]
    print(f"{args.dataset}: {len(sections)} stations on {len(keys)} segments; "
          f"dev {counts['dev']['stations']} on {counts['dev']['images']} images, "
          f"test {counts['test']['stations']} on {counts['test']['images']} images; "
          f"edges {protocol['selection']['edges']}")
    if committed.exists():
        doc = annotation.read(committed)
        if not same_sections(doc["sections"], sections) or doc["protocol"].get("frames") != protocol["frames"]:
            print(f"{committed.relative_to(REPO)}: its stations differ from the ones recomputed", file=sys.stderr)
            return 1
        print(f"{committed.relative_to(REPO)}: its stations reproduce")
    elif args.check:
        print(f"{committed.relative_to(REPO)}: missing", file=sys.stderr)
        return 1
    if args.check:
        return 0
    if not committed.exists():
        committed.parent.mkdir(parents=True, exist_ok=True)
        annotation.Session.create(committed, sections, protocol)
        print(f"wrote {committed.relative_to(REPO)}: frozen, commit it before annotating")
    out = args.data_root / args.dataset / "crosssections"
    def load(image_id):
        return LOADERS[args.dataset](args.data_root / args.dataset, image_id)
    write_data(out, args.dataset, load, curves, keys, cfg, entry["citation"])
    print(f"wrote {out / 'sequence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
