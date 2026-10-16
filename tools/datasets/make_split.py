#!/usr/bin/env python3
"""Write a dataset's dev/test split file, `splits/<id>.json`, from its fetched data.

    python3 tools/datasets/make_split.py damsegment
    python3 tools/datasets/make_split.py hpatches --check   # recompute, compare, write nothing

A split holds IDs only, never data. The rule is fixed and stated in the file, so anyone
with the data reproduces it:
- the IDs are grouped into strata (DamSegment: its difficulty; HPatches: illumination
  or viewpoint; HRF: the subject group);
- within a stratum, IDs are ranked by `sha256("<seed>|<id>")`, and the first
  `round(dev_fraction * n)` are dev, the rest test.

A split is frozen once committed: settings are chosen on dev before test is ever run.
This script therefore refuses to overwrite an existing split file. `--check` recomputes
it from the data and fails on any difference.

Standard library only (Python 3.9+).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
SEED = "vision-metrology-split-v1"
DEV_FRACTION = 0.3


def natural(s: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def damsegment_ids(folder: pathlib.Path) -> dict[str, list[str]]:
    """Images, as `<difficulty>/<stem>`, per difficulty. One image holds several crack
    paths, so the split is by image: paths of one image never straddle dev and test."""
    root = folder / "extracted" / "Damage Segmentaion"
    out = {}
    for diff in ("Easy", "Medium", "Hard"):
        stems = sorted((p.stem for p in (root / diff / "Images").glob("*.jpg")), key=natural)
        out[diff] = [f"{diff}/{s}" for s in stems]
    return out


def hpatches_ids(folder: pathlib.Path) -> dict[str, list[str]]:
    """Sequences, by their folder names, per kind: `i_` illumination, `v_` viewpoint.
    Every pair is drawn within one sequence, so the split is by sequence."""
    root = folder / "extracted" / "hpatches-sequences-release"
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    return {
        "illumination": [n for n in names if n.startswith("i_")],
        "viewpoint": [n for n in names if n.startswith("v_")],
    }


def hrf_ids(folder: pathlib.Path) -> dict[str, list[str]]:
    """Images, by stem (`01_h`), per group: healthy `_h`, diabetic retinopathy `_dr` and
    glaucoma `_g`. One image yields many cross-sections, so the split is by image."""
    root = folder / "extracted" / "images"
    stems = sorted((p.stem for p in root.iterdir() if p.suffix.lower() == ".jpg"), key=natural)
    return {
        group: [s for s in stems if s.endswith(f"_{suffix}")]
        for group, suffix in (("healthy", "h"), ("diabetic_retinopathy", "dr"), ("glaucoma", "g"))
    }


LISTERS = {
    "damsegment": (damsegment_ids, "image"),
    "hrf": (hrf_ids, "image"),
    "hpatches": (hpatches_ids, "sequence"),
}


def rank(ids: list[str]) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(f"{SEED}|{i}".encode("utf-8")).hexdigest())


def build(dataset: str, strata: dict[str, list[str]], unit: str) -> dict:
    dev: list[str] = []
    test: list[str] = []
    counts = {}
    for name, ids in strata.items():
        if not ids:
            sys.exit(f"{dataset}: stratum '{name}' is empty; is the data extracted?")
        ranked = rank(ids)
        k = round(DEV_FRACTION * len(ranked))
        dev.extend(sorted(ranked[:k], key=natural))
        test.extend(sorted(ranked[k:], key=natural))
        counts[name] = {"dev": k, "test": len(ranked) - k}
    return {
        "dataset": dataset,
        "unit": unit,
        "rule": (
            f"per stratum, rank IDs by sha256('{SEED}|<id>'); the first "
            f"round({DEV_FRACTION} * n) are dev, the rest test"
        ),
        "strata": counts,
        "dev": dev,
        "test": test,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("id", choices=sorted(LISTERS))
    parser.add_argument("--check", action="store_true", help="recompute and compare; write nothing")
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=REPO / "data",
        help="the folder that holds data/<id>/ (default: data/ in this checkout)",
    )
    args = parser.parse_args()
    lister, unit = LISTERS[args.id]
    split = build(args.id, lister(args.data_root / args.id), unit)
    path = HERE / "splits" / f"{args.id}.json"
    text = json.dumps(split, indent=1) + "\n"
    if args.check:
        if not path.is_file():
            sys.exit(f"{path.name}: missing")
        if path.read_text(encoding="utf-8") != text:
            print(f"{path.name}: differs from the split recomputed from the data", file=sys.stderr)
            return 1
        print(f"{path.name}: reproduces ({len(split['dev'])} dev, {len(split['test'])} test)")
        return 0
    if path.exists():
        sys.exit(f"{path.name} exists and is frozen; use --check to compare")
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.name}: {len(split['dev'])} dev, {len(split['test'])} test, {split['strata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
