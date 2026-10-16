"""SYNTHETIC annotations for testing `crosssection_eval.py`: mask walls plus noise.

Not annotations. For every frozen station it takes the two walls of the dataset's own
mask along the station's normal, where the bilinear mask crosses 0.5 on either side of
the run through the station, and adds independent Gaussian noise per pass. Both passes
are written under the annotator `synthetic-mask`, and the protocol is marked
`"synthetic": true`. The file is written outside the repository only: it must never be
committed or reported as a measurement.

    python -I tools/bead_eval/crosssection_synth.py --manifest data/hrf/crosssections/sequence.json \\
        --annotations tools/datasets/annotations/hrf/annotations.json --out /tmp/hrf-synthetic.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "registration_eval"))

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

import annotation  # noqa: E402
from manifest import Sequence, load_array  # noqa: E402

REPO = HERE.parent.parent
ANNOTATOR = "synthetic-mask"


def mask_walls(mask, station, normal, half_length):
    """(left, right): the mask's 0.5 crossings either side of the run through the station,
    as offsets along the normal, or None where the run reaches the profile's end."""
    t = annotation.offsets(half_length)
    points = np.asarray(station, float) + t[:, None] * np.asarray(normal, float)
    values = ndimage.map_coordinates(mask, [points[:, 1], points[:, 0]], order=1, mode="nearest")
    inside = values > 0.5
    if not inside.any():
        return None, None
    centre = int(np.argmin(np.where(inside, np.abs(t), np.inf)))
    lo = hi = centre
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    while hi < len(t) - 1 and inside[hi + 1]:
        hi += 1

    def cross(i, j):  # the 0.5 crossing between samples i (outside) and j (inside)
        return float(t[i] + (0.5 - values[i]) / (values[j] - values[i]) * (t[j] - t[i]))
    left = cross(lo - 1, lo) if lo > 0 else None
    right = cross(hi + 1, hi) if hi < len(t) - 1 else None
    return left, right


def synthesize(sequence, doc, curves, sigma, seed):
    """A synthetic annotations document: the frozen sections, two noisy mask passes each."""
    half = float(doc["protocol"].get("half_length_px", annotation.HALF_LENGTH))
    rng = np.random.default_rng(seed)
    masks: dict[str, np.ndarray] = {}
    out = {"version": annotation.VERSION, "sections": doc["sections"], "annotations": [],
           "protocol": {**doc["protocol"], "synthetic": True,
                        "synthetic_note": f"mask walls plus N(0, {sigma} px) per pass; NOT annotations"}}
    for sec in doc["sections"]:
        rel = curves[sec["frame"]]["mask"]
        if rel not in masks:
            masks[rel] = (load_array(sequence.root / rel) > 0).astype(np.float32)
        left, right = mask_walls(masks[rel], sec["station"], sec["normal"], half)
        for pass_ in (1, 2):
            noisy = [None if v is None else v + rng.normal(0, sigma) for v in (left, right)]
            if None not in noisy and noisy[0] >= noisy[1]:
                noisy = [None, None]
            out["annotations"].append(annotation.record(sec, noisy[0], noisy[1], ANNOTATOR, pass_,
                                                        "1970-01-01T00:00:00+00:00"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--annotations", type=pathlib.Path, required=True, help="the frozen stations")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="outside the repository")
    parser.add_argument("--sigma", type=float, default=0.25, help="noise per pass, px (default 0.25)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(REPO):
        parser.error("synthetic annotations are written outside the repository only")
    sequence = Sequence(args.manifest)
    doc = annotation.read(args.annotations, len(sequence))
    curves_path = sequence.root / doc["protocol"].get("curves", "curves.json")
    curves = {c["frame"]: c for c in json.loads(curves_path.read_text())["curves"]}
    annotation.write(args.out, synthesize(sequence, doc, curves, args.sigma, args.seed))
    print(f"wrote {args.out}: SYNTHETIC, {len(doc['sections'])} sections x 2 passes")


if __name__ == "__main__":
    main()
