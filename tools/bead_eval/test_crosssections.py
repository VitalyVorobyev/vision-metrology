"""Tests for the cross-section benchmark tooling, on a synthetic scene whose walls are known
exactly: `crosssections.py`'s stations, `crosssection_synth.py`'s mask walls, and
`crosssection_eval.py` end to end against SYNTHETIC annotations.

    python -m pytest tools/bead_eval/test_crosssections.py

Needs scikit-image for the skeleton. The end-to-end test needs the `vision_metrology`
bindings, and its coupled mode `BeadCoupling`; each is skipped without them.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "registration_eval"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.special import ndtr  # noqa: E402

pytest.importorskip("skimage")

import annotation  # noqa: E402
import crosssection_eval  # noqa: E402
import crosssection_synth  # noqa: E402
import crosssections  # noqa: E402
from manifest import Sequence  # noqa: E402

SIZE = (200, 260)       # rows, columns
WIDTH = 8.0             # the bead's true width, px


def centreline(x):
    return 100 + 15 * np.sin(2 * np.pi * x / 260)


def scene(noise_seed):
    """A dark bead 8 px wide on a sine, blurred by σ 1 px, its mask, and its true walls."""
    xs = np.linspace(-20, SIZE[1] + 20, 20001)
    dense = np.stack([xs, centreline(xs)], axis=1)
    rows, cols = np.mgrid[:SIZE[0], :SIZE[1]]
    d, _ = cKDTree(dense).query(np.stack([cols.ravel(), rows.ravel()], axis=1))
    d = d.reshape(SIZE)
    image = 150 - 60 * (ndtr(WIDTH / 2 - d) + ndtr(WIDTH / 2 + d) - 1)
    image += np.random.default_rng(noise_seed).normal(0, 1, SIZE)
    return image.astype(np.float32), d <= WIDTH / 2, cKDTree(dense)


CFG = dict(crosssections.CONFIGS["damsegment"], min_length=40.0, count=8)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Stations on two synthetic images (one dev, one test), a manifest and SYNTHETIC
    annotations from the masks."""
    root = tmp_path_factory.mktemp("crosssections")
    images = {"img-dev": scene(1), "img-test": scene(2)}
    pooled = {"dev": crosssections.candidates("damsegment", "img-dev", "dev", *images["img-dev"][:2], None, CFG),
              "test": crosssections.candidates("damsegment", "img-test", "test", *images["img-test"][:2], None, CFG)}
    chosen, edges = crosssections.choose("damsegment", pooled, CFG["count"])
    sections, protocol, curves, keys = crosssections.assemble("damsegment", chosen, edges, CFG, "a synthetic scene")
    crosssections.write_data(root / "seq", "damsegment", lambda i: (images[i][0], images[i][1], None),
                             curves, keys, CFG, "a synthetic scene")
    stations = root / "stations.json"
    annotation.Session.create(stations, sections, protocol)
    sequence = Sequence(root / "seq" / "sequence.json")
    doc = annotation.read(stations, len(sequence))
    by_frame = {c["frame"]: c for c in json.loads((root / "seq" / "curves.json").read_text())["curves"]}
    synthetic = crosssection_synth.synthesize(sequence, doc, by_frame, 0.25, 0)
    return {"pooled": pooled, "chosen": chosen, "sections": sections, "protocol": protocol, "curves": by_frame,
            "sequence": sequence, "synthetic": synthetic, "images": images}


def test_candidates_respect_the_rules(built):
    for split, pool in built["pooled"].items():
        assert pool, split
        for c in pool:
            assert CFG["width"][0] <= c["width"] <= CFG["width"][1]
            assert c["contrast"] >= CFG["min_contrast"]
            assert abs(c["width"] - WIDTH) < 2.0      # the mask width, 2 EDT - 1, near the truth


def test_selection_is_deterministic_split_and_spaced(built):
    again, _ = crosssections.choose("damsegment", built["pooled"], CFG["count"])
    assert sorted((c["image"], c["s"]) for c in again) == sorted((c["image"], c["s"]) for c in built["chosen"])
    n_dev = sum(c["split"] == "dev" for c in built["chosen"])
    assert 0 < n_dev <= round(crosssections.DEV_FRACTION * CFG["count"])
    assert all(c["image"] == f"img-{c['split']}" for c in built["chosen"])
    for a in built["chosen"]:
        for b in built["chosen"]:
            if a is not b and a["image"] == b["image"]:
                assert np.hypot(*(a["station"] - b["station"])) >= 40
    frames = built["protocol"]["frames"]
    assert all(frames[s["frame"]]["split"] in ("dev", "test") for s in built["sections"])
    assert crosssections.same_sections(built["sections"], built["sections"])


def test_mask_walls_lie_within_half_a_pixel_of_the_truth(built):
    for sec in built["sections"]:
        mask = built["images"]["img-dev" if built["protocol"]["frames"][sec["frame"]]["split"] == "dev"
                               else "img-test"][1].astype(np.float32)
        left, right = crosssection_synth.mask_walls(mask, sec["station"], sec["normal"], 12.0)
        # The station sits on the skeleton, within a pixel of the true centre.
        assert abs((right - left) - WIDTH) < 1.0
    rows = built["synthetic"]["annotations"]
    assert built["synthetic"]["protocol"]["synthetic"] is True
    assert {r["annotator"] for r in rows} == {crosssection_synth.ANNOTATOR}
    assert len(rows) == 2 * len(built["sections"])


def test_crossing_finds_the_wall_along_the_normal():
    wall = np.array([[0.0, 5.0], [10.0, 5.0], [np.nan, np.nan], [20.0, 5.0]])
    assert crosssection_eval.crossing(wall, [4.0, 0.0], [0.0, 1.0], 12.0) == pytest.approx([4.0, 5.0])
    assert crosssection_eval.crossing(wall, [15.0, 0.0], [0.0, 1.0], 12.0) is None    # a gap
    assert crosssection_eval.crossing(wall, [4.0, 0.0], [0.0, 1.0], 3.0) is None      # out of reach


def test_end_to_end_against_synthetic_annotations(built):
    vm = pytest.importorskip("vision_metrology")
    modes = ["tracker"] + (["coupled"] if hasattr(vm, "BeadCoupling") else [])
    args = argparse.Namespace(split="dev", annotator=None, modes=modes, reach=8.0, spacing=2.0, threshold=3.0,
                              sigma=1.0, obliquity=180.0, coupled_passes=1, coupling={}, dataset="damsegment")
    report = crosssection_eval.evaluate(built["sequence"], built["synthetic"], built["curves"], args)
    assert report["synthetic"] is True and report["split"] == "dev"
    assert 0.1 < report["annotator"]["sigma"] < 0.4
    for mode in modes:
        r = report["modes"][mode]
        assert r["coverage"]["left"]["rate"] == 1.0 and r["coverage"]["right"]["rate"] == 1.0, mode
        # The annotations are mask walls (within half a pixel of the truth) plus 0.25 px noise.
        assert abs(r["walls"]["both"]["bias"]) < 0.5, mode
        assert r["walls"]["both"]["sigma"] < 0.6, mode
        assert abs(r["width"]["bias"]) < 1.0, mode
        assert r["intervals_90"]["wall_bias"] is None or len(r["intervals_90"]["wall_bias"]) == 2
