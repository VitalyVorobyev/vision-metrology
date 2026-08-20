"""Replays `lab/contract/fixtures/*.json` against a fresh FastAPI backend and asserts
the response matches the committed golden within a float tolerance.

This is the browser-path half of the W6 anti-drift gate (plan decision 7) — the Tauri
side is `lab/frontend/src-tauri/tests/contract_parity.rs`, replaying the same fixtures
through the native command handlers. Both exist so a change to `vm_lab`'s response
shape *or* to the Rust command layer is caught here rather than discovered as a UI bug
in only one of the two shells.

Uses `export_contract_fixtures.Run` — the exact same operation sequence that captured
the goldens — so this test is "does the current backend still agree with what
generation captured", not a hand-maintained parallel implementation that could itself
drift from the generator. See `lab/contract/README.md` for the normalization rules.
"""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_contract_fixtures as gen  # noqa: E402

_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "contract" / "fixtures"

REL_TOL = 1e-3
ABS_TOL = 1e-3


def _assert_close(actual: Any, expected: Any, path: str = "$") -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        assert math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL), f"{path}: {actual} != {expected}"
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected a dict, got {type(actual)}"
        assert actual.keys() == expected.keys(), f"{path}: key mismatch: {sorted(actual)} vs {sorted(expected)}"
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected a list, got {type(actual)}"
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _golden(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES_DIR / f"{name}.json").read_text())


def test_committed_synthetic_images_match_the_generator():
    """The PNGs under `lab/contract/fixtures/` must be exactly what
    `export_contract_fixtures` would render today — otherwise a golden JSON response
    would silently describe a different input than what ships in the repo."""
    disc = gen._png_bytes(gen._disc_array(gen.IMG_SIZE, gen.IMG_SIZE, *gen.DISC_CENTER, gen.DISC_RADIUS))
    frame_a = gen._png_bytes(gen._texture_frame(gen.TEXTURE_SIZE, 0.0, 0.0))
    frame_b = gen._png_bytes(gen._texture_frame(gen.TEXTURE_SIZE, gen.SHIFT_DX, gen.SHIFT_DY))

    committed_disc = np.asarray(PILImage.open(_FIXTURES_DIR / "disc.png"))
    committed_a = np.asarray(PILImage.open(_FIXTURES_DIR / "frame_a.png"))
    committed_b = np.asarray(PILImage.open(_FIXTURES_DIR / "frame_b.png"))

    assert np.array_equal(np.asarray(PILImage.open(io.BytesIO(disc))), committed_disc)
    assert np.array_equal(np.asarray(PILImage.open(io.BytesIO(frame_a))), committed_a)
    assert np.array_equal(np.asarray(PILImage.open(io.BytesIO(frame_b))), committed_b)


def test_contract_fixtures_replay_within_tolerance():
    client, tmp = gen.isolated_client()
    with tmp:
        run = gen.Run(client)

    for name, actual_fixture in run.fixtures.items():
        golden = _golden(name)
        assert actual_fixture["operation"] == golden["operation"] == name
        _assert_close(actual_fixture["request"], golden["request"], f"{name}.request")
        _assert_close(actual_fixture["response"], golden["response"], f"{name}.response")


def test_rectify_crop_pixels_match_the_golden_within_tolerance():
    """The crop PNG's *decoded pixels*, not its compressed bytes — re-encoding the
    same array is not guaranteed byte-identical across PIL/zlib versions, but the
    geometry the crop encodes is what this fixture actually gates."""
    client, tmp = gen.isolated_client()
    with tmp:
        run = gen.Run(client)

    actual = np.asarray(PILImage.open(io.BytesIO(run.crop_png)))
    golden = np.asarray(PILImage.open(_FIXTURES_DIR / "rectify_crop.png"))
    assert actual.shape == golden.shape
    diff = np.abs(actual.astype(np.int16) - golden.astype(np.int16))
    assert diff.mean() < 0.5, f"mean abs diff {diff.mean()} too high"
    assert diff.max() <= 2, f"max abs diff {diff.max()} too high"


def test_calibration_fixture_matches_the_upstream_vision_metrology_fixture():
    """`lab/contract/fixtures/calibration.json` is a copy of
    `crates/vision-metrology/tests/fixtures/table_calibration.json` — catches the copy
    silently going stale if the upstream fixture is ever regenerated."""
    committed = (_FIXTURES_DIR / "calibration.json").read_bytes()
    assert committed == gen._CAL_SRC.read_bytes()
