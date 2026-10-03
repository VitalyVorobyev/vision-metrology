#!/usr/bin/env python3
"""Write the CaliperBench golden fixture for `tests/caliperbench_protocol.rs`.

Each case is a small 8-bit image (inline, at most 16 x 64), a CaliperBench request, a
method and its parameters, and what CaliperBench's own baseline returns for them: the
prediction (status, reason, edges) and the lab trace (profile, smoothed profile, gradient
and candidates, or end levels). The Rust test runs `examples/common/caliperbench.rs` on
the same inputs and compares.

The cases are tie-free: no comparison CaliperBench makes (peak against neighbour, peak
against `min_response`, two candidate strengths, profile against the midpoint level,
contrast against `min_contrast`) is closer than `MARGIN`, so `f32` and `float64` cannot
decide one differently. The script refuses to write a fixture that breaks this.

Run it with CaliperBench's environment (it imports `caliperbench`):

    uv run --directory /path/to/caliperbench python "$PWD/tools/gen_caliperbench_golden.py"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import caliperbench
from caliperbench.baseline import BaselineParams, predict
from caliperbench.schema import Request

OUT = (
    Path(__file__).resolve().parent.parent
    / "crates/vision-metrology/tests/fixtures/caliperbench_golden.json"
)
MARGIN = 1e-6
H, W = 16, 64


def render(levels, edges, angle_deg=0.0, size=(H, W)):
    """Anti-aliased image of parallel straight edges, 8 x 8 supersampled per pixel.

    `edges` are increasing positions along the edge normal (at `angle_deg` from +x),
    measured from the image origin; `levels` has one more entry than `edges` and gives
    the intensity between them, in 8-bit numbers."""
    h, w = size
    a = np.deg2rad(angle_deg)
    nx, ny = np.cos(a), np.sin(a)
    sub = (np.arange(8) + 0.5) / 8 - 0.5
    y, x = np.mgrid[:h, :w].astype(float)
    acc = np.zeros((h, w))
    for dy in sub:
        for dx in sub:
            s = (x + dx) * nx + (y + dy) * ny
            band = np.searchsorted(np.asarray(edges, dtype=float), s.ravel()).reshape(h, w)
            acc += np.asarray(levels, dtype=float)[band]
    return np.clip(np.round(acc / 64), 0, 255).astype(np.uint8)


def ramp(lo, hi, size=(H, W)):
    h, w = size
    row = np.round(np.linspace(lo, hi, w))
    return np.tile(row, (h, 1)).astype(np.uint8)


def flat(v, size=(H, W)):
    return np.full(size, v, dtype=np.uint8)


def strip(start, end, samples, width_px=1.0, across=1):
    return {
        "start_xy": [float(start[0]), float(start[1])],
        "end_xy": [float(end[0]), float(end[1])],
        "width_px": float(width_px),
        "samples": int(samples),
        "across": int(across),
    }


BAR = render([40, 200, 40], [15.3, 47.7])
# Phases .3 and .7 mirror each other, so BAR's two edges are equally strong; where
# either polarity may win, the edges need different strengths.
SKEWED_BAR = render([40, 200, 55], [15.3, 47.62])
STEP = render([60, 190], [30.6])
OBLIQUE = render([50, 180], [31.0], angle_deg=20.0)
FAINT = render([100, 108], [30.6])
EDGE_NEGATIVE = render([70, 150], [36.2])

CASES = [
    ("pair_phases", BAR, "gradient_parabolic", {}, strip((1, 7), (62, 7), 62), ["rising", "falling"]),
    ("pair_reversed", BAR, "gradient_parabolic", {}, strip((62, 7), (1, 7), 62), ["rising", "falling"]),
    ("pair_half_pixel", BAR, "gradient_parabolic", {}, strip((1, 7), (62, 7), 123), ["rising", "falling"]),
    ("pair_integer_params", SKEWED_BAR, "gradient_integer",
     {"sigma": 1.5, "radius": 4, "min_response": 0.02}, strip((1, 7), (62, 7), 62), ["either", "either"]),
    ("oblique_wide", OBLIQUE, "gradient_parabolic", {}, strip((6, 4), (56, 11), 101, 3.0, 3), ["rising"]),
    ("width_override_one_line", OBLIQUE, "gradient_parabolic",
     {"width_px": 3.0, "across": 1}, strip((4, 8), (58, 8), 55), ["rising"]),
    ("negative_flat", flat(120), "gradient_parabolic", {}, strip((4, 8), (58, 8), 55), []),
    ("negative_ramp", ramp(60, 120), "gradient_parabolic", {}, strip((4, 8), (58, 8), 55), []),
    ("negative_edge", EDGE_NEGATIVE, "gradient_parabolic", {}, strip((4, 8), (58, 8), 55), []),
    ("missing_peak", STEP, "gradient_parabolic", {}, strip((4, 8), (58, 8), 55), ["falling"]),
    ("out_of_bounds", STEP, "gradient_parabolic", {}, strip((-0.5, 8), (40, 8), 41), ["rising"]),
    ("midpoint_ok", STEP, "midpoint_crossing", {}, strip((4, 8), (58, 8), 55), ["rising"]),
    ("midpoint_low_contrast", FAINT, "midpoint_crossing", {}, strip((4, 8), (58, 8), 55), ["rising"]),
    ("midpoint_wrong_polarity", STEP, "midpoint_crossing", {}, strip((4, 8), (58, 8), 55), ["falling"]),
    ("midpoint_one_edge_rule", BAR, "midpoint_crossing", {}, strip((1, 7), (62, 7), 62), ["rising", "falling"]),
]


def check_tie_free(name, trace, params, polarities):
    """Refuse a case whose outcome hangs on a comparison closer than MARGIN."""
    def close(a, b):
        return abs(a - b) < MARGIN

    if "gradient" in trace:
        g = np.asarray(trace["gradient"])
        for sign in (1, -1):
            r = sign * g
            for i in range(1, len(r) - 1):
                if r[i] < params.min_response / 2:
                    continue
                for other in (r[i - 1], r[i + 1], params.min_response):
                    assert not close(r[i], other), f"{name}: near-tie at sample {i}"
        # Candidates compete when a task admits both polarities, else within one.
        either = not polarities or "either" in polarities
        for pol in ("any",) if either else ("rising", "falling"):
            strengths = sorted(c[0] for c in trace["candidates"] if either or c[2] == pol)
            for a, b in zip(strengths, strengths[1:]):
                assert not close(a, b), f"{name}: two candidates of equal strength"
    if "levels" in trace:
        before, after = trace["levels"]
        assert not close(abs(after - before), params.min_contrast), f"{name}: contrast tie"
        if len(polarities) == 1:
            for v in trace["smooth"]:
                assert not close(v, trace["threshold"]), f"{name}: sample on the level"


def main():
    cases = []
    for name, pixels, method, raw_params, strip_spec, polarities in CASES:
        h, w = pixels.shape
        assert h <= 16 and w <= 64
        params = BaselineParams.from_dict(raw_params)
        request = Request.model_validate(
            {
                "sample_id": name,
                "image": f"golden/{name}.png",
                "image_sha256": "0" * 64,
                "strip": strip_spec,
                "polarities": polarities,
            }
        )
        trace = {}
        prediction = predict(pixels.astype(float) / 255, request, method, params, trace)
        check_tie_free(name, trace, params, polarities)
        cases.append(
            {
                "name": name,
                "width": w,
                "height": h,
                "pixels": pixels.ravel().tolist(),
                "method": method,
                "params": raw_params,
                "request": request.model_dump(),
                "expected": {
                    "status": prediction.status,
                    "reason": prediction.reason,
                    "edges_px": prediction.edges_px,
                },
                "trace": trace,
            }
        )
        print(f"{name:26} {prediction.status:6} {prediction.reason or prediction.edges_px}")
    header = {
        "generator": "tools/gen_caliperbench_golden.py",
        "caliperbench": caliperbench.__version__,
        "numpy": np.__version__,
    }
    body = ",\n".join(json.dumps(c, allow_nan=False) for c in cases)
    OUT.write_text(
        "{\n"
        + "".join(f"  {json.dumps(k)}: {json.dumps(v)},\n" for k, v in header.items())
        + '  "cases": [\n'
        + body
        + "\n  ]\n}\n"
    )
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
