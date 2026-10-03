"""The measure router's caliper list and overlay, read from one `explain` pass.

The same checks as the Tauri crate's `commands::measure` unit test, so both shells build
the caliper list the same way: a box where its caliper looked, linked to its row by id,
and each hit's residual against the fit and its edge amplitude.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import vision_metrology as vm

from vm_lab.routers.measure import _caliper_results


def _disc(size: int = 128, r: float = 30.0) -> np.ndarray:
    """A bright disc of radius `r` centred at (64, 64) on a dark ground, anti-aliased."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    cover = np.clip(r + 0.5 - np.hypot(xs - 64.0, ys - 64.0), 0.0, 1.0)
    return (20.0 + 180.0 * cover).round().astype(np.uint8)


def _traces() -> list[vm.ObjectTrace]:
    model = vm.MetrologyModel()
    model.add(vm.MetrologyObject(shape=vm.MetrologyShape.circle((64.0, 64.0), 30.0), n_calipers=8))
    model.add(
        vm.MetrologyObject(
            shape=vm.MetrologyShape.line((8.0, 10.0), (8.0, 40.0)), n_calipers=3, caliper_len=4.0
        )
    )
    return model.explain(_disc(), x=0.0, y=0.0)


def test_hits_carry_their_edge_their_residual_and_their_span() -> None:
    rim = _traces()[0]
    calipers, overlay = _caliper_results(rim, 0)
    assert [c.status for c in calipers] == ["hit"] * 8
    for c, hit in zip(calipers, rim.result.hits, strict=True):
        edge = c.profile.edges[0]
        assert edge.pos_px == hit.t
        assert edge.amplitude == hit.amplitude
        # The residual is the edge's distance from the fit, so none exceeds `max_dev`.
        assert c.residual is not None and abs(c.residual) <= rim.result.max_dev + 1e-5
        # The default caliper reaches 10 px either side of the rim, where `t` is 0.
        assert (c.profile.start_px, c.profile.end_px) == (-10.0, 10.0)


def test_each_box_sits_on_the_rim_and_shares_its_id_with_its_edge_mark() -> None:
    _, overlay = _caliper_results(_traces()[0], 0)
    for i in range(8):
        box, edge = overlay[2 * i], overlay[2 * i + 1]
        assert (box.kind, edge.kind) == ("caliper", "point")
        assert box.id == edge.id == f"caliper-0-{i}"
        # Not at the circle's centre: a radial placement's `center` is the circle's.
        assert math.hypot(box.cx - 64.0, box.cy - 64.0) == pytest.approx(30.0, abs=1e-3)


def test_rejections_have_a_reason_and_no_residual() -> None:
    flat = _traces()[1]
    calipers, overlay = _caliper_results(flat, 1)
    assert [(c.index, c.status, c.reason, c.residual) for c in calipers] == [
        (i, "rejected", "no_edge", None) for i in range(3)
    ]
    assert [o.id for o in overlay] == ["caliper-1-0", "caliper-1-1", "caliper-1-2"]
