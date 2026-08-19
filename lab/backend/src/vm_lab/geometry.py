"""The fixture-pose transform, replicated from the Rust side.

`vision_metrology::matching::ShapeMatch::pose` maps a model-frame point into the source
image as `position + scale · R(angle) · (point − origin)` (radians; `R` is the standard
`[[cos,-sin],[sin,cos]]` matrix — see `crates/vm-primitives/src/core/geom/transform.rs`).

The Python binding `vm.MetrologyModel.apply(image, x, y, angle, scale)` builds its own
fixture as `scale · R(angle) · point + (x, y)` — it does **not** subtract `origin` first.
That only equals the Rust `pose` above when `origin == (0, 0)`. `correct_translation`
below computes the `(x, y)` to hand `apply` so that its un-origin-aware formula produces
the correct, origin-aware pose — see the model-frame gotcha noted in lab/README.md.

`to_scene` performs the same transform directly, used by this module's own caliper
placement (`measure.py`), which must mirror `MetrologyModel::measure_one` in
`crates/vision-metrology/src/measure/model.rs` exactly to report accurate per-caliper
hit/reject reasons and profiles — the Rust `apply` binding does not expose those.
"""

from __future__ import annotations

import math


def rotate(vx: float, vy: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return c * vx - s * vy, s * vx + c * vy


def perp(vx: float, vy: float) -> tuple[float, float]:
    """+90 degrees, matching `vm_primitives::Vec2fExt::perp`."""
    return -vy, vx


def to_scene(
    px: float, py: float, origin: tuple[float, float], x: float, y: float, angle: float, scale: float
) -> tuple[float, float]:
    ox, oy = origin
    rx, ry = rotate(px - ox, py - oy, angle)
    return x + scale * rx, y + scale * ry


def correct_translation(
    origin: tuple[float, float], x: float, y: float, angle: float, scale: float
) -> tuple[float, float]:
    ox, oy = origin
    rx, ry = rotate(ox, oy, angle)
    return x - scale * rx, y - scale * ry
