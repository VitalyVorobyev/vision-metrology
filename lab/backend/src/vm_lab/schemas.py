"""Request/response models for the `/api` surface. Pydantic, not the raw PyO3 classes —
keeps the wire contract stable even as the Rust-side types evolve, and gives FastAPI's
OpenAPI schema (consumed by the frontend for types) something concrete to describe."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Roi = tuple[float, float, float, float]  # (x, y, w, h), source-image pixels
AngleRange = tuple[float, float]  # degrees


# -- images --------------------------------------------------------------------------


class ImageOut(BaseModel):
    id: str
    filename: str
    width: int
    height: int
    sha256: str


# -- models (teach) --------------------------------------------------------------------


class ModelCreateRequest(BaseModel):
    image_id: str
    roi: Roi
    min_contrast: float = Field(default=0.1, gt=0, le=1, description="fraction of dtype range")
    num_levels: int | None = None


class ModelOut(BaseModel):
    id: str
    image_id: str
    roi: Roi
    min_contrast: float
    num_levels: int | None
    origin: tuple[float, float]
    num_levels_built: int
    point_counts: list[int]


# -- find ------------------------------------------------------------------------------


class FindRequest(BaseModel):
    image_id: str
    model_id: str
    min_score: float = Field(default=0.7, ge=0, le=1)
    max_matches: int | None = None
    roi: Roi | None = None
    angle_range: AngleRange | None = None


class MatchOut(BaseModel):
    x: float
    y: float
    angle: float
    scale: float
    score: float
    support: int
    level: int


class FindResponse(BaseModel):
    matches: list[MatchOut]


# -- measure -----------------------------------------------------------------------------

MeasureShapeKind = Literal["circle", "line"]


class MeasureConfigIn(BaseModel):
    """Overrides for `vm.MeasureConfig`; unset fields fall back to library defaults."""

    sigma: float | None = None
    threshold: float | None = None
    polarity: Literal["bright_to_dark", "dark_to_bright", "either"] | None = None
    max_obliquity_deg: float | None = None


class FitConfigIn(BaseModel):
    """Overrides for `vm.FitConfig`."""

    loss: Literal["l2", "huber", "tukey"] | None = None
    inlier_tol: float | None = None


class MeasureObjectIn(BaseModel):
    """One metrology object, nominal geometry given in **model frame** (the taught
    `ShapeModel`'s own coordinates, i.e. relative to its `origin`)."""

    kind: MeasureShapeKind
    label: str | None = None

    # circle
    cx: float | None = None
    cy: float | None = None
    r: float | None = None
    arc: AngleRange | None = None  # degrees, optional partial-circle span

    # line
    ax: float | None = None
    ay: float | None = None
    bx: float | None = None
    by: float | None = None

    n_calipers: int = 12
    caliper_len: float = 20.0
    caliper_width: float = 8.0
    measure: MeasureConfigIn = Field(default_factory=MeasureConfigIn)
    fit: FitConfigIn = Field(default_factory=FitConfigIn)


class FixtureIn(BaseModel):
    x: float
    y: float
    angle: float = 0.0
    scale: float = 1.0


class MeasureRequest(BaseModel):
    image_id: str
    model_id: str
    """Whose frame `objects`' nominal geometry is expressed in — required even when
    `fixture` is given explicitly, since the model's `origin` anchors that frame."""
    fixture: FixtureIn | None = None
    """Explicit pose. If omitted, `model_id` is auto-found against `image_id` and the
    top-scoring match (by `min_score`) supplies the pose."""
    min_score: float = 0.7
    objects: list[MeasureObjectIn]


class EdgeMarkOut(BaseModel):
    pos_px: float
    polarity: str


class CaliperProfileOut(BaseModel):
    values: list[float]
    step_px: float
    edges: list[EdgeMarkOut]


class CaliperResultOut(BaseModel):
    index: int
    status: Literal["hit", "rejected"]
    reason: str | None = None
    profile: CaliperProfileOut


class OverlayPrimitiveOut(BaseModel):
    """Mirrors `@vitavision/lab-ui`'s `MeasurePrimitive` union exactly (field names and
    units — angles in radians) so the frontend can pass this straight through."""

    kind: Literal["point", "segment", "circle", "arc", "caliper", "dimension"]
    tone: Literal["signal", "normal", "defect", "warn", "muted"] | None = None

    x: float | None = None
    y: float | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    cx: float | None = None
    cy: float | None = None
    r: float | None = None
    startAngle: float | None = None
    endAngle: float | None = None
    width: float | None = None
    height: float | None = None
    angle: float | None = None
    label: str | None = None
    dashed: bool | None = None
    cross: bool | None = None


class MeasureObjectResultOut(BaseModel):
    kind: MeasureShapeKind | Literal["error"]
    label: str | None = None
    # fitted params (circle)
    circle_cx: float | None = None
    circle_cy: float | None = None
    circle_r: float | None = None
    # fitted params (line)
    line_px: float | None = None
    line_py: float | None = None
    line_dx: float | None = None
    line_dy: float | None = None

    rms: float | None = None
    max_dev: float | None = None
    n_used: int | None = None
    message: str | None = None

    calipers: list[CaliperResultOut] = Field(default_factory=list)
    overlay: list[OverlayPrimitiveOut] = Field(default_factory=list)


class MeasureResponse(BaseModel):
    fixture: FixtureIn
    fixture_source: Literal["explicit", "auto_find"]
    objects: list[MeasureObjectResultOut]
