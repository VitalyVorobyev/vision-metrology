"""Type stubs for `vision_metrology` — PyO3 bindings for the vision-metrology
library. See crates/vm-python/README.md for the guide; this file is the
contract IDEs and mypy see.
"""

# Hand-maintained: keep in step with `src/lib.rs`'s `#[pymodule]` registration
# list whenever a class or free function is added, renamed or removed.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Every entry point generic over `vm_primitives::Pixel` accepts any of these
# three dtypes, dispatching at the Rust boundary; an unsupported dtype raises
# `ValueError`. Entry points whose Rust counterpart is genuinely `u8`-only
# (segmentation, morphology) are typed `ImageU8` instead.
ImageU8 = npt.NDArray[np.uint8]
ImageAny = Union[npt.NDArray[np.uint8], npt.NDArray[np.uint16], npt.NDArray[np.float32]]
PointsF32 = npt.NDArray[np.float32]  # (N, 2)

# ---------------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------------

class EdgeConfig:
    smooth_kind: str
    low_thresh: Optional[float]
    high_thresh: Optional[float]
    border_mode: str
    border_constant: float
    subpix: str
    def __init__(
        self,
        smooth_kind: Optional[str] = ...,
        low_thresh: Optional[float] = ...,
        high_thresh: Optional[float] = ...,
        border_mode: Optional[str] = ...,
        border_constant: Optional[float] = ...,
        subpix: Optional[str] = ...,
    ) -> None: ...

class LsdConfig:
    downscale_levels: int
    pre_smooth: str
    ang_th: float
    log_eps: float
    density_th: float
    n_bins: int
    min_length: float
    def __init__(
        self,
        downscale_levels: Optional[int] = ...,
        pre_smooth: Optional[str] = ...,
        ang_th: Optional[float] = ...,
        log_eps: Optional[float] = ...,
        density_th: Optional[float] = ...,
        n_bins: Optional[int] = ...,
        min_length: Optional[float] = ...,
    ) -> None: ...

class FitConfig:
    loss: str
    loss_scale: float
    ransac_iters: int
    inlier_tol: float
    min_inliers: int
    seed: int
    max_iters: int
    tol: float
    def __init__(
        self,
        loss: Optional[str] = ...,
        loss_scale: Optional[float] = ...,
        ransac_iters: Optional[int] = ...,
        inlier_tol: Optional[float] = ...,
        min_inliers: Optional[int] = ...,
        seed: Optional[int] = ...,
        max_iters: Optional[int] = ...,
        tol: Optional[float] = ...,
    ) -> None: ...

class Contrast:
    """Tagged `min_contrast` unit — construct with `raw` or `fraction_of_range`,
    never a bare float."""

    @staticmethod
    def raw(value: float) -> Contrast: ...
    @staticmethod
    def fraction_of_range(value: float) -> Contrast: ...

class ShapeSearchTuning:
    greediness: float
    angle_step: Optional[float]
    scale_step: Optional[float]
    last_level: int
    max_candidates: int
    coarse_score_factor: float
    def __init__(
        self,
        greediness: Optional[float] = ...,
        angle_step: Optional[float] = ...,
        scale_step: Optional[float] = ...,
        last_level: Optional[int] = ...,
        max_candidates: Optional[int] = ...,
        coarse_score_factor: Optional[float] = ...,
    ) -> None: ...

class ShapeModelConfig:
    num_levels: Optional[int]
    edge: EdgeConfig
    pre_smooth: str
    min_contrast: Contrast
    max_points: Optional[int]
    origin: Optional[Tuple[float, float]]
    reference_angle: float
    angle_min: float
    angle_max: float
    scale_min: float
    scale_max: float
    polarity: str
    min_points_per_level: int
    def __init__(
        self,
        num_levels: Optional[int] = ...,
        edge: Optional[EdgeConfig] = ...,
        pre_smooth: Optional[str] = ...,
        min_contrast: Optional[Contrast] = ...,
        max_points: Optional[int] = ...,
        origin: Optional[Tuple[float, float]] = ...,
        reference_angle: Optional[float] = ...,
        angle_min: Optional[float] = ...,
        angle_max: Optional[float] = ...,
        scale_min: Optional[float] = ...,
        scale_max: Optional[float] = ...,
        polarity: Optional[str] = ...,
        min_points_per_level: Optional[int] = ...,
    ) -> None: ...

class ShapeSearchConfig:
    min_score: float
    max_matches: Optional[int]
    max_overlap: float
    roi: Optional[Tuple[float, float, float, float]]
    angle_range: Optional[Tuple[float, float]]
    scale_range: Optional[Tuple[float, float]]
    min_contrast: Contrast
    refinement: str
    tuning: ShapeSearchTuning
    def __init__(
        self,
        min_score: Optional[float] = ...,
        max_matches: Optional[int] = ...,
        max_overlap: Optional[float] = ...,
        roi: Optional[Tuple[float, float, float, float]] = ...,
        angle_range: Optional[Tuple[float, float]] = ...,
        scale_range: Optional[Tuple[float, float]] = ...,
        min_contrast: Optional[Contrast] = ...,
        refinement: Optional[str] = ...,
        tuning: Optional[ShapeSearchTuning] = ...,
    ) -> None: ...

class CropSpec:
    """Fixed crop geometry for `ShapeMatch.model_frame_map` — rectify a
    located match into a canonical, model-frame patch. `rect` is
    `(x, y, width, height)` in model-frame coordinates. Output pixel size
    (`output_size`) depends only on `rect` and `px_per_unit`, never on any
    particular match."""

    rect: Tuple[float, float, float, float]
    px_per_unit: float
    normalize_scale: bool
    def __init__(
        self,
        rect: Tuple[float, float, float, float],
        px_per_unit: float,
        normalize_scale: bool = ...,
    ) -> None: ...
    @property
    def output_size(self) -> Tuple[int, int]: ...

class CorrTemplateTuning:
    coarse_angle_step_deg: float
    min_angle_step_deg: float
    fill_value: int
    precompute_coarsest: bool
    def __init__(
        self,
        coarse_angle_step_deg: Optional[float] = ...,
        min_angle_step_deg: Optional[float] = ...,
        fill_value: Optional[int] = ...,
        precompute_coarsest: Optional[bool] = ...,
    ) -> None: ...

class CorrTemplateConfig:
    rotation: bool
    max_levels: Optional[int]
    tuning: CorrTemplateTuning
    def __init__(
        self,
        rotation: Optional[bool] = ...,
        max_levels: Optional[int] = ...,
        tuning: Optional[CorrTemplateTuning] = ...,
    ) -> None: ...

class CorrSearchTuning:
    parallel: bool
    max_image_levels: Optional[int]
    beam_width: int
    per_angle_topk: int
    nms_radius: int
    roi_radius: int
    angle_half_range_steps: int
    min_var_i: float
    def __init__(
        self,
        parallel: Optional[bool] = ...,
        max_image_levels: Optional[int] = ...,
        beam_width: Optional[int] = ...,
        per_angle_topk: Optional[int] = ...,
        nms_radius: Optional[int] = ...,
        roi_radius: Optional[int] = ...,
        angle_half_range_steps: Optional[int] = ...,
        min_var_i: Optional[float] = ...,
    ) -> None: ...

class CorrConfig:
    rotation: bool
    metric: str
    min_score: Optional[float]
    tuning: CorrSearchTuning
    def __init__(
        self,
        rotation: Optional[bool] = ...,
        metric: Optional[str] = ...,
        min_score: Optional[float] = ...,
        tuning: Optional[CorrSearchTuning] = ...,
    ) -> None: ...

class Refine:
    """Tagged `DisplacementConfig.refine` — construct with `Refine.none()` or
    `Refine.lucas_kanade(iters=...)`, never a bare string."""

    @staticmethod
    def none() -> Refine: ...
    @staticmethod
    def lucas_kanade(iters: int = ...) -> Refine: ...

class DisplacementConfig:
    window: Tuple[float, float, float, float]
    search: Tuple[int, int]
    refine: Refine
    min_score: float
    def __init__(
        self,
        window: Tuple[float, float, float, float],
        search: Tuple[int, int] = ...,
        refine: Optional[Refine] = ...,
        min_score: float = ...,
    ) -> None: ...

class MomentScaleConfig:
    """`estimate_scale_moments` parameters."""

    polarity: str  # 'dark_on_bright' or 'bright_on_dark'
    min_area: int
    def __init__(
        self, polarity: Optional[str] = ..., min_area: Optional[int] = ...
    ) -> None: ...

class LogPolarScaleConfig:
    """`estimate_scale_logpolar` parameters."""

    scale_search: Tuple[float, float]
    angle_margin: Optional[float]
    edge: EdgeConfig
    def __init__(
        self,
        scale_search: Optional[Tuple[float, float]] = ...,
        angle_margin: Optional[float] = ...,
        edge: Optional[EdgeConfig] = ...,
    ) -> None: ...

class ScaleInvariantConfig:
    """`find_scale_invariant_roi`/`find_scale_invariant_center` parameters."""

    moments: MomentScaleConfig
    logpolar: LogPolarScaleConfig
    search: ShapeSearchConfig
    def __init__(
        self,
        moments: Optional[MomentScaleConfig] = ...,
        logpolar: Optional[LogPolarScaleConfig] = ...,
        search: Optional[ShapeSearchConfig] = ...,
    ) -> None: ...

class Locate:
    """How a caliper locates an edge on its profile. Construct with
    `Locate.gradient_peak(refine=...)`, `Locate.midpoint_crossing(...)` or
    `Locate.half_contrast(...)`."""

    kind: str
    refine: str
    centroid_radius: int
    endpoint_samples: int
    min_contrast: float
    flank_px: Tuple[float, float]
    tol_px: float
    max_iter: int
    @staticmethod
    def gradient_peak(refine: str = ..., centroid_radius: int = ...) -> Locate:
        """A local extremum of the derivative. `refine` is "none", "parabolic"
        (default), "gaussian" (log-parabola) or "centroid"."""
        ...
    @staticmethod
    def midpoint_crossing(endpoint_samples: int = ..., min_contrast: float = ...) -> Locate:
        """One edge where the smoothed profile crosses the mean of its end levels
        (medians of the first and last `endpoint_samples` samples), nearest the
        middle. Raises `MeasureRejected("low_contrast")` when the levels differ by
        less than `min_contrast`, `"wrong_polarity"` when their order is not the
        configured polarity, and `"no_crossing"` when there is no crossing."""
        ...
    @staticmethod
    def half_contrast(
        flank_px: Tuple[float, float] = ...,
        tol_px: float = ...,
        max_iter: int = ...,
        min_contrast: float = ...,
    ) -> Locate:
        """Gradient peaks, each moved to the crossing of its local half-contrast
        level: the mean of the medians `flank_px[0]` to `flank_px[1]` pixels before
        and after it, iterated until it moves by `tol_px` or less."""
        ...

class MeasureConfig:
    sigma: float
    threshold: float
    polarity: str
    select: str
    sequence: List[str]
    step: float
    max_obliquity_deg: float
    border_mode: str
    border_constant: float
    derivative: str
    kernel_radius_px: float
    locate: Locate
    off_image: str
    def __init__(
        self,
        sigma: Optional[float] = ...,
        threshold: Optional[float] = ...,
        polarity: Optional[str] = ...,
        select: Optional[str] = ...,
        sequence: Optional[List[str]] = ...,
        step: Optional[float] = ...,
        max_obliquity_deg: Optional[float] = ...,
        border_mode: Optional[str] = ...,
        border_constant: Optional[float] = ...,
        derivative: Optional[str] = ...,
        kernel_radius_px: Optional[float] = ...,
        locate: Optional[Locate] = ...,
        off_image: Optional[str] = ...,
    ) -> None:
        """`select` is "all" (default), "first", "last", "strongest" or
        "in_order"; "in_order" needs `sequence`, one or two of "rising",
        "falling" and "either", found in scan order, each the strongest edge of
        its polarity after the previous one (equal strength: the earlier edge).
        `derivative` is "dog" (derivative of Gaussian, default) or
        "smooth_central" (Gaussian of half-width `kernel_radius_px`, then central
        differences). `off_image` is "fill" (default: sample outside the image with
        `border_mode` and measure) or "reject" (raise `MeasureRejected("off_image")`
        whenever any sample lies outside the image). The string fields accept only
        these names, here and on assignment; anything else raises `ValueError`."""
        ...

class BeadCaliper:
    """The caliper one stage of a `BeadTracker` measures with. The default is the
    tracking stage's (`max_offset` 15 px, `half_width` 2 px); `BeadConfig()` measures
    with a stricter one. Every strip keeps all edges of both polarities. The string
    fields take the names `MeasureConfig`'s do; `locate` may not be
    `Locate.midpoint_crossing()`."""

    max_offset: float
    half_width: float
    threshold: float
    locate: Locate
    max_obliquity_deg: float
    sigma: float
    step: float
    border_mode: str
    border_constant: float
    derivative: str
    kernel_radius_px: float
    off_image: str
    def __init__(
        self,
        max_offset: Optional[float] = ...,
        half_width: Optional[float] = ...,
        threshold: Optional[float] = ...,
        locate: Optional[Locate] = ...,
        max_obliquity_deg: Optional[float] = ...,
        sigma: Optional[float] = ...,
        step: Optional[float] = ...,
        border_mode: Optional[str] = ...,
        border_constant: Optional[float] = ...,
        derivative: Optional[str] = ...,
        kernel_radius_px: Optional[float] = ...,
        off_image: Optional[str] = ...,
    ) -> None: ...
    def to_measure_config(self) -> MeasureConfig:
        """The `MeasureConfig` each strip of this stage measures with: every edge
        (`select="all"`) of either polarity (`polarity="any"`)."""
        ...

class BeadTuning:
    """How hard a `BeadTracker` works and how stiff its corrections are. `loss` is
    "none", "huber" (default, `loss_scale` 1 px) or "tukey", applied by `irls_iters`
    reweighted solves after a least-squares one. Corrections shorter than about
    `2*pi*bending_px` are suppressed in one pass and decay over passes. `passes` and
    `irls_iters` must be at least 1, or the config raises `ValueError` when used."""

    passes: int
    tol: float
    damping: float
    tension_px: float
    bending_px: float
    loss: str
    loss_scale: float
    irls_iters: int
    tangent_window_px: float
    min_support: float
    def __init__(
        self,
        passes: Optional[int] = ...,
        tol: Optional[float] = ...,
        damping: Optional[float] = ...,
        tension_px: Optional[float] = ...,
        bending_px: Optional[float] = ...,
        loss: Optional[str] = ...,
        loss_scale: Optional[float] = ...,
        irls_iters: Optional[int] = ...,
        tangent_window_px: Optional[float] = ...,
        min_support: Optional[float] = ...,
    ) -> None: ...

class BeadConfig:
    """What a `BeadTracker` tracks and measures. `polarity` is "light" (default) or
    "dark". `clearance` and `min_margin` are off when `None`. `track`, `measure` and
    `tuning` are copies: assign a whole new value to change one."""

    polarity: str
    min_width: float
    max_width: float
    spacing: float
    clearance: Optional[float]
    min_margin: Optional[float]
    track: BeadCaliper
    measure: BeadCaliper
    tuning: BeadTuning
    def __init__(
        self,
        polarity: Optional[str] = ...,
        min_width: Optional[float] = ...,
        max_width: Optional[float] = ...,
        spacing: Optional[float] = ...,
        clearance: Optional[float] = ...,
        min_margin: Optional[float] = ...,
        track: Optional[BeadCaliper] = ...,
        measure: Optional[BeadCaliper] = ...,
        tuning: Optional[BeadTuning] = ...,
    ) -> None: ...

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class Edgel:
    x: float
    y: float
    nx: float
    ny: float
    strength: float

class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    nfa: float
    angle: float
    length: float

class Circle:
    cx: float
    cy: float
    r: float
    rms: float
    max_dev: float
    n_used: int

class Ellipse:
    cx: float
    cy: float
    a: float
    b: float
    angle: float
    rms: float
    max_dev: float
    n_used: int

class Line:
    px: float
    py: float
    dx: float
    dy: float
    rms: float
    max_dev: float
    n_used: int

class ShapeMatch:
    x: float
    y: float
    angle: float
    scale: float
    score: float
    support: int
    level: int
    def matrix(self, origin: Tuple[float, float]) -> List[List[float]]: ...
    def model_frame_map(self, spec: CropSpec) -> Map: ...
    def model_frame_pose(self, spec: CropSpec) -> List[List[float]]: ...

class ComponentStats:
    label: int
    pixel_count: int
    cx: float
    cy: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float

class MeasureEdge:
    """One caliper edge. `t` is the position along the scan, in pixels: the
    distance from `start` for a strip, the signed distance from the centre for
    rect and radial calipers, the arc length from `angle_start` for an arc."""

    x: float
    y: float
    t: float
    amplitude: float
    polarity: str

class MeasurePair:
    first: MeasureEdge
    second: MeasureEdge
    cx: float
    cy: float
    width: float

class CorrMatch:
    """One `corr.find`/`find_topk` match. Not `ShapeMatch`: `score` is a raw
    correlation coefficient, not `1 - occluded_fraction`, and there is no
    scale."""

    x: float
    y: float
    angle: float
    score: float

class Displacement:
    """`corr.displacement` result: `(dx, dy)` in pixels plus the stage-1
    (ZNCC) score — Lucas-Kanade refines the shift only, it has no score of
    its own."""

    dx: float
    dy: float
    score: float

class ScaleEstimate:
    """One `estimate_scale_*` answer. `score`'s meaning differs between
    estimators (a fill-fraction diagnostic for moments, a ZNCC correlation
    score for log-polar) -- see each function's own docs, not this class."""

    scale: float
    angle: Optional[float]
    score: float

# ---------------------------------------------------------------------------
# measure / metrology
# ---------------------------------------------------------------------------

class MetrologyShape:
    kind: str
    a: Tuple[float, float]
    b: Tuple[float, float]
    center: Tuple[float, float]
    radius: float
    arc: Optional[Tuple[float, float]]
    @staticmethod
    def line(a: Tuple[float, float], b: Tuple[float, float]) -> MetrologyShape: ...
    @staticmethod
    def circle(
        center: Tuple[float, float],
        radius: float,
        arc: Optional[Tuple[float, float]] = ...,
    ) -> MetrologyShape: ...

class MetrologyObject:
    shape: MetrologyShape
    n_calipers: int
    caliper_len: float
    caliper_width: float
    measure: MeasureConfig
    fit: FitConfig
    def __init__(
        self,
        shape: MetrologyShape,
        n_calipers: Optional[int] = ...,
        caliper_len: Optional[float] = ...,
        caliper_width: Optional[float] = ...,
        measure: Optional[MeasureConfig] = ...,
        fit: Optional[FitConfig] = ...,
    ) -> None: ...

class MetrologyResult:
    kind: str
    line: Optional[Line]
    circle: Optional[Circle]
    rms: float
    max_dev: float
    n_used: int
    hits: List[MeasureEdge]

class MetrologyError:
    message: str

class CaliperPlacement:
    """Where one caliper of a `MetrologyModel` sits at a fixture pose, without
    measuring — see `MetrologyModel.layout`. `kind` is `"rect"` or `"radial"`;
    `radius` is set only for `"radial"` placements, where `center` is the
    *circle's* own centre (not the caliper's position on the circle)."""

    object_index: int
    caliper_index: int
    kind: str
    center: Tuple[float, float]
    angle: float
    half_len: float
    half_width: float
    radius: Optional[float]

class MetrologyModel:
    def __init__(self) -> None: ...
    def add(self, object: MetrologyObject) -> int: ...
    @property
    def num_objects(self) -> int: ...
    def apply(
        self,
        image: ImageAny,
        x: float,
        y: float,
        angle: float = ...,
        scale: float = ...,
        origin: Tuple[float, float] = ...,
    ) -> List[Union[MetrologyResult, MetrologyError]]: ...
    def layout(
        self,
        x: float,
        y: float,
        angle: float = ...,
        scale: float = ...,
        origin: Tuple[float, float] = ...,
    ) -> List[CaliperPlacement]: ...
    def explain(
        self,
        image: ImageAny,
        x: float,
        y: float,
        angle: float = ...,
        scale: float = ...,
        origin: Tuple[float, float] = ...,
    ) -> List[ObjectTrace]:
        """`apply` and `Caliper.explain` in one pass: per object, what `apply`
        returns plus every caliper's placement and trace."""
        ...

class ObjectTrace:
    """One object of a `MetrologyModel`, measured and explained. `result` is
    what `apply` returns for it; `placements` and `calipers` are parallel, one
    entry per caliper. A caliper hit when its trace has `edges`, and its first
    edge is the one the fit used."""

    object_index: int
    result: Union[MetrologyResult, MetrologyError]
    placements: List[CaliperPlacement]
    calipers: List[CaliperTrace]

class Caliper:
    """A reusable caliper. Construct with `rect`, `arc`, `radial` or `strip`;
    `move_to_rect`, `move_to_arc`, `move_to_radial` and `move_to_strip`
    reposition it, keeping its config and scratch buffers."""

    @staticmethod
    def rect(
        center: Tuple[float, float],
        angle: float,
        half_len: float,
        half_width: float,
        config: Optional[MeasureConfig] = ...,
    ) -> Caliper: ...
    @staticmethod
    def arc(
        center: Tuple[float, float],
        radius: float,
        angle_start: float,
        angle_extent: float,
        half_width: float,
        config: Optional[MeasureConfig] = ...,
    ) -> Caliper: ...
    @staticmethod
    def radial(
        center: Tuple[float, float],
        radius: float,
        angle: float,
        half_len: float,
        half_width: float,
        config: Optional[MeasureConfig] = ...,
    ) -> Caliper: ...
    @staticmethod
    def strip(
        start: Tuple[float, float],
        end: Tuple[float, float],
        half_width: float = ...,
        samples: Optional[int] = ...,
        across: Optional[int] = ...,
        config: Optional[MeasureConfig] = ...,
    ) -> Caliper:
        """A strip from `start` to `end`. `samples` points along it include both
        endpoints; `across` lines are spread evenly over `+-half_width`. `None`
        follows the config's `step` along and about one line per pixel across.
        An edge's `t` is its distance from `start`."""
        ...
    def move_to_rect(
        self,
        center: Tuple[float, float],
        angle: float,
        half_len: float,
        half_width: float,
    ) -> None: ...
    def move_to_arc(
        self,
        center: Tuple[float, float],
        radius: float,
        angle_start: float,
        angle_extent: float,
        half_width: float,
    ) -> None: ...
    def move_to_radial(
        self,
        center: Tuple[float, float],
        radius: float,
        angle: float,
        half_len: float,
        half_width: float,
    ) -> None: ...
    def move_to_strip(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        half_width: float = ...,
        samples: Optional[int] = ...,
        across: Optional[int] = ...,
    ) -> None: ...
    def measure(self, img: ImageAny) -> List[MeasureEdge]: ...
    def measure_pairs(self, img: ImageAny) -> List[MeasurePair]: ...
    def profile(self) -> List[float]: ...
    def levels(self) -> List[LevelEdge]:
        """The level crossings behind the last `measure` call's edges (none for
        `Locate.gradient_peak`); `x` is in profile samples."""
        ...
    def spacing(self) -> float:
        """Distance between profile samples, in pixels: the scan's extent divided
        by `samples - 1`, which differs from the config's `step` whenever the
        extent is not a whole number of steps. Index `x` of the profile (a
        `LevelEdge.x`) sits `x * spacing` from the first sample."""
        ...
    def explain(self, img: ImageAny) -> CaliperTrace:
        """Measure and keep every intermediate. Never raises `MeasureRejected`:
        a rejection is the trace's `reject`."""
        ...

class CaliperTrace:
    """Everything one caliper measurement computed. `edges` and `reject` are
    exactly what `measure` returns or raises; `spacing` is what
    `Caliper.spacing()` reports."""

    spacing: float
    samples: int
    across: int
    threshold: float
    profile: npt.NDArray[np.float32]
    smoothed: npt.NDArray[np.float32]
    response: npt.NDArray[np.float32]
    candidates: List[MeasureEdge]
    levels: List[LevelEdge]
    edges: List[MeasureEdge]
    reject: Optional[str]

class LevelEdge:
    """An edge located as a level crossing."""

    x: float
    before: float
    after: float
    level: float
    iterations: int

class MeasureRejected(Exception):
    """Raised by `Caliper.measure`; `args[0]` is one of `"profile_too_short"`,
    `"no_edge"`, `"wrong_polarity"`, `"too_oblique"`, `"off_image"`,
    `"incomplete_sequence"`, `"low_contrast"`, `"no_crossing"`."""

class BeadSolve:
    """One pass's solve and the correction it applied. Lengths are in px;
    `step_scale` is the fraction of the solved correction applied, below 1 when the
    full step would fold the curve; `irls_iters` counts the reweighted solves after the
    least-squares one."""

    correction_rms: float
    correction_max: float
    residual_rms: float
    residual_max: float
    step_scale: float
    irls_iters: int

class BeadPass:
    """One tracking pass. `longest_gap` is in px; `solve` is `None` when the pass
    found too few pairs to solve and left the curve where it was."""

    n_valid: int
    support: float
    longest_gap: float
    solve: Optional[BeadSolve]
    rejects: Dict[str, int]

class TrackedBead:
    """A tracked bead, one array row per station of the refined curve. `centerline`
    is the next call's prior as it stands. `offset`, `width`, `confidence`, `center`,
    `first` and `second` (the edges on the -n and +n sides) come from the final stage
    and are NaN where it rejected; `reject` names the reason there. The statistics are
    `None` without a hit. `stop` is "converged" (the last solved correction was below
    `tol` and applied in full), "pass_limit" or "too_few_valid"; it says the loop
    stopped, while `center_rms` and `center_max_dev` say whether the curve sits on the
    bead. `rejects` counts the final stage's rejections by reason, in a fixed order."""

    centerline: npt.NDArray[np.float32]
    spacing: float
    normals: npt.NDArray[np.float32]
    offset: npt.NDArray[np.float32]
    width: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]
    center: npt.NDArray[np.float32]
    first: npt.NDArray[np.float32]
    second: npt.NDArray[np.float32]
    reject: List[Optional[str]]
    support: float
    longest_gap: float
    n_used: int
    center_rms: Optional[float]
    center_max_dev: Optional[float]
    width_mean: Optional[float]
    width_std: Optional[float]
    width_min: Optional[float]
    width_max: Optional[float]
    rejects: Dict[str, int]
    stop: str
    passes: List[BeadPass]

class BeadTracker:
    """Tracks a bead along a prior curve and measures its position and width along the
    refined curve. Raises `ValueError` for an invalid config."""

    config: BeadConfig
    def __init__(self, config: Optional[BeadConfig] = ...) -> None: ...
    def track(
        self, image: ImageAny, prior: Union[PointsF32, npt.NDArray[np.float64]]
    ) -> TrackedBead:
        """Track from `prior`, an (N, 2) float32 or float64 polyline in any memory
        layout. A missing bead is a result
        with every station rejected; a prior with fewer than two points, a non-finite
        point or no length raises `ValueError`. Reject reasons are the caliper's
        (`"no_edge"`, `"off_image"`, ...) or the pair gates' (`"no_pair"`, `"width"`,
        `"offset"`, `"clearance"`, `"ambiguous"`)."""
        ...
    def explain(
        self, image: ImageAny, prior: Union[PointsF32, npt.NDArray[np.float64]]
    ) -> BeadTrace:
        """`track` with every station's evidence kept: `result` is what `track`
        returns for the same arguments. A rejected station is part of the trace,
        never an exception; a bad prior raises `ValueError` as in `track`."""
        ...

class BeadTrace:
    """A bead tracker's run, explained. `result` is what `track` returns;
    `passes` has one `BeadPassTrace` per tracking pass, parallel to
    `result.passes`; `measure` has one `BeadStationTrace` per station of the
    final stage, parallel to `result`'s arrays."""

    result: TrackedBead
    passes: List[BeadPassTrace]
    measure: List[BeadStationTrace]

class BeadPassTrace:
    """One tracking pass, one array row or list entry per station, as the pass
    measured it, before it moved it. `points`, `tangents`, `normals`,
    `strip_starts` and `strip_ends` are (N, 2) `(x, y)`; `windows` is (N, 2) of
    `(lo, hi)`, the offsets in px along the normal that a pair's midpoint had to
    fall in. `observed` is each station's pair offset, NaN where it was
    rejected; `weights` is the weight each observation carried in the pass's
    last solve, 0 at a rejected station; `corrections` is the correction the
    pass applied, in px along the normal. `reject` names each rejection (`None`
    at a hit), and `calipers` holds each strip's `CaliperTrace`."""

    points: npt.NDArray[np.float32]
    tangents: npt.NDArray[np.float32]
    normals: npt.NDArray[np.float32]
    windows: npt.NDArray[np.float32]
    strip_starts: npt.NDArray[np.float32]
    strip_ends: npt.NDArray[np.float32]
    observed: npt.NDArray[np.float32]
    weights: npt.NDArray[np.float32]
    corrections: npt.NDArray[np.float32]
    reject: List[Optional[str]]
    calipers: List[CaliperTrace]

class BeadStationTrace:
    """One station of the final stage, explained. Points and directions are
    `(x, y)`; the strip scans from `strip_start`, its -n end, to `strip_end`;
    `window` is `(lo, hi)` in px along the normal. `caliper` is everything the
    strip's caliper computed. At a hit `pair`, `offset` and `confidence`
    describe the bead's pair and `reject` is `None`; at a rejection they are
    `None` and `reject` names the reason."""

    point: Tuple[float, float]
    tangent: Tuple[float, float]
    normal: Tuple[float, float]
    window: Tuple[float, float]
    strip_start: Tuple[float, float]
    strip_end: Tuple[float, float]
    caliper: CaliperTrace
    pair: Optional[MeasurePair]
    offset: Optional[float]
    confidence: Optional[float]
    reject: Optional[str]

# ---------------------------------------------------------------------------
# Detectors, fitters, matchers, segmentation
# ---------------------------------------------------------------------------

class EdgeDetector:
    def __init__(self, config: Optional[EdgeConfig] = ...) -> None: ...
    def detect(self, img: ImageAny) -> List[Edgel]: ...
    def config(self) -> EdgeConfig: ...
    def set_config(self, config: EdgeConfig) -> None: ...

class LsdDetector:
    def __init__(self, config: Optional[LsdConfig] = ...) -> None: ...
    def detect(self, img: ImageAny) -> List[LineSegment]: ...
    def config(self) -> LsdConfig: ...
    def set_config(self, config: LsdConfig) -> None: ...

class Fitter:
    def __init__(self, config: Optional[FitConfig] = ...) -> None: ...
    def fit_ellipse(self, pts: PointsF32) -> Optional[Ellipse]: ...
    def fit_circle(self, pts: PointsF32) -> Optional[Circle]: ...
    def fit_line(self, pts: PointsF32) -> Optional[Line]: ...
    def config(self) -> FitConfig: ...
    def set_config(self, config: FitConfig) -> None: ...

class ShapeModel:
    def __init__(
        self,
        image: ImageAny,
        roi: Tuple[float, float, float, float],
        config: Optional[ShapeModelConfig] = ...,
        mask: Optional[ImageU8] = ...,
    ) -> None:
        """Build a model from a reference image and a rectangular ROI.

        `mask`, when given, is a uint8 array the same size as `image`: an edge
        point enters the model only where the mask is non-zero. Use it when the
        part is not rectangular -- the ROI's corners otherwise contribute
        background edges that no instance can match, and each one dilutes the
        score, whose denominator is the model's own point count."""

    @property
    def num_levels(self) -> int: ...
    @property
    def origin(self) -> Tuple[float, float]: ...
    @property
    def point_counts(self) -> List[int]: ...
    def reference_points(self) -> List[Tuple[float, float]]: ...
    @property
    def reference_angle(self) -> float:
        """The canonical orientation, radians, the model frame is rotated onto
        relative to the reference image (see ShapeModelConfig.reference_angle)."""

    def model_geometry(self, level: int) -> List[Tuple[float, float, float, float]]:
        """Model points at `level` as `(x, y, dx, dy)` in **model-frame**
        coordinates -- the frame a match's pose consumes, so these are what to
        transform by `ShapeMatch.matrix` to draw a found instance. Every level
        is reported in level-0 units. Empty when `level >= num_levels`."""

    def reference_geometry(self, level: int) -> List[Tuple[float, float, float, float]]:
        """`model_geometry` with `reference_angle` undone -- what to draw over
        the image the model was taught from."""

    def reference_frame_map(self, spec: CropSpec) -> Map:
        """The `dst -> src` map rectifying this model's own reference image
        into the same canonical crop `ShapeMatch.model_frame_map` produces for
        a found instance -- the untouched half of a side-by-side."""

    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> ShapeModel: ...
    @property
    def teach_point_count(self) -> int:
        """Points `resample_at` has to work with; 0 for a model loaded from
        a format-3 document (predating that stored data)."""

    def resample_at(self, s: float) -> ShapeModel:
        """Rebuild this model with every point resampled at scale `s`
        (estimate-then-verify). The result's own scale_range is
        a narrow band around 1.0 -- search that, and multiply a found
        match's own `scale` by `s` to recover scale relative to *this*
        model. Raises ValueError if `s` is not finite/positive, or this
        model has no stored teach data."""

class ShapeMatcher:
    def __init__(self, config: Optional[ShapeSearchConfig] = ...) -> None: ...
    def find(self, image: ImageAny, model: ShapeModel) -> List[ShapeMatch]: ...
    @property
    def truncated(self) -> bool: ...
    @property
    def config(self) -> ShapeSearchConfig: ...
    @config.setter
    def config(self, value: ShapeSearchConfig) -> None: ...

class Segmenter:
    def __init__(self) -> None: ...
    def otsu_threshold(self, img: ImageU8) -> int: ...
    def threshold_binary(self, img: ImageU8, threshold: int) -> ImageU8: ...
    def label_components(
        self, img: ImageU8, connectivity: Optional[int] = ...
    ) -> Tuple[npt.NDArray[np.int32], int]: ...
    def component_stats(
        self,
        label_img: npt.NDArray[np.int32],
        n_labels: int,
        min_area: Optional[int] = ...,
    ) -> List[ComponentStats]: ...

class CorrTemplate:
    """Compiled `corr` matching assets for one reference patch (corrmatch's
    pyramid, plus an angle bank when `rotation=True`). `image` must be
    `uint8` — see the `corr` module docs for why."""

    def __init__(
        self,
        image: ImageU8,
        rect: Tuple[float, float, float, float],
        config: Optional[CorrTemplateConfig] = ...,
    ) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def is_rotated(self) -> bool: ...

class ContourGraph:
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def num_junctions(self) -> int: ...
    def polylines(self) -> List[PointsF32]: ...
    def edge_lengths(self) -> List[float]: ...

# A (3, 3) float32 homogeneous matrix, row-major (numpy's default layout).
Matrix3x3 = npt.NDArray[np.float32]

class Map:
    """A precomputed `dst -> src` coordinate map: build once with `affine`,
    `projective` or `polar`, then call `apply`/`apply_with_mask` per frame.
    Every builder states the mapping from a destination pixel to the source
    coordinate it samples — see the Rust `warp` module docs for the
    dst -> src convention and how to invert a forward transform."""

    @staticmethod
    def affine(w: int, h: int, matrix: Matrix3x3) -> Map: ...
    @staticmethod
    def projective(w: int, h: int, matrix: Matrix3x3) -> Map: ...
    @staticmethod
    def polar(
        center: Tuple[float, float],
        r: Tuple[float, float],
        phi: Tuple[float, float],
        w: int,
        h: int,
    ) -> Map: ...
    @staticmethod
    def log_polar(
        center: Tuple[float, float],
        r: Tuple[float, float],
        phi: Tuple[float, float],
        w: int,
        h: int,
    ) -> Map:
        """Log-polar unwrap: `x` sweeps `phi` linearly, `y` sweeps `r`
        logarithmically -- a uniform scale change becomes a constant shift
        along `y` (Fourier-Mellin, no FFT; see `estimate_scale_logpolar`).
        `r[0]` must be > 0 and < `r[1]`."""

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def apply(
        self,
        img: ImageAny,
        interp: str = ...,
        border_mode: str = ...,
        border_constant: float = ...,
    ) -> ImageAny: ...
    def apply_with_mask(
        self,
        img: ImageAny,
        interp: str = ...,
        border_mode: str = ...,
        border_constant: float = ...,
    ) -> Tuple[ImageAny, ImageU8]: ...

# ---------------------------------------------------------------------------
# metric: the calibration bridge
# ---------------------------------------------------------------------------

# A (4, 4) float64 homogeneous camera-from-reference transform, row-major:
# top-left 3x3 rotation, last column translation in millimetres. `Pose3` has
# no dedicated Python class -- this is the numpy-friendly representation
# every function below takes and returns instead.
Pose3 = npt.NDArray[np.float64]

class PinholeIntrinsics:
    """Pinhole intrinsics, in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    skew: float
    def __init__(self, fx: float, fy: float, cx: float, cy: float, skew: float = ...) -> None: ...

class BrownConrady5:
    """Brown-Conrady 5-parameter radial-tangential distortion."""

    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    def __init__(
        self, k1: float = ..., k2: float = ..., k3: float = ..., p1: float = ..., p2: float = ...
    ) -> None: ...

class CameraModel:
    """A calibrated camera: intrinsics + distortion."""

    intrinsics: PinholeIntrinsics
    distortion: BrownConrady5
    def __init__(
        self, intrinsics: PinholeIntrinsics, distortion: Optional[BrownConrady5] = ...
    ) -> None: ...

class Plane3:
    """An oriented plane in the reference frame: `n . X + d = 0`."""

    n: Tuple[float, float, float]
    d: float
    def __init__(self, n: Tuple[float, float, float], d: float) -> None: ...
    @staticmethod
    def xy() -> Plane3: ...

class PlaneGrid:
    """A metric raster on the reference frame's `z = 0` plane."""

    origin_mm: Tuple[float, float]
    mm_per_px: float
    w: int
    h: int
    def __init__(
        self, origin_mm: Tuple[float, float], mm_per_px: float, w: int, h: int
    ) -> None: ...

def pixel_to_plane(
    camera: CameraModel, pose: Pose3, plane: Plane3, pixels: PointsF32
) -> npt.NDArray[np.float64]:
    """Project `(N, 2)` pixel coordinates onto `plane`, vectorized. Returns
    `(N, 2)` plane-frame `(x_mm, y_mm)`; a pixel whose ray misses the plane
    gets a NaN row rather than raising."""

def plane_grid_map(camera: CameraModel, pose: Pose3, grid: PlaneGrid) -> Map:
    """The runtime bird's-eye `dst -> src` map: destination is `grid` pixel
    coordinates, source is the raw (distorted) camera image."""

def undistort_map(camera: CameraModel, w: int, h: int) -> Map:
    """Build a `dst -> src` undistortion map for a `w x h` image shot by
    `camera`."""

def project_plane_points(
    camera: CameraModel, pose: Pose3, points_mm: PointsF32
) -> npt.NDArray[np.float64]:
    """Project `(N, 2)` reference-frame `z = 0` plane points (`x_mm, y_mm`)
    into `camera`'s raw (distorted) pixel space: `pose * (x_mm, y_mm, 0)`,
    perspective divide, then lens distortion, the forward geometry
    `plane_grid_map` uses, point by point. Returns `(N, 2)` pixel
    coordinates; a point behind the camera (`z <= 0`) gets a NaN row."""

def load_rig_extrinsics(source: Union[str, bytes]) -> List[Tuple[CameraModel, Pose3]]:
    """Load a calibration-rs `RigExtrinsicsExport` JSON document (file path
    or raw bytes). One `(CameraModel, pose)` pair per camera."""

def load_table_calibration(source: Union[str, bytes]) -> List[Tuple[CameraModel, Pose3]]:
    """Load a `table_calibration` `calibration.json` document (file path or
    raw bytes). One `(CameraModel, pose)` pair per camera, sorted by index."""

# ---------------------------------------------------------------------------
# corr: cross-correlation matching + displacement
# ---------------------------------------------------------------------------

def find(
    template: CorrTemplate, scene: ImageU8, config: Optional[CorrConfig] = ...
) -> CorrMatch:
    """Finds `template`'s best match in a `uint8` scene."""

def find_topk(
    template: CorrTemplate,
    scene: ImageU8,
    k: int,
    config: Optional[CorrConfig] = ...,
) -> List[CorrMatch]:
    """Finds up to `k` matches, best score first."""

def displacement(
    prev: ImageU8, curr: ImageU8, config: DisplacementConfig
) -> Displacement:
    """Two-stage subpixel inter-frame shift: bounded ZNCC search + (by
    default) translation-only Lucas-Kanade refinement."""

# ---------------------------------------------------------------------------
# scale: estimate-then-verify
# ---------------------------------------------------------------------------

def estimate_scale_moments(
    model: ShapeModel,
    scene: ImageU8,
    roi: Tuple[float, float, float, float],
    config: Optional[MomentScaleConfig] = ...,
) -> ScaleEstimate:
    """Estimate scale from a segmented scene blob's spatial spread vs. the
    taught model's own. Works on any model."""

def estimate_scale_logpolar(
    model: ShapeModel,
    scene: ImageU8,
    approx_center: Tuple[float, float],
    config: Optional[LogPolarScaleConfig] = ...,
) -> ScaleEstimate:
    """Estimate scale (and, with `config.angle_margin` set, rotation) via
    log-polar ZNCC correlation. Requires `model.teach_point_count > 0`
    (teach data, from format 4 or later)."""

def find_scale_invariant_roi(
    model: ShapeModel,
    scene: ImageU8,
    roi: Tuple[float, float, float, float],
    config: Optional[ScaleInvariantConfig] = ...,
) -> List[ShapeMatch]:
    """Estimate scale via `estimate_scale_moments` over `roi`, resample
    `model` at that estimate, and verify in a narrow band -- one call for
    the whole estimate-then-verify strategy. Empty list, not an error, when
    nothing scores above `config.search.min_score`."""

def find_scale_invariant_center(
    model: ShapeModel,
    scene: ImageU8,
    center: Tuple[float, float],
    config: Optional[ScaleInvariantConfig] = ...,
) -> List[ShapeMatch]:
    """Same as `find_scale_invariant_roi`, estimating scale via
    `estimate_scale_logpolar` around `center` instead of segmenting `roi`."""

# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def detect_edges(img: ImageAny, config: EdgeConfig) -> List[Edgel]: ...
def detect_line_segments(img: ImageAny, config: LsdConfig) -> List[LineSegment]: ...
def fit_ellipse(pts: PointsF32, config: FitConfig) -> Optional[Ellipse]: ...
def fit_line(pts: PointsF32, config: FitConfig) -> Optional[Line]: ...
def find_shape_model(
    model_image: ImageAny,
    roi: Tuple[float, float, float, float],
    scene_image: ImageAny,
    model_config: Optional[ShapeModelConfig] = ...,
    search_config: Optional[ShapeSearchConfig] = ...,
    model_mask: Optional[ImageU8] = ...,
) -> List[ShapeMatch]: ...
def otsu_threshold(img: ImageU8) -> int: ...
def threshold_binary(img: ImageU8, threshold: int) -> ImageU8: ...
def label_components(
    img: ImageU8, connectivity: int = ...
) -> Tuple[npt.NDArray[np.int32], int]: ...
def component_stats(
    label_img: npt.NDArray[np.int32], n_labels: int, min_area: int = ...
) -> List[ComponentStats]: ...
def build_contour_graph(
    img: ImageU8,
    edge_config: Optional[EdgeConfig] = ...,
    connectivity: Optional[str] = ...,
    min_component_size: Optional[int] = ...,
    record_geometry: Optional[bool] = ...,
    thin: Optional[bool] = ...,
) -> ContourGraph: ...
def smooth_polyline(points: PointsF32, sigma: float) -> PointsF32: ...
def erode(img: ImageU8, shape: str = ..., radius: int = ...) -> ImageU8: ...
def dilate(img: ImageU8, shape: str = ..., radius: int = ...) -> ImageU8: ...
def open(img: ImageU8, shape: str = ..., radius: int = ...) -> ImageU8: ...
def close(img: ImageU8, shape: str = ..., radius: int = ...) -> ImageU8: ...
def thin(img: ImageU8) -> ImageU8: ...
def chamfer_distance(img: ImageU8) -> npt.NDArray[np.float32]: ...
