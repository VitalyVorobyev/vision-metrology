"""Type stubs for `vision_metrology` — PyO3 bindings for the vision-metrology
workspace. See crates/vm-python/README.md for the guide; this file is the
contract IDEs and mypy see.

Hand-maintained: keep in step with `src/lib.rs`'s `#[pymodule]` registration
list whenever a class or free function is added, renamed or removed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

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

class MeasureConfig:
    sigma: float
    threshold: float
    polarity: str
    select: str
    step: float
    max_obliquity_deg: float
    border_mode: str
    border_constant: float
    def __init__(
        self,
        sigma: Optional[float] = ...,
        threshold: Optional[float] = ...,
        polarity: Optional[str] = ...,
        select: Optional[str] = ...,
        step: Optional[float] = ...,
        max_obliquity_deg: Optional[float] = ...,
        border_mode: Optional[str] = ...,
        border_constant: Optional[float] = ...,
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

class Caliper:
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
    def measure(self, img: ImageAny) -> List[MeasureEdge]: ...
    def measure_pairs(self, img: ImageAny) -> List[MeasurePair]: ...
    def profile(self) -> List[float]: ...

class MeasureRejected(Exception):
    """Raised by `Caliper.measure`; `args[0]` is one of `"profile_too_short"`,
    `"no_edge"`, `"wrong_polarity"`, `"too_oblique"`, `"off_image"`."""

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
    ) -> None: ...
    @property
    def num_levels(self) -> int: ...
    @property
    def origin(self) -> Tuple[float, float]: ...
    @property
    def point_counts(self) -> List[int]: ...
    def reference_points(self) -> List[Tuple[float, float]]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> ShapeModel: ...

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
