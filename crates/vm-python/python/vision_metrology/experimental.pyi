"""Type stubs for `vision_metrology.experimental`: outside the compatibility promise.

Hand-maintained: keep in step with the `_experimental` submodule in `src/lib.rs`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from . import TranslationConfig

__all__ = ["Similarity", "SimilarityConfig", "SimilarityEstimator"]

class SimilarityConfig:
    translation: TranslationConfig
    max_angle: float
    max_log_scale: float
    def __init__(self, *, translation: Optional[TranslationConfig] = None, max_angle: float = 0.05, max_log_scale: float = 0.05) -> None: ...

class Similarity:
    """Only status='accepted' permits using the previous-to-current transform."""
    transform: Optional[Tuple[float, float, float, float]]  # tx, ty, angle radians, scale
    score: Optional[float]
    margin: Optional[float]
    support: int
    residual_rms: Optional[float]
    residual_max: Optional[float]
    gain: Optional[float]
    offset: Optional[float]
    covariance: Optional[List[List[float]]]  # centre dx/dy, angle, log-scale
    regions: int
    region_error: Optional[float]
    status: str

class SimilarityEstimator:
    def __init__(self, config: Optional[SimilarityConfig] = None) -> None: ...
    def estimate(self, previous: np.ndarray, current: np.ndarray, *, previous_mask: Optional[np.ndarray] = None, current_mask: Optional[np.ndarray] = None, prediction: Optional[Tuple[float, float, float, float]] = None) -> Similarity: ...
