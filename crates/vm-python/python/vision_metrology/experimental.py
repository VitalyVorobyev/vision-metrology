"""Experimental API, outside the compatibility promise.

`SimilarityEstimator`, `SimilarityConfig` and `Similarity` (local masked similarity
registration) may change or be removed in any release, until the evidence supports
promotion.
"""

from .vision_metrology import _experimental as _native

Similarity = _native.Similarity
SimilarityConfig = _native.SimilarityConfig
SimilarityEstimator = _native.SimilarityEstimator

__all__ = ["Similarity", "SimilarityConfig", "SimilarityEstimator"]
