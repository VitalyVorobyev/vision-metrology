//! Configuration for multi-scale edge detection.

use vm_primitives::edge::edge2d::Edge2DConfig;

/// Configuration for [`MultiScaleEdgeDetector`].
///
/// [`MultiScaleEdgeDetector`]: crate::MultiScaleEdgeDetector
///
/// # Threshold scaling
///
/// The edge detector auto-thresholds when `edge.low_thresh == 0.0` and
/// `edge.high_thresh == 0.0` (recommended: works well across all levels).
///
/// When explicit thresholds are set, they are halved at each level:
/// at level `l` the detector sees `low_thresh / 2^l` and `high_thresh / 2^l`.
/// This compensates for the gradient magnitude reduction caused by 2×2 mean
/// downsampling and keeps detection sensitivity roughly uniform across levels.
///
/// # Deduplication
///
/// When `merge_duplicates` is true, edgels occupying the same level-0 pixel
/// cell are deduplicated: the finest-scale detection (lowest `level`) is kept.
/// This prevents the same physical edge from appearing once per pyramid level.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiScaleConfig {
    /// Total number of pyramid levels to process, **including level 0**
    /// (the original resolution image).
    ///
    /// `num_levels = 1`: only the original image is processed.
    /// `num_levels = 3`: original + two half-sized downsampled levels.
    ///
    /// Processing stops early if an intermediate level would be smaller than 2×2.
    pub num_levels: usize,

    /// Base Gaussian sigma used to annotate edgels.
    ///
    /// The effective sigma at level `l` is `base_sigma * 2^l`. This field is
    /// purely informational — it does not affect the detection thresholds.
    /// Set to the sigma of the edge detector's implicit pre-smoothing
    /// (typically `~1.0` for `SmoothKind::Binomial3`).
    pub base_sigma: f32,

    /// Edge detector configuration applied at level 0.
    ///
    /// Thresholds are auto-scaled per level as described above.
    /// All other fields (`pre_smooth`, `smooth_kind`, `border`, `subpix`) are
    /// shared across all levels unchanged.
    pub edge: Edge2DConfig,

    /// When `true`, suppress edgels that fall in the same level-0 pixel cell
    /// as a finer-scale (lower-level) edgel.
    ///
    /// Recommended: `true` for shape detection (avoids double-counting edges),
    /// `false` when scale redundancy is useful (e.g. confidence estimation).
    pub merge_duplicates: bool,
}

impl Default for MultiScaleConfig {
    fn default() -> Self {
        Self {
            num_levels: 3,
            base_sigma: 1.0,
            edge: Edge2DConfig::default(), // auto-threshold, Binomial3 smooth, parabolic subpix
            merge_duplicates: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use vm_primitives::edge::edge2d::Edge2DConfig;

    use super::MultiScaleConfig;

    /// The documented contract of the defaults: three levels including the
    /// original, informational sigma 1.0, auto-thresholding edge config, and
    /// per-cell dedup on. Changing any of these is a behavior change for every
    /// downstream user and must be deliberate.
    #[test]
    fn default_values_are_the_documented_contract() {
        let cfg = MultiScaleConfig::default();
        assert_eq!(cfg.num_levels, 3);
        assert_eq!(cfg.base_sigma, 1.0);
        assert!(cfg.merge_duplicates);
        assert_eq!(cfg.edge, Edge2DConfig::default());
        // Auto-threshold mode is what makes the per-level halving irrelevant
        // by default; pin that the default edge config actually is auto.
        assert_eq!(cfg.edge.low_thresh, 0.0);
        assert_eq!(cfg.edge.high_thresh, 0.0);
    }
}
