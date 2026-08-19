//! Multi-scale edge detector implementation.

use vm_primitives::Pyramid;
use vm_primitives::edge::edge2d::{Edge2DConfig, Edge2DDetector};
use vm_primitives::{ImageView, Point2f, Vec2f};

use crate::{MultiScaleConfig, ScaleAnnotatedEdgel};

/// Combines [`Pyramid`] and [`Edge2DDetector`] to produce edgels across
/// multiple pyramid levels in a single call.
///
/// Owns all scratch buffers internally so allocations happen only on the first
/// call (or when the image size changes). Subsequent calls with the same
/// image dimensions are allocation-free beyond the output `Vec`.
///
/// # Example
///
/// ```
/// use vm_primitives::Image;
/// use vision_metrology::{MultiScaleConfig, MultiScaleEdgeDetector};
///
/// // Synthetic 32×32 step-edge image.
/// let mut data = vec![0u8; 32 * 32];
/// for y in 0..32 { for x in 16..32 { data[y * 32 + x] = 200; } }
/// let img = Image::from_vec(32, 32, data).unwrap();
///
/// let mut det = MultiScaleEdgeDetector::new();
/// let cfg = MultiScaleConfig { num_levels: 2, ..MultiScaleConfig::default() };
/// let edgels = det.detect_u8(&img.as_view(), &cfg);
///
/// // At least one edgel should be found on the vertical step.
/// assert!(!edgels.is_empty());
/// // All positions are in level-0 coordinates (x ∈ [0, 32)).
/// for e in &edgels { assert!(e.p.x < 32.0 && e.p.y < 32.0); }
/// ```
#[derive(Debug)]
pub struct MultiScaleEdgeDetector {
    pyramid: Pyramid,
    detector: Edge2DDetector,
    /// Scratch bitmask: `cell_used[y * base_w + x]` is set when that level-0
    /// pixel cell has already been claimed by a finer-scale detection.
    cell_used: Vec<bool>,
}

impl Default for MultiScaleEdgeDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiScaleEdgeDetector {
    /// Create a new detector with empty scratch buffers.
    pub fn new() -> Self {
        Self {
            pyramid: Pyramid::new(),
            detector: Edge2DDetector::new(),
            cell_used: Vec::new(),
        }
    }

    /// Detect edgels across pyramid levels from a `u8` source image.
    pub fn detect_u8(
        &mut self,
        img: &ImageView<'_, u8>,
        cfg: &MultiScaleConfig,
    ) -> Vec<ScaleAnnotatedEdgel> {
        self.pyramid.build(img, cfg.num_levels);
        self.detect_from_pyramid(img.width(), img.height(), cfg)
    }

    /// Detect edgels across pyramid levels from a `u16` source image.
    pub fn detect_u16(
        &mut self,
        img: &ImageView<'_, u16>,
        cfg: &MultiScaleConfig,
    ) -> Vec<ScaleAnnotatedEdgel> {
        self.pyramid.build(img, cfg.num_levels);
        self.detect_from_pyramid(img.width(), img.height(), cfg)
    }

    /// Detect edgels across pyramid levels from an `f32` source image.
    pub fn detect_f32(
        &mut self,
        img: &ImageView<'_, f32>,
        cfg: &MultiScaleConfig,
    ) -> Vec<ScaleAnnotatedEdgel> {
        self.pyramid.build(img, cfg.num_levels);
        self.detect_from_pyramid(img.width(), img.height(), cfg)
    }

    /// Core implementation — runs after the pyramid has been built.
    fn detect_from_pyramid(
        &mut self,
        base_w: usize,
        base_h: usize,
        cfg: &MultiScaleConfig,
    ) -> Vec<ScaleAnnotatedEdgel> {
        let actual_levels = self.pyramid.num_levels();
        let mut out: Vec<ScaleAnnotatedEdgel> = Vec::new();

        for level in 0..actual_levels {
            let scale_factor = 1usize << level; // 2^level
            let scale_factor_f = scale_factor as f32;

            // Build a per-level edge config with scaled thresholds.
            let level_cfg = scaled_config(&cfg.edge, level);

            // Detect on the pyramid level (always f32 after pyramid build).
            let level_img = self
                .pyramid
                .level(level)
                .expect("level within actual_levels");
            let edgels = self.detector.detect_f32(&level_img.as_view(), &level_cfg);

            // Map edgels to level-0 coordinates and annotate.
            for e in edgels {
                // Level-0 integer cell: multiply index by scale factor.
                let lx = e.idx.0.saturating_mul(scale_factor);
                let ly = e.idx.1.saturating_mul(scale_factor);

                // Skip if outside level-0 bounds (can happen at boundary).
                if lx >= base_w || ly >= base_h {
                    continue;
                }

                out.push(ScaleAnnotatedEdgel {
                    p: Point2f {
                        x: e.p.x * scale_factor_f,
                        y: e.p.y * scale_factor_f,
                    },
                    n: Vec2f { x: e.n.x, y: e.n.y },
                    strength: e.strength,
                    scale: cfg.base_sigma * scale_factor_f,
                    level,
                    idx: (lx, ly),
                });
            }
        }

        // Deduplication: keep finest-scale detection per level-0 pixel cell.
        // Edgels were pushed finest-first (level 0 before level 1, etc.),
        // so the first occurrence of each cell is always the finest.
        if cfg.merge_duplicates && base_w > 0 && base_h > 0 {
            let cell_count = base_w * base_h;
            self.cell_used.clear();
            self.cell_used.resize(cell_count, false);

            let cell_used = &mut self.cell_used;
            out.retain(|e| {
                let cell = e.idx.1 * base_w + e.idx.0;
                if cell_used[cell] {
                    false
                } else {
                    cell_used[cell] = true;
                    true
                }
            });
        }

        out
    }
}

/// Build a modified `Edge2DConfig` with thresholds halved `level` times.
///
/// When both thresholds are zero (auto-threshold mode), they are left at zero
/// so the detector continues to auto-threshold at each level independently.
fn scaled_config(base: &Edge2DConfig, level: usize) -> Edge2DConfig {
    if level == 0 || (base.low_thresh == 0.0 && base.high_thresh == 0.0) {
        return base.clone();
    }
    let scale = (1u32 << level) as f32;
    Edge2DConfig {
        low_thresh: base.low_thresh / scale,
        high_thresh: base.high_thresh / scale,
        ..base.clone()
    }
}

#[cfg(test)]
mod tests {
    use vm_primitives::Image;

    use super::MultiScaleEdgeDetector;
    use crate::MultiScaleConfig;

    fn step_image(w: usize, h: usize) -> Image<u8> {
        let data: Vec<u8> = (0..w * h)
            .map(|i| if (i % w) >= w / 2 { 200u8 } else { 0u8 })
            .collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    #[test]
    fn detects_step_edge_at_level0() {
        let img = step_image(64, 64);
        let mut det = MultiScaleEdgeDetector::new();
        let cfg = MultiScaleConfig {
            num_levels: 1,
            ..MultiScaleConfig::default()
        };
        let edgels = det.detect_u8(&img.as_view(), &cfg);

        assert!(!edgels.is_empty(), "no edgels found on a clear step edge");
        // All edgels should be level 0 when only one level is requested.
        assert!(edgels.iter().all(|e| e.level == 0));
        // Edgels near the step at x ≈ 32.
        let near_step = edgels.iter().filter(|e| (e.p.x - 32.0).abs() < 2.0).count();
        assert!(near_step > 0, "no edgels found near the step at x=32");
    }

    #[test]
    fn positions_in_level0_coords() {
        // All positions must be within the original image bounds regardless of level.
        let img = step_image(64, 64);
        let mut det = MultiScaleEdgeDetector::new();
        let cfg = MultiScaleConfig {
            num_levels: 3,
            ..MultiScaleConfig::default()
        };
        let edgels = det.detect_u8(&img.as_view(), &cfg);

        for e in &edgels {
            assert!(
                e.p.x >= 0.0 && e.p.x < 64.0,
                "p.x={} out of range for level {}",
                e.p.x,
                e.level
            );
            assert!(
                e.p.y >= 0.0 && e.p.y < 64.0,
                "p.y={} out of range for level {}",
                e.p.y,
                e.level
            );
        }
    }

    #[test]
    fn merge_duplicates_reduces_count() {
        let img = step_image(64, 64);
        let mut det = MultiScaleEdgeDetector::new();

        let cfg_merge = MultiScaleConfig {
            num_levels: 3,
            merge_duplicates: true,
            ..MultiScaleConfig::default()
        };
        let cfg_all = MultiScaleConfig {
            num_levels: 3,
            merge_duplicates: false,
            ..MultiScaleConfig::default()
        };

        let merged = det.detect_u8(&img.as_view(), &cfg_merge);
        let all = det.detect_u8(&img.as_view(), &cfg_all);

        assert!(
            merged.len() <= all.len(),
            "merge should not increase count: merged={} all={}",
            merged.len(),
            all.len()
        );
    }

    #[test]
    fn merge_dedup_no_duplicate_cells() {
        let img = step_image(64, 64);
        let mut det = MultiScaleEdgeDetector::new();
        let cfg = MultiScaleConfig {
            num_levels: 3,
            merge_duplicates: true,
            ..MultiScaleConfig::default()
        };
        let edgels = det.detect_u8(&img.as_view(), &cfg);

        // No two edgels should share the same level-0 grid cell.
        let mut seen = std::collections::HashSet::new();
        for e in &edgels {
            assert!(seen.insert(e.idx), "duplicate cell {:?}", e.idx);
        }
    }

    #[test]
    fn scale_annotation_matches_level() {
        let img = step_image(64, 64);
        let mut det = MultiScaleEdgeDetector::new();
        let base_sigma = 1.2_f32;
        let cfg = MultiScaleConfig {
            num_levels: 3,
            base_sigma,
            merge_duplicates: false,
            ..MultiScaleConfig::default()
        };
        let edgels = det.detect_u8(&img.as_view(), &cfg);

        for e in &edgels {
            let expected_scale = base_sigma * (1 << e.level) as f32;
            assert!(
                (e.scale - expected_scale).abs() < 1e-6,
                "level={} expected scale={} got {}",
                e.level,
                expected_scale,
                e.scale
            );
        }
    }

    #[test]
    fn zero_levels_returns_empty() {
        let img = step_image(32, 32);
        let mut det = MultiScaleEdgeDetector::new();
        let cfg = MultiScaleConfig {
            num_levels: 0,
            ..MultiScaleConfig::default()
        };
        let edgels = det.detect_u8(&img.as_view(), &cfg);
        assert!(edgels.is_empty());
    }

    #[test]
    fn u16_and_f32_detect_paths_return_results() {
        let img_u16 = Image::from_vec(
            32,
            32,
            (0..32 * 32)
                .map(|i| if (i % 32) >= 16 { 50000u16 } else { 0 })
                .collect(),
        )
        .unwrap();
        let img_f32 = Image::from_vec(
            32,
            32,
            (0..32 * 32)
                .map(|i| if (i % 32) >= 16 { 200.0f32 } else { 0.0 })
                .collect(),
        )
        .unwrap();

        let mut det = MultiScaleEdgeDetector::new();
        let cfg = MultiScaleConfig {
            num_levels: 2,
            ..MultiScaleConfig::default()
        };

        assert!(!det.detect_u16(&img_u16.as_view(), &cfg).is_empty());
        assert!(!det.detect_f32(&img_f32.as_view(), &cfg).is_empty());
    }
}
