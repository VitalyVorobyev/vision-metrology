//! [`ShapeModel::resample_at`] — rebuild a model's pyramid levels at an
//! arbitrary uniform scale, from its own stored teach-time edge points.
//!
//! # Why this exists (roadmap W7, "estimate-then-verify")
//!
//! A discrete `scale_range` scan is one search per scale step, which is
//! linear in how wide a range is searched — expensive for a genuinely wide
//! range, and the wrong shape of cost for a problem where the scale is
//! *approximately known* (from a segmentable blob's area, or a coarse
//! log-polar correlation — `vision_metrology::scale`) rather than unknown.
//! `resample_at` is the other half of that strategy: given an estimate `ŝ`,
//! rebuild a model whose points already sit at `ŝ`, then search a **narrow**
//! band around it — one anchor per estimate, not a static bank of models at
//! K fixed scales, and not a wider scan.
//!
//! # What it needs, and why a format-3 model cannot do this
//!
//! Every [`ShapeModel`] already stores its level-0 points *after* grid
//! decimation (`level(0).points()`), but decimation is exactly what a
//! different scale needs to redo — invariant 4 requires spatially uniform
//! decimation, and a cell sized for scale 1.0 is the wrong size at scale
//! 0.4 or 2.5. So the model additionally stores the *pre*-decimation level-0
//! set (`TeachPoint`, format 4) — the same data
//! [`ShapeModelBuilder::build`]/`from_edgels`/`from_directed_points`/
//! `from_polylines` already compute internally before their own one-time
//! assembly, just not previously kept around afterward. A model loaded from
//! a format-3 document predates this field and returns
//! [`Error::InvalidConfig`] here rather than resampling from decimated (and
//! therefore already-scale-1.0-shaped) points, which would silently bias
//! the result.
//!
//! # How the rebuild works
//!
//! Reuses `matching::build`'s own geometry-only assembly path
//! (`from_raw`/`assemble`/`merge_into_cells`/`decimate`/`stratify`) verbatim
//! — scaling every teach point's offset by `s` and feeding the result back
//! through exactly the pipeline [`ShapeModel::from_directed_points`] already
//! uses is simpler and safer than a second implementation of "decimate a
//! point cloud into levels". Offsets scale cleanly because they are
//! *differences* from the origin: the pyramid coordinate map
//! (`(v - (2^l-1)/2) / 2^l`, invariant 2) is affine, so subtracting two
//! points at the same level cancels its additive term and leaves a clean
//! `/ 2^l` — which is exactly what dividing a level-0 offset by `2^l` gives
//! directly, without re-deriving an absolute origin position at every level.

use vm_primitives::{Edge2DConfig, Error, Point2f};

use super::build::{MAX_LEVELS, RawPoint, from_raw};
use super::config::{Contrast, ShapeModelConfig};
use super::model::ShapeModel;

/// A narrow scale_range on the resampled model, centered on 1.0 — the
/// "verify" half of estimate-then-verify. Widening it defeats the point of
/// resampling: the constant-cost claim rests on searching a fixed, small
/// number of scale steps regardless of how far `s` was from 1.0.
const VERIFY_SCALE_RANGE: (f32, f32) = (0.95, 1.05);

impl ShapeModel {
    /// Rebuild this model with every point resampled at scale `s`.
    ///
    /// The result's own [`scale_range`](ShapeModel::scale_range) is a narrow
    /// band around 1.0 (`(0.95, 1.05)`) — a caller doing estimate-then-verify
    /// should search that band, not widen it, and multiply a found match's
    /// own [`ShapeMatch::scale`](super::ShapeMatch::scale) by `s` to recover
    /// the scale relative to the *original* model. `angle_range` and
    /// `polarity` carry over unchanged (rotation and photometric polarity
    /// are not affected by a uniform rescale); `pre_smooth`/`smooth` carry
    /// over too, so the resampled model stays compatible with a scene
    /// decimated the same way as the original (invariant 3).
    ///
    /// Deterministic: two calls with the same `s` produce identical models
    /// (no RNG anywhere in the pipeline, per invariant 12).
    ///
    /// # Errors
    /// - [`Error::InvalidConfig`] if `s` is not finite and positive, or if
    ///   this model has no stored teach data (`teach_point_count() == 0` —
    ///   either loaded from a format-3 document, or built with too few
    ///   surviving level-0 points to store any).
    /// - [`Error::InsufficientData`] if too few points survive resampling
    ///   at `s` to build even one level (e.g. `s` so small the whole model
    ///   collapses under the spatial decimation grid).
    pub fn resample_at(&self, s: f32) -> Result<ShapeModel, Error> {
        if !(s.is_finite() && s > 0.0) {
            return Err(Error::InvalidConfig(
                "resample_at: scale must be finite and positive",
            ));
        }
        let teach = self.teach_points().ok_or(Error::InvalidConfig(
            "resample_at: this model has no stored teach-time edge data — it was loaded from a \
             format-3 document (predating that field) or built with none surviving; \
             rebuild/retrain with the current format to use resample_at",
        ))?;

        // `origin: Some((0, 0))` is the trick that lets `from_raw`'s
        // existing per-level `to_level` + `merge_into_cells` pipeline treat
        // these already-relative offsets as if they were absolute level-0
        // positions: `to_level` is affine with the same additive term for
        // every point, so it cancels in `from_raw`'s own `p - ref_l`
        // subtraction and what survives is exactly `offset / 2^level` — see
        // the module doc.
        let raw: Vec<RawPoint> = teach
            .iter()
            .map(|tp| RawPoint {
                p: Point2f::new(tp.d.x * s, tp.d.y * s),
                t: tp.t,
                strength: tp.strength,
            })
            .collect();

        // `from_raw` never runs an edge detector — it only reads
        // `edge.smooth_kind` (which feeds the built model's own `smooth()`)
        // off this config; every other `Edge2DConfig` field is inert here.
        let edge = Edge2DConfig {
            smooth_kind: self.smooth(),
            ..Edge2DConfig::default()
        };
        let cfg = ShapeModelConfig {
            num_levels: core::num::NonZeroUsize::new(self.num_levels().min(MAX_LEVELS)),
            edge,
            pre_smooth: self.pre_smooth(),
            min_contrast: Contrast::Raw(0.0),
            max_points: core::num::NonZeroUsize::new(self.point_count(0).max(8)),
            origin: Some(Point2f::new(0.0, 0.0)),
            angle_range: self.angle_range(),
            scale_range: VERIFY_SCALE_RANGE,
            polarity: self.polarity(),
            min_points_per_level: 8,
        };

        from_raw(raw, &cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::VERIFY_SCALE_RANGE;
    use crate::matching::{ShapeModel, ShapeModelBuilder, ShapeModelConfig};
    use vm_primitives::{Image, Rect2f};

    fn l_disc(size: usize, cx: f32, cy: f32, r: f32) -> Image<u8> {
        let mut data = vec![20u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if (dx * dx + dy * dy).sqrt() < r {
                    data[y * size + x] = 220;
                }
            }
        }
        Image::from_vec(size, size, data).expect("valid image")
    }

    fn build_disc_model() -> ShapeModel {
        let img = l_disc(200, 100.0, 100.0, 40.0);
        let roi = Rect2f {
            x: 40.0,
            y: 40.0,
            width: 120.0,
            height: 120.0,
        };
        ShapeModelBuilder::new()
            .build(&img.as_view(), roi, &ShapeModelConfig::default())
            .expect("disc model builds")
    }

    #[test]
    fn a_freshly_built_model_carries_teach_points() {
        let model = build_disc_model();
        assert!(model.teach_point_count() > 0);
    }

    #[test]
    fn resample_at_one_reproduces_roughly_the_same_geometry() {
        let model = build_disc_model();
        let same = model.resample_at(1.0).expect("resample at 1.0");
        assert_eq!(same.scale_range(), VERIFY_SCALE_RANGE);
        // Level-0 radius should match the original to within the grid
        // decimation's own jitter — resampling at 1.0 reruns the same
        // assembly, not an identity copy.
        let r0 = model.level(0).unwrap().radius();
        let r1 = same.level(0).unwrap().radius();
        assert!((r0 - r1).abs() < 1.0, "r0={r0} r1={r1}");
    }

    #[test]
    fn resample_at_half_halves_the_radius() {
        let model = build_disc_model();
        let half = model.resample_at(0.5).expect("resample at 0.5");
        let r0 = model.level(0).unwrap().radius();
        let r1 = half.level(0).unwrap().radius();
        assert!((r1 - 0.5 * r0).abs() < 1.0, "r0={r0} r1={r1}");
    }

    #[test]
    fn resample_at_rejects_non_positive_or_non_finite_scale() {
        let model = build_disc_model();
        assert!(model.resample_at(0.0).is_err());
        assert!(model.resample_at(-1.0).is_err());
        assert!(model.resample_at(f32::NAN).is_err());
        assert!(model.resample_at(f32::INFINITY).is_err());
    }

    // A format-3 document (no `teach_points` field at all) is the only way
    // to reach `resample_at`'s "no teach data" branch through the public
    // API; that path is exercised in `tests/shape_matching.rs` (`serde`
    // feature), next to the rest of the format-version tests.
}
