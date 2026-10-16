//! Cross-correlation matching and inter-frame subpixel displacement.
//!
//! `corr` is a thin wrapper over [`corrmatch`], the project's standard
//! cross-correlation engine. It does **not** reimplement corrmatch's
//! pyramid, angle bank, beam search, or subpixel refinement: [`CorrTemplate`]
//! adapts this crate's [`ImageView`](crate::ImageView) into corrmatch's own,
//! and [`find`] /
//! [`find_topk`] map its `Match` back into this crate's coordinate and angle
//! conventions (pixel-centre coordinates, radians — the same conventions
//! `tests/corrmatch_bridge.rs` pins for the `matching` module).
//!
//! # `corr` vs `matching`
//!
//! [`crate::matching`]'s `ShapeMatcher` is a gradient-orientation similarity
//! measure: illumination-invariant, occlusion-robust
//! (`score ≈ 1 - occluded_fraction`), and searches translation + rotation +
//! **uniform scale**. `corr`'s [`CorrMatch`] is a raw intensity correlation:
//! cheaper, but sensitive to illumination change (`Zncc` tolerates
//! offset/gain, not more) and has **no scale search at all** — corrmatch's
//! published API does not offer one. Use `corr` for a known, roughly
//! constant-scale target (tracking, small inter-frame motion, template
//! confirmation at a pose another stage already found) and `matching` for
//! teach-and-find under unknown pose.
//!
//! # Pixel types
//!
//! Both [`CorrTemplate::from_image`] and [`displacement`] take
//! `ImageView<'_, u8>` only — no `Pixel` generic pretense. corrmatch's
//! published API (0.2.5) is `u8`-only; `u16`/`f32` support belongs in
//! corrmatch itself, not in a wrapper that would have to quantize its way
//! around it and misreport the precision.
//!
//! [`TranslationEstimator`] is a specialized native two-mask registration operator,
//! generic over `u8`, `u16` and `f32`. It reports typed acceptance, photometry,
//! covariance and regional consistency. The experimental `SimilarityEstimator`
//! (feature `corr-experimental`) shares that narrow native exception for local
//! prediction-guided rotation/scale; neither changes the ordinary matcher.
//!
//! # Quick start
//!
//! ```no_run
//! use vision_metrology::corr::{CorrConfig, CorrTemplate, CorrTemplateConfig, find};
//! use vision_metrology::{Image, Rect2f};
//!
//! # fn main() -> Result<(), vision_metrology::Error> {
//! let reference: Image<u8> = Image::new_fill(640, 480, 0);
//! let rect = Rect2f { x: 280.0, y: 200.0, width: 64.0, height: 64.0 };
//! let template = CorrTemplate::from_image(&reference.as_view(), rect, &CorrTemplateConfig::default())?;
//!
//! let scene: Image<u8> = Image::new_fill(640, 480, 0);
//! let m = find(&template, &scene.as_view(), &CorrConfig::default())?;
//! println!("score {:.3} at {:?}", m.score, m.position);
//! # Ok(())
//! # }
//! ```

mod adapter;
mod config;
mod displacement;
mod plane;
#[cfg(feature = "corr-experimental")]
mod similarity;
mod template;
mod translation;

pub use config::{
    CorrConfig, CorrMetric, CorrSearchTuning, CorrTemplateConfig, CorrTemplateTuning,
    DisplacementConfig, Refine,
};
pub use displacement::{Displacement, displacement};
#[cfg(feature = "corr-experimental")]
pub use similarity::{Similarity, SimilarityConfig, SimilarityEstimator};
pub use template::{CorrMatch, CorrTemplate, find, find_topk};

pub use translation::{
    RegionCheck, Translation, TranslationConfig, TranslationEstimator, TranslationStatus,
    TranslationTuning,
};
