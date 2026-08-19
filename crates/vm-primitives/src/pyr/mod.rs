//! Image pyramids for coarse-to-fine processing.
//!
//! A [`Pyramid`] halves the image at every level with a 2×2 box mean, and is
//! generic over the source [`Pixel`](crate::core::Pixel) type: levels are always
//! `f32`, inputs may be `u8`, `u16` or `f32`.
//!
//! ## Decimation policy
//! - Output size is `(w / 2, h / 2)`; a trailing odd column or row is dropped.
//! - Each destination pixel is the mean of one non-overlapping 2×2 block, so
//!   level `L+1` summarises non-overlapping 2×2 neighbourhoods of level `L`.
//! - Integer types round half-up; `f32` scales by `0.25`.
//!
//! ## Coordinates
//! [`level_to_base`] and [`base_to_level`] are the single implementation of the
//! level↔level-0 mapping (system-design invariant 2). Never re-derive it inline.
//!
//! ## Aliasing
//! A box mean has no stop-band. [`PreSmooth::Binomial121`] adds a symmetric
//! 3-tap pre-filter for content that would otherwise alias at coarse levels.
//! It is **off by default**: a stored shape model and the scene it is searched
//! in must share the same kernel (invariant 3).

mod downsample;
mod pyramid;

pub use pyramid::{PreSmooth, Pyramid, PyramidConfig, base_to_level, level_to_base};

/// Benchmark hook for the 2×2 box-mean kernel. **Not public API.**
///
/// The kernel itself is `pub(crate)` — [`Pyramid`] is the entry point. This
/// exists only so `benches/downsample.rs` can measure kernel throughput per
/// pixel type without the level-0 widening pass that a full [`Pyramid::build`]
/// necessarily includes. It carries no stability promise whatsoever.
///
/// # Errors
/// Returns [`Error`](crate::core::Error) unless `dst` is exactly
/// `(src.width() / 2, src.height() / 2)`.
#[doc(hidden)]
pub fn bench_downsample2x2_mean_to_f32_into<P: crate::core::Pixel>(
    src: &crate::core::ImageView<'_, P>,
    dst: &mut crate::core::ImageViewMut<'_, f32>,
) -> Result<(), crate::core::Error> {
    downsample::downsample2x2_mean_to_f32_into(src, dst)
}
