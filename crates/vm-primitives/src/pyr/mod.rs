//! Fast image pyramid primitives for coarse-to-fine vision processing.
//!
//! `vm-pyr` uses a fixed 2x2 mean downsample (box filter) and aggressively
//! favors throughput.
//!
//! Drop-odd policy:
//! - Output size is `(src.width() / 2, src.height() / 2)`.
//! - If source width or height is odd, the last column/row is dropped.
//!
//! Representational meaning:
//! - Each destination pixel is the arithmetic mean of one 2x2 source block.
//! - Level `L+1` therefore summarizes non-overlapping 2x2 neighborhoods from
//!   level `L`.

mod downsample;
mod pyramid;

pub use downsample::{
    downsample2x2_mean_f32, downsample2x2_mean_u8, downsample2x2_mean_u8_to_f32,
    downsample2x2_mean_u16, downsample2x2_mean_u16_to_f32,
};
pub use pyramid::PyramidF32;
