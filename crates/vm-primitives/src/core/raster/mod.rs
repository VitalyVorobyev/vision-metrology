//! The raster layer: pixels, buffers, borders, sampling. **No nalgebra.**
//!
//! Everything here describes how image memory is laid out and read, and
//! nothing here knows what a point or a transform is. That is a deliberate
//! boundary rather than an accident of file placement: this workspace is not
//! the only consumer of an `ImageView` — the ecosystem around it already
//! carries five near-duplicates, one of which (`rtvt-image`) exists purely
//! because it is pinned to a different nalgebra major version. A raster layer
//! that mentions no linear-algebra type at all is the piece that could be
//! shared across those version boundaries.
//!
//! The boundary is enforced by reading, not by the compiler: if a signature
//! here grows a `Point2f`, the layer stops being extractable. Coordinates
//! cross it as bare `f32` pairs — see [`sample_bilinear_f32`] — and the
//! `Point2f`-shaped conveniences live one module over, in `geom`.
//!
//! ## Image views and stride
//! Images use element stride (not byte stride). `stride` is the distance, in
//! elements, between adjacent row starts and may be greater than `width`.
//! This allows borrowed views over padded buffers and subviews.
//!
//! ## Border modes
//! Sampling supports clamp, constant fill, and reflect-101 behavior.
//! Reflect-101 mirrors around edge pixels without repeating edge elements.
//!
//! ## Sampling coordinates
//! Sampling uses pixel-center coordinates where integer coordinates refer to
//! pixel centers. Nearest-neighbor uses round-to-nearest integer indices;
//! bilinear uses the standard floor-based 2×2 interpolation neighborhood.

mod border;
mod error;
mod image;
mod pixel;
mod sample;

pub use border::{BorderMode, map_index};
pub use error::Error;
pub use image::{Image, ImageView, ImageViewMut, to_f32, to_f32_u16};
pub use pixel::Pixel;
pub use sample::{sample_bilinear_f32, sample_nearest};
