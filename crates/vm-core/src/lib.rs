//! Foundational primitives for machine-vision metrology.
//!
//! ## Image Views and Stride
//! Images use element stride (not byte stride). `stride` is the distance, in
//! elements, between adjacent row starts and may be greater than `width`.
//! This allows borrowed views over padded buffers and subviews.
//!
//! ## Border Modes
//! Sampling supports clamp, constant fill, and reflect-101 behavior.
//! Reflect-101 mirrors around edge pixels without repeating edge elements.
//!
//! ## Sampling Coordinates
//! Sampling uses pixel-center coordinates where integer coordinates refer to
//! pixel centers. Nearest-neighbor uses round-to-nearest integer indices;
//! bilinear uses the standard floor-based 2x2 interpolation neighborhood.

mod border;
mod error;
mod geom;
mod image;
mod sample;

pub use border::{BorderMode, map_index};
pub use error::Error;
pub use geom::{Line2f, Point2f, Polyline2f, Vec2f};
pub use image::{Image, ImageView, ImageViewMut, to_f32, to_f32_u16};
pub use sample::{sample_bilinear_f32, sample_nearest};
