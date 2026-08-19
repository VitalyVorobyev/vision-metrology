//! Low-level building blocks for vision metrology.
//!
//! `vm-primitives` bundles four foundational layers:
//!
//! | Module  | Content |
//! |---------|---------|
//! | [`core`]  | Image views, sampling, border modes, geometry + nalgebra type aliases |
//! | [`pyr`]   | 2×2 mean image pyramid, generic over pixel type, optional anti-alias pre-smooth |
//! | [`edge`]  | 1-D/2-D subpixel edge (DoG / Scharr) detection, edgels, edge-pairs, dense direction fields |
//! | [`morph`] | Binary morphology, chamfer distance transform, Zhang-Suen thinning |
//!
//! ## Coordinate convention
//! Integer coordinates refer to **pixel centers**: pixel at index `i` is
//! located at position `i` (not `i + 0.5`).
//!
//! ## Quick start
//! ```no_run
//! use vm_primitives::{Image, Edge2DConfig, Edge2DDetector};
//!
//! let img: Image<u8> = Image::from_vec(640, 480, vec![0u8; 640 * 480]).unwrap();
//! let mut det = Edge2DDetector::new();
//! let edgels = det.detect(&img.as_view(), &Edge2DConfig::default());
//! println!("{} edgels found", edgels.len());
//! ```

pub mod core;
pub mod edge;
pub mod morph;
pub mod pyr;

/// The working set, for `use vm_primitives::prelude::*;`.
///
/// Everything here is also reachable by its module path — the prelude is a
/// convenience, not a second API. Types that only a few callers need
/// (`DoGKernel1D`, `StructuringElement`, the morphology functions) are
/// deliberately left out; import those from their module.
pub mod prelude {
    pub use crate::core::{
        BorderMode, Circle2f, Conic2f, Ellipse2f, Error, Image, ImageView, ImageViewMut, Line2f,
        Pixel, Point2f, Rect2f, Similarity2f, Vec2f, Vec2fExt,
    };
    pub use crate::edge::{
        DirectionField, Edge1DConfig, Edge1DDetector, Edge2DConfig, Edge2DDetector, EdgePolarity,
        Edgel, SmoothKind,
    };
    pub use crate::pyr::{PreSmooth, Pyramid, PyramidConfig};
}

// ---------------------------------------------------------------------------
// Flat re-exports — everything available at crate root
// ---------------------------------------------------------------------------

pub use core::{
    Affine2f, Angle, BorderMode, Circle2f, Conic2f, Ellipse2f, Error, Image, ImageView,
    ImageViewMut, Isometry2f, Line2f, Pixel, Point2f, Polyline2f, Projective2f, Rect2f,
    Similarity2f, Vec2f, Vec2fExt, map_index, parabolic_peak_offset, sample_bilinear_f32,
    sample_nearest, similarity_from_parts, similarity_parts, to_f32, to_f32_u16, transform_point,
    transform_vec, wrap_angle,
};
pub use edge::{
    DirectionField, DoGKernel1D, Edge1DConfig, Edge1DDetector, Edge2DConfig, Edge2DDetector,
    EdgePair1D, EdgePairConfig, EdgePeak, EdgePolarity, Edgel, GradientBuffers, SmoothKind,
    Subpix2D, SubpixRefine, best_edge_pair, best_edge_pair_in_row_u8,
};
pub use morph::{
    StructuringElement, chamfer_distance_u8, close_binary_u8, close3x3_binary_u8, dilate_binary_u8,
    dilate3x3_binary_u8, erode_binary_u8, erode3x3_binary_u8, open_binary_u8, open3x3_binary_u8,
    thin_binary_u8,
};
pub use pyr::{PreSmooth, Pyramid, PyramidConfig, base_to_level, level_to_base};
