#![doc = include_str!("../README.md")]

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
    ImageViewMut, Isometry2f, Isometry3f, Line2f, Pixel, Point2f, Point3f, Polyline2f,
    Projective2f, Rect2f, Similarity2f, Vec2f, Vec2fExt, Vec3f, map_index, parabolic_peak_offset,
    sample_bilinear_at, sample_bilinear_f32, sample_nearest, similarity_from_parts,
    similarity_parts, to_f32, to_f32_u16, transform_point, transform_vec, wrap_angle,
};
pub use edge::{
    DirectionField, DoGKernel1D, Edge1DConfig, Edge1DDetector, Edge2DConfig, Edge2DDetector,
    EdgePair1D, EdgePairConfig, EdgePeak, EdgePolarity, Edgel, GradientBuffers, Hysteresis,
    SmoothKind, Subpix2D, SubpixRefine, TiledField, best_edge_pair, best_edge_pair_in_row_u8,
};
pub use morph::{
    StructuringElement, chamfer_distance_u8, close_binary_u8, close3x3_binary_u8, dilate_binary_u8,
    dilate3x3_binary_u8, erode_binary_u8, erode3x3_binary_u8, open_binary_u8, open3x3_binary_u8,
    thin_binary_u8,
};
pub use pyr::{PreSmooth, Pyramid, PyramidConfig, base_to_level, level_to_base};
