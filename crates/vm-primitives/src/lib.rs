//! Low-level building blocks for vision metrology.
//!
//! `vm-primitives` bundles four foundational layers:
//!
//! | Module  | Content |
//! |---------|---------|
//! | [`core`]  | Image views, sampling, border modes, geometry + nalgebra type aliases |
//! | [`pyr`]   | Ultra-fast 2×2 mean image pyramid |
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
//! let edgels = det.detect_u8(&img.as_view(), &Edge2DConfig::default());
//! println!("{} edgels found", edgels.len());
//! ```

pub mod core;
pub mod edge;
pub mod morph;
pub mod pyr;

// ---------------------------------------------------------------------------
// Flat re-exports — everything available at crate root
// ---------------------------------------------------------------------------

pub use core::{
    Affine2f, Angle, BorderMode, Error, Image, ImageView, ImageViewMut, Isometry2f, Line2f,
    Point2f, Polyline2f, Projective2f, Rect2f, Similarity2f, Vec2f, from_na_point, from_na_vec,
    map_index, parabolic_peak_offset, sample_bilinear_f32, sample_nearest, similarity_from_parts,
    similarity_parts, to_f32, to_f32_u16, to_na_point, to_na_vec, transform_point,
    transform_point_iso, transform_vec, wrap_angle,
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
pub use pyr::{
    PyramidF32, downsample2x2_mean_f32, downsample2x2_mean_u8, downsample2x2_mean_u8_to_f32,
    downsample2x2_mean_u16, downsample2x2_mean_u16_to_f32,
};
