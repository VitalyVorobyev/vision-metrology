//! Industrial machine-vision metrology library.
//!
//! `vision-metrology` provides a complete pipeline for high-precision image
//! analysis in industrial settings. It builds on [`vm_primitives`], which it
//! re-exports in full, so this is the only dependency you need.
//!
//! ## Modules
//!
//! | Module        | Content |
//! |---------------|---------|
//! | [`contour`]   | Junction-aware contour graph extraction from 2D edgels |
//! | [`corr`]      | Cross-correlation matching (corrmatch) and inter-frame subpixel displacement |
//! | [`fit`]       | Robust line / circle / ellipse fitting with reported residuals |
//! | [`laser`]     | Laser stripe extraction using opposite-polarity edge pairs |
//! | [`lsd`]       | LSD line-segment detection |
//! | [`matching`]  | Shape-based object detection: gradient-orientation model matching |
//! | [`measure`]   | Calipers and metrology models — measuring a located part |
//! | [`metric`]    | The calibration bridge: pixel ↔ millimetre via a mirrored camera model |
//! | [`segment`]   | Otsu / adaptive threshold, CCL, watershed, region growing |
//! | [`warp`]      | Image warping: build a `dst → src` map once, apply per frame |
//!
//! ## Features
//!
//! Every domain module is opt-in, so a build compiles only what it uses. All
//! are on by default:
//!
//! ```toml
//! vision-metrology = { version = "0.2", default-features = false,
//!                      features = ["matching", "lsd"] }
//! ```
//!
//! | Feature | Module | Implies |
//! |---|---|---|
//! | `contour` | [`contour`] | — |
//! | `corr` | [`corr`] | — |
//! | `fit` | [`fit`] | — |
//! | `laser` | [`laser`] | — |
//! | `lsd` | [`lsd`] | — |
//! | `matching` | [`matching`] | `warp` (`ShapeMatch` rectifies into a canonical crop) |
//! | `measure` | [`measure`] | `fit` (measured points are fitted) |
//! | `metric` | [`metric`] | `warp` (`plane_grid_map`/`undistort_map` build a `warp::Map`) |
//! | `segment` | [`segment`] | `contour` (region growing consumes a `ContourGraph`) |
//! | `warp` | [`warp`] | — |
//! | `serde` | `ShapeModel` persistence | `matching` |
//!
//! ## Importing
//!
//! `use vision_metrology::prelude::*;` brings in the working set, including
//! `vm_primitives`' own prelude — one dependency is enough. The full lower
//! crate is reachable as [`vm_primitives`] for anything outside that set.

#[cfg(feature = "contour")]
pub mod contour;
#[cfg(feature = "corr")]
pub mod corr;
#[cfg(feature = "fit")]
pub mod fit;
#[cfg(feature = "laser")]
pub mod laser;
#[cfg(feature = "lsd")]
pub mod lsd;
#[cfg(feature = "matching")]
pub mod matching;
#[cfg(feature = "measure")]
pub mod measure;
#[cfg(feature = "metric")]
pub mod metric;
#[cfg(feature = "segment")]
pub mod segment;
#[cfg(feature = "warp")]
pub mod warp;

/// The lower crate, re-exported whole so one dependency is enough.
///
/// Reach anything not in the curated list below through this: e.g.
/// `vision_metrology::vm_primitives::morph::chamfer_distance_u8`.
pub use vm_primitives;

/// The working set, for `use vision_metrology::prelude::*;`.
///
/// Each group follows its module's feature gate, so the prelude shrinks with
/// the build rather than failing it. Everything here is also reachable at its
/// module path — the prelude is a convenience, not a second API (invariant 17).
pub mod prelude {
    pub use vm_primitives::prelude::*;

    #[cfg(feature = "contour")]
    pub use crate::contour::{
        Connectivity, ContourBuildConfig, ContourGraph, build_graph_from_edgels,
    };
    #[cfg(feature = "corr")]
    pub use crate::corr::{
        CorrConfig, CorrMatch, CorrTemplate, CorrTemplateConfig, Displacement, DisplacementConfig,
        displacement, find, find_topk,
    };
    #[cfg(feature = "fit")]
    pub use crate::fit::{
        Fit, FitConfig, RansacConfig, RobustLoss, fit_circle, fit_ellipse, fit_line,
    };
    #[cfg(feature = "laser")]
    pub use crate::laser::{LaserExtractConfig, LaserExtractor, LaserLine};
    #[cfg(feature = "lsd")]
    pub use crate::lsd::{LineSegment2f, LsdConfig, LsdDetector};
    #[cfg(feature = "matching")]
    pub use crate::matching::{
        CropSpec, Polarity, ShapeMatch, ShapeMatcher, ShapeModel, ShapeModelBuilder,
        ShapeModelConfig, ShapeSearchConfig,
    };
    #[cfg(feature = "measure")]
    pub use crate::measure::{
        Caliper, EdgeSelect, MeasureConfig, MeasureEdge, MetrologyFit, MetrologyModel,
        MetrologyObject, MetrologyResult, MetrologyShape, PolaritySelect, RejectReason,
    };
    #[cfg(feature = "metric")]
    pub use crate::metric::{
        BrownConrady5, CameraModel, PinholeIntrinsics, Plane3, PlaneGrid, Pose3, distort_pixel,
        homography_plane_to_image, pixel_to_plane, pixel_to_ray, plane_grid_map,
        ray_plane_intersect, undistort_map, undistort_pixel,
    };
    #[cfg(feature = "segment")]
    pub use crate::segment::{
        AdaptiveThreshConfig, CcLabel, ComponentStats, adaptive_threshold_u8,
        label_connected_components_u8, otsu_threshold_u8,
    };
    #[cfg(feature = "warp")]
    pub use crate::warp::{Interp, Map};
}

// The primitives most callers of this crate need by name. Deliberately an
// explicit list rather than a glob: a glob would make every addition to
// `vm-primitives` a potential name collision here, and hide what this crate's
// surface actually is.
pub use vm_primitives::{
    Affine2f, Angle, BorderMode, Circle2f, Conic2f, Edge1DConfig, Edge1DDetector, Edge2DConfig,
    Edge2DDetector, EdgePolarity, Edgel, Ellipse2f, Error, Image, ImageView, ImageViewMut,
    Isometry2f, Isometry3f, Line2f, Pixel, Point2f, Point3f, Polyline2f, PreSmooth, Projective2f,
    Pyramid, PyramidConfig, Rect2f, Similarity2f, SmoothKind, SubpixRefine, Vec2f, Vec2fExt, Vec3f,
    sample_bilinear_at, sample_bilinear_f32, sample_nearest, similarity_from_parts,
    similarity_parts, transform_point, transform_vec, wrap_angle,
};
