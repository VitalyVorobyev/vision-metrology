#![doc = include_str!("../README.md")]

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
#[cfg(feature = "scale")]
pub mod scale;
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
/// module path — the prelude is a convenience, not a second API.
// Invariant 17.
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
        BeadConfig, BeadTracker, Caliper, EdgeSelect, MeasureConfig, MeasureEdge, MetrologyFit,
        MetrologyModel, MetrologyObject, MetrologyResult, MetrologyShape, PolaritySelect,
        RejectReason, TrackedBead,
    };
    #[cfg(feature = "metric")]
    pub use crate::metric::{
        BrownConrady5, CameraModel, PinholeIntrinsics, Plane3, PlaneGrid, Pose3, distort_pixel,
        homography_plane_to_image, pixel_to_plane, pixel_to_ray, plane_grid_map,
        ray_plane_intersect, undistort_map, undistort_pixel,
    };
    #[cfg(feature = "scale")]
    pub use crate::scale::{
        ScaleEstimate, estimate_scale_logpolar, estimate_scale_moments, find_scale_invariant,
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
