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
/// module path — the prelude is a convenience, not a second API. Free
/// functions whose names say nothing out of context (`corr::find`,
/// `corr::displacement`) stay at their module path.
// Invariant 17: an explicit list, also for the primitives, so nothing added to
// `vm-primitives` lands here unseen.
pub mod prelude {
    pub use vm_primitives::{
        BorderMode, Circle2f, Conic2f, Edge1DConfig, Edge1DDetector, Edge2DConfig, Edge2DDetector,
        EdgePolarity, Edgel, Ellipse2f, Error, Image, ImageView, ImageViewMut, Line2f, Pixel,
        Point2f, PreSmooth, Pyramid, PyramidConfig, Rect2f, Similarity2f, SmoothKind, Vec2f,
        Vec2fExt,
    };

    #[cfg(feature = "contour")]
    pub use crate::contour::{
        Connectivity, ContourBuildConfig, ContourGraph, build_graph_from_edgels,
    };
    #[cfg(feature = "corr")]
    pub use crate::corr::{
        CorrConfig, CorrMatch, CorrTemplate, CorrTemplateConfig, Displacement, DisplacementConfig,
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
        BeadConfig, BeadTracker, Caliper, EdgeSelect, EdgeSequence, MeasureConfig, MeasureEdge,
        MeasureRect, MeasureStrip, MetrologyFit, MetrologyModel, MetrologyObject, MetrologyResult,
        MetrologyShape, PolaritySelect, RejectReason, TrackedBead,
    };
    #[cfg(feature = "metric")]
    pub use crate::metric::{
        BrownConrady5, CameraModel, PinholeIntrinsics, Plane3, PlaneGrid, Pose3, distort_pixel,
        homography_plane_to_image, pixel_to_plane, pixel_to_ray, plane_grid_map,
        ray_plane_intersect, undistort_map, undistort_pixel,
    };
    #[cfg(feature = "scale")]
    pub use crate::scale::{
        ScaleEstimate, ScaleHint, ScaleInvariantConfig, estimate_scale_logpolar,
        estimate_scale_moments, find_scale_invariant,
    };
    #[cfg(feature = "segment")]
    pub use crate::segment::{
        AdaptiveThreshConfig, CcLabel, ComponentStats, adaptive_threshold_u8, component_stats,
        label_connected_components_u8, otsu_threshold_u8,
    };
    #[cfg(feature = "warp")]
    pub use crate::warp::{Interp, Map};
}

// The primitive *types* most callers of this crate need by name (invariant 17).
// Deliberately an explicit list rather than a glob: a glob would make every
// addition to `vm-primitives` a potential name collision here, and hide what
// this crate's surface actually is. Free functions (sampling, angle and
// similarity helpers) stay at their `vm_primitives` path.
pub use vm_primitives::{
    Affine2f, BorderMode, Circle2f, Conic2f, Edge1DConfig, Edge1DDetector, Edge2DConfig,
    Edge2DDetector, EdgePolarity, Edgel, Ellipse2f, Error, Image, ImageView, ImageViewMut,
    Isometry2f, Isometry3f, Line2f, Pixel, Point2f, Point3f, Polyline2f, PreSmooth, Projective2f,
    Pyramid, PyramidConfig, Rect2f, Similarity2f, SmoothKind, SubpixRefine, Vec2f, Vec2fExt, Vec3f,
};
