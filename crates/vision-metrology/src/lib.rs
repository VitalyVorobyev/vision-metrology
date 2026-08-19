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
//! | [`laser`]     | Laser stripe extraction using opposite-polarity edge pairs |
//! | [`matching`]  | Shape-based object detection: gradient-orientation model matching |
//! | [`segment`]   | Otsu / adaptive threshold, CCL, watershed, region growing |
//! | [`fit`]       | Robust line / circle / ellipse fitting with reported residuals |
//! | [`measure`]   | Calipers and metrology models — measuring a located part |
//! | [`shape`]     | LSD line-segment detection |
//!
//! ## Features
//!
//! Every domain module is opt-in, so a build compiles only what it uses. All
//! are on by default:
//!
//! ```toml
//! vision-metrology = { version = "0.2", default-features = false,
//!                      features = ["matching", "shape"] }
//! ```
//!
//! | Feature | Module | Implies |
//! |---|---|---|
//! | `contour` | [`contour`] | — |
//! | `fit` | [`fit`] | — |
//! | `measure` | [`measure`] | `fit` (measured points are fitted) |
//! | `laser` | [`laser`] | — |
//! | `matching` | [`matching`] | — |
//! | `segment` | [`segment`] | `contour` (region growing consumes a `ContourGraph`) |
//! | `shape` | [`shape`] | — |
//! | `serde` | `ShapeModel` persistence | `matching` |
//!
//! ## Importing
//!
//! `use vision_metrology::prelude::*;` brings in the working set, including
//! `vm_primitives`' own prelude — one dependency is enough. The full lower
//! crate is reachable as [`vm_primitives`] for anything outside that set.

#[cfg(feature = "contour")]
pub mod contour;
#[cfg(feature = "fit")]
pub mod fit;
#[cfg(feature = "laser")]
pub mod laser;
#[cfg(feature = "matching")]
pub mod matching;
#[cfg(feature = "measure")]
pub mod measure;
#[cfg(feature = "segment")]
pub mod segment;
#[cfg(feature = "shape")]
pub mod shape;

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
    #[cfg(feature = "fit")]
    pub use crate::fit::{
        Fit, FitConfig, RansacConfig, RobustLoss, fit_circle, fit_ellipse, fit_line,
    };
    #[cfg(feature = "laser")]
    pub use crate::laser::{LaserExtractConfig, LaserExtractor, LaserLine};
    #[cfg(feature = "matching")]
    pub use crate::matching::{
        Polarity, ShapeMatch, ShapeMatcher, ShapeModel, ShapeModelBuilder, ShapeModelConfig,
        ShapeSearchConfig,
    };
    #[cfg(feature = "measure")]
    pub use crate::measure::{
        Caliper, EdgeSelect, MeasureConfig, MeasureEdge, MetrologyFit, MetrologyModel,
        MetrologyObject, MetrologyResult, MetrologyShape, PolaritySelect, RejectReason,
    };
    #[cfg(feature = "segment")]
    pub use crate::segment::{
        AdaptiveThreshConfig, CcLabel, ComponentStats, adaptive_threshold_u8,
        label_connected_components_u8, otsu_threshold_u8,
    };
    #[cfg(feature = "shape")]
    pub use crate::shape::{LineSegment2f, LsdConfig, LsdDetector};
}

// The primitives most callers of this crate need by name. Deliberately an
// explicit list rather than a glob: a glob would make every addition to
// `vm-primitives` a potential name collision here, and hide what this crate's
// surface actually is.
pub use vm_primitives::{
    Angle, BorderMode, Circle2f, Conic2f, Edge1DConfig, Edge1DDetector, Edge2DConfig,
    Edge2DDetector, EdgePolarity, Edgel, Ellipse2f, Error, Image, ImageView, ImageViewMut,
    Isometry2f, Line2f, Pixel, Point2f, Polyline2f, PreSmooth, Pyramid, PyramidConfig, Rect2f,
    Similarity2f, SmoothKind, SubpixRefine, Vec2f, Vec2fExt, sample_bilinear_at,
    sample_bilinear_f32, sample_nearest, similarity_from_parts, similarity_parts, transform_point,
    transform_vec, wrap_angle,
};
