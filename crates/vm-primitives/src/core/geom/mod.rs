//! The geometry layer: nalgebra aliases, transforms, and the shapes the
//! fitters produce.
//!
//! Everything in this crate that names a `Point2f`, a `Similarity2f`, or any
//! other nalgebra type lives here or above it. Keeping it out of
//! [`raster`](super::raster) is what makes the raster layer shareable across
//! nalgebra major versions — see that module's own note.
//!
//! Points and vectors are nalgebra *aliases*, not wrappers (invariant 13), so
//! they cross into `calibration-rs`, `corrmatch` and `chess-corners-rs`
//! untouched. The one behaviour that is not nalgebra's is
//! [`Vec2fExt::normalized_or_zero`], which restores zero-on-degenerate where
//! nalgebra's `normalize` yields `NaN`.

mod shapes;
mod transform;

pub use shapes::{Circle2f, Conic2f, Ellipse2f};
pub use transform::{
    Affine2f, Angle, Isometry2f, Isometry3f, Line2f, Point2f, Point3f, Polyline2f, Projective2f,
    Rect2f, Similarity2f, Vec2f, Vec2fExt, Vec3f, parabolic_peak_offset, similarity_from_parts,
    similarity_parts, transform_point, transform_vec, wrap_angle,
};
