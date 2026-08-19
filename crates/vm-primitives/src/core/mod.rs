//! Foundational primitives for machine-vision metrology.
//!
//! Split in two, and the split is load-bearing:
//!
//! | Submodule | Contents | nalgebra |
//! |---|---|---|
//! | `raster` | `Image`/`ImageView`/`ImageViewMut`, `Pixel`, `BorderMode`, sampling, `Error` | **none** |
//! | `geom` | nalgebra aliases, `Vec2fExt`, transforms, `Circle2f`/`Ellipse2f`/`Conic2f` | yes |
//!
//! Both submodules are private: every name keeps one canonical path, `core::…`
//! (invariant 17). The division is a rule about *dependencies*, not about
//! import paths — the raster layer must stay free of linear algebra so it
//! could be shared with crates pinned to a different nalgebra major version.
//! Read the `raster` module's own note (in the source) before adding
//! anything to it.
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
//! bilinear uses the standard floor-based 2x2 interpolation neighborhood.

mod geom;
mod raster;

pub use geom::{
    Affine2f, Angle, Circle2f, Conic2f, Ellipse2f, Isometry2f, Line2f, Point2f, Polyline2f,
    Projective2f, Rect2f, Similarity2f, Vec2f, Vec2fExt, parabolic_peak_offset,
    similarity_from_parts, similarity_parts, transform_point, transform_vec, wrap_angle,
};
pub use raster::{
    BorderMode, Error, Image, ImageView, ImageViewMut, Pixel, map_index, sample_bilinear_f32,
    sample_nearest, to_f32, to_f32_u16,
};

/// Bilinear sample at a [`Point2f`] — the geometry-side convenience over
/// [`sample_bilinear_f32`].
///
/// The raster layer takes a bare `(x, y)` pair on purpose: it must not name a
/// nalgebra type. Callers that already hold a point should not have to
/// destructure it, so the convenience lives here, on the geometry side of the
/// boundary.
#[inline]
pub fn sample_bilinear_at<T: Pixel>(
    img: &ImageView<'_, T>,
    p: Point2f,
    border: BorderMode<f32>,
) -> f32 {
    sample_bilinear_f32(img, p.x, p.y, border)
}

#[cfg(test)]
mod tests {
    use super::{BorderMode, Image, Point2f, sample_bilinear_at, sample_bilinear_f32};

    /// The convenience must be the raster call and nothing else — it exists to
    /// keep a nalgebra type out of the raster signature, not to change results.
    #[test]
    fn sampling_at_a_point_matches_sampling_at_the_pair() {
        let img = Image::from_vec(3, 2, vec![0u8, 10, 20, 30, 40, 50]).expect("valid image");
        let v = img.as_view();
        for (x, y) in [(0.0, 0.0), (1.25, 0.5), (2.0, 1.0), (-3.0, 7.0)] {
            assert_eq!(
                sample_bilinear_at(&v, Point2f::new(x, y), BorderMode::Clamp),
                sample_bilinear_f32(&v, x, y, BorderMode::Clamp),
                "at ({x}, {y})"
            );
        }
    }
}
