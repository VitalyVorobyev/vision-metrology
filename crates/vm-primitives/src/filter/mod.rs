//! Linear filters (ADR-0023): exact box and binomial smoothing on `u8`, Gaussian on `f32`.
//!
//! - [`box_mean_u8`]: a `(2r+1)²` box mean, rounded half up, exact in integers.
//! - [`binomial_u8`]: binomial smoothing of even order, exact in integers.
//! - [`gaussian_f32`]: Gaussian by σ. The exact truncated FIR is used below σ = 8, and a
//!   three-box cascade above it; its error is stated in ADR-0023.
//!
//! Every function takes a [`BorderMode`] for out-of-image samples and returns a new image.
//! Scratch buffers are internal for now; a reusable-scratch API comes with the hot-path work.

mod cascade;
mod fir;
mod gaussian;
mod integer;
mod kernel;

use crate::core::{BorderMode, Error, Image, ImageView};

use gaussian::{GaussianScratch, gaussian_into};
use integer::{IntScratch, binomial_into, box_mean_into};

// Crate-internal building blocks for other modules (the pyramid pre-smooth uses the FIR).
pub(crate) use fir::{FirScratch, separable_fir_f32};
pub(crate) use kernel::Kernel1D;

/// Copies a strided view into one contiguous row-major buffer.
fn gather<T: Copy>(src: &ImageView<'_, T>) -> Vec<T> {
    let mut out = Vec::with_capacity(src.width() * src.height());
    for y in 0..src.height() {
        out.extend_from_slice(src.row(y));
    }
    out
}

/// Box mean of width `2·radius + 1` on `u8`, rounded half up. Each output is
/// `round(sum / (2r+1)²)` over the window, computed exactly in integers.
///
/// Out-of-image samples are resolved by `border`. A `Constant` border supplies its value.
///
/// # Errors
/// Returns [`Error::InvalidConfig`] when `(2·radius + 1)² ≥ 2²²`, the largest window for which
/// the rounding divide is exact.
pub fn box_mean_u8(
    src: &ImageView<'_, u8>,
    radius: usize,
    border: BorderMode<u8>,
) -> Result<Image<u8>, Error> {
    let win = 2 * radius + 1;
    if win * win >= 1 << 22 {
        return Err(Error::InvalidConfig(
            "box window too large for exact rounding",
        ));
    }
    let (w, h) = (src.width(), src.height());
    let data = gather(src);
    let mut dst = vec![0u8; w * h];
    box_mean_into(
        &data,
        w,
        h,
        radius,
        &border,
        &mut IntScratch::default(),
        &mut dst,
    );
    Image::from_vec(w, h, dst)
}

/// Binomial smoothing on `u8`: the separable kernel `C(order, k) / 2^order` in each
/// direction, rounded half up. Exact in integers. The kernel has radius `order / 2`.
///
/// # Errors
/// Returns [`Error::InvalidConfig`] when `order` is odd or outside `2..=12`.
pub fn binomial_u8(
    src: &ImageView<'_, u8>,
    order: usize,
    border: BorderMode<u8>,
) -> Result<Image<u8>, Error> {
    if !(order.is_multiple_of(2) && (2..=12).contains(&order)) {
        return Err(Error::InvalidConfig(
            "binomial order must be even and in 2..=12",
        ));
    }
    let (w, h) = (src.width(), src.height());
    let data = gather(src);
    let mut dst = vec![0u8; w * h];
    binomial_into(
        &data,
        w,
        h,
        order,
        &border,
        &mut IntScratch::default(),
        &mut dst,
    );
    Image::from_vec(w, h, dst)
}

/// Gaussian of standard deviation `sigma` (in pixels) on `f32`. Below σ = 8 this is the exact
/// FIR truncated at `ceil(3σ)`; from σ = 8 on it is a three-box cascade.
///
/// # Errors
/// Returns [`Error::InvalidConfig`] when `sigma` is not finite and positive.
pub fn gaussian_f32(
    src: &ImageView<'_, f32>,
    sigma: f64,
    border: BorderMode<f32>,
) -> Result<Image<f32>, Error> {
    if !(sigma.is_finite() && sigma > 0.0) {
        return Err(Error::InvalidConfig("sigma must be finite and > 0"));
    }
    let (w, h) = (src.width(), src.height());
    let data = gather(src);
    let mut dst = vec![0.0f32; w * h];
    gaussian_into(
        &data,
        w,
        h,
        sigma,
        &border,
        &mut GaussianScratch::default(),
        &mut dst,
    );
    Image::from_vec(w, h, dst)
}

#[cfg(test)]
mod api_tests {
    use super::*;

    fn img(w: usize, h: usize) -> Image<u8> {
        Image::from_vec(w, h, (0..w * h).map(|i| ((i * 37) % 256) as u8).collect()).unwrap()
    }

    #[test]
    fn public_box_matches_the_internal_kernel() {
        let src = img(20, 11);
        let out = box_mean_u8(&src.as_view(), 2, BorderMode::Clamp).unwrap();
        let mut want = vec![0u8; 20 * 11];
        box_mean_into(
            src.data(),
            20,
            11,
            2,
            &BorderMode::Clamp,
            &mut IntScratch::default(),
            &mut want,
        );
        assert_eq!(out.data(), &want[..]);
    }

    #[test]
    fn public_functions_reject_bad_parameters() {
        let src = img(8, 8);
        assert!(binomial_u8(&src.as_view(), 3, BorderMode::Clamp).is_err());
        assert!(binomial_u8(&src.as_view(), 14, BorderMode::Clamp).is_err());
        assert!(box_mean_u8(&src.as_view(), 1 << 11, BorderMode::Clamp).is_err());
        let f = Image::from_vec(4, 4, vec![1.0f32; 16]).unwrap();
        assert!(gaussian_f32(&f.as_view(), 0.0, BorderMode::Clamp).is_err());
        assert!(gaussian_f32(&f.as_view(), f64::NAN, BorderMode::Clamp).is_err());
    }

    #[test]
    fn public_functions_accept_a_strided_view() {
        // A 6x5 view with stride 9 inside a larger buffer.
        let data: Vec<u8> = (0..9 * 5).map(|i| (i % 251) as u8).collect();
        let view = ImageView::from_slice(6, 5, 9, &data).unwrap();
        let out = binomial_u8(&view, 2, BorderMode::Clamp).unwrap();
        assert_eq!((out.width(), out.height()), (6, 5));
        let contiguous: Vec<u8> = (0..5)
            .flat_map(|y| data[y * 9..y * 9 + 6].to_vec())
            .collect();
        let copy = Image::from_vec(6, 5, contiguous).unwrap();
        assert_eq!(
            out.data(),
            binomial_u8(&copy.as_view(), 2, BorderMode::Clamp)
                .unwrap()
                .data()
        );
    }
}
