//! Gaussian by σ with the method chosen by measurement (ADR-0023, Phase 3).
//!
//! Below [`CROSSOVER_SIGMA`] the exact folded FIR is used, so the result is the truncated
//! Gaussian to `f32` rounding. From the crossover on, the three-box cascade is used: it costs
//! the same for every σ, and its error against the untruncated Gaussian is stated in ADR-0023.

use crate::core::BorderMode;

use super::cascade::{CascadeScratch, gaussian_cascade_f32};
use super::fir::{FirScratch, separable_fir_f32};
use super::kernel::Kernel1D;

/// σ from which the cascade is faster than the exact FIR. Measured on the Apple M4 Pro,
/// 1280×1024 `f32`, best of 25 (ADR-0023): the FIR is 4.9 ms at σ = 8 and the cascade 4.7 ms.
pub(crate) const CROSSOVER_SIGMA: f64 = 8.0;

/// Reusable scratch for both methods.
#[derive(Default, Debug)]
pub(crate) struct GaussianScratch {
    fir: FirScratch,
    cascade: CascadeScratch,
}

/// Gaussian of standard deviation `sigma` on an `f32` image, `Auto` method.
///
/// # Panics
/// Panics when the slices do not match `w · h`, or `sigma` is not finite and positive.
pub(crate) fn gaussian_into(
    src: &[f32],
    w: usize,
    h: usize,
    sigma: f64,
    border: &BorderMode<f32>,
    scratch: &mut GaussianScratch,
    dst: &mut [f32],
) {
    if sigma < CROSSOVER_SIGMA {
        let kernel = Kernel1D::gaussian(sigma);
        separable_fir_f32(src, w, h, &kernel, *border, &mut scratch.fir, dst);
    } else {
        gaussian_cascade_f32(src, w, h, sigma, border, &mut scratch.cascade, dst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_uses_the_fir_below_the_crossover_and_the_cascade_above() {
        let (w, h) = (40usize, 30usize);
        let src: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 256) as f32).collect();
        let mut scratch = GaussianScratch::default();
        for sigma in [1.0f64, CROSSOVER_SIGMA + 2.0] {
            let mut auto = vec![0.0f32; w * h];
            gaussian_into(
                &src,
                w,
                h,
                sigma,
                &BorderMode::Clamp,
                &mut scratch,
                &mut auto,
            );
            let mut want = vec![0.0f32; w * h];
            if sigma < CROSSOVER_SIGMA {
                separable_fir_f32(
                    &src,
                    w,
                    h,
                    &Kernel1D::gaussian(sigma),
                    BorderMode::Clamp,
                    &mut FirScratch::default(),
                    &mut want,
                );
            } else {
                gaussian_cascade_f32(
                    &src,
                    w,
                    h,
                    sigma,
                    &BorderMode::Clamp,
                    &mut CascadeScratch::default(),
                    &mut want,
                );
            }
            assert_eq!(auto, want, "sigma {sigma}");
        }
    }
}
