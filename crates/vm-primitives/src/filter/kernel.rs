//! Symmetric 1-D kernels, stored folded.

/// A symmetric kernel of radius `r`, stored folded: `taps[j]` is the weight at offsets
/// `±j`, for `j = 0..=r`. Normalised in f64 so that `taps[0] + 2·Σ taps[1..] == 1`, then
/// rounded to f32 once.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Kernel1D {
    taps: Vec<f32>,
}

impl Kernel1D {
    /// Gaussian with `radius = ceil(3·sigma)`, the truncation used across the crate.
    ///
    /// # Panics
    /// Panics when `sigma` is not finite or not positive.
    pub(crate) fn gaussian(sigma: f64) -> Self {
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "sigma must be > 0 and finite"
        );
        let r = (3.0 * sigma).ceil() as usize;
        let g: Vec<f64> = (0..=r)
            .map(|j| (-(j as f64).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect();
        Self::from_folded(&g)
    }

    /// Kernel from folded weights `g[0..=r]` (centre first). Any non-zero scale is accepted;
    /// the result is normalised.
    ///
    /// # Panics
    /// Panics when `g` is empty, has a non-finite entry, or sums to zero.
    pub(crate) fn from_folded(g: &[f64]) -> Self {
        assert!(!g.is_empty(), "a kernel needs at least the centre tap");
        assert!(
            g.iter().all(|v| v.is_finite()),
            "kernel taps must be finite"
        );
        let total = g[0] + 2.0 * g[1..].iter().sum::<f64>();
        assert!(total != 0.0, "kernel must not sum to zero");
        Self {
            taps: g.iter().map(|v| (v / total) as f32).collect(),
        }
    }

    /// Half-width `r`; the full kernel has `2r + 1` taps.
    pub(crate) fn radius(&self) -> usize {
        self.taps.len() - 1
    }

    /// Folded taps, centre first, length `r + 1`.
    pub(crate) fn taps(&self) -> &[f32] {
        &self.taps
    }
}

#[cfg(test)]
mod tests {
    use super::Kernel1D;

    #[test]
    fn gaussian_is_normalised_and_symmetric_in_radius() {
        let k = Kernel1D::gaussian(1.5);
        assert_eq!(k.radius(), 5, "ceil(3 * 1.5)");
        let sum: f64 =
            k.taps()[0] as f64 + 2.0 * k.taps()[1..].iter().map(|&v| v as f64).sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-6, "sum {sum}");
    }

    #[test]
    fn gaussian_matches_the_formula() {
        let sigma = 2.0f64;
        let k = Kernel1D::gaussian(sigma);
        let g: Vec<f64> = (0..=k.radius())
            .map(|j| (-(j as f64).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect();
        let total = g[0] + 2.0 * g[1..].iter().sum::<f64>();
        for (j, &t) in k.taps().iter().enumerate() {
            assert!((t as f64 - g[j] / total).abs() < 1e-7, "tap {j}");
        }
    }

    #[test]
    #[should_panic(expected = "sigma must be > 0")]
    fn zero_sigma_is_rejected() {
        let _ = Kernel1D::gaussian(0.0);
    }
}
