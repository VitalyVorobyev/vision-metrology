/// 1D Gaussian and first-derivative-of-Gaussian kernels.
///
/// Conventions:
/// - `radius = ceil(3*sigma)`, minimum 1, unless built with [`DoGKernel1D::with_radius`].
/// - `g` is normalized such that `sum(g) ~= 1`.
/// - `dg[i] = -(x/sigma^2) * g[i]` (using normalized `g`).
/// - `dg` is not normalized to unit sum; numerically `sum(dg) ~= 0`.
#[derive(Debug, Clone)]
pub struct DoGKernel1D {
    /// Standard deviation of the Gaussian in pixels.
    pub sigma: f32,
    /// Half-width of both kernels: `ceil(3 * sigma)` (minimum 1) or the explicit radius.
    pub radius: usize,
    /// Gaussian kernel coefficients, length `2 * radius + 1`, normalised to sum ≈ 1.
    pub g: Vec<f32>,
    /// First-derivative-of-Gaussian coefficients; `dg[i] = -(x/σ²) * g[i]`.
    pub dg: Vec<f32>,
}

impl DoGKernel1D {
    /// Construct a kernel pair for the given Gaussian sigma.
    ///
    /// # Panics
    /// Panics when `sigma <= 0` or `sigma` is not finite.
    pub fn new(sigma: f32) -> Self {
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "sigma must be > 0 and finite"
        );
        Self::with_radius(sigma, ((3.0 * sigma).ceil() as usize).max(1))
    }

    /// Construct a kernel pair with an explicit half-width `radius` (in samples)
    /// instead of `ceil(3·sigma)`.
    ///
    /// # Panics
    /// Panics when `sigma <= 0`, `sigma` is not finite, or `radius == 0`.
    pub fn with_radius(sigma: f32, radius: usize) -> Self {
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "sigma must be > 0 and finite"
        );
        assert!(radius > 0, "radius must be at least 1");
        let len = 2 * radius + 1;

        let sigma2 = sigma * sigma;
        let mut g = vec![0.0f32; len];
        for (i, gi) in g.iter_mut().enumerate() {
            let x = i as isize - radius as isize;
            let xf = x as f32;
            *gi = (-(xf * xf) / (2.0 * sigma2)).exp();
        }

        let sum_g: f32 = g.iter().sum();
        for gi in &mut g {
            *gi /= sum_g;
        }

        let mut dg = vec![0.0f32; len];
        for (i, dgi) in dg.iter_mut().enumerate() {
            let x = i as isize - radius as isize;
            let xf = x as f32;
            *dgi = -(xf / sigma2) * g[i];
        }

        Self {
            sigma,
            radius,
            g,
            dg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DoGKernel1D;

    #[test]
    fn gaussian_and_derivative_properties() {
        let k = DoGKernel1D::new(1.2);

        let sum_g: f32 = k.g.iter().sum();
        assert!((sum_g - 1.0).abs() < 1e-5);

        let sum_dg: f32 = k.dg.iter().sum();
        assert!(sum_dg.abs() < 1e-6);

        for i in 1..=k.radius {
            let pos = k.radius + i;
            let neg = k.radius - i;
            assert!((k.dg[pos] + k.dg[neg]).abs() < 1e-6);
        }
    }
}
