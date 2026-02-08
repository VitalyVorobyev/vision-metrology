/// 1D Gaussian and first-derivative-of-Gaussian kernels.
///
/// Conventions:
/// - `radius = ceil(3*sigma)`, minimum 1.
/// - `g` is normalized such that `sum(g) ~= 1`.
/// - `dg[i] = -(x/sigma^2) * g[i]` (using normalized `g`).
/// - `dg` is not normalized to unit sum; numerically `sum(dg) ~= 0`.
#[derive(Debug, Clone)]
pub struct DoGKernel1D {
    pub sigma: f32,
    pub radius: usize,
    pub g: Vec<f32>,
    pub dg: Vec<f32>,
}

impl DoGKernel1D {
    pub fn new(sigma: f32) -> Self {
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "sigma must be > 0 and finite"
        );

        let radius = ((3.0 * sigma).ceil() as usize).max(1);
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
