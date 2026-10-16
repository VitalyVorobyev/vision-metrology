//! Continuous derivative-of-Gaussian profiles along arbitrary normals.
//!
//! [`ProfileSampler`] evaluates, at any continuous point `p` and direction `n`,
//!
//! ```text
//! value(p)      = Σ_q I(q) w(p − q) / Σ_q w(p − q)
//! derivative(p) = Σ_q (I(q) − value(p)) · ((q − p)·n / σ²) · w(p − q) / C
//! w(d)          = exp(−|d|² / 2σ²) · (1 − |d|² / r²)²,   r = 4σ
//! ```
//!
//! over the pixels `q` with `|p − q| < r`. The taper takes the kernel and its first
//! derivative to zero at `r`, so both outputs are C¹ in `p`: no pixel enters or leaves the
//! support with a jump. `C` is the kernel's continuous second moment along `n`, so the
//! derivative of a unit ramp is 1. Subtracting `value(p)` makes the derivative of a
//! constant image exactly zero.
//!
//! The Gaussian factor is separable: `exp(−dx²/2σ²) · exp(−dy²/2σ²)`, each a 1-D
//! Gaussian at the continuous offsets of one support row or column, built by a
//! two-exponential recurrence. Only the taper is evaluated per pixel, as a polynomial.

use crate::core::{Error, ImageView, Pixel, Point2f, Vec2f};

/// Where a profile runs: a point, and the direction its offsets and derivative follow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileStation {
    /// The point at offset 0, in image coordinates (pixel centres at integers).
    pub p: Point2f,
    /// The direction of the profile. Normalised before use; it must be finite and
    /// non-zero.
    pub n: Vec2f,
}

/// The Gaussian-smoothed intensity at one point and its derivative along the station's
/// direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileSample {
    /// The smoothed intensity, on the input pixel scale.
    pub value: f32,
    /// The derivative along the station's direction, in input units per pixel.
    pub derivative: f32,
}

/// The kernel a [`ProfileSampler`] evaluates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProfileSamplerConfig {
    /// Standard deviation of the Gaussian, in pixels. The support reaches `4σ` from each
    /// sample. Below about 0.7 px the discrete sums start to alias.
    pub sigma: f32,
}

impl Default for ProfileSamplerConfig {
    fn default() -> Self {
        Self { sigma: 1.0 }
    }
}

/// Samples continuous derivative-of-Gaussian profiles along station directions.
///
/// It holds the kernel constants and the 1-D Gaussian scratch, so a reused sampler
/// allocates nothing after its first call at the largest σ it has seen.
///
/// A sample is `None` when its support (every pixel closer than `4σ`) reaches outside the
/// image, touches a pixel the mask excludes, or reads a non-finite pixel. Nothing is
/// extrapolated: border fill is never evidence.
///
/// # Example
/// ```
/// use vm_primitives::edge::{ProfileSampler, ProfileSamplerConfig, ProfileStation};
/// use vm_primitives::{Image, Point2f, Vec2f};
///
/// // A vertical step between columns 15 and 16: 0 on the left, 100 on the right.
/// let data: Vec<f32> = (0..32 * 32).map(|i| if i % 32 >= 16 { 100.0 } else { 0.0 }).collect();
/// let img = Image::from_vec(32, 32, data).unwrap();
/// let station = ProfileStation { p: Point2f::new(15.5, 16.0), n: Vec2f::new(1.0, 0.0) };
/// let offsets = [-1.0, 0.0, 1.0];
/// let mut out = [None; 3];
/// let mut sampler = ProfileSampler::new();
/// sampler
///     .sample(&img.as_view(), None, &[station], &offsets, &ProfileSamplerConfig::default(), &mut out)
///     .unwrap();
/// let mid = out[1].expect("inside the image");
/// assert!((mid.value - 50.0).abs() < 1e-3);
/// // The derivative peaks on the step, symmetric about it.
/// let (left, right) = (out[0].unwrap().derivative, out[2].unwrap().derivative);
/// assert!(mid.derivative > left && (left - right).abs() < 1e-3);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ProfileSampler {
    kernel: Option<Kernel>,
    gx: Vec<f64>,
    gy: Vec<f64>,
    /// `qx − p.x` per support column.
    dx: Vec<f64>,
}

/// Constants of the kernel for one σ.
#[derive(Debug, Clone, Copy)]
struct Kernel {
    sigma_bits: u32,
    /// Support radius `4σ`, its square and inverse square.
    r: f64,
    r2: f64,
    inv_r2: f64,
    /// `1 / 2σ²`.
    inv_2s2: f64,
    /// `exp(−1/σ²)`, the ratio of successive ratios of the 1-D Gaussian on a unit grid.
    decay: f64,
    /// `1 / (σ² C)`: turns `Σ (I − v)(q − p)·n w` into a derivative in units per pixel.
    norm: f64,
}

impl Kernel {
    fn new(sigma: f32) -> Self {
        let s = f64::from(sigma);
        let r = 4.0 * s;
        // C = ∫∫ (u²/σ²) w du dv. In polar coordinates with x = ρ²/2σ² (so ρ²/r² = x/8)
        // it is 2πσ² ∫₀⁸ x e⁻ˣ (1 − x/8)² dx; expanding the square gives incomplete
        // gamma integrals of x, x² and x³.
        let j1 = lower_gamma_int(1, 8.0) - lower_gamma_int(2, 8.0) / 4.0
            + lower_gamma_int(3, 8.0) / 64.0;
        let c = 2.0 * std::f64::consts::PI * s * s * j1;
        Self {
            sigma_bits: sigma.to_bits(),
            r,
            r2: r * r,
            inv_r2: 1.0 / (r * r),
            inv_2s2: 1.0 / (2.0 * s * s),
            decay: (-1.0 / (s * s)).exp(),
            norm: 1.0 / (s * s * c),
        }
    }
}

/// `∫₀ᵃ xᵏ e⁻ˣ dx = k! (1 − e⁻ᵃ Σ_{j ≤ k} aʲ / j!)`.
fn lower_gamma_int(k: u32, a: f64) -> f64 {
    let (mut term, mut sum, mut fact) = (1.0, 1.0, 1.0);
    for j in 1..=k {
        term *= a / f64::from(j);
        sum += term;
        fact *= f64::from(j);
    }
    fact * (1.0 - (-a).exp() * sum)
}

impl ProfileSampler {
    /// A sampler with empty scratch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sample every station at every offset.
    ///
    /// Sample `j` of station `i` sits at `p + offsets[j] · n̂` (`n̂` the normalised
    /// direction) and is written to `out[i · offsets.len() + j]`. `mask`, when given, has
    /// the image's size; a non-zero pixel is valid.
    ///
    /// # Errors
    /// [`Error::InvalidConfig`] when σ is not finite and positive, or a station or offset
    /// is not finite, or a direction is zero; [`Error::SizeMismatch`] when `out` does not
    /// hold `stations.len() · offsets.len()` samples or the mask's size differs from the
    /// image's. `out` is untouched on error.
    pub fn sample<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        mask: Option<&ImageView<'_, u8>>,
        stations: &[ProfileStation],
        offsets: &[f32],
        cfg: &ProfileSamplerConfig,
        out: &mut [Option<ProfileSample>],
    ) -> Result<(), Error> {
        if !(cfg.sigma.is_finite() && cfg.sigma > 0.0) {
            return Err(Error::InvalidConfig(
                "profile sigma must be finite and positive",
            ));
        }
        let expected = stations.len() * offsets.len();
        if out.len() != expected {
            return Err(Error::SizeMismatch {
                expected,
                actual: out.len(),
            });
        }
        if let Some(m) = mask
            && (m.width(), m.height()) != (img.width(), img.height())
        {
            return Err(Error::SizeMismatch {
                expected: img.width() * img.height(),
                actual: m.width() * m.height(),
            });
        }
        if offsets.iter().any(|o| !o.is_finite()) {
            return Err(Error::InvalidConfig("profile offsets must be finite"));
        }
        for s in stations {
            let finite = s.p.x.is_finite() && s.p.y.is_finite();
            let norm = f64::from(s.n.x).hypot(f64::from(s.n.y));
            if !(finite && norm.is_finite() && norm > 0.0) {
                return Err(Error::InvalidConfig(
                    "profile stations need a finite point and a finite, non-zero direction",
                ));
            }
        }
        let k = self.kernel_for(cfg.sigma);
        // The most support columns (rows) a sample can have: the integers strictly inside
        // an open interval of length 2r.
        let span = 2 * (k.r.ceil() as usize) + 1;
        if self.gx.len() < span {
            self.gx.resize(span, 0.0);
            self.gy.resize(span, 0.0);
            self.dx.resize(span, 0.0);
        }
        for (s, row) in stations
            .iter()
            .zip(out.chunks_exact_mut(offsets.len().max(1)))
        {
            let (px, py) = (f64::from(s.p.x), f64::from(s.p.y));
            let norm = f64::from(s.n.x).hypot(f64::from(s.n.y));
            let n = (f64::from(s.n.x) / norm, f64::from(s.n.y) / norm);
            for (o, slot) in offsets.iter().zip(row.iter_mut()) {
                let o = f64::from(*o);
                *slot = self.sample_at(img, mask, &k, (px + o * n.0, py + o * n.1), n);
            }
        }
        Ok(())
    }

    /// The kernel for `sigma`, rebuilt only when σ changed.
    fn kernel_for(&mut self, sigma: f32) -> Kernel {
        match self.kernel {
            Some(k) if k.sigma_bits == sigma.to_bits() => k,
            _ => {
                let k = Kernel::new(sigma);
                self.kernel = Some(k);
                k
            }
        }
    }

    /// One sample at `p`, direction `n` (unit), or `None` when its support is not all
    /// valid image.
    fn sample_at<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        mask: Option<&ImageView<'_, u8>>,
        k: &Kernel,
        p: (f64, f64),
        n: (f64, f64),
    ) -> Option<ProfileSample> {
        // The support's bounding box: integers strictly within r of p on each axis.
        let x0 = (p.0 - k.r).floor() as i64 + 1;
        let x1 = (p.0 + k.r).ceil() as i64 - 1;
        let y0 = (p.1 - k.r).floor() as i64 + 1;
        let y1 = (p.1 + k.r).ceil() as i64 - 1;
        let (nx, ny) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        gaussian_1d(&mut self.gx[..nx], x0 as f64 - p.0, k);
        gaussian_1d(&mut self.gy[..ny], y0 as f64 - p.1, k);
        for (kx, dx) in self.dx[..nx].iter_mut().enumerate() {
            *dx = (x0 + kx as i64) as f64 - p.0;
        }
        let (w, h) = (img.width() as i64, img.height() as i64);

        let (mut sw, mut siw, mut sdw, mut sidw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (ky, &gy) in self.gy[..ny].iter().enumerate() {
            let qy = y0 + ky as i64;
            let dy = qy as f64 - p.1;
            let dy2 = dy * dy;
            // This row's part of the disc, |dx|² < r² − dy², narrowed from the box by the
            // exact test so that validity does not depend on a rounded square root.
            let inside = |dx: f64| dx * dx + dy2 < k.r2;
            let Some(lo) = self.dx[..nx].iter().position(|&dx| inside(dx)) else {
                continue;
            };
            let hi = nx
                - self.dx[..nx]
                    .iter()
                    .rev()
                    .position(|&dx| inside(dx))
                    .unwrap_or(0);
            let (qlo, qhi) = (x0 + lo as i64, x0 + hi as i64 - 1);
            if qy < 0 || qy >= h || qlo < 0 || qhi >= w {
                return None;
            }
            let cols = qlo as usize..qhi as usize + 1;
            if mask.is_some_and(|m| m.row(qy as usize)[cols.clone()].contains(&0)) {
                return None;
            }
            let pixels = &img.row(qy as usize)[cols];
            let dyn_ = dy * n.1;
            let (mut rw, mut riw, mut rdw, mut ridw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for ((v, &gx), &dx) in pixels.iter().zip(&self.gx[lo..hi]).zip(&self.dx[lo..hi]) {
                let t = 1.0 - (dx * dx + dy2) * k.inv_r2;
                let wq = gx * t * t;
                let v = f64::from(v.to_f32());
                let dw = (dx * n.0 + dyn_) * wq;
                rw += wq;
                riw += v * wq;
                rdw += dw;
                ridw += v * dw;
            }
            sw += gy * rw;
            siw += gy * riw;
            sdw += gy * rdw;
            sidw += gy * ridw;
        }
        let value = siw / sw;
        let derivative = (sidw - value * sdw) * k.norm;
        (value.is_finite() && derivative.is_finite()).then_some(ProfileSample {
            value: value as f32,
            derivative: derivative as f32,
        })
    }
}

/// `out[i] = exp(−(d0 + i)² / 2σ²)`, by the recurrence `g[i+1] = g[i]·a[i]`,
/// `a[i+1] = a[i]·exp(−1/σ²)`: two exponentials per row or column.
fn gaussian_1d(out: &mut [f64], d0: f64, k: &Kernel) {
    let mut g = (-d0 * d0 * k.inv_2s2).exp();
    let mut a = (-(2.0 * d0 + 1.0) * k.inv_2s2).exp();
    for slot in out {
        *slot = g;
        g *= a;
        a *= k.decay;
    }
}

#[cfg(test)]
mod tests {
    use super::{Kernel, ProfileSample, ProfileSampler, ProfileSamplerConfig, ProfileStation};
    use crate::core::{Error, Image, Point2f, Vec2f};

    fn station(x: f32, y: f32, nx: f32, ny: f32) -> ProfileStation {
        ProfileStation {
            p: Point2f::new(x, y),
            n: Vec2f::new(nx, ny),
        }
    }

    fn image(w: usize, h: usize, f: impl Fn(f64, f64) -> f64) -> Image<f32> {
        let data = (0..w * h)
            .map(|i| f((i % w) as f64, (i / w) as f64) as f32)
            .collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    /// A direct evaluation of the module formula, with one `exp` per pixel.
    fn reference(img: &Image<f32>, p: (f64, f64), n: (f64, f64), sigma: f64) -> (f64, f64) {
        let r = 4.0 * sigma;
        let k = Kernel::new(sigma as f32);
        let (mut sw, mut siw, mut sdw, mut sidw) = (0.0, 0.0, 0.0, 0.0);
        for qy in 0..img.height() {
            for qx in 0..img.width() {
                let (dx, dy) = (qx as f64 - p.0, qy as f64 - p.1);
                let d2 = dx * dx + dy * dy;
                if d2 >= r * r {
                    continue;
                }
                let w = (-d2 / (2.0 * sigma * sigma)).exp() * (1.0 - d2 / (r * r)).powi(2);
                let v = f64::from(img.as_view().row(qy)[qx]);
                let dn = dx * n.0 + dy * n.1;
                sw += w;
                siw += v * w;
                sdw += dn * w;
                sidw += v * dn * w;
            }
        }
        let value = siw / sw;
        (value, (sidw - value * sdw) * k.norm)
    }

    fn sample_one(
        img: &Image<f32>,
        mask: Option<&Image<u8>>,
        s: ProfileStation,
        sigma: f32,
    ) -> Option<ProfileSample> {
        let mut out = [None];
        ProfileSampler::new()
            .sample(
                &img.as_view(),
                mask.map(|m| m.as_view()).as_ref(),
                &[s],
                &[0.0],
                &ProfileSamplerConfig { sigma },
                &mut out,
            )
            .expect("valid input");
        out[0]
    }

    #[test]
    fn matches_a_direct_evaluation() {
        let img = image(40, 40, |x, y| {
            50.0 + 30.0 * (0.3 * x + 0.17 * y).sin() + x * y * 0.01
        });
        let mut worst = 0.0f64;
        for &sigma in &[0.8f32, 1.0, 1.7, 2.5] {
            for &(x, y) in &[(20.0f64, 20.0f64), (19.37, 21.81), (15.5, 24.25)] {
                for &deg in &[0.0f64, 30.0, 117.0] {
                    let n = (deg.to_radians().cos(), deg.to_radians().sin());
                    let got = sample_one(
                        &img,
                        None,
                        station(x as f32, y as f32, n.0 as f32, n.1 as f32),
                        sigma,
                    )
                    .expect("inside");
                    let (v, d) = reference(
                        &img,
                        (f64::from(x as f32), f64::from(y as f32)),
                        n,
                        f64::from(sigma),
                    );
                    worst = worst
                        .max((f64::from(got.value) - v).abs() / v.abs().max(1.0))
                        .max((f64::from(got.derivative) - d).abs() / d.abs().max(1.0));
                }
            }
        }
        assert!(
            worst < 1e-6,
            "the recurrence must match one exp per pixel, worst relative error {worst:.2e}"
        );
    }

    #[test]
    fn a_ramp_has_unit_derivative_and_its_own_value() {
        // I = 3x − 2y + 10: the derivative along n is 3nx − 2ny, the value I(p).
        let img = image(48, 48, |x, y| 3.0 * x - 2.0 * y + 100.0);
        for &sigma in &[1.0f32, 1.5, 2.0] {
            for &deg in &[0.0f64, 40.0, 200.0] {
                let n = (deg.to_radians().cos(), deg.to_radians().sin());
                let (x, y) = (23.3f32, 24.6f32);
                let got = sample_one(&img, None, station(x, y, n.0 as f32, n.1 as f32), sigma)
                    .expect("inside");
                let want_d = 3.0 * n.0 - 2.0 * n.1;
                let want_v = 3.0 * f64::from(x) - 2.0 * f64::from(y) + 100.0;
                assert!(
                    (f64::from(got.derivative) - want_d).abs() < 1e-4,
                    "σ {sigma}, {deg}°: derivative {} != {want_d}",
                    got.derivative
                );
                assert!(
                    (f64::from(got.value) - want_v).abs() < 1e-3,
                    "σ {sigma}: value {} != {want_v}",
                    got.value
                );
            }
        }
    }

    #[test]
    fn a_constant_has_zero_derivative() {
        let img = image(20, 20, |_, _| 77.0);
        let got = sample_one(&img, None, station(9.31, 10.77, 0.6, 0.8), 0.6).expect("inside");
        assert!((got.value - 77.0).abs() < 1e-5, "got {}", got.value);
        assert!(got.derivative.abs() < 1e-9, "got {}", got.derivative);
    }

    #[test]
    fn the_response_is_continuous_as_pixels_enter_the_support() {
        // A rough texture; p steps across many support boundaries in 1e-3 px increments. A
        // hard cut-off at 4σ jumps by a pixel's weight there (about 2e-3 of the response's
        // scale here), which shows in the second differences; with the C¹ taper they stay
        // at the size of the curvature times the step squared.
        let img = image(40, 40, |x, y| {
            100.0 + 60.0 * ((x * 12.9898 + y * 78.233).sin() * 43758.5453).fract()
        });
        let mut sampler = ProfileSampler::new();
        let s = [station(20.0, 20.0, 1.0, 0.0)];
        let offsets: Vec<f32> = (0..=2000).map(|i| i as f32 * 1e-3).collect();
        let mut out = vec![None; offsets.len()];
        let cfg = ProfileSamplerConfig { sigma: 1.0 };
        sampler
            .sample(&img.as_view(), None, &s, &offsets, &cfg, &mut out)
            .expect("valid");
        let d: Vec<f32> = out.iter().map(|o| o.expect("inside").derivative).collect();
        let worst = d
            .windows(3)
            .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
            .fold(0.0f32, f32::max);
        let scale = d.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            worst < 1e-5 * scale,
            "the response must be C¹: second difference {worst} at scale {scale}"
        );
    }

    #[test]
    fn support_outside_the_image_or_the_mask_is_invalid() {
        let img = image(30, 30, |x, _| x);
        // σ 1: the support holds the pixels closer than 4 px. Column −1 is 3.9 px from
        // x = 2.9 and exactly 4 px from x = 3.0; column 30, past the last one, likewise.
        assert!(sample_one(&img, None, station(2.9, 15.0, 1.0, 0.0), 1.0).is_none());
        assert!(sample_one(&img, None, station(3.0, 15.0, 1.0, 0.0), 1.0).is_some());
        assert!(sample_one(&img, None, station(26.0, 15.0, 1.0, 0.0), 1.0).is_some());
        assert!(sample_one(&img, None, station(26.01, 15.0, 1.0, 0.0), 1.0).is_none());
        // The support is the disc, not its bounding box: from (2.99, 15.5) column −1 is in
        // the box, but its nearest pixels are sqrt(3.99² + 0.5²) ≈ 4.02 px away.
        assert!(sample_one(&img, None, station(2.99, 15.5, 1.0, 0.0), 1.0).is_some());
        assert!(sample_one(&img, None, station(2.99, 15.0, 1.0, 0.0), 1.0).is_none());

        let mut mask = Image::new_fill(30, 30, 255u8);
        mask.as_view_mut().row_mut(15)[20] = 0;
        assert!(sample_one(&img, Some(&mask), station(16.5, 15.0, 1.0, 0.0), 1.0).is_none());
        assert!(sample_one(&img, Some(&mask), station(16.0, 15.0, 1.0, 0.0), 1.0).is_some());

        let mut nan = img.clone();
        nan.as_view_mut().row_mut(15)[15] = f32::NAN;
        assert!(sample_one(&nan, None, station(15.0, 15.0, 1.0, 0.0), 1.0).is_none());
    }

    #[test]
    fn layout_and_errors() {
        let img = image(30, 30, |x, y| x + 2.0 * y);
        let stations = [
            station(10.0, 10.0, 2.0, 0.0),
            station(15.0, 12.0, 0.0, -1.0),
        ];
        let offsets = [-1.0f32, 0.0, 2.5];
        let mut out = vec![None; 6];
        let cfg = ProfileSamplerConfig::default();
        let mut sampler = ProfileSampler::new();
        sampler
            .sample(&img.as_view(), None, &stations, &offsets, &cfg, &mut out)
            .expect("valid");
        // Station 0 runs along +x (normalised), station 1 along −y.
        let values: Vec<f32> = out.iter().map(|o| o.expect("inside").value).collect();
        let want = [29.0, 30.0, 32.5, 41.0, 39.0, 34.0];
        for (g, w) in values.iter().zip(want) {
            assert!((g - w).abs() < 1e-3, "{values:?} != {want:?}");
        }
        assert!((out[0].unwrap().derivative - 1.0).abs() < 1e-4);
        assert!((out[3].unwrap().derivative + 2.0).abs() < 1e-4);

        let mut short = vec![None; 5];
        assert_eq!(
            sampler.sample(&img.as_view(), None, &stations, &offsets, &cfg, &mut short),
            Err(Error::SizeMismatch {
                expected: 6,
                actual: 5
            })
        );
        let bad = [station(10.0, 10.0, 0.0, 0.0)];
        let mut one = vec![None; 3];
        assert!(matches!(
            sampler.sample(&img.as_view(), None, &bad, &offsets, &cfg, &mut one),
            Err(Error::InvalidConfig(_))
        ));
        let sigma = ProfileSamplerConfig { sigma: 0.0 };
        assert!(matches!(
            sampler.sample(&img.as_view(), None, &stations, &offsets, &sigma, &mut out),
            Err(Error::InvalidConfig(_))
        ));
        let mask = Image::new_fill(29, 30, 1u8);
        assert!(matches!(
            sampler.sample(
                &img.as_view(),
                Some(&mask.as_view()),
                &stations,
                &offsets,
                &cfg,
                &mut out
            ),
            Err(Error::SizeMismatch { .. })
        ));
    }
}
