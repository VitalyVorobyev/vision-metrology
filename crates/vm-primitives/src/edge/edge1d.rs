use crate::core::BorderMode;

use super::conv1d::convolve_f32;
use super::kernels1d::DoGKernel1D;

/// Subpixel refinement method applied to raw DoG peak positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubpixRefine {
    /// No subpixel refinement; the reported position is the integer peak index.
    None,
    /// Fit a parabola to the three samples around the peak and use its vertex.
    Parabolic3,
    /// Intensity-weighted centroid over `±radius` samples around the peak.
    Centroid {
        /// Half-width of the centroid window in samples.
        radius: usize,
    },
}

/// Configuration for the 1-D DoG edge detector.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge1DConfig {
    /// Standard deviation of the Gaussian smoothing kernel in pixels.
    pub sigma: f32,
    /// Border extension mode applied during convolution. Default: `Clamp`.
    pub border: BorderMode<f32>,
    /// Minimum positive DoG response to report a rising edge peak.
    pub pos_thresh: f32,
    /// Minimum absolute negative DoG response to report a falling edge peak.
    pub neg_thresh: f32,
    /// Subpixel refinement method.
    pub refine: SubpixRefine,
}

impl Default for Edge1DConfig {
    fn default() -> Self {
        Self {
            sigma: 1.2,
            border: BorderMode::Clamp,
            pos_thresh: 0.0,
            neg_thresh: 0.0,
            refine: SubpixRefine::Parabolic3,
        }
    }
}

/// Polarity of a 1-D edge (sign of the first derivative of intensity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgePolarity {
    /// Positive edge: intensity increases (dark-to-bright transition).
    Rising,
    /// Negative edge: intensity decreases (bright-to-dark transition).
    Falling,
}

/// A detected 1-D edge peak with subpixel position and strength.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgePeak {
    /// Subpixel position in pixels along the scanned signal.
    pub x: f32,
    /// Integer sample index of the peak in the DoG response buffer.
    pub idx: usize,
    /// DoG response value at the peak (signed; positive for rising edges).
    pub value: f32,
    /// Absolute DoG response strength (`|value|`).
    pub strength: f32,
    /// Whether this is a rising or falling intensity transition.
    pub polarity: EdgePolarity,
}

/// Reusable 1-D edge detector based on first-derivative-of-Gaussian (DoG) convolution.
///
/// Scratch buffers are owned and reused across `detect_*` calls to avoid per-call allocation.
#[derive(Debug, Clone)]
pub struct Edge1DDetector {
    kernel: DoGKernel1D,
    tmp: Vec<f32>,
    resp: Vec<f32>,
    peaks: Vec<EdgePeak>,
}

impl Edge1DDetector {
    /// Create a new detector with a DoG kernel of the given Gaussian sigma.
    pub fn new(sigma: f32) -> Self {
        Self {
            kernel: DoGKernel1D::new(sigma),
            tmp: Vec::new(),
            resp: Vec::new(),
            peaks: Vec::new(),
        }
    }

    /// Update the Gaussian sigma. Rebuilds the kernel only if `sigma` changed.
    pub fn set_sigma(&mut self, sigma: f32) {
        if (sigma - self.kernel.sigma).abs() > f32::EPSILON {
            self.kernel = DoGKernel1D::new(sigma);
        }
    }

    /// Detect edges in an `f32` signal; returns an owned `Vec<EdgePeak>`.
    pub fn detect_in_f32(&mut self, signal: &[f32], cfg: &Edge1DConfig) -> Vec<EdgePeak> {
        self.detect_in_f32_borrowed(signal, cfg).to_vec()
    }

    /// Detect edges in an `f32` signal; borrows the internal peak buffer.
    ///
    /// The returned slice is valid until the next call to any `detect_*` method.
    pub fn detect_in_f32_ref<'a>(
        &'a mut self,
        signal: &[f32],
        cfg: &Edge1DConfig,
    ) -> &'a [EdgePeak] {
        self.detect_in_f32_borrowed(signal, cfg)
    }

    /// Detect edges in a `u8` signal; returns an owned `Vec<EdgePeak>`.
    pub fn detect_in_u8(&mut self, signal: &[u8], cfg: &Edge1DConfig) -> Vec<EdgePeak> {
        self.detect_in_u8_borrowed(signal, cfg).to_vec()
    }

    /// Detect edges in a `u8` signal; borrows the internal peak buffer.
    ///
    /// The returned slice is valid until the next call to any `detect_*` method.
    pub fn detect_in_u8_ref<'a>(&'a mut self, signal: &[u8], cfg: &Edge1DConfig) -> &'a [EdgePeak] {
        self.detect_in_u8_borrowed(signal, cfg)
    }

    /// Detect edges in a `u16` signal; returns an owned `Vec<EdgePeak>`.
    pub fn detect_in_u16(&mut self, signal: &[u16], cfg: &Edge1DConfig) -> Vec<EdgePeak> {
        self.detect_in_u16_borrowed(signal, cfg).to_vec()
    }

    /// Detect edges in a `u16` signal; borrows the internal peak buffer.
    ///
    /// The returned slice is valid until the next call to any `detect_*` method.
    pub fn detect_in_u16_ref<'a>(
        &'a mut self,
        signal: &[u16],
        cfg: &Edge1DConfig,
    ) -> &'a [EdgePeak] {
        self.detect_in_u16_borrowed(signal, cfg)
    }

    pub(crate) fn detect_in_u8_borrowed<'a>(
        &'a mut self,
        signal: &[u8],
        cfg: &Edge1DConfig,
    ) -> &'a [EdgePeak] {
        self.tmp.resize(signal.len(), 0.0);
        for (dst, &src) in self.tmp.iter_mut().zip(signal.iter()) {
            *dst = src as f32;
        }
        self.detect_tmp(cfg)
    }

    pub(crate) fn detect_in_u16_borrowed<'a>(
        &'a mut self,
        signal: &[u16],
        cfg: &Edge1DConfig,
    ) -> &'a [EdgePeak] {
        self.tmp.resize(signal.len(), 0.0);
        for (dst, &src) in self.tmp.iter_mut().zip(signal.iter()) {
            *dst = src as f32;
        }
        self.detect_tmp(cfg)
    }

    pub(crate) fn detect_in_f32_borrowed<'a>(
        &'a mut self,
        signal: &[f32],
        cfg: &Edge1DConfig,
    ) -> &'a [EdgePeak] {
        self.set_sigma(cfg.sigma);

        self.resp.resize(signal.len(), 0.0);
        if signal.is_empty() {
            self.peaks.clear();
            return &self.peaks;
        }

        convolve_f32(
            signal,
            &self.kernel.dg,
            self.kernel.radius,
            cfg.border.clone(),
            &mut self.resp,
        );

        self.find_local_extrema(cfg)
    }

    fn detect_tmp(&mut self, cfg: &Edge1DConfig) -> &[EdgePeak] {
        self.set_sigma(cfg.sigma);

        self.resp.resize(self.tmp.len(), 0.0);
        if self.tmp.is_empty() {
            self.peaks.clear();
            return &self.peaks;
        }

        convolve_f32(
            &self.tmp,
            &self.kernel.dg,
            self.kernel.radius,
            cfg.border.clone(),
            &mut self.resp,
        );

        self.find_local_extrema(cfg)
    }

    fn find_local_extrema(&mut self, cfg: &Edge1DConfig) -> &[EdgePeak] {
        self.peaks.clear();

        if self.resp.len() < 3 {
            return &self.peaks;
        }

        for i in 1..(self.resp.len() - 1) {
            let a = self.resp[i - 1];
            let b = self.resp[i];
            let c = self.resp[i + 1];

            if b >= a && b > c && b > cfg.pos_thresh {
                let x = refine_x(&self.resp, i, cfg.refine);
                self.peaks.push(EdgePeak {
                    x,
                    idx: i,
                    value: b,
                    strength: b.abs(),
                    polarity: EdgePolarity::Rising,
                });
            }

            if b <= a && b < c && -b > cfg.neg_thresh {
                let x = refine_x(&self.resp, i, cfg.refine);
                self.peaks.push(EdgePeak {
                    x,
                    idx: i,
                    value: b,
                    strength: b.abs(),
                    polarity: EdgePolarity::Falling,
                });
            }
        }

        &self.peaks
    }
}

fn refine_x(resp: &[f32], idx: usize, method: SubpixRefine) -> f32 {
    match method {
        SubpixRefine::None => idx as f32,
        SubpixRefine::Parabolic3 => {
            let ym1 = resp[idx - 1];
            let y0 = resp[idx];
            let yp1 = resp[idx + 1];
            let denom = ym1 - 2.0 * y0 + yp1;
            if denom.abs() < 1e-12 {
                idx as f32
            } else {
                let delta = (0.5 * (ym1 - yp1) / denom).clamp(-1.0, 1.0);
                idx as f32 + delta
            }
        }
        SubpixRefine::Centroid { radius } => {
            let start = idx.saturating_sub(radius);
            let end = (idx + radius).min(resp.len() - 1);
            let mut sum_w = 0.0f32;
            let mut sum_xw = 0.0f32;
            for (j, &rv) in resp.iter().enumerate().take(end + 1).skip(start) {
                let w = rv.abs();
                sum_w += w;
                sum_xw += (j as f32) * w;
            }
            if sum_w <= f32::EPSILON {
                idx as f32
            } else {
                sum_xw / sum_w
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::BorderMode;

    use super::{Edge1DConfig, Edge1DDetector, EdgePolarity, SubpixRefine};
    use crate::DoGKernel1D;
    use crate::edge::conv1d::convolve_f32;

    fn stripe_signal(len: usize, x_l: f32, x_r: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; len];
        for (i, dst) in out.iter_mut().enumerate() {
            let x0 = i as f32 - 0.5;
            let x1 = i as f32 + 0.5;
            let overlap = (x1.min(x_r) - x0.max(x_l)).max(0.0);
            *dst = overlap.clamp(0.0, 1.0);
        }
        out
    }

    fn blur(signal: &[f32], sigma: f32) -> Vec<f32> {
        let k = DoGKernel1D::new(sigma);
        let mut out = vec![0.0f32; signal.len()];
        convolve_f32(signal, &k.g, k.radius, BorderMode::Clamp, &mut out);
        out
    }

    fn nearest_peak_x(peaks: &[crate::EdgePeak], polarity: EdgePolarity, target: f32) -> f32 {
        peaks
            .iter()
            .filter(|p| p.polarity == polarity)
            .min_by(|a, b| {
                (a.x - target)
                    .abs()
                    .partial_cmp(&(b.x - target).abs())
                    .expect("finite compare")
            })
            .expect("peak for polarity should exist")
            .x
    }

    #[test]
    fn detects_stripe_edges_subpixel() {
        let sigma = 1.2;
        let x_l = 20.3;
        let x_r = 35.7;
        let sig = blur(&stripe_signal(96, x_l, x_r), sigma);

        let mut det = Edge1DDetector::new(sigma);
        let mut cfg = Edge1DConfig {
            sigma,
            border: BorderMode::Clamp,
            pos_thresh: 0.01,
            neg_thresh: 0.01,
            refine: SubpixRefine::None,
        };

        let peaks = det.detect_in_f32(&sig, &cfg);
        let rise = nearest_peak_x(&peaks, EdgePolarity::Rising, x_l);
        let fall = nearest_peak_x(&peaks, EdgePolarity::Falling, x_r);
        // Integer-only extrema are quantized to pixel centers.
        assert!((rise - x_l).abs() <= 0.3);
        assert!((fall - x_r).abs() <= 0.3);

        cfg.refine = SubpixRefine::Parabolic3;
        let peaks_ref = det.detect_in_f32(&sig, &cfg);
        let rise_ref = nearest_peak_x(&peaks_ref, EdgePolarity::Rising, x_l);
        let fall_ref = nearest_peak_x(&peaks_ref, EdgePolarity::Falling, x_r);
        assert!((rise_ref - x_l).abs() <= 0.1);
        assert!((fall_ref - x_r).abs() <= 0.1);
    }
}
