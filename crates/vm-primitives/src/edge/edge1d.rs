use crate::core::{BorderMode, Pixel};

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

    /// Detect edges in a 1-D signal of any [`Pixel`] type; owned result.
    ///
    /// Prefer [`detect_in_ref`](Self::detect_in_ref) in a scan loop — it hands
    /// back the internal buffer instead of allocating per line.
    pub fn detect_in<P: Pixel>(&mut self, signal: &[P], cfg: &Edge1DConfig) -> Vec<EdgePeak> {
        self.detect_in_ref(signal, cfg).to_vec()
    }

    /// Detect edges in a 1-D signal of any [`Pixel`] type, borrowing the
    /// internal peak buffer.
    ///
    /// The returned slice is valid until the next `detect_in*` call. An `f32`
    /// signal is convolved in place; `u8`/`u16` are widened into scratch first.
    pub fn detect_in_ref<'a, P: Pixel>(
        &'a mut self,
        signal: &[P],
        cfg: &Edge1DConfig,
    ) -> &'a [EdgePeak] {
        // `f32` needs no widening, and a laser scan calls this once per row.
        if let Some(direct) = P::as_f32_slice(signal) {
            return self.detect_f32_slice(direct, cfg);
        }
        self.tmp.resize(signal.len(), 0.0);
        for (dst, src) in self.tmp.iter_mut().zip(signal) {
            *dst = src.to_f32();
        }
        self.detect_tmp(cfg)
    }

    fn detect_f32_slice<'a>(&'a mut self, signal: &[f32], cfg: &Edge1DConfig) -> &'a [EdgePeak] {
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

        let peaks = det.detect_in(&sig, &cfg);
        let rise = nearest_peak_x(&peaks, EdgePolarity::Rising, x_l);
        let fall = nearest_peak_x(&peaks, EdgePolarity::Falling, x_r);
        // Integer-only extrema are quantized to pixel centers.
        assert!((rise - x_l).abs() <= 0.3);
        assert!((fall - x_r).abs() <= 0.3);

        cfg.refine = SubpixRefine::Parabolic3;
        let peaks_ref = det.detect_in(&sig, &cfg);
        let rise_ref = nearest_peak_x(&peaks_ref, EdgePolarity::Rising, x_l);
        let fall_ref = nearest_peak_x(&peaks_ref, EdgePolarity::Falling, x_r);
        assert!((rise_ref - x_l).abs() <= 0.1);
        assert!((fall_ref - x_r).abs() <= 0.1);
    }

    #[test]
    fn centroid_refinement_locates_the_stripe_edges() {
        // The centroid window integrates the (single-signed) DoG lobe around
        // each edge, so on a clean blurred step it should land close to the
        // true edge -- looser than Parabolic3, but well under a pixel.
        let sigma = 1.2;
        let x_l = 20.3;
        let x_r = 35.7;
        let sig = blur(&stripe_signal(96, x_l, x_r), sigma);

        let mut det = Edge1DDetector::new(sigma);
        let cfg = Edge1DConfig {
            sigma,
            border: BorderMode::Clamp,
            pos_thresh: 0.01,
            neg_thresh: 0.01,
            refine: SubpixRefine::Centroid { radius: 2 },
        };

        let peaks = det.detect_in(&sig, &cfg);
        let rise = nearest_peak_x(&peaks, EdgePolarity::Rising, x_l);
        let fall = nearest_peak_x(&peaks, EdgePolarity::Falling, x_r);
        assert!((rise - x_l).abs() <= 0.25, "rise {rise} vs {x_l}");
        assert!((fall - x_r).abs() <= 0.25, "fall {fall} vs {x_r}");
    }

    #[test]
    fn u8_and_u16_inputs_agree_with_f32() {
        // A uniform intensity scale does not move DoG extrema, so the three
        // typed entry points must report the same subpixel positions on the
        // same underlying stripe.
        let sigma = 1.2;
        let (x_l, x_r) = (20.3, 35.7);
        let sig_f = blur(&stripe_signal(96, x_l, x_r), sigma);
        let sig_u8: Vec<u8> = sig_f.iter().map(|&v| (v * 200.0).round() as u8).collect();
        let sig_u16: Vec<u16> = sig_f
            .iter()
            .map(|&v| (v * 50_000.0).round() as u16)
            .collect();

        let cfg = Edge1DConfig {
            sigma,
            border: BorderMode::Clamp,
            pos_thresh: 0.01,
            neg_thresh: 0.01,
            refine: SubpixRefine::Parabolic3,
        };
        // Scale-invariant thresholds: keep them below the weakest response in
        // every scaling.
        let mut det = Edge1DDetector::new(sigma);
        let f_rise = nearest_peak_x(&det.detect_in(&sig_f, &cfg), EdgePolarity::Rising, x_l);
        let f_fall = nearest_peak_x(&det.detect_in(&sig_f, &cfg), EdgePolarity::Falling, x_r);

        let cfg_u = Edge1DConfig {
            pos_thresh: 1.0,
            neg_thresh: 1.0,
            ..cfg.clone()
        };
        let u8_rise = nearest_peak_x(&det.detect_in(&sig_u8, &cfg_u), EdgePolarity::Rising, x_l);
        let u8_fall = nearest_peak_x(&det.detect_in(&sig_u8, &cfg_u), EdgePolarity::Falling, x_r);
        let u16_rise = nearest_peak_x(&det.detect_in(&sig_u16, &cfg_u), EdgePolarity::Rising, x_l);
        let u16_fall = nearest_peak_x(&det.detect_in(&sig_u16, &cfg_u), EdgePolarity::Falling, x_r);

        // u8 quantization moves the parabola vertex slightly; u16 barely.
        assert!((u8_rise - f_rise).abs() <= 0.05, "{u8_rise} vs {f_rise}");
        assert!((u8_fall - f_fall).abs() <= 0.05, "{u8_fall} vs {f_fall}");
        assert!((u16_rise - f_rise).abs() <= 0.01);
        assert!((u16_fall - f_fall).abs() <= 0.01);
    }

    #[test]
    fn thresholds_reject_weak_peaks() {
        // Two stripes: full-contrast and 10%-contrast. A threshold between
        // their DoG responses must keep the strong pair and drop the weak one.
        let sigma = 1.2;
        let strong = stripe_signal(96, 20.0, 30.0);
        let weak: Vec<f32> = stripe_signal(96, 60.0, 70.0)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        let combined: Vec<f32> = strong.iter().zip(&weak).map(|(a, b)| a + b).collect();
        let sig = blur(&combined, sigma);

        let mut det = Edge1DDetector::new(sigma);
        let permissive = Edge1DConfig {
            sigma,
            border: BorderMode::Clamp,
            pos_thresh: 0.0,
            neg_thresh: 0.0,
            refine: SubpixRefine::Parabolic3,
        };
        let all = det.detect_in(&sig, &permissive);
        let strong_rise = all
            .iter()
            .filter(|p| p.polarity == EdgePolarity::Rising)
            .map(|p| p.strength)
            .fold(0.0f32, f32::max);
        let weak_rise = all
            .iter()
            .filter(|p| p.polarity == EdgePolarity::Rising && (p.x - 60.0).abs() < 3.0)
            .map(|p| p.strength)
            .fold(0.0f32, f32::max);
        assert!(
            weak_rise > 0.0,
            "weak edge must be found without thresholds"
        );
        assert!(weak_rise < strong_rise);

        let thr = 0.5 * (weak_rise + strong_rise);
        let strict = Edge1DConfig {
            pos_thresh: thr,
            neg_thresh: thr,
            ..permissive
        };
        let kept = det.detect_in(&sig, &strict);
        assert!(!kept.is_empty());
        for p in &kept {
            assert!(
                (p.x - 60.0).abs() > 3.0 && (p.x - 70.0).abs() > 3.0,
                "weak-stripe peak at {} survived the threshold",
                p.x
            );
        }
    }

    #[test]
    fn empty_and_short_signals_yield_no_peaks() {
        let mut det = Edge1DDetector::new(1.2);
        let cfg = Edge1DConfig::default();

        assert!(det.detect_in::<f32>(&[], &cfg).is_empty());
        assert!(det.detect_in(&[1.0f32, 2.0], &cfg).is_empty());
        assert!(det.detect_in::<u8>(&[], &cfg).is_empty());
        assert!(det.detect_in(&[10u8, 200], &cfg).is_empty());
        assert!(det.detect_in::<u16>(&[], &cfg).is_empty());
    }

    #[test]
    fn detector_reuse_across_sigmas_is_consistent() {
        // One detector reused with a changed (and then unchanged) sigma must
        // give the same answers as a fresh detector at each sigma.
        let (x_l, x_r) = (20.3, 35.7);
        for &sigma in &[1.0f32, 2.0, 2.0] {
            let sig = blur(&stripe_signal(96, x_l, x_r), sigma);
            let cfg = Edge1DConfig {
                sigma,
                border: BorderMode::Clamp,
                pos_thresh: 0.005,
                neg_thresh: 0.005,
                refine: SubpixRefine::Parabolic3,
            };

            let mut reused = Edge1DDetector::new(0.8);
            reused.detect_in(&[0.0; 16], &Edge1DConfig::default());
            let a = nearest_peak_x(&reused.detect_in(&sig, &cfg), EdgePolarity::Rising, x_l);

            let mut fresh = Edge1DDetector::new(sigma);
            let b = nearest_peak_x(&fresh.detect_in(&sig, &cfg), EdgePolarity::Rising, x_l);
            assert_eq!(a, b, "sigma {sigma}");
        }
    }
}
