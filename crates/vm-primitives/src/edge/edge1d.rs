use std::num::NonZeroUsize;

use crate::core::{BorderMode, Pixel};

use super::conv1d::convolve_f32;
use super::kernels1d::DoGKernel1D;

/// Subpixel refinement method applied to raw derivative peak positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubpixRefine {
    /// No subpixel refinement; the reported position is the integer peak index.
    None,
    /// Fit a parabola to the three samples around the peak and use its vertex.
    ///
    /// At a strict local maximum the vertex always lies within ±0.5 samples.
    Parabolic3,
    /// Fit a parabola to the logarithm of the three samples around the peak (a
    /// Gaussian fit), which is exact for a Gaussian-shaped derivative peak.
    ///
    /// Falls back to [`Parabolic3`](Self::Parabolic3) when a neighbour of the peak
    /// is not strictly positive in the peak's own polarity.
    Gaussian3,
    /// Intensity-weighted centroid over `±radius` samples around the peak.
    Centroid {
        /// Half-width of the centroid window in samples.
        radius: usize,
    },
}

/// How the 1-D derivative is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Derivative1D {
    /// Convolve with the analytic first derivative of a Gaussian, radius `ceil(3σ)`.
    #[default]
    DerivativeOfGaussian,
    /// Smooth with a normalised Gaussian of the given half-width (in samples), then
    /// take central differences `½(s[i+1] − s[i−1])`, one-sided at the two ends.
    ///
    /// This is the textbook "Gaussian, then finite differences" operator (numpy's
    /// `np.gradient` of a smoothed profile).
    SmoothThenCentral {
        /// Half-width of the smoothing kernel, in samples.
        radius: NonZeroUsize,
    },
}

/// Configuration for the 1-D edge detector.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge1DConfig {
    /// Standard deviation of the Gaussian smoothing kernel in samples.
    pub sigma: f32,
    /// How the derivative is computed.
    pub derivative: Derivative1D,
    /// Border extension mode applied during convolution. Default: `Clamp`.
    pub border: BorderMode<f32>,
    /// Minimum positive derivative response to report a rising edge peak.
    pub pos_thresh: f32,
    /// Minimum absolute negative derivative response to report a falling edge peak.
    pub neg_thresh: f32,
    /// Subpixel refinement method.
    pub refine: SubpixRefine,
}

impl Default for Edge1DConfig {
    fn default() -> Self {
        Self {
            sigma: 1.2,
            derivative: Derivative1D::default(),
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
    /// Subpixel position in samples along the scanned signal.
    pub x: f32,
    /// Integer sample index of the peak in the derivative response buffer.
    pub idx: usize,
    /// Derivative response at the peak (signed; positive for rising edges).
    pub value: f32,
    /// Absolute derivative response strength (`|value|`).
    pub strength: f32,
    /// Whether this is a rising or falling intensity transition.
    pub polarity: EdgePolarity,
}

/// Reusable 1-D edge detector: smooth and differentiate, then find extrema.
///
/// Scratch buffers are owned and reused across `detect_*` calls to avoid per-call allocation.
#[derive(Debug, Clone)]
pub struct Edge1DDetector {
    kernel: DoGKernel1D,
    tmp: Vec<f32>,
    smooth: Vec<f32>,
    resp: Vec<f32>,
    peaks: Vec<EdgePeak>,
}

impl Edge1DDetector {
    /// Create a new detector with a kernel of the given Gaussian sigma (in samples).
    pub fn new(sigma: f32) -> Self {
        Self {
            kernel: DoGKernel1D::new(sigma),
            tmp: Vec::new(),
            smooth: Vec::new(),
            resp: Vec::new(),
            peaks: Vec::new(),
        }
    }

    /// Update the Gaussian sigma. Rebuilds the kernel only if `sigma` changed.
    pub fn set_sigma(&mut self, sigma: f32) {
        self.ensure_kernel(sigma, None);
    }

    /// Build the kernel for `sigma` and, if given, an explicit radius; reuse the
    /// cached one when both already match.
    fn ensure_kernel(&mut self, sigma: f32, radius: Option<usize>) {
        let default_radius = || ((3.0 * sigma).ceil() as usize).max(1);
        let want_radius = radius.unwrap_or_else(default_radius);
        if (sigma - self.kernel.sigma).abs() > f32::EPSILON || want_radius != self.kernel.radius {
            self.kernel = match radius {
                Some(r) => DoGKernel1D::with_radius(sigma, r),
                None => DoGKernel1D::new(sigma),
            };
        }
    }

    /// The derivative response from the last `detect_*` call, one value per sample.
    pub fn response(&self) -> &[f32] {
        &self.resp
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
    /// signal is read in place; `u8`/`u16` are widened into scratch first.
    pub fn detect_in_ref<'a, P: Pixel>(
        &'a mut self,
        signal: &[P],
        cfg: &Edge1DConfig,
    ) -> &'a [EdgePeak] {
        // `f32` needs no widening, and a laser scan calls this once per row.
        if let Some(direct) = P::as_f32_slice(signal) {
            self.respond(direct, cfg);
        } else {
            let mut tmp = std::mem::take(&mut self.tmp);
            tmp.clear();
            tmp.extend(signal.iter().map(|v| v.to_f32()));
            self.respond(&tmp, cfg);
            self.tmp = tmp;
        }
        self.find_local_extrema(cfg)
    }

    /// Fill `resp` with the derivative of `signal` under `cfg`.
    fn respond(&mut self, signal: &[f32], cfg: &Edge1DConfig) {
        self.resp.clear();
        self.resp.resize(signal.len(), 0.0);
        if signal.is_empty() {
            return;
        }
        match cfg.derivative {
            Derivative1D::DerivativeOfGaussian => {
                self.ensure_kernel(cfg.sigma, None);
                convolve_f32(
                    signal,
                    &self.kernel.dg,
                    self.kernel.radius,
                    cfg.border,
                    &mut self.resp,
                );
            }
            Derivative1D::SmoothThenCentral { radius } => {
                self.ensure_kernel(cfg.sigma, Some(radius.get()));
                self.smooth.clear();
                self.smooth.resize(signal.len(), 0.0);
                convolve_f32(
                    signal,
                    &self.kernel.g,
                    self.kernel.radius,
                    cfg.border,
                    &mut self.smooth,
                );
                central_difference(&self.smooth, &mut self.resp);
            }
        }
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
                let x = refine_x(&self.resp, i, 1.0, cfg.refine);
                self.peaks.push(EdgePeak {
                    x,
                    idx: i,
                    value: b,
                    strength: b.abs(),
                    polarity: EdgePolarity::Rising,
                });
            }

            if b <= a && b < c && -b > cfg.neg_thresh {
                let x = refine_x(&self.resp, i, -1.0, cfg.refine);
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

/// `out[i] = ½(s[i+1] − s[i−1])` inside, `s[1] − s[0]` and `s[n−1] − s[n−2]` at the
/// ends (zero for a single sample).
fn central_difference(s: &[f32], out: &mut [f32]) {
    let n = s.len();
    match n {
        0 => {}
        1 => out[0] = 0.0,
        _ => {
            out[0] = s[1] - s[0];
            out[n - 1] = s[n - 1] - s[n - 2];
            for i in 1..n - 1 {
                out[i] = 0.5 * (s[i + 1] - s[i - 1]);
            }
        }
    }
}

/// Vertex offset of the parabola through `(−1, a)`, `(0, b)`, `(1, c)`, clamped to
/// ±0.5 (which a strict local maximum never exceeds).
fn parabola_offset(a: f32, b: f32, c: f32) -> f32 {
    let denom = a - 2.0 * b + c;
    if denom.abs() < 1e-12 {
        0.0
    } else {
        (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
    }
}

/// Subpixel position of the peak at `idx`; `sign` is +1 for a rising (maximum) and
/// −1 for a falling (minimum) peak.
fn refine_x(resp: &[f32], idx: usize, sign: f32, method: SubpixRefine) -> f32 {
    match method {
        SubpixRefine::None => idx as f32,
        SubpixRefine::Parabolic3 => {
            idx as f32 + parabola_offset(resp[idx - 1], resp[idx], resp[idx + 1])
        }
        SubpixRefine::Gaussian3 => {
            let (a, b, c) = (sign * resp[idx - 1], sign * resp[idx], sign * resp[idx + 1]);
            if a > 0.0 && b > 0.0 && c > 0.0 {
                idx as f32 + parabola_offset(a.ln(), b.ln(), c.ln())
            } else {
                idx as f32 + parabola_offset(resp[idx - 1], resp[idx], resp[idx + 1])
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

    use std::num::NonZeroUsize;

    use super::{
        Derivative1D, Edge1DConfig, Edge1DDetector, EdgePolarity, SubpixRefine, central_difference,
        parabola_offset, refine_x,
    };
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
            ..Edge1DConfig::default()
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
            ..Edge1DConfig::default()
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
            ..Edge1DConfig::default()
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
            ..Edge1DConfig::default()
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
                ..Edge1DConfig::default()
            };

            let mut reused = Edge1DDetector::new(0.8);
            reused.detect_in(&[0.0; 16], &Edge1DConfig::default());
            let a = nearest_peak_x(&reused.detect_in(&sig, &cfg), EdgePolarity::Rising, x_l);

            let mut fresh = Edge1DDetector::new(sigma);
            let b = nearest_peak_x(&fresh.detect_in(&sig, &cfg), EdgePolarity::Rising, x_l);
            assert_eq!(a, b, "sigma {sigma}");
        }
    }

    /// numpy's `np.gradient`: central inside, one-sided first-order at the ends.
    #[test]
    fn central_difference_matches_np_gradient_on_a_ramp() {
        let s: Vec<f32> = (0..6).map(|i| 2.0 * i as f32 + 1.0).collect();
        let mut out = vec![0.0f32; 6];
        central_difference(&s, &mut out);
        assert_eq!(
            out,
            vec![2.0; 6],
            "a ramp of slope 2 everywhere, ends included"
        );

        let s = [0.0f32, 1.0, 4.0, 9.0];
        let mut out = vec![0.0f32; 4];
        central_difference(&s, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 4.0, 5.0], "np.gradient([0,1,4,9])");

        let mut one = vec![7.0f32];
        central_difference(&[3.0], &mut one);
        assert_eq!(one, vec![0.0]);
    }

    /// Gaussian (radius 1, σ = 1, edge padding), then `np.gradient`, on a unit step:
    /// `g = [e^-½, 1, e^-½] / (1 + 2e^-½)`, so the smoothed step is
    /// `[0, 0, g0, 1 − g0, 1, 1]` and its gradient
    /// `[0, g0/2, (1 − g0)/2, (1 − g0)/2, g0/2, 0]` with `g0 = 0.274068…`.
    #[test]
    fn smooth_then_central_is_the_textbook_operator() {
        let mut det = Edge1DDetector::new(1.0);
        let cfg = Edge1DConfig {
            sigma: 1.0,
            derivative: Derivative1D::SmoothThenCentral {
                radius: NonZeroUsize::new(1).expect("nonzero"),
            },
            pos_thresh: 0.01,
            neg_thresh: 0.01,
            ..Edge1DConfig::default()
        };
        let peaks = det.detect_in(&[0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], &cfg);
        let g0 = (-0.5f64).exp() / (1.0 + 2.0 * (-0.5f64).exp());
        let expected = [
            0.0,
            0.5 * g0,
            0.5 * (1.0 - g0),
            0.5 * (1.0 - g0),
            0.5 * g0,
            0.0,
        ];
        for (i, (&r, &e)) in det.response().iter().zip(&expected).enumerate() {
            assert!(
                (f64::from(r) - e).abs() < 1e-6,
                "response[{i}] = {r}, expected {e}"
            );
        }
        // The plateau [2, 3] peaks at its second sample; the parabola puts the edge
        // exactly between them.
        assert_eq!(peaks.len(), 1, "{peaks:?}");
        assert_eq!(peaks[0].idx, 3);
        assert!((peaks[0].x - 2.5).abs() < 1e-6, "x = {}", peaks[0].x);
    }

    /// At a strict local maximum (`b ≥ a`, `b > c`) the parabola vertex never leaves
    /// ±0.5, in value space or log space.
    #[test]
    fn subpixel_offsets_stay_within_half_a_sample() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32) / (1u64 << 24) as f32
        };
        for _ in 0..20_000 {
            let b = 0.05 + next();
            let a = b * next();
            let c = b * next() * 0.999_999;
            for d in [
                parabola_offset(a, b, c),
                parabola_offset(a.ln(), b.ln(), c.ln()),
            ] {
                assert!(d.abs() <= 0.5, "a={a} b={b} c={c} -> {d}");
            }
            assert!(parabola_offset(b, b, c) <= 0.0 && parabola_offset(b, b, c) >= -0.5);
        }
    }

    /// A sampled Gaussian peak is recovered exactly by the log-parabola, not by the
    /// plain parabola.
    #[test]
    fn gaussian3_recovers_a_sampled_gaussian_peak() {
        for &x0 in &[10.0f32, 10.13, 10.37, 9.71] {
            let resp: Vec<f32> = (0..21)
                .map(|i| (-((i as f32 - x0).powi(2)) / (2.0 * 1.3 * 1.3)).exp())
                .collect();
            let idx = x0.round() as usize;
            let g = refine_x(&resp, idx, 1.0, SubpixRefine::Gaussian3);
            let p = refine_x(&resp, idx, 1.0, SubpixRefine::Parabolic3);
            assert!((g - x0).abs() < 1e-4, "gaussian3 {g} vs {x0}");
            if (x0 - x0.round()).abs() > 0.1 {
                assert!(
                    (p - x0).abs() > 1e-3,
                    "parabolic3 is biased here: {p} vs {x0}"
                );
            }
            // A falling peak is the same fit on the negated response.
            let neg: Vec<f32> = resp.iter().map(|v| -v).collect();
            assert_eq!(refine_x(&neg, idx, -1.0, SubpixRefine::Gaussian3), g);
        }
    }

    /// With a non-positive neighbour the logarithm is undefined; Gaussian3 falls back to
    /// the plain parabola.
    #[test]
    fn gaussian3_falls_back_to_the_parabola() {
        let resp = [0.0f32, 2.0, 1.0];
        assert_eq!(
            refine_x(&resp, 1, 1.0, SubpixRefine::Gaussian3),
            refine_x(&resp, 1, 1.0, SubpixRefine::Parabolic3)
        );
    }
}
