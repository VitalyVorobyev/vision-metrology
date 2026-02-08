use vm_core::BorderMode;

use crate::conv1d::convolve_f32;
use crate::kernels1d::DoGKernel1D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubpixRefine {
    None,
    Parabolic3,
    Centroid { radius: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edge1DConfig {
    pub sigma: f32,
    pub border: BorderMode<f32>,
    pub pos_thresh: f32,
    pub neg_thresh: f32,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgePolarity {
    Rising,
    Falling,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgePeak {
    pub x: f32,
    pub idx: usize,
    pub value: f32,
    pub strength: f32,
    pub polarity: EdgePolarity,
}

#[derive(Debug, Clone)]
pub struct Edge1DDetector {
    kernel: DoGKernel1D,
    tmp: Vec<f32>,
    resp: Vec<f32>,
    peaks: Vec<EdgePeak>,
}

impl Edge1DDetector {
    pub fn new(sigma: f32) -> Self {
        Self {
            kernel: DoGKernel1D::new(sigma),
            tmp: Vec::new(),
            resp: Vec::new(),
            peaks: Vec::new(),
        }
    }

    pub fn set_sigma(&mut self, sigma: f32) {
        if (sigma - self.kernel.sigma).abs() > f32::EPSILON {
            self.kernel = DoGKernel1D::new(sigma);
        }
    }

    pub fn detect_in_f32(&mut self, signal: &[f32], cfg: &Edge1DConfig) -> Vec<EdgePeak> {
        self.detect_in_f32_borrowed(signal, cfg).to_vec()
    }

    pub fn detect_in_u8(&mut self, signal: &[u8], cfg: &Edge1DConfig) -> Vec<EdgePeak> {
        self.detect_in_u8_borrowed(signal, cfg).to_vec()
    }

    pub fn detect_in_u16(&mut self, signal: &[u16], cfg: &Edge1DConfig) -> Vec<EdgePeak> {
        self.detect_in_u16_borrowed(signal, cfg).to_vec()
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
    use vm_core::BorderMode;

    use crate::conv1d::convolve_f32;
    use crate::edge1d::{Edge1DConfig, Edge1DDetector, EdgePolarity, SubpixRefine};
    use crate::kernels1d::DoGKernel1D;

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

    fn nearest_peak_x(
        peaks: &[crate::edge1d::EdgePeak],
        polarity: EdgePolarity,
        target: f32,
    ) -> f32 {
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
