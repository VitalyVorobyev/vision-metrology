use vm_core::ImageView;

use crate::edge1d::{Edge1DConfig, Edge1DDetector, EdgePeak, EdgePolarity};

#[derive(Debug, Clone, PartialEq)]
pub struct EdgePair1D {
    pub left: EdgePeak,
    pub right: EdgePeak,
    pub center_x: f32,
    pub width: f32,
    pub score: f32,
    pub bright_on_dark: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgePairConfig {
    pub min_width: f32,
    pub max_width: f32,
    pub prefer_bright_on_dark: bool,
}

impl Default for EdgePairConfig {
    fn default() -> Self {
        Self {
            min_width: 1.0,
            max_width: 50.0,
            prefer_bright_on_dark: true,
        }
    }
}

pub fn best_edge_pair(peaks: &[EdgePeak], cfg: &EdgePairConfig) -> Option<EdgePair1D> {
    let mut best: Option<EdgePair1D> = None;

    for i in 0..peaks.len() {
        for j in (i + 1)..peaks.len() {
            if peaks[i].polarity == peaks[j].polarity {
                continue;
            }

            let (left, right) = if peaks[i].x <= peaks[j].x {
                (&peaks[i], &peaks[j])
            } else {
                (&peaks[j], &peaks[i])
            };

            let width = right.x - left.x;
            if width < cfg.min_width || width > cfg.max_width {
                continue;
            }

            let bright_on_dark =
                left.polarity == EdgePolarity::Rising && right.polarity == EdgePolarity::Falling;

            // Simple score favoring strong edges with a mild width penalty.
            let width_penalty = 0.02 * width;
            let mut score = (left.strength + right.strength) / (1.0 + width_penalty);

            if cfg.prefer_bright_on_dark {
                if bright_on_dark {
                    score *= 1.10;
                } else {
                    score *= 0.95;
                }
            }

            let cand = EdgePair1D {
                left: left.clone(),
                right: right.clone(),
                center_x: 0.5 * (left.x + right.x),
                width,
                score,
                bright_on_dark,
            };

            if best.as_ref().is_none_or(|b| cand.score > b.score) {
                best = Some(cand);
            }
        }
    }

    best
}

pub fn best_edge_pair_in_row_u8(
    detector: &mut Edge1DDetector,
    img: &ImageView<'_, u8>,
    y: usize,
    edge_cfg: &Edge1DConfig,
    pair_cfg: &EdgePairConfig,
) -> Option<EdgePair1D> {
    let row = img.row(y);
    let peaks = detector.detect_in_u8_borrowed(row, edge_cfg);
    best_edge_pair(peaks, pair_cfg)
}

#[cfg(test)]
mod tests {
    use vm_core::{BorderMode, Image};

    use crate::conv1d::convolve_f32;
    use crate::edge1d::{Edge1DConfig, Edge1DDetector, SubpixRefine};
    use crate::kernels1d::DoGKernel1D;
    use crate::laser1d::{EdgePairConfig, best_edge_pair, best_edge_pair_in_row_u8};

    fn stripe_signal(len: usize, x_l: f32, x_r: f32, amp: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; len];
        for (i, dst) in out.iter_mut().enumerate() {
            let x0 = i as f32 - 0.5;
            let x1 = i as f32 + 0.5;
            let overlap = (x1.min(x_r) - x0.max(x_l)).max(0.0);
            *dst = amp * overlap.clamp(0.0, 1.0);
        }
        out
    }

    fn blur(signal: &[f32], sigma: f32) -> Vec<f32> {
        let k = DoGKernel1D::new(sigma);
        let mut out = vec![0.0f32; signal.len()];
        convolve_f32(signal, &k.g, k.radius, BorderMode::Clamp, &mut out);
        out
    }

    #[test]
    fn best_pair_with_reflection_artifact() {
        let sigma = 1.2;
        let x_l = 20.3;
        let x_r = 35.7;

        let mut sig = stripe_signal(96, x_l, x_r, 1.0);
        let refl = stripe_signal(96, 43.2, 47.6, 0.22);
        for (s, r) in sig.iter_mut().zip(refl.iter()) {
            *s += *r;
        }
        let sig = blur(&sig, sigma);

        let mut det = Edge1DDetector::new(sigma);
        let edge_cfg = Edge1DConfig {
            sigma,
            border: BorderMode::Clamp,
            pos_thresh: 0.01,
            neg_thresh: 0.01,
            refine: SubpixRefine::Parabolic3,
        };
        let pair_cfg = EdgePairConfig {
            min_width: 4.0,
            max_width: 30.0,
            prefer_bright_on_dark: true,
        };

        let peaks = det.detect_in_f32(&sig, &edge_cfg);
        let pair = best_edge_pair(&peaks, &pair_cfg).expect("pair should exist");

        assert!((pair.left.x - x_l).abs() <= 0.2);
        assert!((pair.right.x - x_r).abs() <= 0.2);
        assert!((pair.width - (x_r - x_l)).abs() <= 0.3);
        assert!(pair.bright_on_dark);
    }

    #[test]
    fn rowwise_u8_pair_center() {
        let sigma = 1.2;
        let x_l = 18.6;
        let x_r = 40.2;
        let center = 0.5 * (x_l + x_r);

        let row_f = blur(&stripe_signal(80, x_l, x_r, 255.0), sigma);

        let mut data = vec![0u8; 80 * 3];
        for (i, &v) in row_f.iter().enumerate() {
            data[80 + i] = v.round().clamp(0.0, 255.0) as u8;
        }
        let img = Image::from_vec(80, 3, data).expect("valid image");

        let mut det = Edge1DDetector::new(sigma);
        let edge_cfg = Edge1DConfig {
            sigma,
            border: BorderMode::Clamp,
            pos_thresh: 1.0,
            neg_thresh: 1.0,
            refine: SubpixRefine::Parabolic3,
        };
        let pair_cfg = EdgePairConfig::default();

        let pair = best_edge_pair_in_row_u8(&mut det, &img.as_view(), 1, &edge_cfg, &pair_cfg)
            .expect("pair should exist");

        assert!((pair.center_x - center).abs() <= 0.2);
    }
}
