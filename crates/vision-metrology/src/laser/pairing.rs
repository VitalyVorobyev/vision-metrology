//! Bright-on-dark edge-pair selection with a continuity prior.

use vm_primitives::{Edge1DDetector, EdgePair1D, EdgePeak, EdgePolarity, Pixel};

use super::types::{LaserExtractConfig, LaserSample};

/// Detect edge peaks on one scan line and pick the best stripe pair.
pub(super) fn detect_pair<P: Pixel>(
    detector: &mut Edge1DDetector,
    line: &[P],
    predicted: f32,
    tracking: bool,
    cfg: &LaserExtractConfig,
    x_offset: usize,
    scan_i: usize,
) -> Option<LaserSample> {
    let peaks = detector.detect_in_ref(line, &cfg.tuning.edge_cfg);
    let pair = best_pair_with_prior_offset(
        peaks,
        x_offset as f32,
        cfg.min_width,
        cfg.max_width,
        predicted,
        cfg.tuning.prior_weight,
    )?;

    accept_pair(pair, predicted, tracking, cfg, scan_i)
}

fn accept_pair(
    pair: EdgePair1D,
    predicted: f32,
    tracking: bool,
    cfg: &LaserExtractConfig,
    scan_i: usize,
) -> Option<LaserSample> {
    if pair.score < cfg.min_score {
        return None;
    }

    if tracking && (pair.center_x - predicted).abs() > cfg.tuning.max_jump_px {
        return None;
    }

    Some(LaserSample {
        scan_i,
        center: pair.center_x,
        width: pair.width,
        score: pair.score,
        left: pair.left.x,
        right: pair.right.x,
        valid: true,
    })
}

fn best_pair_with_prior_offset(
    peaks: &[EdgePeak],
    x_offset: f32,
    min_width: f32,
    max_width: f32,
    predicted_center: f32,
    prior_weight: f32,
) -> Option<EdgePair1D> {
    let mut best: Option<EdgePair1D> = None;
    let xoff_idx = x_offset.round() as usize;

    for left in peaks.iter().filter(|p| p.polarity == EdgePolarity::Rising) {
        for right in peaks.iter().filter(|p| p.polarity == EdgePolarity::Falling) {
            let lx = left.x + x_offset;
            let rx = right.x + x_offset;
            if rx <= lx {
                continue;
            }

            let width = rx - lx;
            if width < min_width || width > max_width {
                continue;
            }

            let center = 0.5 * (lx + rx);
            let base_score = left.strength + right.strength;
            let total_score = base_score - prior_weight * (center - predicted_center).abs();

            let cand = EdgePair1D {
                left: EdgePeak {
                    x: lx,
                    idx: left.idx + xoff_idx,
                    value: left.value,
                    strength: left.strength,
                    polarity: left.polarity,
                },
                right: EdgePeak {
                    x: rx,
                    idx: right.idx + xoff_idx,
                    value: right.value,
                    strength: right.strength,
                    polarity: right.polarity,
                },
                center_x: center,
                width,
                score: total_score,
                bright_on_dark: true,
            };

            if best.as_ref().is_none_or(|b| cand.score > b.score) {
                best = Some(cand);
            }
        }
    }

    best
}
