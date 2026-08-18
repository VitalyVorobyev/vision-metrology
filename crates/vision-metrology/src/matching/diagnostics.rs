//! Per-point match diagnostics.
//!
//! A [`ShapeMatch`] reports one aggregate score; for auditing a result you
//! want to see *which parts of the contour* carried it. This module computes
//! the individual score term of every level-0 model point at a recovered
//! pose, using the same sampling and polarity rules as the matcher's own
//! final scoring — the mean of the returned terms is the match score.
//!
//! This is a diagnostics path: it builds what it needs from the scene image
//! on every call and does not reuse matcher scratch. Do not put it in a
//! per-frame loop.

use vm_primitives::{DirectionField, Image, ImageView};

use super::config::Polarity;
use super::matcher::ShapeMatch;
use super::model::ShapeModel;

/// Individual score terms of every level-0 model point at `m`'s pose.
///
/// The value per point is `t'ᵢ · ĝ(pᵢ)` under the model's polarity rules:
/// `IgnoreLocal` reports `|c|`; `Match` and `IgnoreGlobal` report the signed
/// term. A value of exactly `0.0` means the point landed on a below-contrast
/// pixel or outside the image — the same "no evidence" convention the score
/// uses, so `mean(terms) == m.score` for `Match`/`IgnoreLocal` (and
/// `|mean|` for `IgnoreGlobal`).
///
/// `min_contrast` should be the [`ShapeSearchConfig::min_contrast`] the match
/// was found with, so the gate matches what the matcher saw.
///
/// [`ShapeSearchConfig::min_contrast`]: super::config::ShapeSearchConfig::min_contrast
pub fn match_point_scores(
    scene: &ImageView<'_, u8>,
    model: &ShapeModel,
    m: &ShapeMatch,
    min_contrast: f32,
) -> Vec<f32> {
    let Some(level0) = model.level(0) else {
        return Vec::new();
    };
    let points = &level0.points;
    if points.is_empty() {
        return Vec::new();
    }

    // Level-0 field, built lazily only around the match.
    let (w, h) = (scene.width(), scene.height());
    let mut data = vec![0.0f32; w * h];
    for y in 0..h {
        let row = scene.row(y);
        for (d, &s) in data[y * w..(y + 1) * w].iter_mut().zip(row.iter()) {
            *d = f32::from(s);
        }
    }
    let img = Image::from_vec(w, h, data).expect("scene dimensions are valid");

    let mut field = DirectionField::new();
    field.begin_tiled_f32(&img, model.smooth(), min_contrast);
    let reach = (level0.radius * m.scale() * 1.1).ceil() as i32 + 4;
    let (cx, cy) = (m.position.x.round() as i32, m.position.y.round() as i32);
    field.ensure_rect_f32(&img, cx - reach, cy - reach, cx + reach + 1, cy + reach + 1);

    let (sn, cs) = m.angle().sin_cos();
    let scale = m.scale();
    let (qx, qy) = (m.position.x, m.position.y);
    let dir = field.dir();

    points
        .iter()
        .map(|p| {
            let px = (qx + scale * (cs * p.d.x - sn * p.d.y)).round();
            let py = (qy + scale * (sn * p.d.x + cs * p.d.y)).round();
            if px < 0.0 || py < 0.0 {
                return 0.0;
            }
            let (ux, uy) = (px as usize, py as usize);
            if ux >= w || uy >= h {
                return 0.0;
            }
            let k = 2 * (uy * w + ux);
            let tx = cs * p.t.x - sn * p.t.y;
            let ty = sn * p.t.x + cs * p.t.y;
            let c = tx * dir[k] + ty * dir[k + 1];
            if model.polarity() == Polarity::IgnoreLocal {
                c.abs()
            } else {
                c
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use vm_primitives::{Image, Rect2f};

    use super::match_point_scores;
    use crate::matching::{ShapeMatcher, ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig};

    /// Bright rectangle on dark ground — every model point sits on a clean
    /// edge, so every term must be strong and their mean must reproduce the
    /// match score.
    #[test]
    fn terms_mean_reproduces_the_score() {
        let (w, h) = (256usize, 256usize);
        let mut data = vec![30u8; w * h];
        for y in 90..170 {
            for x in 80..180 {
                data[y * w + x] = 220;
            }
        }
        let img = Image::from_vec(w, h, data).unwrap();

        let roi = Rect2f {
            x: 70.0,
            y: 80.0,
            width: 120.0,
            height: 100.0,
        };
        let model = ShapeModelBuilder::new()
            .build_u8(&img.as_view(), roi, &ShapeModelConfig::default())
            .expect("model builds");

        let cfg = ShapeSearchConfig::default();
        let m = ShapeMatcher::new()
            .find_u8(&img.as_view(), &model, &cfg)
            .into_iter()
            .next()
            .expect("self-match");

        let terms = match_point_scores(&img.as_view(), &model, &m, cfg.min_contrast);
        assert_eq!(terms.len(), model.point_count(0));

        let mean = terms.iter().sum::<f32>() / terms.len() as f32;
        assert!(
            (mean - m.score).abs() < 1e-5,
            "mean of terms {mean} vs score {}",
            m.score
        );
        // On a clean self-match, the vast majority of points contribute
        // strongly.
        let strong = terms.iter().filter(|&&t| t > 0.8).count();
        assert!(strong * 10 >= terms.len() * 8, "{strong}/{}", terms.len());
    }
}
