//! Two-stage subpixel inter-frame shift.
//!
//! Stage 1 is a rotation-off ZNCC search (corrmatch, bounded to a small
//! margin around the window's previous position) with corrmatch's own
//! quadratic-peak subpixel refinement. Stage 2 (default) is
//! translation-only inverse-compositional Lucas-Kanade, implemented here:
//! a 2x2 normal-equations solve on the window's own Scharr gradient,
//! bilinear-sampling `curr` each iteration. Stage 1 alone is what a plain
//! correlation tracker reports; stage 2 removes its bias toward integer
//! pixel positions ("pixel-locking") — see `tests/accuracy.rs`'s
//! `displacement_quadratic` vs `displacement_lk` rows for the measured
//! difference.

use vm_primitives::{BorderMode, Error, ImageView, Point2f, Vec2f, sample_bilinear_f32};

use super::adapter::{copy_bounds_u8, corr_view, map_corr_err, rect_bounds};
use super::config::{DisplacementConfig, Refine};

/// Result of [`displacement`]: subpixel shift from `prev` to `curr`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Displacement {
    /// `curr` position minus `prev` position, in pixels.
    pub shift: Vec2f,
    /// Stage-1 (corrmatch ZNCC) score at the accepted match. Stage 2
    /// (Lucas-Kanade) only refines `shift` — it has no score of its own, so
    /// this is unchanged by which [`Refine`] mode ran.
    pub score: f32,
}

/// Tracks `cfg.window` (in `prev`) into `curr`.
///
/// Allocates per call (a copy of the window, a copy of the bounded search
/// region, and — under [`Refine::LucasKanade`] — one gradient buffer sized
/// to the window); no persistent scratch is kept, since this is a plain
/// function rather than a reusable extractor.
///
/// # Errors
/// [`Error::InvalidConfig`] if `cfg.window` does not overlap `prev`, is
/// smaller than 2x2, or its bounded search region does not fit inside
/// `curr`. [`Error::Degenerate`] if the stage-1 score is below
/// `cfg.min_score`.
pub fn displacement(
    prev: &ImageView<'_, u8>,
    curr: &ImageView<'_, u8>,
    cfg: &DisplacementConfig,
) -> Result<Displacement, Error> {
    let (wx0, wy0, tw, th) = rect_bounds(prev.width(), prev.height(), cfg.window)?;
    if tw < 2 || th < 2 {
        return Err(Error::InvalidConfig(
            "corr::displacement: window must be at least 2x2",
        ));
    }
    let tpl_data = copy_bounds_u8(prev, wx0, wy0, tw, th);

    // Stage 1: bounded ZNCC search in a `search`-pixel margin around the
    // window's `prev` position — never the whole scene, so this stays cheap
    // and cannot lock onto a distant, unrelated peak.
    let template = corrmatch::Template::new(tpl_data.clone(), tw, th).map_err(map_corr_err)?;
    let compiled = corrmatch::CompiledTemplate::compile_unrotated(
        &template,
        corrmatch::CompileConfigNoRot::default(),
    )
    .map_err(map_corr_err)?;

    let (sx, sy) = cfg.search;
    let roi_x0 = (wx0 as i32 - sx).max(0);
    let roi_y0 = (wy0 as i32 - sy).max(0);
    let roi_x1 = ((wx0 + tw) as i32 + sx).min(curr.width() as i32);
    let roi_y1 = ((wy0 + th) as i32 + sy).min(curr.height() as i32);
    if roi_x1 - roi_x0 < tw as i32 || roi_y1 - roi_y0 < th as i32 {
        return Err(Error::InvalidConfig(
            "corr::displacement: bounded search region does not fit inside curr",
        ));
    }
    let (roi_w, roi_h) = ((roi_x1 - roi_x0) as usize, (roi_y1 - roi_y0) as usize);
    let (roi_x0, roi_y0) = (roi_x0 as usize, roi_y0 as usize);
    let scene_data = copy_bounds_u8(curr, roi_x0, roi_y0, roi_w, roi_h);
    let view = corr_view(&scene_data, roi_w, roi_h)?;

    let mut match_cfg = corrmatch::MatchConfig::default();
    match_cfg.rotation = corrmatch::RotationMode::Disabled;
    match_cfg.metric = corrmatch::Metric::Zncc;
    let matcher = corrmatch::Matcher::new(compiled).with_config(match_cfg);
    let m = matcher.match_image(view).map_err(map_corr_err)?;

    if m.score < cfg.min_score {
        return Err(Error::Degenerate(
            "corr::displacement: stage-1 score below min_score",
        ));
    }

    let found_topleft = Point2f::new(roi_x0 as f32 + m.x, roi_y0 as f32 + m.y);
    let window_origin = Point2f::new(wx0 as f32, wy0 as f32);
    let shift0 = found_topleft - window_origin;

    let shift = match cfg.refine {
        Refine::None => shift0,
        Refine::LucasKanade { iters } => {
            lucas_kanade_translation(&tpl_data, tw, th, curr, window_origin, shift0, iters)
        }
    };

    Ok(Displacement {
        shift,
        score: m.score,
    })
}

/// Translation-only inverse-compositional Lucas-Kanade, seeded at `p0`.
///
/// The Hessian and the steepest-descent images are built once from the
/// **template**'s own Scharr gradient (`tpl` is `tw x th`, row-major `u8`,
/// the same window `displacement` extracted from `prev`) — the
/// inverse-compositional trick that keeps each iteration to one bilinear
/// sample of `curr` per template pixel plus a 2x2 solve, with no
/// re-differentiation of `curr`.
fn lucas_kanade_translation(
    tpl: &[u8],
    tw: usize,
    th: usize,
    curr: &ImageView<'_, u8>,
    window_origin: Point2f,
    p0: Vec2f,
    iters: u32,
) -> Vec2f {
    let grad = scharr_u8(tpl, tw, th);

    let (mut hxx, mut hxy, mut hyy) = (0.0f64, 0.0f64, 0.0f64);
    for &(gx, gy) in &grad {
        hxx += (gx * gx) as f64;
        hxy += (gx * gy) as f64;
        hyy += (gy * gy) as f64;
    }
    let det = hxx * hyy - hxy * hxy;
    if det.abs() < 1e-6 {
        // Flat window (no gradient signal): nothing to refine, keep stage 1.
        return p0;
    }
    let inv = [[hyy / det, -hxy / det], [-hxy / det, hxx / det]];

    let mut p = p0;
    for _ in 0..iters {
        let (mut bx, mut by) = (0.0f64, 0.0f64);
        for y in 0..th {
            for x in 0..tw {
                let t = tpl[y * tw + x] as f32;
                let sx = window_origin.x + x as f32 + p.x;
                let sy = window_origin.y + y as f32 + p.y;
                let i = sample_bilinear_f32(curr, sx, sy, BorderMode::Clamp);
                let err = (i - t) as f64;
                let (gx, gy) = grad[y * tw + x];
                bx += gx as f64 * err;
                by += gy as f64 * err;
            }
        }
        let dpx = (inv[0][0] * bx + inv[0][1] * by) as f32;
        let dpy = (inv[1][0] * bx + inv[1][1] * by) as f32;
        // Inverse-compositional update for a translation (Abelian) warp:
        // p <- p - dp.
        p.x -= dpx;
        p.y -= dpy;
    }
    p
}

/// Local copy of the crate's dense 3x3 Scharr kernel (clamped border, same
/// kernel constants as `vm_primitives::edge::gradient::dense_scharr`).
/// That function is `pub(super)`-visible only inside its own module, and
/// this is a ~15-line, window-sized variant — not worth widening a private
/// boundary for.
fn scharr_u8(data: &[u8], w: usize, h: usize) -> Vec<(f32, f32)> {
    let at = |x: usize, y: usize| data[y * w + x] as f32;
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(h - 1);
        for x in 0..w {
            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(w - 1);
            let p00 = at(xm1, ym1);
            let p01 = at(x, ym1);
            let p02 = at(xp1, ym1);
            let p10 = at(xm1, y);
            let p12 = at(xp1, y);
            let p20 = at(xm1, yp1);
            let p21 = at(x, yp1);
            let p22 = at(xp1, yp1);
            let gx = (3.0 * p02 + 10.0 * p12 + 3.0 * p22) - (3.0 * p00 + 10.0 * p10 + 3.0 * p20);
            let gy = (3.0 * p20 + 10.0 * p21 + 3.0 * p22) - (3.0 * p00 + 10.0 * p01 + 3.0 * p02);
            out.push((gx, gy));
        }
    }
    out
}
