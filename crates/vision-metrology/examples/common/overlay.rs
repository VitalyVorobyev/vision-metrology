//! Shared overlay rendering for the shape-matching examples.
//!
//! Two products:
//! - [`overview`] — the classic side-by-side: reference with ROI + model
//!   points on the left, scene with every located instance on the right.
//! - [`diagnostic`] — the audit view for one match, three panels at 2×
//!   magnification:
//!   1. **Pose panel**: scene crop with the transformed ROI quad, an axes
//!      arrow along the model's +x direction (length ∝ scale) and the origin
//!      cross — translation and rotation readable at a glance.
//!   2. **Registration panel**: checkerboard composite alternating 8×8 px
//!      blocks of the scene and the reference patch warped by the recovered
//!      pose. Sub-pixel misregistration shows as edge breaks at every block
//!      boundary; a correct pose makes the checker seams invisible.
//!   3. **Contribution panel**: the per-point score terms painted on the
//!      contour — green where the scene gradient agrees with the model,
//!      red where a point contributes nothing. Occlusion, clutter and model
//!      defects become visible instead of being averaged into one number.

// This module is `#[path]`-included from more than one example root; each
// root uses a subset of it, so per-root dead-code analysis is meaningless.
#![allow(dead_code)]

use vision_metrology::{
    BorderMode, Image, Point2f, Rect2f, ShapeMatch, ShapeModel, sample_bilinear_f32,
};

pub const GREEN: [u8; 3] = [40, 220, 60];
pub const RED: [u8; 3] = [235, 60, 50];
pub const YELLOW: [u8; 3] = [235, 210, 40];
pub const CYAN: [u8; 3] = [60, 200, 220];
pub const ORANGE: [u8; 3] = [245, 150, 40];

/// Side-by-side overview: reference + ROI + model points | scene + matches.
pub fn overview(
    reference: &Image<u8>,
    scene: &Image<u8>,
    roi: Rect2f,
    model: &ShapeModel,
    matches: &[ShapeMatch],
) -> image::RgbImage {
    let (w, h) = (
        reference.width().max(scene.width()),
        reference.height().max(scene.height()),
    );
    let mut canvas = image::RgbImage::new((2 * w) as u32, h as u32);

    blit(&mut canvas, reference, 0);
    blit(&mut canvas, scene, w);

    rect(&mut canvas, roi, 0, CYAN);
    for p in model.reference_points() {
        dot(&mut canvas, p.x, p.y, 0, GREEN);
    }
    cross(&mut canvas, model.origin().x, model.origin().y, 0, RED);

    for m in matches {
        for p in model.reference_points() {
            let q = m.pose * nalgebra::Point2::new(p.x, p.y);
            dot(&mut canvas, q.x, q.y, w, GREEN);
        }
        for (a, b) in rect_edges(roi) {
            let pa = m.pose * nalgebra::Point2::new(a.x, a.y);
            let pb = m.pose * nalgebra::Point2::new(b.x, b.y);
            line(&mut canvas, pa.x, pa.y, pb.x, pb.y, w, YELLOW);
        }
        axes_arrow(&mut canvas, m, w);
        cross(&mut canvas, m.position.x, m.position.y, w, RED);
    }
    canvas
}

/// Magnification of the diagnostic panels.
const ZOOM: usize = 2;
/// Checker block size in scene pixels.
const CHECKER: usize = 8;
const GUTTER: usize = 6;

/// Three-panel audit view for one match. `terms` are the per-point score
/// contributions from `vision_metrology::match_point_scores`.
pub fn diagnostic(
    reference: &Image<u8>,
    scene: &Image<u8>,
    model: &ShapeModel,
    m: &ShapeMatch,
    terms: &[f32],
) -> image::RgbImage {
    // Crop window: the model's reach at this pose plus a margin.
    let radius = model
        .levels()
        .first()
        .map_or(64.0, |l| l.radius * m.scale());
    let half = (radius + 16.0).ceil() as i32;
    let cx = m.position.x.round() as i32;
    let cy = m.position.y.round() as i32;
    let x0 = (cx - half).max(0);
    let y0 = (cy - half).max(0);
    let x1 = (cx + half + 1).min(scene.width() as i32);
    let y1 = (cy + half + 1).min(scene.height() as i32);
    let (cw, ch) = ((x1 - x0) as usize, (y1 - y0) as usize);

    let (pw, ph) = (cw * ZOOM, ch * ZOOM);
    let mut canvas = image::RgbImage::from_pixel(
        (3 * pw + 2 * GUTTER) as u32,
        ph as u32,
        image::Rgb([24, 24, 28]),
    );

    let inv = m.pose.inverse();
    let scene_view = scene.as_view();
    let ref_view = reference.as_view();

    for py in 0..ph {
        for px in 0..pw {
            // Scene coordinates of this panel pixel (pixel centres).
            let sx = x0 as f32 + px as f32 / ZOOM as f32;
            let sy = y0 as f32 + py as f32 / ZOOM as f32;
            let sv = sample_bilinear_f32(&scene_view, sx, sy, BorderMode::Clamp)
                .round()
                .clamp(0.0, 255.0) as u8;

            // Panel 1: plain scene crop (annotations drawn after).
            canvas.put_pixel(px as u32, py as u32, image::Rgb([sv, sv, sv]));

            // Panel 2: checkerboard scene / warped reference.
            let cell = (sx as usize / CHECKER + sy as usize / CHECKER).is_multiple_of(2);
            let v2 = if cell {
                sv
            } else {
                let r = inv * nalgebra::Point2::new(sx, sy);
                if r.x >= 0.0
                    && r.y >= 0.0
                    && r.x < reference.width() as f32
                    && r.y < reference.height() as f32
                {
                    sample_bilinear_f32(&ref_view, r.x, r.y, BorderMode::Clamp)
                        .round()
                        .clamp(0.0, 255.0) as u8
                } else {
                    40
                }
            };
            canvas.put_pixel(
                (pw + GUTTER + px) as u32,
                py as u32,
                image::Rgb([v2, v2, v2]),
            );

            // Panel 3: dimmed scene, points painted on top afterwards.
            let dim = sv / 2;
            canvas.put_pixel(
                (2 * (pw + GUTTER) + px) as u32,
                py as u32,
                image::Rgb([dim, dim, dim]),
            );
        }
    }

    // Panel 1 annotations in panel coordinates.
    let to_panel = |p: Point2f| {
        Point2f::new(
            (p.x - x0 as f32) * ZOOM as f32,
            (p.y - y0 as f32) * ZOOM as f32,
        )
    };
    {
        let o = to_panel(m.position);
        // Axes arrow along the model +x axis.
        let len = radius * 0.6 * ZOOM as f32;
        let (snn, css) = m.angle().sin_cos();
        let tip = Point2f::new(o.x + css * len, o.y + snn * len);
        line_px(&mut canvas, o.x, o.y, tip.x, tip.y, ORANGE);
        let (hx, hy) = (-css * 6.0, -snn * 6.0);
        line_px(
            &mut canvas,
            tip.x,
            tip.y,
            tip.x + hx - hy * 0.6,
            tip.y + hy + hx * 0.6,
            ORANGE,
        );
        line_px(
            &mut canvas,
            tip.x,
            tip.y,
            tip.x + hx + hy * 0.6,
            tip.y + hy - hx * 0.6,
            ORANGE,
        );
        cross_px(&mut canvas, o.x, o.y, RED);
    }

    // Panel 3: per-point contributions.
    let pts = model.reference_points();
    for (p, &t) in pts.iter().zip(terms.iter()) {
        let q = m.pose * nalgebra::Point2::new(p.x, p.y);
        let pp = to_panel(Point2f::new(q.x, q.y));
        let c = term_color(t);
        for dy in 0..2i32 {
            for dx in 0..2i32 {
                put_px(
                    &mut canvas,
                    2.0 * (pw + GUTTER) as f32 + pp.x + dx as f32,
                    pp.y + dy as f32,
                    c,
                );
            }
        }
    }

    canvas
}

/// Red (no evidence) → yellow (weak) → green (strong agreement).
fn term_color(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        let k = t / 0.5;
        [
            RED[0] as f32 + (YELLOW[0] as f32 - RED[0] as f32) * k,
            RED[1] as f32 + (YELLOW[1] as f32 - RED[1] as f32) * k,
            RED[2] as f32 + (YELLOW[2] as f32 - RED[2] as f32) * k,
        ]
    } else {
        let k = (t - 0.5) / 0.5;
        [
            YELLOW[0] as f32 + (GREEN[0] as f32 - YELLOW[0] as f32) * k,
            YELLOW[1] as f32 + (GREEN[1] as f32 - YELLOW[1] as f32) * k,
            YELLOW[2] as f32 + (GREEN[2] as f32 - YELLOW[2] as f32) * k,
        ]
    }
    .map(|v| v.round().clamp(0.0, 255.0) as u8)
}

fn axes_arrow(canvas: &mut image::RgbImage, m: &ShapeMatch, x_off: usize) {
    let len = 45.0 * m.scale();
    let (snn, css) = m.angle().sin_cos();
    let (ox, oy) = (m.position.x, m.position.y);
    let (tx, ty) = (ox + css * len, oy + snn * len);
    line(canvas, ox, oy, tx, ty, x_off, ORANGE);
    let (hx, hy) = (-css * 7.0, -snn * 7.0);
    line(
        canvas,
        tx,
        ty,
        tx + hx - hy * 0.6,
        ty + hy + hx * 0.6,
        x_off,
        ORANGE,
    );
    line(
        canvas,
        tx,
        ty,
        tx + hx + hy * 0.6,
        ty + hy - hx * 0.6,
        x_off,
        ORANGE,
    );
}

// ── primitives ──────────────────────────────────────────────────────────────

fn blit(canvas: &mut image::RgbImage, src: &Image<u8>, x_off: usize) {
    for y in 0..src.height() {
        let row = src.as_view().row(y);
        for (x, &v) in row.iter().enumerate() {
            canvas.put_pixel((x + x_off) as u32, y as u32, image::Rgb([v, v, v]));
        }
    }
}

fn put(canvas: &mut image::RgbImage, x: f32, y: f32, x_off: usize, c: [u8; 3]) {
    let (xi, yi) = (x.round(), y.round());
    if xi < 0.0 || yi < 0.0 {
        return;
    }
    let (xi, yi) = (xi as u32 + x_off as u32, yi as u32);
    if xi < canvas.width() && yi < canvas.height() {
        canvas.put_pixel(xi, yi, image::Rgb(c));
    }
}

fn put_px(canvas: &mut image::RgbImage, x: f32, y: f32, c: [u8; 3]) {
    put(canvas, x, y, 0, c);
}

/// A 2x2 block, so a single model point stays visible when the overlay is
/// viewed scaled down.
fn dot(canvas: &mut image::RgbImage, x: f32, y: f32, x_off: usize, c: [u8; 3]) {
    for dy in 0..2 {
        for dx in 0..2 {
            put(canvas, x + dx as f32, y + dy as f32, x_off, c);
        }
    }
}

fn cross(canvas: &mut image::RgbImage, x: f32, y: f32, x_off: usize, c: [u8; 3]) {
    for d in -7..=7 {
        put(canvas, x + d as f32, y, x_off, c);
        put(canvas, x, y + d as f32, x_off, c);
    }
}

fn cross_px(canvas: &mut image::RgbImage, x: f32, y: f32, c: [u8; 3]) {
    cross(canvas, x, y, 0, c);
}

fn line(
    canvas: &mut image::RgbImage,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x_off: usize,
    c: [u8; 3],
) {
    let n = ((x1 - x0).abs().max((y1 - y0).abs()) * 2.0).ceil().max(1.0) as usize;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        put(canvas, x0 + t * (x1 - x0), y0 + t * (y1 - y0), x_off, c);
    }
}

fn line_px(canvas: &mut image::RgbImage, x0: f32, y0: f32, x1: f32, y1: f32, c: [u8; 3]) {
    line(canvas, x0, y0, x1, y1, 0, c);
}

fn rect(canvas: &mut image::RgbImage, r: Rect2f, x_off: usize, c: [u8; 3]) {
    for (a, b) in rect_edges(r) {
        line(canvas, a.x, a.y, b.x, b.y, x_off, c);
    }
}

fn rect_edges(r: Rect2f) -> [(Point2f, Point2f); 4] {
    let (x0, y0) = (r.x, r.y);
    let (x1, y1) = (r.x + r.width, r.y + r.height);
    let p = |x, y| Point2f::new(x, y);
    [
        (p(x0, y0), p(x1, y0)),
        (p(x1, y0), p(x1, y1)),
        (p(x1, y1), p(x0, y1)),
        (p(x0, y1), p(x0, y0)),
    ]
}
