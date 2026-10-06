//! Drawing a tracked bead: the prior, the final stage's strips, the refined centreline,
//! the final edges, and every rejected station coloured by its reason.
//!
//! `#[path]`-included next to `overlay.rs`, whose helpers it draws with.

// Each root that includes this module uses a subset of it.
#![allow(dead_code)]

use vision_metrology::measure::BeadReject;
use vision_metrology::measure::diagnostics::BeadTrace;
use vision_metrology::{Image, Point2f};

use super::overlay::{CYAN, GREEN, ORANGE, RED, YELLOW, blit, dot, line};

pub const PRIOR: [u8; 3] = [105, 105, 105];
pub const STRIP: [u8; 3] = [90, 140, 220];
pub const MAGENTA: [u8; 3] = [220, 80, 220];
pub const WHITE: [u8; 3] = [240, 240, 240];

/// The colour a rejected station is drawn in.
pub fn reason_colour(r: BeadReject) -> [u8; 3] {
    match r {
        BeadReject::Caliper(_) => RED,
        BeadReject::Clearance => ORANGE,
        BeadReject::Offset => CYAN,
        BeadReject::Width => MAGENTA,
        _ => WHITE,
    }
}

/// `c` blended over the pixel nearest `(x, y)` with weight `alpha`.
fn blend(canvas: &mut image::RgbImage, x: f32, y: f32, c: [u8; 3], alpha: f32) {
    let (xi, yi) = (x.round(), y.round());
    if xi < 0.0 || yi < 0.0 || xi >= canvas.width() as f32 || yi >= canvas.height() as f32 {
        return;
    }
    let px = canvas.get_pixel_mut(xi as u32, yi as u32);
    for (v, &k) in px.0.iter_mut().zip(&c) {
        *v = (f32::from(*v) * (1.0 - alpha) + f32::from(k) * alpha).round() as u8;
    }
}

/// A segment blended at `alpha`, one sample per pixel.
fn faint_line(canvas: &mut image::RgbImage, a: Point2f, b: Point2f, c: [u8; 3], alpha: f32) {
    let n = (b - a).norm().ceil().max(1.0) as usize;
    for i in 0..=n {
        let p = a + (b - a) * (i as f32 / n as f32);
        blend(canvas, p.x, p.y, c, alpha);
    }
}

fn polyline(canvas: &mut image::RgbImage, pts: &[Point2f], c: [u8; 3]) {
    for w in pts.windows(2) {
        line(canvas, w[0].x, w[0].y, w[1].x, w[1].y, 0, c);
    }
}

/// `img` with the bead `trace` tracked from `prior` drawn over it.
pub fn draw(img: &Image<u8>, prior: &[Point2f], trace: &BeadTrace) -> image::RgbImage {
    let mut canvas = image::RgbImage::new(img.width() as u32, img.height() as u32);
    blit(&mut canvas, img, 0);
    for st in &trace.measure {
        faint_line(&mut canvas, st.strip.start, st.strip.end, STRIP, 0.3);
    }
    polyline(&mut canvas, prior, PRIOR);
    let bead = &trace.result;
    polyline(&mut canvas, &bead.centerline, GREEN);
    for smp in &bead.samples {
        match smp.hit {
            Ok(hit) => {
                for e in [hit.pair.first, hit.pair.second] {
                    dot(&mut canvas, e.p.x - 0.5, e.p.y - 0.5, 0, YELLOW);
                }
            }
            Err(r) => {
                let c = reason_colour(r);
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let (x, y) = (smp.point.x + dx as f32, smp.point.y + dy as f32);
                        blend(&mut canvas, x, y, c, 1.0);
                    }
                }
            }
        }
    }
    canvas
}
