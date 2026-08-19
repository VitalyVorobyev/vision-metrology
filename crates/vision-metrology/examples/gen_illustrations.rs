//! Deterministically render the doc illustrations in `docs/assets/`.
//!
//! Every image comes from a **synthetic fixture** run through the real
//! algorithm — no private dataset frames are ever committed (the canend data
//! is private). Re-run this after a change to any module it exercises:
//!
//! ```text
//! cargo run --release --example gen_illustrations --all-features
//! ```
//!
//! Drawing reuses the public helpers in `examples/common/overlay.rs`
//! (`blit`, `dot`, `line`, `rect`, `put_px`, the colour palette) plus a tiny
//! local bitmap font for score/level labels. Palette and line weights are
//! deliberately plain — legible and correct first, polish later.

use std::path::Path;

use vision_metrology::contour::{
    Connectivity, ContourBuildConfig, NodeKind, build_graph_from_edgels,
};
use vision_metrology::fit::{FitConfig, RobustLoss, fit_circle};
use vision_metrology::laser::{LaserExtractConfig, LaserExtractor};
use vision_metrology::matching::{
    Refinement, ShapeMatcher, ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig,
};
use vision_metrology::measure::{Caliper, MeasureConfig, MeasureRect};
use vision_metrology::{Edgel, Image, Point2f, Pyramid, Rect2f, Vec2f, Vec2fExt};

#[path = "common/overlay.rs"]
mod overlay;
use overlay::{CYAN, GREEN, ORANGE, RED, YELLOW, blit, dot, line, put_px};

const OUT_DIR: &str = "docs/assets";
const BG: image::Rgb<u8> = image::Rgb([24, 24, 28]);

fn main() {
    std::fs::create_dir_all(OUT_DIR).expect("create docs/assets");
    shape_matching_illustration();
    caliper_anatomy_illustration();
    laser_stripe_illustration();
    circle_fit_illustration();
    contour_graph_illustration();
    pyramid_levels_illustration();
}

fn save(img: &image::RgbImage, name: &str) {
    let path = Path::new(OUT_DIR).join(name);
    img.save(&path)
        .unwrap_or_else(|e| panic!("saving {}: {e}", path.display()));
    println!(
        "wrote {} ({}x{})",
        path.display(),
        img.width(),
        img.height()
    );
}

// ── (a) shape matching ──────────────────────────────────────────────────────

fn sdf_box(p: (f32, f32), half: (f32, f32)) -> f32 {
    let dx = p.0.abs() - half.0;
    let dy = p.1.abs() - half.1;
    (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() + dx.max(dy).min(0.0)
}

fn sdf_bracket(p: (f32, f32)) -> f32 {
    sdf_box((p.0, p.1 + 30.0), (40.0, 10.0)).min(sdf_box((p.0 + 30.0, p.1), (10.0, 40.0)))
}

fn stamp(
    data: &mut [u8],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    angle: f32,
    sdf: &dyn Fn((f32, f32)) -> f32,
) {
    let (sn, cs) = angle.sin_cos();
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let m = (cs * dx + sn * dy, -sn * dx + cs * dy);
            let t = ((-sdf(m) + 1.0) / 2.0).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            if t > 0.0 {
                let v = (35.0 + 175.0 * t).round() as u8;
                data[y * w + x] = data[y * w + x].max(v);
            }
        }
    }
}

/// Model contour overlaid on a synthetic scene, with found poses and score
/// labels — the same L-bracket fixture `examples/shape_matching.rs` uses for
/// its self-asserting synthetic demo.
fn shape_matching_illustration() {
    const W: usize = 640;
    const H: usize = 480;

    let mut refdata = vec![35u8; W * H];
    stamp(&mut refdata, W, H, 150.0, 130.0, 0.0, &sdf_bracket);
    let reference = Image::from_vec(W, H, refdata).expect("valid image");
    let roi = Rect2f {
        x: 95.0,
        y: 75.0,
        width: 112.0,
        height: 112.0,
    };

    let truth = [(150.0f32, 130.0f32, 0.0f32), (460.0, 150.0, 1.1)];
    let mut scene_data = vec![35u8; W * H];
    for &(cx, cy, a) in &truth {
        stamp(&mut scene_data, W, H, cx, cy, a, &sdf_bracket);
    }
    // Light clutter: a bar and a disc that share edge orientations with the
    // model but do not match its shape.
    stamp(&mut scene_data, W, H, 80.0, 400.0, -0.9, &|p| {
        sdf_box(p, (9.0, 55.0))
    });
    stamp(&mut scene_data, W, H, 470.0, 320.0, 0.0, &|p| {
        (p.0 * p.0 + p.1 * p.1).sqrt() - 35.0
    });
    let scene = Image::from_vec(W, H, scene_data).expect("valid image");

    let model = ShapeModelBuilder::new()
        .build(&reference.as_view(), roi, &ShapeModelConfig::default())
        .expect("model build");
    let cfg = ShapeSearchConfig {
        min_score: 0.6,
        max_matches: None,
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };
    let matches = ShapeMatcher::new().find(&scene.as_view(), &model, &cfg);
    assert_eq!(matches.len(), 2, "expected both bracket instances");

    let mut canvas = image::RgbImage::from_pixel(W as u32, H as u32, BG);
    blit(&mut canvas, &scene, 0);
    for m in &matches {
        for p in model.reference_points() {
            let q = m.pose * nalgebra::Point2::new(p.x, p.y);
            dot(&mut canvas, q.x, q.y, 0, GREEN);
        }
        draw_text(
            &mut canvas,
            m.position.x - 16.0,
            m.position.y - 60.0,
            &format!("{:.2}", m.score),
            ORANGE,
            2.0,
        );
    }
    save(&canvas, "shape-matching.png");
}

// ── (b) caliper anatomy ──────────────────────────────────────────────────────

/// A rect caliper box over a synthetic vertical step edge, with the
/// cross-averaged 1-D profile plotted alongside it and the detected subpixel
/// edge marked on both panels.
fn caliper_anatomy_illustration() {
    const RW: usize = 200;
    const RH: usize = 200;

    // A softly anti-aliased step (~2 px transition), so the plotted profile
    // shows a real subpixel edge rather than an instantaneous jump.
    let mut data = vec![0u8; RW * RH];
    for row in data.chunks_mut(RW) {
        for (x, v) in row.iter_mut().enumerate() {
            let t = ((x as f32 - 99.5) / 2.0 + 0.5).clamp(0.0, 1.0);
            *v = (30.0 + 190.0 * t).round() as u8;
        }
    }
    let img = Image::from_vec(RW, RH, data).expect("valid image");

    let rect = MeasureRect {
        center: Point2f::new(100.0, 100.0),
        angle: 0.0,
        half_len: 70.0,
        half_width: 30.0,
    };
    let mut cal = Caliper::rect(rect, MeasureConfig::default());
    let edge = cal.measure(&img.as_view()).expect("an edge")[0];
    let profile = cal.profile().to_vec();

    let panel_w = 220usize;
    let gutter = 16usize;
    let total_w = RW + gutter + panel_w;
    let mut canvas = image::RgbImage::from_pixel(total_w as u32, RH as u32, BG);

    // Left panel: raster crop with the caliper box.
    blit(&mut canvas, &img, 0);
    let (u, n) = (Vec2f::new(1.0, 0.0), Vec2f::new(0.0, 1.0)); // rect.angle == 0
    let (hl, hw) = (rect.half_len, rect.half_width);
    let corners = [
        rect.center - u * hl - n * hw,
        rect.center + u * hl - n * hw,
        rect.center + u * hl + n * hw,
        rect.center - u * hl + n * hw,
    ];
    for i in 0..4 {
        let (a, b) = (corners[i], corners[(i + 1) % 4]);
        line(&mut canvas, a.x, a.y, b.x, b.y, 0, CYAN);
    }
    let arrow_tip = rect.center + u * (hl * 0.9);
    line(
        &mut canvas,
        rect.center.x,
        rect.center.y,
        arrow_tip.x,
        arrow_tip.y,
        0,
        ORANGE,
    );
    dot(&mut canvas, edge.p.x, edge.p.y, 0, RED);

    // Right panel: the cross-averaged profile, plotted intensity vs. position
    // along the scan axis.
    let px0 = RW + gutter;
    let n_samples = profile.len();
    let sample_xy = |i: usize| -> (f32, f32) {
        let x = px0 as f32 + i as f32 / (n_samples - 1) as f32 * panel_w as f32;
        let y = RH as f32 - 1.0 - (profile[i] / 255.0) * (RH as f32 - 1.0);
        (x, y)
    };
    for i in 0..n_samples - 1 {
        let (x0, y0) = sample_xy(i);
        let (x1, y1) = sample_xy(i + 1);
        line(&mut canvas, x0, y0, x1, y1, 0, YELLOW);
    }
    // The edge, at the same horizontal fraction along the panel as it sits
    // along the caliper's own axis.
    let frac = (edge.t + hl) / (2.0 * hl);
    let ex = px0 as f32 + frac * panel_w as f32;
    line(&mut canvas, ex, 0.0, ex, RH as f32 - 1.0, 0, RED);

    save(&canvas, "caliper-anatomy.png");
}

// ── (c) laser stripe extraction ─────────────────────────────────────────────

/// A synthetic laser stripe (Gaussian cross-section, wandering centre) with
/// the extracted subpixel centerline drawn over it.
fn laser_stripe_illustration() {
    const W: usize = 480;
    const H: usize = 220;

    let mut data = vec![18u8; W * H];
    for y in 0..H {
        let cx = 240.0 + 90.0 * (y as f32 / 40.0).sin();
        for x in 0..W {
            let d = x as f32 - cx;
            let v = 18.0 + 220.0 * (-(d * d) / (2.0 * 2.2 * 2.2)).exp();
            data[y * W + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    let img = Image::from_vec(W, H, data).expect("valid image");

    let mut extractor = LaserExtractor::new(1.2);
    let stripe = extractor
        .extract_line(&img.as_view(), 0..H, &LaserExtractConfig::default(), None)
        .expect("row scan never fails");
    assert!(
        stripe.points.len() > H / 2,
        "expected most rows to find the stripe, got {}",
        stripe.points.len()
    );

    let mut canvas = image::RgbImage::from_pixel(W as u32, H as u32, BG);
    blit(&mut canvas, &img, 0);
    for w in stripe.points.windows(2) {
        line(&mut canvas, w[0].x, w[0].y, w[1].x, w[1].y, 0, GREEN);
    }
    for (i, p) in stripe.points.iter().enumerate() {
        if i % 8 == 0 {
            dot(&mut canvas, p.x, p.y, 0, ORANGE);
        }
    }
    save(&canvas, "laser-stripe.png");
}

// ── (d) robust circle fit ───────────────────────────────────────────────────

/// A tiny seeded LCG — deterministic noise without an external RNG
/// dependency (invariant 12: no ambient randomness in what these fixtures
/// produce).
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f32 {
        // xorshift64*
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Noisy points on a circle plus gross outliers, a Tukey-robust fit, and
/// residual whiskers from each inlier to the fitted circle.
fn circle_fit_illustration() {
    const W: usize = 360;
    const H: usize = 300;

    let center = Point2f::new(180.0, 150.0);
    let radius = 100.0f32;
    let mut rng = Lcg(0x2545_f491_4f6c_dd1d);

    let mut pts = Vec::new();
    for i in 0..48 {
        let phi = i as f32 / 48.0 * core::f32::consts::TAU;
        let noise = (rng.next_unit() - 0.5) * 7.0;
        pts.push(center + Vec2f::new(phi.cos(), phi.sin()) * (radius + noise));
    }
    // Gross outliers, well off the ring in either direction (~40-60 px
    // residual) rather than near-misses — Tukey's rejection radius is only
    // a few pixels.
    for &(dx, dy) in &[(140.0, -40.0), (-40.0, 15.0), (10.0, 140.0)] {
        pts.push(center + Vec2f::new(dx, dy));
    }

    let cfg = FitConfig {
        loss: RobustLoss::Tukey { c: 3.0 },
        ..FitConfig::default()
    };
    let fit = fit_circle(&pts, &cfg).expect("enough points to fit");
    assert!(
        (fit.model.radius - radius).abs() < 2.0,
        "the robust fit should land close to the true radius, got {}",
        fit.model.radius
    );

    let mut canvas = image::RgbImage::from_pixel(W as u32, H as u32, BG);
    let steps = 240;
    for i in 0..steps {
        let phi = i as f32 / steps as f32 * core::f32::consts::TAU;
        let p = fit.model.center + Vec2f::new(phi.cos(), phi.sin()) * fit.model.radius;
        put_px(&mut canvas, p.x, p.y, CYAN);
    }
    let inliers: std::collections::HashSet<u32> = fit.inliers.iter().copied().collect();
    for (i, p) in pts.iter().enumerate() {
        let is_inlier = fit.inliers.is_empty() || inliers.contains(&(i as u32));
        dot(
            &mut canvas,
            p.x,
            p.y,
            0,
            if is_inlier { GREEN } else { RED },
        );
        if is_inlier {
            let dir = (p - fit.model.center).normalized_or_zero();
            let on_circle = fit.model.center + dir * fit.model.radius;
            line(&mut canvas, p.x, p.y, on_circle.x, on_circle.y, 0, YELLOW);
        }
    }
    save(&canvas, "circle-fit.png");
}

// ── (e) contour graph ────────────────────────────────────────────────────────

/// A hand-built T-junction: a horizontal chain and a vertical chain sharing
/// one pixel, so the traced graph has one genuine degree-3 junction node and
/// three degree-1 endpoints. Built directly as edgels (as
/// `contour::build`'s own geometry tests do) rather than through raster edge
/// detection, because the boundary of *any* simply-connected filled blob —
/// however T-shaped its silhouette — traces as a single loop with no
/// branching; a real junction needs a genuine 1-pixel-wide branching curve.
fn contour_graph_illustration() {
    const W: usize = 260;
    const H: usize = 280;

    let e = |x: i32, y: i32| Edgel {
        p: Point2f::new(x as f32, y as f32),
        n: Vec2f::new(1.0, 0.0),
        strength: 1.0,
        idx: (x as usize, y as usize),
    };
    let mut edgels = Vec::new();
    for x in 20..=220 {
        edgels.push(e(x, 100));
    }
    // Vertical stem, sharing pixel (120, 100) with the bar rather than
    // duplicating it.
    for y in 101..=260 {
        edgels.push(e(120, y));
    }

    let cfg = ContourBuildConfig {
        connectivity: Connectivity::C8,
        min_component_size: 2,
        record_strengths: false,
        record_geometry: false,
        thin: false, // already exactly one pixel wide by construction
    };
    let graph = build_graph_from_edgels(W, H, &edgels, &cfg);
    assert_eq!(
        graph.num_junctions(),
        1,
        "a T has exactly one branching node, got {}",
        graph.num_junctions()
    );
    assert_eq!(
        graph.edges.len(),
        3,
        "three arms meet at the junction, got {}",
        graph.edges.len()
    );

    let mut canvas = image::RgbImage::from_pixel(W as u32, H as u32, BG);
    let palette = [GREEN, CYAN, YELLOW];
    for (i, ge) in graph.edges.iter().enumerate() {
        let c = palette[i % palette.len()];
        for w in ge.points.windows(2) {
            line(&mut canvas, w[0].x, w[0].y, w[1].x, w[1].y, 0, c);
        }
    }
    for node in &graph.nodes {
        let c = if node.kind == NodeKind::Junction {
            RED
        } else {
            ORANGE
        };
        dot(&mut canvas, node.p.x, node.p.y, 0, c);
    }
    save(&canvas, "contour-graph.png");
}

// ── (f) pyramid levels ───────────────────────────────────────────────────────

/// A checkerboard-and-rings fixture (so every level keeps visible structure)
/// carried through a 5-level pyramid, levels laid side by side.
fn pyramid_levels_illustration() {
    const W: usize = 256;
    const H: usize = 256;

    let data: Vec<u8> = (0..H)
        .flat_map(|y| {
            (0..W).map(move |x| {
                let (fx, fy) = (x as f32 - 128.0, y as f32 - 128.0);
                let r = (fx * fx + fy * fy).sqrt();
                let ring = (0.5 + 0.5 * (r / 9.0).sin()) * 110.0;
                let checker = if (x / 16 + y / 16).is_multiple_of(2) {
                    40.0
                } else {
                    0.0
                };
                (55.0 + ring + checker).round().clamp(0.0, 255.0) as u8
            })
        })
        .collect();
    let img = Image::from_vec(W, H, data).expect("valid image");

    let mut pyr = Pyramid::new();
    pyr.build(&img.as_view(), 5);

    const GAP: usize = 10;
    const LABEL_H: usize = 20;
    let total_w: usize = (0..pyr.num_levels())
        .map(|i| pyr.level(i).expect("level exists").width() + GAP)
        .sum();
    let strip_h = H + LABEL_H;
    let mut canvas = image::RgbImage::from_pixel(total_w as u32, strip_h as u32, BG);

    let mut x_off = 0usize;
    for i in 0..pyr.num_levels() {
        let level = pyr.level(i).expect("level exists");
        let (lw, lh) = (level.width(), level.height());
        for y in 0..lh {
            for x in 0..lw {
                let v = level.data()[y * lw + x].round().clamp(0.0, 255.0) as u8;
                canvas.put_pixel((x_off + x) as u32, y as u32, image::Rgb([v, v, v]));
            }
        }
        draw_text(
            &mut canvas,
            x_off as f32 + 2.0,
            (H + 4) as f32,
            &format!("L{i}"),
            ORANGE,
            2.0,
        );
        x_off += lw + GAP;
    }
    save(&canvas, "pyramid-levels.png");
}

// ── tiny bitmap font, for score/level labels ────────────────────────────────

const FONT_COLS: usize = 3;

fn glyph(c: char) -> [&'static str; 5] {
    match c {
        '0' => ["111", "101", "101", "101", "111"],
        '1' => ["010", "110", "010", "010", "111"],
        '2' => ["111", "001", "111", "100", "111"],
        '3' => ["111", "001", "111", "001", "111"],
        '4' => ["101", "101", "111", "001", "001"],
        '5' => ["111", "100", "111", "001", "111"],
        '6' => ["111", "100", "111", "101", "111"],
        '7' => ["111", "001", "010", "010", "010"],
        '8' => ["111", "101", "111", "101", "111"],
        '9' => ["111", "101", "111", "001", "111"],
        '.' => ["000", "000", "000", "000", "010"],
        'L' => ["100", "100", "100", "100", "111"],
        _ => ["000", "000", "000", "000", "000"],
    }
}

/// Draw `text` at top-left `(x, y)`, each glyph pixel scaled to a `scale`
/// device-pixel block.
fn draw_text(canvas: &mut image::RgbImage, x: f32, y: f32, text: &str, color: [u8; 3], scale: f32) {
    let mut cx = x;
    for ch in text.chars() {
        for (row, bits) in glyph(ch).iter().enumerate() {
            for (col, bit) in bits.chars().enumerate() {
                if bit != '1' {
                    continue;
                }
                let (px0, py0) = (cx + col as f32 * scale, y + row as f32 * scale);
                let n = scale.ceil().max(1.0) as i32;
                for dy in 0..n {
                    for dx in 0..n {
                        put_px(canvas, px0 + dx as f32, py0 + dy as f32, color);
                    }
                }
            }
        }
        cx += (FONT_COLS as f32 + 1.0) * scale;
    }
}
