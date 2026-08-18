//! Example: shape-based object detection.
//!
//! Two modes.
//!
//! **Synthetic** (the default, and a self-asserting integration test):
//! renders three L-brackets plus clutter, finds all three, and checks the
//! recovered poses against the ground truth.
//!
//! ```text
//! cargo run -p vision-metrology --example shape_matching
//! ```
//!
//! **Real data**: builds a model from a rectangular ROI of one image and
//! searches for it in another, writing a side-by-side overlay PNG — model image
//! with its ROI on the left, scene with the located contour on the right.
//!
//! ```text
//! cargo run --release -p vision-metrology --example shape_matching -- \
//!     --model-image ref.bmp --roi 420,350,420,320 --scene scene.bmp \
//!     --out overlay.png --polarity ignore-global
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use image::ImageReader;
use vision_metrology::{
    Image, Point2f, Polarity, Rect2f, Refinement, ShapeMatch, ShapeMatcher, ShapeModel,
    ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig,
};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum PolarityArg {
    Match,
    IgnoreGlobal,
    IgnoreLocal,
}

impl From<PolarityArg> for Polarity {
    fn from(p: PolarityArg) -> Self {
        match p {
            PolarityArg::Match => Polarity::Match,
            PolarityArg::IgnoreGlobal => Polarity::IgnoreGlobal,
            PolarityArg::IgnoreLocal => Polarity::IgnoreLocal,
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Build a shape model from a reference ROI and locate it in a scene")]
struct Args {
    /// Reference image the model is built from. Omit for the synthetic demo.
    #[arg(long)]
    model_image: Option<PathBuf>,

    /// Model ROI as `x,y,width,height` in reference-image pixels.
    #[arg(long, value_parser = parse_roi)]
    roi: Option<Rect2f>,

    /// Scene image to search.
    #[arg(long)]
    scene: Option<PathBuf>,

    /// Search every image in this directory instead of a single scene.
    #[arg(long)]
    scene_dir: Option<PathBuf>,

    /// Overlay PNG to write (single-scene mode).
    #[arg(long, default_value = "shape_match_overlay.png")]
    out: PathBuf,

    /// Directory to write one overlay per frame (batch mode). Omit for stats only.
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Contrast-reversal tolerance.
    #[arg(long, value_enum, default_value_t = PolarityArg::Match)]
    polarity: PolarityArg,

    /// Minimum score in [0, 1].
    #[arg(long, default_value_t = 0.5)]
    min_score: f32,

    /// Maximum instances to report (0 = unlimited).
    #[arg(long, default_value_t = 1)]
    max_matches: usize,

    /// Greedy early termination: 0 is exhaustive, 1 is fastest.
    #[arg(long, default_value_t = 0.9)]
    greediness: f32,

    /// Scene gradient floor, in Scharr response units on the input pixel scale.
    #[arg(long, default_value_t = 10.0)]
    min_contrast: f32,

    /// Model points per pyramid level.
    #[arg(long, default_value_t = 512)]
    max_points: usize,

    /// Gradient floor for a reference-image edge to enter the model.
    ///
    /// Raise it on low-relief parts, where faint surface shading produces
    /// edgels that do not repeat between frames and only dilute the score.
    #[arg(long, default_value_t = 0.0)]
    model_min_contrast: f32,
}

fn parse_roi(s: &str) -> Result<Rect2f, String> {
    let v: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    match v[..] {
        [x, y, width, height] => Ok(Rect2f {
            x,
            y,
            width,
            height,
        }),
        _ => Err("expected x,y,width,height".into()),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match (&args.model_image, args.roi) {
        (Some(m), Some(roi)) => {
            let model = build_model(&args, m, roi)?;
            match (&args.scene, &args.scene_dir) {
                (Some(s), None) => single(&args, m, &model, roi, s),
                (None, Some(d)) => batch(&args, m, &model, roi, d),
                _ => bail!("give exactly one of --scene or --scene-dir"),
            }
        }
        (None, None) if args.scene.is_none() && args.scene_dir.is_none() => synthetic(),
        _ => bail!("--model-image and --roi must be given together"),
    }
}

// ── real data ─────────────────────────────────────────────────────────────────

fn load_gray(path: &Path) -> Result<Image<u8>> {
    let img = ImageReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .decode()
        .with_context(|| format!("decoding {}", path.display()))?
        .to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Image::from_vec(w, h, img.into_raw()).map_err(Into::into)
}

fn build_model(args: &Args, model_path: &Path, roi: Rect2f) -> Result<ShapeModel> {
    let reference = load_gray(model_path)?;
    let cfg = ShapeModelConfig {
        max_points: args.max_points,
        polarity: args.polarity.into(),
        min_contrast: args.model_min_contrast,
        ..Default::default()
    };
    let t0 = Instant::now();
    let model = ShapeModelBuilder::new().build_u8(&reference.as_view(), roi, &cfg)?;
    println!(
        "model from {}: {} levels, built in {:.1} ms",
        model_path.display(),
        model.num_levels(),
        t0.elapsed().as_secs_f64() * 1e3
    );
    for (i, lvl) in model.levels().iter().enumerate() {
        println!(
            "  level {i}: {:4} points, radius {:6.1} px, angle step {:.4} rad",
            lvl.points.len(),
            lvl.radius,
            lvl.angle_step
        );
    }
    Ok(model)
}

fn search_config(args: &Args) -> ShapeSearchConfig {
    ShapeSearchConfig {
        min_score: args.min_score,
        greediness: args.greediness,
        max_matches: args.max_matches,
        min_contrast: args.min_contrast,
        refinement: Refinement::LeastSquares,
        ..Default::default()
    }
}

fn single(
    args: &Args,
    model_path: &Path,
    model: &ShapeModel,
    roi: Rect2f,
    scene_path: &Path,
) -> Result<()> {
    let reference = load_gray(model_path)?;
    let scene = load_gray(scene_path)?;
    let search = search_config(args);

    let mut matcher = ShapeMatcher::new();
    // One warm-up call so the reported time excludes first-touch allocation.
    let _ = matcher.find_u8(&scene.as_view(), model, &search);
    let t1 = Instant::now();
    let matches = matcher.find_u8(&scene.as_view(), model, &search);
    let find_ms = t1.elapsed().as_secs_f64() * 1e3;

    println!(
        "search: {} match(es) in {find_ms:.1} ms{}",
        matches.len(),
        if matcher.truncated() {
            " (candidate list truncated)"
        } else {
            ""
        }
    );
    for m in &matches {
        print_match(m, model.point_count(0));
    }

    draw_overlay(&reference, &scene, roi, model, &matches)
        .save(&args.out)
        .with_context(|| format!("writing {}", args.out.display()))?;
    println!("overlay -> {}", args.out.display());

    if matches.is_empty() {
        bail!("no match above min_score = {}", args.min_score);
    }
    Ok(())
}

fn print_match(m: &ShapeMatch, n: usize) {
    println!(
        "  score {:.3}  support {:4}/{:4}  position ({:8.2}, {:8.2})  angle {:7.2} deg  scale {:.4}",
        m.score,
        m.support,
        n,
        m.position.x,
        m.position.y,
        m.angle().to_degrees(),
        m.scale()
    );
}

/// Run the model against every image in a directory and report the spread.
///
/// This is the acceptance test for real data: a model built from one frame of a
/// production line must be found in every other frame of that line.
fn batch(
    args: &Args,
    model_path: &Path,
    model: &ShapeModel,
    roi: Rect2f,
    dir: &Path,
) -> Result<()> {
    let reference = load_gray(model_path)?;
    let search = search_config(args);
    let mut matcher = ShapeMatcher::new();

    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && !p
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'))
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no images in {}", dir.display());
    }

    if let Some(od) = &args.out_dir {
        std::fs::create_dir_all(od)?;
    }

    let mut scores: Vec<f64> = Vec::new();
    let mut times: Vec<f64> = Vec::new();
    let mut angles: Vec<f64> = Vec::new();
    let mut positions: Vec<Point2f> = Vec::new();
    let mut missed: Vec<PathBuf> = Vec::new();

    for path in &paths {
        let Ok(scene) = load_gray(path) else { continue };
        let t = Instant::now();
        let matches = matcher.find_u8(&scene.as_view(), model, &search);
        times.push(t.elapsed().as_secs_f64() * 1e3);

        match matches.first() {
            Some(m) => {
                scores.push(f64::from(m.score));
                angles.push(f64::from(m.angle().to_degrees()));
                positions.push(m.position);
            }
            None => missed.push(path.clone()),
        }
        if let Some(od) = &args.out_dir {
            let name = path.file_stem().unwrap_or_default().to_string_lossy();
            let out = od.join(format!("{name}.png"));
            draw_overlay(&reference, &scene, roi, model, &matches).save(&out)?;
        }
    }

    let n = paths.len();
    println!(
        "\n{} / {} frames found in {}",
        scores.len(),
        n,
        dir.display()
    );
    if !scores.is_empty() {
        println!(
            "  score    min {:.3}  median {:.3}  max {:.3}",
            min_of(&scores),
            median(&mut scores.clone()),
            max_of(&scores)
        );
        println!(
            "  angle    min {:7.2} deg  max {:7.2} deg  (span {:.1} deg)",
            min_of(&angles),
            max_of(&angles),
            max_of(&angles) - min_of(&angles)
        );
        let xs: Vec<f64> = positions.iter().map(|p| f64::from(p.x)).collect();
        let ys: Vec<f64> = positions.iter().map(|p| f64::from(p.y)).collect();
        println!(
            "  position x {:.1} +- {:.1} px,  y {:.1} +- {:.1} px",
            mean(&xs),
            0.5 * (max_of(&xs) - min_of(&xs)),
            mean(&ys),
            0.5 * (max_of(&ys) - min_of(&ys))
        );
    }
    println!(
        "  time     median {:.1} ms  max {:.1} ms",
        median(&mut times.clone()),
        max_of(&times)
    );
    for p in &missed {
        println!("  MISS {}", p.display());
    }
    if let Some(od) = &args.out_dir {
        println!("  overlays -> {}", od.display());
    }
    Ok(())
}

fn min_of(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_of(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v[v.len() / 2]
}

// ── overlay rendering ─────────────────────────────────────────────────────────

const GREEN: [u8; 3] = [40, 230, 60];
const YELLOW: [u8; 3] = [250, 210, 40];
const RED: [u8; 3] = [240, 60, 60];
const CYAN: [u8; 3] = [60, 200, 240];

/// Side-by-side: reference with its ROI on the left, scene with the located
/// model contour on the right.
fn draw_overlay(
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

    // Left: the ROI the model was cut from, and the model points themselves.
    rect(&mut canvas, roi, 0, CYAN);
    for p in model.reference_points() {
        dot(&mut canvas, p.x, p.y, 0, GREEN);
    }
    cross(&mut canvas, model.origin().x, model.origin().y, 0, RED);

    // Right: each located instance.
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
        cross(&mut canvas, m.position.x, m.position.y, w, RED);
    }
    canvas
}

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

fn rect_edges(r: Rect2f) -> [(Point2f, Point2f); 4] {
    let (x0, y0) = (r.x, r.y);
    let (x1, y1) = (r.x + r.width, r.y + r.height);
    let p = |x, y| Point2f { x, y };
    [
        (p(x0, y0), p(x1, y0)),
        (p(x1, y0), p(x1, y1)),
        (p(x1, y1), p(x0, y1)),
        (p(x0, y1), p(x0, y0)),
    ]
}

fn rect(canvas: &mut image::RgbImage, r: Rect2f, x_off: usize, c: [u8; 3]) {
    for (a, b) in rect_edges(r) {
        line(canvas, a.x, a.y, b.x, b.y, x_off, c);
    }
}

// ── synthetic demo ────────────────────────────────────────────────────────────

const SW: usize = 640;
const SH: usize = 480;

fn sdf_box(p: (f32, f32), half: (f32, f32)) -> f32 {
    let dx = p.0.abs() - half.0;
    let dy = p.1.abs() - half.1;
    (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() + dx.max(dy).min(0.0)
}

fn sdf_bracket(p: (f32, f32)) -> f32 {
    sdf_box((p.0, p.1 + 30.0), (40.0, 10.0)).min(sdf_box((p.0 + 30.0, p.1), (10.0, 40.0)))
}

fn stamp(data: &mut [u8], cx: f32, cy: f32, angle: f32, sdf: &dyn Fn((f32, f32)) -> f32) {
    let (sn, cs) = angle.sin_cos();
    for y in 0..SH {
        for x in 0..SW {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let m = (cs * dx + sn * dy, -sn * dx + cs * dy);
            let t = ((-sdf(m) + 1.0) / 2.0).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            if t > 0.0 {
                let v = (40.0 + 170.0 * t).round() as u8;
                data[y * SW + x] = data[y * SW + x].max(v);
            }
        }
    }
}

fn synthetic() -> Result<()> {
    let truth = [
        (150.0f32, 130.0f32, 0.0f32),
        (460.0, 150.0, 1.1),
        (300.0, 350.0, -2.4),
    ];

    let mut refdata = vec![40u8; SW * SH];
    stamp(&mut refdata, 150.0, 130.0, 0.0, &sdf_bracket);
    let reference = Image::from_vec(SW, SH, refdata)?;
    let roi = Rect2f {
        x: 95.0,
        y: 75.0,
        width: 112.0,
        height: 112.0,
    };

    let mut scene_data = vec![40u8; SW * SH];
    for &(cx, cy, a) in &truth {
        stamp(&mut scene_data, cx, cy, a, &sdf_bracket);
    }
    // Clutter: bars and a disc that share edge orientations with the model.
    stamp(&mut scene_data, 560.0, 400.0, 0.4, &|p| {
        sdf_box(p, (50.0, 9.0))
    });
    stamp(&mut scene_data, 80.0, 400.0, -0.9, &|p| {
        sdf_box(p, (9.0, 55.0))
    });
    stamp(&mut scene_data, 470.0, 300.0, 0.0, &|p| {
        (p.0 * p.0 + p.1 * p.1).sqrt() - 35.0
    });
    let scene = Image::from_vec(SW, SH, scene_data)?;

    let model = ShapeModelBuilder::new().build_u8(
        &reference.as_view(),
        roi,
        &ShapeModelConfig::default(),
    )?;
    println!(
        "model: {} levels, {} points at level 0",
        model.num_levels(),
        model.point_count(0)
    );

    let cfg = ShapeSearchConfig {
        min_score: 0.6,
        max_matches: 0,
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };
    let t0 = Instant::now();
    let matches = ShapeMatcher::new().find_u8(&scene.as_view(), &model, &cfg);
    println!(
        "found {} instances in {:.1} ms",
        matches.len(),
        t0.elapsed().as_secs_f64() * 1e3
    );

    assert_eq!(matches.len(), 3, "expected exactly the three brackets");
    for &(cx, cy, angle) in &truth {
        // The reference bracket was stamped with its own coordinate origin at
        // (150, 130), so that point must map onto the instance's centre.
        let hit = matches.iter().any(|m| {
            let q = m.pose * nalgebra::Point2::new(150.0f32, 130.0);
            (q.x - cx).abs() < 1.5
                && (q.y - cy).abs() < 1.5
                && vision_metrology::wrap_angle(m.angle() - angle).abs() < 0.05
        });
        assert!(hit, "no match for ground truth ({cx}, {cy}, {angle})");
        println!("  ok: ({cx:.0}, {cy:.0}) at {:.1} deg", angle.to_degrees());
    }
    for m in &matches {
        println!(
            "  score {:.3}  position ({:6.2}, {:6.2})  angle {:7.2} deg",
            m.score,
            m.position.x,
            m.position.y,
            m.angle().to_degrees()
        );
    }
    println!("All assertions passed.");
    Ok(())
}
