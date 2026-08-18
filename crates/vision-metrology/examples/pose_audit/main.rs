//! Pose auditing for shape-based matching, on real image folders.
//!
//! Two subcommands:
//!
//! - `audit` — run the shape matcher over a folder and, per frame, compute an
//!   **independent ZNCC score** of the recovered pose via `corrmatch` (a
//!   different algorithm over different data: raw intensities instead of
//!   gradient directions). Renders three-panel diagnostic overlays
//!   (pose / checkerboard registration / per-point contributions) for the
//!   first frames and every miss. With `--rim-radius`, also measures
//!   **repeatability** by fitting the can rim with the RANSAC ellipse fitter
//!   and expressing the tab pose in rim-centred coordinates.
//!
//! - `xcheck` — run corrmatch's own full rotation search next to the shape
//!   matcher on the same frames and report position/angle disagreement
//!   statistics. Catches systematic pose bias neither library can see about
//!   itself.
//!
//! ```text
//! cargo run --release -p vision-metrology --example pose_audit -- audit \
//!   --model-image ref.bmp --roi 420,350,420,320 --scene-dir frames/ \
//!   --model-min-contrast 400 --out-dir audit_out --rim-radius 367
//! ```

#[path = "../common/corr.rs"]
mod corr;
#[path = "../common/overlay.rs"]
mod overlay;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use corrmatch::{
    CompileConfig, MatchConfig as CorrMatchConfig, Matcher as CorrMatcher, RotationMode,
    Template as CorrTemplate,
};
use vision_metrology::{
    ConicFitConfig, ConicFitter, Edge2DConfig, Edge2DDetector, Image, Point2f, Polarity, Rect2f,
    ShapeMatch, ShapeMatcher, ShapeModel, ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig,
    match_point_scores, wrap_angle,
};

#[derive(Parser)]
#[command(about = "Audit shape-matching poses with independent checks")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Per-frame ZNCC + diagnostic overlays + optional rim repeatability.
    Audit(AuditArgs),
    /// corrmatch full search vs the shape matcher, disagreement stats.
    Xcheck(XcheckArgs),
}

#[derive(Args)]
struct CommonArgs {
    /// Reference image the model is built from.
    #[arg(long)]
    model_image: PathBuf,
    /// Model ROI as `x,y,width,height` in the reference image.
    #[arg(long, value_parser = parse_roi)]
    roi: Rect2f,
    /// Directory of scene images.
    #[arg(long)]
    scene_dir: PathBuf,
    /// Gradient floor for model points (0 = detector auto).
    #[arg(long, default_value_t = 0.0)]
    model_min_contrast: f32,
    /// Search polarity: match | ignore-global | ignore-local.
    #[arg(long, default_value = "match")]
    polarity: String,
    /// Minimum accepted score.
    #[arg(long, default_value_t = 0.5)]
    min_score: f32,
}

#[derive(Args)]
struct AuditArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Output directory for diagnostic overlays.
    #[arg(long, default_value = "pose_audit_out")]
    out_dir: PathBuf,
    /// Render diagnostic overlays for the first N found frames (misses always).
    #[arg(long, default_value_t = 3)]
    diag_count: usize,
    /// Approximate rim radius in px; enables rim-relative repeatability.
    #[arg(long)]
    rim_radius: Option<f32>,
}

#[derive(Args)]
struct XcheckArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Check at most this many frames (corrmatch full search is ~0.1 s/frame).
    #[arg(long, default_value_t = 20)]
    max_frames: usize,
}

fn parse_roi(s: &str) -> Result<Rect2f, String> {
    let v: Vec<f32> = s
        .split(',')
        .map(|t| t.trim().parse::<f32>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    if v.len() != 4 {
        return Err("expected x,y,width,height".into());
    }
    Ok(Rect2f {
        x: v[0],
        y: v[1],
        width: v[2],
        height: v[3],
    })
}

fn parse_polarity(s: &str) -> Result<Polarity> {
    Ok(match s {
        "match" => Polarity::Match,
        "ignore-global" => Polarity::IgnoreGlobal,
        "ignore-local" => Polarity::IgnoreLocal,
        other => bail!("unknown polarity {other:?}"),
    })
}

fn load_gray(path: &Path) -> Result<Image<u8>> {
    let img = image::open(path)
        .with_context(|| format!("loading {}", path.display()))?
        .to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Image::from_vec(w, h, img.into_raw()).context("image buffer")
}

fn frames_in(dir: &Path) -> Result<Vec<PathBuf>> {
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
    Ok(paths)
}

struct Setup {
    reference: Image<u8>,
    model: ShapeModel,
    search: ShapeSearchConfig,
    roi: Rect2f,
}

fn setup(common: &CommonArgs) -> Result<Setup> {
    let reference = load_gray(&common.model_image)?;
    let polarity = parse_polarity(&common.polarity)?;
    let model_cfg = ShapeModelConfig {
        min_contrast: common.model_min_contrast,
        polarity,
        ..Default::default()
    };
    let model = ShapeModelBuilder::new()
        .build_u8(&reference.as_view(), common.roi, &model_cfg)
        .map_err(|e| anyhow::anyhow!("model build: {e}"))?;
    let search = ShapeSearchConfig {
        min_score: common.min_score,
        ..Default::default()
    };
    Ok(Setup {
        reference,
        model,
        search,
        roi: common.roi,
    })
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Audit(a) => audit(a),
        Cmd::Xcheck(a) => xcheck(a),
    }
}

// ── audit ───────────────────────────────────────────────────────────────────

fn audit(args: AuditArgs) -> Result<()> {
    let s = setup(&args.common)?;
    let frames = frames_in(&args.common.scene_dir)?;
    std::fs::create_dir_all(&args.out_dir)?;

    let roi_center = Point2f {
        x: s.roi.x + 0.5 * s.roi.width,
        y: s.roi.y + 0.5 * s.roi.height,
    };

    let mut matcher = ShapeMatcher::new();
    let mut det = Edge2DDetector::new();
    let mut fitter = ConicFitter::new();

    let mut scores = Vec::new();
    let mut znccs = Vec::new();
    let mut times = Vec::new();
    let mut rim_r = Vec::new();
    let mut rim_dphi = Vec::new();
    let mut missed = Vec::new();
    let mut rendered = 0usize;

    println!(
        "{:>4}  {:>7} {:>7}  {:>8} {:>8}  {:>7}  frame",
        "#", "score", "zncc", "angle", "support", "time"
    );

    for (i, path) in frames.iter().enumerate() {
        let scene = load_gray(path)?;
        let t = Instant::now();
        let found = matcher.find_u8(&scene.as_view(), &s.model, &s.search);
        let dt = t.elapsed().as_secs_f64() * 1e3;
        times.push(dt);
        let name = path.file_stem().unwrap_or_default().to_string_lossy();

        let Some(m) = found.first() else {
            println!(
                "{i:>4}  {:>7} {:>7}  {:>8} {:>8}  {dt:6.1}m  {name}  MISS",
                "-", "-", "-", "-"
            );
            missed.push(path.clone());
            let ov = overlay::overview(&s.reference, &scene, s.roi, &s.model, &[]);
            ov.save(args.out_dir.join(format!("{name}_MISS.png")))?;
            continue;
        };

        // Independent ZNCC of this pose.
        let c = m.pose * nalgebra::Point2::new(roi_center.x, roi_center.y);
        let zncc = corr::zncc_at_pose(
            &scene,
            &s.reference,
            s.roi,
            Point2f { x: c.x, y: c.y },
            m.angle().to_degrees(),
        );

        scores.push(f64::from(m.score));
        if let Some(z) = zncc {
            znccs.push(f64::from(z));
        }

        // Rim-relative repeatability.
        if let Some(r_hint) = args.rim_radius
            && let Some((rr, dphi)) = rim_relative(&mut det, &mut fitter, &scene, m, r_hint)
        {
            rim_r.push(f64::from(rr));
            rim_dphi.push(f64::from(dphi));
        }

        println!(
            "{i:>4}  {:7.3} {:>7}  {:7.2}° {:>5}/{:<3}  {dt:6.1}m  {name}",
            m.score,
            zncc.map_or("-".into(), |z| format!("{z:.3}")),
            m.angle().to_degrees(),
            m.support,
            s.model.point_count(0),
            i = i
        );

        if rendered < args.diag_count {
            let terms = match_point_scores(&scene.as_view(), &s.model, m, s.search.min_contrast);
            let diag = overlay::diagnostic(&s.reference, &scene, &s.model, m, &terms);
            diag.save(args.out_dir.join(format!("{name}_diag.png")))?;
            let ov = overlay::overview(&s.reference, &scene, s.roi, &s.model, &found);
            ov.save(args.out_dir.join(format!("{name}_overview.png")))?;
            rendered += 1;
        }
    }

    println!(
        "\n{} / {} frames found; misses: {}",
        scores.len(),
        frames.len(),
        missed.len()
    );
    summary("shape score", &scores);
    summary("zncc", &znccs);
    summary("time ms", &times);
    if !rim_r.is_empty() {
        summary("rim: tab radius px", &rim_r);
        summary("rim: phi - angle deg", &rim_dphi);
        println!(
            "  repeatability: sigma(radius) = {:.3} px, sigma(phi-angle) = {:.3} deg over {} frames",
            std_dev(&rim_r),
            std_dev(&rim_dphi),
            rim_r.len()
        );
    }
    println!("overlays -> {}", args.out_dir.display());
    Ok(())
}

/// Fit the rim and express the tab position relative to it.
///
/// Returns `(|position − rim_center|, atan2(position − rim_center) − angle)`,
/// both invariant to where the can sits in the frame. Their spread across
/// frames of one folder is a real repeatability number for the combined
/// rim + tab measurement.
fn rim_relative(
    det: &mut Edge2DDetector,
    fitter: &mut ConicFitter,
    scene: &Image<u8>,
    m: &ShapeMatch,
    r_hint: f32,
) -> Option<(f32, f32)> {
    let edgels = det.detect_u8(&scene.as_view(), &Edge2DConfig::default());
    // The rim circles the image centre region; seed with the frame centre.
    let seed = Point2f {
        x: scene.width() as f32 * 0.5,
        y: scene.height() as f32 * 0.5,
    };
    let band: Vec<Point2f> = edgels
        .iter()
        .map(|e| e.p)
        .filter(|p| {
            let d = ((p.x - seed.x).powi(2) + (p.y - seed.y).powi(2)).sqrt();
            (d - r_hint).abs() < 0.08 * r_hint
        })
        .collect();
    if band.len() < 50 {
        return None;
    }
    let ellipse = fitter
        .fit_ellipse_ransac(&band, &ConicFitConfig::default())
        .ok()?;
    let dx = m.position.x - ellipse.center.x;
    let dy = m.position.y - ellipse.center.y;
    let r = (dx * dx + dy * dy).sqrt();
    let phi = dy.atan2(dx);
    Some((r, wrap_angle(phi - m.angle()).to_degrees()))
}

// ── xcheck ──────────────────────────────────────────────────────────────────

fn xcheck(args: XcheckArgs) -> Result<()> {
    let s = setup(&args.common)?;
    let frames = frames_in(&args.common.scene_dir)?;
    let n = frames.len().min(args.max_frames);

    let roi_center = Point2f {
        x: s.roi.x + 0.5 * s.roi.width,
        y: s.roi.y + 0.5 * s.roi.height,
    };

    // corrmatch: template = the reference ROI patch, full rotation search.
    let (tpl, tw, th) = corr::roi_patch(&s.reference, s.roi);
    let template = CorrTemplate::new(tpl, tw, th).map_err(|e| anyhow::anyhow!("{e}"))?;
    let compiled = template
        .compile(CompileConfig::default())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut corr_cfg = CorrMatchConfig::default();
    corr_cfg.rotation = RotationMode::Enabled;
    let corr = CorrMatcher::new(compiled).with_config(corr_cfg);

    let mut matcher = ShapeMatcher::new();
    let mut dpos = Vec::new();
    let mut dang = Vec::new();

    println!(
        "{:>4}  {:>18} {:>18}  {:>7} {:>8}  frame",
        "#", "shape (x, y)", "corr (x, y)", "dpos", "dangle"
    );

    for (i, path) in frames.iter().take(n).enumerate() {
        let scene = load_gray(path)?;
        let name = path.file_stem().unwrap_or_default().to_string_lossy();

        let Some(m) = matcher
            .find_u8(&scene.as_view(), &s.model, &s.search)
            .into_iter()
            .next()
        else {
            println!("{i:>4}  shape MISS  {name}");
            continue;
        };
        let sc = m.pose * nalgebra::Point2::new(roi_center.x, roi_center.y);

        let scene_view =
            corrmatch::ImageView::from_slice(scene.data(), scene.width(), scene.height())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        let cm = match corr.match_image(scene_view) {
            Ok(c) => c,
            Err(e) => {
                println!("{i:>4}  corrmatch error: {e}  {name}");
                continue;
            }
        };
        let cc = corr::corr_to_center(cm.x, cm.y, tw, th);

        let dp = ((sc.x - cc.x).powi(2) + (sc.y - cc.y).powi(2)).sqrt();
        let da = wrap_angle(cm.angle_deg.to_radians() - m.angle()).to_degrees();
        dpos.push(f64::from(dp));
        dang.push(f64::from(da.abs()));

        println!(
            "{i:>4}  ({:7.2},{:7.2}) ({:7.2},{:7.2})  {dp:6.2}px {da:7.2}°  {name}  [corr zncc {:.3}]",
            sc.x, sc.y, cc.x, cc.y, cm.score
        );
    }

    if dpos.is_empty() {
        bail!("no frames with both results");
    }
    println!("\nagreement over {} frames:", dpos.len());
    println!(
        "  |dpos|   p50 {:.2}  p95 {:.2}  max {:.2} px",
        pct(&dpos, 50.0),
        pct(&dpos, 95.0),
        pct(&dpos, 100.0)
    );
    println!(
        "  |dangle| p50 {:.2}  p95 {:.2}  max {:.2} deg",
        pct(&dang, 50.0),
        pct(&dang, 95.0),
        pct(&dang, 100.0)
    );
    Ok(())
}

// ── stats ───────────────────────────────────────────────────────────────────

fn summary(label: &str, v: &[f64]) {
    if v.is_empty() {
        return;
    }
    println!(
        "  {label:<20} min {:8.3}  p50 {:8.3}  p95 {:8.3}  max {:8.3}",
        pct(v, 0.0),
        pct(v, 50.0),
        pct(v, 95.0),
        pct(v, 100.0)
    );
}

fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let idx = ((p / 100.0) * (s.len() - 1) as f64).round() as usize;
    s[idx.min(s.len() - 1)]
}

fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}
