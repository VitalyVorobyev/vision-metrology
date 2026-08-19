//! End-to-end inspection on real can-end frames: locate → fixture → measure.
//!
//! This is the chain the library exists to support, run on real data rather
//! than a synthetic fixture:
//!
//! ```text
//! ShapeMatcher::find  →  ShapeMatch::pose  →  MetrologyModel::apply  →  Fit + residuals
//!     find the tab          the fixture          calipers on the rim      the measurement
//! ```
//!
//! The interesting part is that the rim is measured **in the tab's frame**. The
//! model is taught once, from the first frame, as "a circle of radius R whose
//! centre is at this offset from the tab". Every later frame re-derives the rim
//! from wherever the tab turned up — which is what a fixture is for, and what
//! makes the spread of the measured radius across frames a real repeatability
//! number rather than a restatement of where the part happened to sit.
//!
//! ## Run
//! ```text
//! cargo run --release -p vision-metrology --example inspect_canend -- \
//!   --scene-dir ~/privatedata/canend/set1/normal/dome \
//!   --roi 420,350,420,320 --rim-radius 367 --tolerance 1.5
//! ```
//!
//! Units are **pixels**. Millimetres arrive with the `metric` module
//! (roadmap B5), which converts a fitted primitive through a calibration.

use std::num::NonZeroUsize;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use vision_metrology::fit::{FitConfig, RansacConfig, RobustLoss, fit_circle};
use vision_metrology::matching::{
    Contrast, ShapeMatcher, ShapeModel, ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig,
};
use vision_metrology::measure::{
    MetrologyFit, MetrologyModel, MetrologyObject, MetrologyResult, MetrologyShape,
};
use vision_metrology::{Edge2DConfig, Edge2DDetector, Image, Point2f, Rect2f};

#[derive(Parser)]
#[command(about = "Locate a can-end tab, then measure the rim in the tab's frame")]
struct Cli {
    /// Directory of frames (`.bmp` / `.png`).
    #[arg(long)]
    scene_dir: PathBuf,
    /// Tab ROI in the first frame, as `x,y,width,height`.
    #[arg(long, value_parser = parse_roi)]
    roi: Rect2f,
    /// Approximate rim radius in pixels, to seed the teach step.
    #[arg(long)]
    rim_radius: f32,
    /// Gradient floor for model points. 400 suits the can-end dataset.
    #[arg(long, default_value_t = 400.0)]
    model_min_contrast: f32,
    /// Reject a frame whose rim `max_dev` exceeds this, in pixels.
    #[arg(long, default_value_t = 2.0)]
    tolerance: f32,
    /// Calipers around the rim.
    #[arg(long, default_value_t = 96)]
    calipers: usize,
    /// Stop after this many frames. `0` runs the folder.
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let frames = frames_in(&args.scene_dir)?;
    if frames.is_empty() {
        bail!("no frames in {}", args.scene_dir.display());
    }

    // ---- Teach --------------------------------------------------------
    // Build the tab model, then measure the rim once in the first frame and
    // express it in the tab's own frame.
    let first = load_gray(&frames[0])?;
    let model = build_model(&first, args.roi, Contrast::Raw(args.model_min_contrast))?;

    let mut matcher = ShapeMatcher::new();
    let search = ShapeSearchConfig {
        min_score: 0.5,
        max_matches: NonZeroUsize::new(1),
        ..ShapeSearchConfig::default()
    };
    let taught_pose = matcher
        .find(&first.as_view(), &model, &search)
        .first()
        .map(|m| m.pose)
        .context("no tab found in the teach frame")?;

    let rim = teach_rim(&first, args.rim_radius)?;
    // Rim geometry in the tab's frame: undo the pose that was in force when it
    // was measured.
    let inv = taught_pose.inverse();
    let rim_center_model = inv * rim.model.center;
    let rim_radius_model = rim.model.radius / taught_pose.scaling();

    println!(
        "teach: rim r={:.3} px at ({:.2}, {:.2}), fit rms {:.3} / max_dev {:.3} over {} pts",
        rim.model.radius, rim.model.center.x, rim.model.center.y, rim.rms, rim.max_dev, rim.n_used
    );
    println!(
        "       in tab frame: centre ({:.2}, {:.2}), r {:.3}\n",
        rim_center_model.x, rim_center_model.y, rim_radius_model
    );

    let mut metrology = MetrologyModel::new();
    let mut object = MetrologyObject::new(MetrologyShape::Circle {
        center: rim_center_model,
        radius: rim_radius_model,
        arc: None,
    });
    object.n_calipers = args.calipers;
    object.caliper_len = 12.0;
    object.caliper_width = 6.0;
    // The rim is bright-on-dark all the way round, but print, dents and the tab
    // itself put other edges nearby: reject the calipers that find them rather
    // than let them bend the circle.
    object.fit = FitConfig {
        loss: RobustLoss::Tukey { c: 2.0 },
        ..FitConfig::default()
    };
    metrology.add(object);

    // ---- Inspect ------------------------------------------------------
    println!(
        "{:>5}  {:>7}  {:>9}  {:>8}  {:>8}  {:>5}  verdict",
        "frame", "score", "radius", "rms", "maxdev", "n"
    );

    let mut radii = Vec::new();
    let (mut passed, mut failed, mut missed) = (0usize, 0usize, 0usize);
    let take = if args.limit == 0 {
        frames.len()
    } else {
        args.limit.min(frames.len())
    };

    for path in frames.iter().take(take) {
        let img = load_gray(path)?;
        let name = path.file_stem().unwrap_or_default().to_string_lossy();

        let Some(m) = matcher
            .find(&img.as_view(), &model, &search)
            .first()
            .cloned()
        else {
            println!("{name:>5}  {:>7}  no tab found", "-");
            missed += 1;
            continue;
        };

        let results = metrology.apply(&img.as_view(), &m.pose);
        let Some(Ok(MetrologyResult {
            fit: MetrologyFit::Circle(fit),
            ..
        })) = results.first()
        else {
            println!("{name:>5}  {:>7.3}  rim not measurable", m.score);
            missed += 1;
            continue;
        };

        let ok = fit.max_dev <= args.tolerance;
        if ok {
            passed += 1;
        } else {
            failed += 1;
        }
        radii.push(fit.model.radius);
        println!(
            "{name:>5}  {:>7.3}  {:>9.3}  {:>8.3}  {:>8.3}  {:>5}  {}",
            m.score,
            fit.model.radius,
            fit.rms,
            fit.max_dev,
            fit.n_used,
            if ok { "pass" } else { "FAIL" }
        );
    }

    // ---- Report -------------------------------------------------------
    println!("\n{passed} pass, {failed} fail, {missed} not measured");
    if radii.len() > 1 {
        let mean = radii.iter().sum::<f32>() / radii.len() as f32;
        let var = radii.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / (radii.len() - 1) as f32;
        let (lo, hi) = radii
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &r| (l.min(r), h.max(r)));
        println!(
            "rim radius over {} frames: mean {mean:.3} px, sigma {:.3} px, range [{lo:.3}, {hi:.3}]",
            radii.len(),
            var.sqrt()
        );
        println!(
            "  sigma is the repeatability of tab-fixture + rim-measurement combined, over\n  \
             different physical cans — an upper bound on the measurement's own repeatability."
        );
    }
    Ok(())
}

/// Fit the rim directly, without a fixture, to teach the model.
fn teach_rim(
    img: &Image<u8>,
    r_hint: f32,
) -> Result<vision_metrology::fit::Fit<vision_metrology::Circle2f>> {
    let mut det = Edge2DDetector::new();
    let edgels = det.detect(&img.as_view(), &Edge2DConfig::default());
    let seed = Point2f::new(img.width() as f32 * 0.5, img.height() as f32 * 0.5);
    let band: Vec<Point2f> = edgels
        .iter()
        .map(|e| e.p)
        .filter(|p| ((p - seed).norm() - r_hint).abs() < 0.08 * r_hint)
        .collect();
    if band.len() < 50 {
        bail!(
            "only {} edgels near r={r_hint}; is --rim-radius right?",
            band.len()
        );
    }
    fit_circle(
        &band,
        &FitConfig {
            ransac: Some(RansacConfig {
                iters: 400,
                inlier_tol: 2.0,
                min_inliers: 50,
                seed: 7,
            }),
            loss: RobustLoss::Tukey { c: 2.0 },
            ..FitConfig::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("rim teach failed: {e}"))
}

fn build_model(img: &Image<u8>, roi: Rect2f, min_contrast: Contrast) -> Result<ShapeModel> {
    ShapeModelBuilder::new()
        .build(
            &img.as_view(),
            roi,
            &ShapeModelConfig {
                min_contrast,
                ..ShapeModelConfig::default()
            },
        )
        .map_err(|e| anyhow::anyhow!("model build failed: {e}"))
}

fn frames_in(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("bmp" | "png" | "BMP" | "PNG")
            )
        })
        .collect();
    out.sort();
    Ok(out)
}

fn load_gray(path: &PathBuf) -> Result<Image<u8>> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Image::from_vec(w, h, img.into_raw()).map_err(|e| anyhow::anyhow!("{e}"))
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
