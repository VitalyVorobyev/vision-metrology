//! End-to-end rectification on real glue-rig frames: teach → find → canonical crop.
//!
//! ```text
//! ShapeModelBuilder::build  →  ShapeMatcher::find  →  ShapeMatch::model_frame_map  →  Map::apply_with_mask
//!        teach the nozzle          locate it per frame        canonical dst -> src map      the rectified crop
//! ```
//!
//! This is the seam `matching` (decision 3, roadmap rectify wave) exists for:
//! a found pose is only useful downstream — a caliper, a variation model, an
//! anomaly detector — once it has been turned into pixels that line up frame
//! to frame. `CropSpec` fixes the output tensor shape once, at teach time, so
//! every frame's rectified crop is directly comparable pixel-for-pixel.
//!
//! The dataset (`data/42781`) has no calibration and three cameras glued
//! vertically into one 320x291 frame (~97 rows per strip, roadmap plan's
//! recon note) — this example picks one strip and teaches the glue nozzle's
//! dome, which has real edges and moves frame to frame as the rig runs.
//!
//! Self-asserting: exits non-zero if the found-rate or mean crop validity
//! drops below its threshold. Set `ALIGN_CROPS_OUT_DIR` to also write each
//! rectified crop (plus its validity mask) as a PNG; unset, this example
//! writes nothing to disk.
//!
//! ## Run
//! ```text
//! cargo run -p vision-metrology --example align_crops
//! cargo run -p vision-metrology --example align_crops -- --data-dir data/42781 --strip 1 --n 20
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use vision_metrology::matching::{CropSpec, ShapeMatcher, ShapeModelBuilder, ShapeModelConfig};
use vision_metrology::warp::Interp;
use vision_metrology::{BorderMode, Image, Rect2f};

#[derive(Parser)]
#[command(about = "Teach the glue nozzle in one camera strip, then rectify N frames to it")]
struct Cli {
    /// Directory of `Image_NNNNN.bmp` frames.
    #[arg(long, default_value = "data/42781")]
    data_dir: PathBuf,
    /// Which of the three vertically-stacked camera strips to use (0, 1, 2).
    #[arg(long, default_value_t = 1)]
    strip: usize,
    /// Row height of one strip.
    #[arg(long, default_value_t = 97)]
    strip_height: usize,
    /// Number of frames (evenly spaced through the sequence, teach frame
    /// excluded) to find and rectify.
    #[arg(long, default_value_t = 20)]
    n: usize,
    /// Minimum acceptable found-rate over the `n` frames.
    #[arg(long, default_value_t = 0.9)]
    min_found_rate: f32,
    /// Minimum acceptable mean crop validity fraction.
    #[arg(long, default_value_t = 0.5)]
    min_validity: f32,
}

/// Teach ROI, in the strip's own local coordinates: the glue nozzle's dome,
/// picked by inspecting `Image_00000.bmp`'s middle strip.
fn teach_roi() -> Rect2f {
    Rect2f {
        x: 110.0,
        y: 30.0,
        width: 100.0,
        height: 67.0,
    }
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let frames = frames_in(&args.data_dir)?;
    if frames.is_empty() {
        bail!("no frames in {}", args.data_dir.display());
    }

    let out_dir = std::env::var_os("ALIGN_CROPS_OUT_DIR").map(PathBuf::from);
    if let Some(dir) = &out_dir {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }

    // ---- Teach ----------------------------------------------------------
    let teach_strip = load_strip(&frames[0], args.strip, args.strip_height)?;
    let roi = teach_roi();
    let model = ShapeModelBuilder::new()
        .build(&teach_strip.as_view(), roi, &ShapeModelConfig::default())
        .map_err(|e| anyhow::anyhow!("model build failed: {e}"))?;
    println!(
        "taught: {} pyramid level(s), {} level-0 points, roi {roi:?}",
        model.num_levels(),
        model.point_count(0)
    );

    let mut matcher = ShapeMatcher::new();
    let search = vision_metrology::matching::ShapeSearchConfig {
        min_score: 0.4,
        ..Default::default()
    };

    // Canonical crop: the taught ROI itself, at native (1x) resolution,
    // rendered at model scale regardless of the found scale.
    let spec = CropSpec::new(roi, 1.0);
    let (cw, ch) = spec.output_size();
    println!(
        "crop spec: {roi:?} @ {}x -> {cw}x{ch} px\n",
        spec.px_per_unit
    );

    // ---- Find + rectify ---------------------------------------------------
    let step = ((frames.len().saturating_sub(1)) / args.n.max(1)).max(1);
    let candidates: Vec<&PathBuf> = frames.iter().skip(1).step_by(step).take(args.n).collect();

    println!(
        "{:>12}  {:>7}  {:>8}  {:>8}  {:>7}  {:>10}",
        "frame", "score", "pos.x", "pos.y", "angle", "validity"
    );

    let mut found = 0usize;
    let mut validity_sum = 0.0f32;
    let mut validity_n = 0usize;

    for path in &candidates {
        let strip = load_strip(path, args.strip, args.strip_height)?;
        let name = path.file_stem().unwrap_or_default().to_string_lossy();

        let Some(m) = matcher
            .find(&strip.as_view(), &model, &search)
            .into_iter()
            .next()
        else {
            println!("{name:>12}  not found");
            continue;
        };
        found += 1;

        let map = m.model_frame_map(&spec);
        let mut dst = vec![0u8; cw * ch];
        let mut mask = vec![0u8; cw * ch];
        map.apply_with_mask(
            &strip.as_view(),
            &mut dst,
            &mut mask,
            Interp::Bilinear,
            BorderMode::Constant(0u8),
        )
        .map_err(|e| anyhow::anyhow!("rectify failed: {e}"))?;

        let valid = mask.iter().filter(|&&v| v == 255).count();
        let validity = valid as f32 / mask.len() as f32;
        validity_sum += validity;
        validity_n += 1;

        println!(
            "{name:>12}  {:>7.3}  {:>8.2}  {:>8.2}  {:>7.2}  {:>10.3}",
            m.score,
            m.position.x,
            m.position.y,
            m.angle().to_degrees(),
            validity
        );

        if let Some(dir) = &out_dir {
            write_png(&dir.join(format!("{name}_crop.png")), cw, ch, &dst)?;
            write_png(&dir.join(format!("{name}_mask.png")), cw, ch, &mask)?;
        }
    }

    let found_rate = found as f32 / candidates.len().max(1) as f32;
    let mean_validity = if validity_n > 0 {
        validity_sum / validity_n as f32
    } else {
        0.0
    };
    println!(
        "\n{found}/{} found (rate {found_rate:.2}), mean crop validity {mean_validity:.3}",
        candidates.len()
    );

    if let Some(dir) = &out_dir {
        println!("crops written to {}", dir.display());
    }

    if found_rate < args.min_found_rate {
        bail!(
            "found-rate {found_rate:.2} below threshold {:.2}",
            args.min_found_rate
        );
    }
    if mean_validity < args.min_validity {
        bail!(
            "mean crop validity {mean_validity:.3} below threshold {:.3}",
            args.min_validity
        );
    }
    Ok(())
}

/// Load `path`, then crop out strip `index` (rows `[index*h, (index+1)*h)`).
fn load_strip(path: &Path, index: usize, h: usize) -> Result<Image<u8>> {
    let img = load_gray(path)?;
    let (w, ih) = (img.width(), img.height());
    let y0 = index * h;
    if y0 + h > ih {
        bail!(
            "strip {index} (rows {y0}..{}) out of bounds for a {w}x{ih} frame",
            y0 + h
        );
    }
    let mut data = vec![0u8; w * h];
    for row in 0..h {
        let src_row = img.data()[(y0 + row) * w..(y0 + row + 1) * w].to_vec();
        data[row * w..(row + 1) * w].copy_from_slice(&src_row);
    }
    Image::from_vec(w, h, data).map_err(|e| anyhow::anyhow!("{e}"))
}

fn load_gray(path: &Path) -> Result<Image<u8>> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Image::from_vec(w, h, img.into_raw()).map_err(|e| anyhow::anyhow!("{e}"))
}

fn frames_in(dir: &Path) -> Result<Vec<PathBuf>> {
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

fn write_png(path: &Path, w: usize, h: usize, data: &[u8]) -> Result<()> {
    image::GrayImage::from_raw(w as u32, h as u32, data.to_vec())
        .context("building output image")?
        .save(path)
        .with_context(|| format!("writing {}", path.display()))
}
