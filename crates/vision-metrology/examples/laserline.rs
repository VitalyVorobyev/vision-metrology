//! Example: laser line detection on a multi-snap image.
//!
//! Loads a horizontally-merged PNG of N equal-width frames, splits it into
//! individual snaps, and runs `LaserExtractor` (column-scan mode) on each.
//! Column scanning is correct for an almost-horizontal laser line: each column
//! is scanned top-to-bottom to find the laser's y-coordinate, yielding one
//! `(col_index, center_y)` point per column.
//!
//! Results are written to a JSON file next to the input image.
//! Per-snap and total timing is printed to stdout.
//!
//! Run from the workspace root:
//!   cargo run -p vision-metrology --example laserline -- --help
//!   cargo run -p vision-metrology --example laserline

use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use image::ImageReader;
use serde::Serialize;
use vision_metrology::{
    ColAccess, Edge1DConfig, Image, LaserExtractConfig, LaserExtractor, ScanAxis,
};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Detect laser lines in a horizontally-merged multi-snap image")]
struct Args {
    /// Path to the merged PNG (default: data/laser_0.png)
    #[arg(long, default_value = "data/laser_0.png")]
    input: String,

    /// Number of equal-width snaps merged in the image
    #[arg(long, default_value_t = 6)]
    n_snaps: usize,

    /// DoG sigma for edge detection
    #[arg(long, default_value_t = 1.2)]
    sigma: f32,

    /// Minimum absolute DoG gradient to count as an edge (applied to both
    /// rising and falling edges). Raise to reject noise on dark backgrounds.
    #[arg(long, default_value_t = 0.0)]
    min_grad: f32,

    /// Output JSON path (default: <input stem>_results.json next to input)
    #[arg(long)]
    out: Option<String>,
}

// ── JSON DTOs ─────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct SampleDto {
    scan_i: usize,
    center: f32,
    width: f32,
    score: f32,
    left: f32,
    right: f32,
    valid: bool,
}

#[derive(Serialize)]
struct PointDto {
    x: f32,
    y: f32,
}

#[derive(Serialize)]
struct SnapResult {
    snap: usize,
    /// Wall-clock time for this snap's extraction, in milliseconds.
    elapsed_ms: f64,
    samples: Vec<SampleDto>,
    points: Vec<PointDto>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Copy one snap (col slice) from the mega-image into a contiguous buffer.
fn extract_snap(
    pixels: &[u8],
    full_width: usize,
    height: usize,
    snap_w: usize,
    snap_idx: usize,
) -> Result<Image<u8>> {
    let mut buf = vec![0u8; snap_w * height];
    let col_offset = snap_idx * snap_w;
    for row in 0..height {
        let src = &pixels[row * full_width + col_offset..row * full_width + col_offset + snap_w];
        buf[row * snap_w..(row + 1) * snap_w].copy_from_slice(src);
    }
    Image::from_vec(snap_w, height, buf).context("building snap Image")
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    let img_path = &args.input;
    let out_path = args.out.unwrap_or_else(|| {
        let p = std::path::Path::new(img_path);
        let stem = p.file_stem().unwrap_or_default().to_string_lossy();
        let dir = p.parent().unwrap_or(std::path::Path::new("."));
        dir.join(format!("{stem}_results.json"))
            .to_string_lossy()
            .into_owned()
    });

    // Load as 8-bit grayscale.
    let gray = ImageReader::open(img_path)
        .with_context(|| format!("opening {img_path}"))?
        .decode()
        .with_context(|| format!("decoding {img_path}"))?
        .into_luma8();

    let full_width = gray.width() as usize;
    let height = gray.height() as usize;
    let n_snaps = args.n_snaps;

    assert!(n_snaps > 0, "n_snaps must be > 0");
    assert_eq!(
        full_width % n_snaps,
        0,
        "image width {full_width} is not divisible by n_snaps={n_snaps}"
    );
    let snap_w = full_width / n_snaps;

    println!(
        "loaded {img_path}: {full_width}x{height}, splitting into {n_snaps} snaps of {snap_w}x{height}"
    );
    println!(
        "config: sigma={:.2}, min_grad={:.2}",
        args.sigma, args.min_grad
    );

    let pixels = gray.as_raw().as_slice();

    // Column-scan: each column is scanned top-to-bottom for the laser y-center.
    let cfg = LaserExtractConfig {
        axis: ScanAxis::Cols {
            access: ColAccess::Gather,
        },
        edge_cfg: Edge1DConfig {
            sigma: args.sigma,
            pos_thresh: args.min_grad,
            neg_thresh: args.min_grad,
            ..Edge1DConfig::default()
        },
        ..LaserExtractConfig::default()
    };
    let mut extractor = LaserExtractor::new(args.sigma);

    let mut results: Vec<SnapResult> = Vec::with_capacity(n_snaps);
    let total_start = Instant::now();

    for snap_idx in 0..n_snaps {
        let snap = extract_snap(pixels, full_width, height, snap_w, snap_idx)?;

        let t0 = Instant::now();
        let line = extractor.extract_line_u8(&snap.as_view(), 0..snap_w, &cfg, None);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;

        let valid_count = line.samples.iter().filter(|s| s.valid).count();
        println!("  snap {snap_idx}: {valid_count}/{snap_w} valid  ({elapsed_ms:.2} ms)");

        let samples = line
            .samples
            .iter()
            .map(|s| SampleDto {
                scan_i: s.scan_i,
                center: s.center,
                width: s.width,
                score: s.score,
                left: s.left,
                right: s.right,
                valid: s.valid,
            })
            .collect();

        let points = line
            .points
            .iter()
            .map(|p| PointDto { x: p.x, y: p.y })
            .collect();

        results.push(SnapResult {
            snap: snap_idx,
            elapsed_ms,
            samples,
            points,
        });
    }

    let total_ms = total_start.elapsed().as_secs_f64() * 1e3;
    println!("total extraction time: {total_ms:.2} ms");

    let out_file =
        std::fs::File::create(&out_path).with_context(|| format!("creating {out_path}"))?;
    serde_json::to_writer_pretty(out_file, &results)
        .with_context(|| format!("writing JSON to {out_path}"))?;

    println!("results written to {out_path}");
    Ok(())
}
