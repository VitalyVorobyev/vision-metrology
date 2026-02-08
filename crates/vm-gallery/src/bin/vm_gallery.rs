use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use image::{GrayImage, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use vm_contour::{Connectivity, ContourBuildConfig, build_graph_from_detector_output};
use vm_core::{BorderMode, Image};
use vm_edge::{
    DoGKernel1D, Edge1DConfig, Edge1DDetector, Edge2DConfig, Edge2DDetector, Subpix2D, SubpixRefine,
};
use vm_laser::{ColAccess, LaserExtractConfig, LaserExtractor, ScanAxis};
use vm_morph::{close3x3_binary_u8, open3x3_binary_u8};
use vm_pyr::PyramidF32;

#[derive(Parser, Debug)]
#[command(name = "vm_gallery")]
#[command(about = "Run vision-metrology algorithms on external fixtures")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[command(name = "morphology")]
    Morphology(MorphologyArgs),
    #[command(name = "pyramid")]
    Pyramid(PyramidArgs),
    #[command(name = "edge1d")]
    Edge1d(Edge1dArgs),
    #[command(name = "laser_rows")]
    LaserRows(LaserArgs),
    #[command(name = "laser_cols")]
    LaserCols(LaserArgs),
    #[command(name = "edgels2d")]
    Edgels2d(Edgels2dArgs),
    #[command(name = "contour_graph")]
    ContourGraph(ContourArgs),
}

#[derive(Args, Debug, Clone)]
struct CommonArgs {
    #[arg(long, required = true)]
    input: PathBuf,
    #[arg(long, required = true)]
    truth: PathBuf,
    #[arg(long, default_value = "docs/fig/raw")]
    out: PathBuf,
}

#[derive(Args, Debug, Clone)]
struct MorphologyArgs {
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Args, Debug, Clone)]
struct PyramidArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, default_value_t = 5)]
    levels: usize,
}

#[derive(Args, Debug, Clone)]
struct Edge1dArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, default_value_t = 1.2)]
    sigma: f32,
    #[arg(long, default_value_t = 0.0)]
    pos_thresh: f32,
    #[arg(long, default_value_t = 0.0)]
    neg_thresh: f32,
    #[arg(long, default_value_t = 1.0)]
    min_width: f32,
    #[arg(long, default_value_t = 60.0)]
    max_width: f32,
}

#[derive(Args, Debug, Clone)]
struct LaserArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, default_value_t = 1.2)]
    sigma: f32,
    #[arg(long, default_value_t = 32)]
    roi_half_width: usize,
    #[arg(long, default_value_t = 8.0)]
    max_jump_px: f32,
    #[arg(long, default_value_t = 5)]
    max_gap_scans: usize,
    #[arg(long, default_value_t = 2.0)]
    min_width: f32,
    #[arg(long, default_value_t = 12.0)]
    max_width: f32,
    #[arg(long, default_value_t = 50.0)]
    min_score: f32,
    #[arg(long, default_value_t = 0.2)]
    prior_weight: f32,
}

#[derive(Args, Debug, Clone)]
struct Edgels2dArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, default_value_t = true)]
    pre_smooth: bool,
    #[arg(long, default_value_t = 0.0)]
    low_thresh: f32,
    #[arg(long, default_value_t = 0.0)]
    high_thresh: f32,
}

#[derive(Args, Debug, Clone)]
struct ContourArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, default_value_t = 0.0)]
    low_thresh: f32,
    #[arg(long, default_value_t = 0.0)]
    high_thresh: f32,
    #[arg(long, default_value_t = 2)]
    min_component_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TruthEnvelope {
    case: String,
    width: usize,
    height: usize,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    truth: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LaserTruthPayload {
    axis: String,
    true_center: Vec<Option<f32>>,
    #[serde(default)]
    true_width: Option<Vec<Option<f32>>>,
}

#[derive(Debug, Clone, Serialize)]
struct MetaMorphology {
    operation: &'static str,
    structuring_element: &'static str,
    se_size: usize,
    pixel_rule: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct MetaPyramid {
    requested_levels: usize,
    built_levels: usize,
    level_sizes: Vec<[usize; 2]>,
    policy: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct Edge1dResult {
    left_x: Option<f32>,
    right_x: Option<f32>,
    center_x: Option<f32>,
    width: Option<f32>,
    score: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct MetaEdge1d {
    sigma: f32,
    border: &'static str,
    pos_thresh: f32,
    neg_thresh: f32,
    refine: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct PointDto {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Serialize)]
struct LaserSampleDto {
    scan_i: usize,
    center: f32,
    width: f32,
    score: f32,
    left: f32,
    right: f32,
    valid: bool,
}

#[derive(Debug, Clone, Serialize)]
struct LaserLineDto {
    axis: &'static str,
    samples: Vec<LaserSampleDto>,
    points: Vec<PointDto>,
}

#[derive(Debug, Clone, Serialize)]
struct MetaLaser {
    axis: &'static str,
    sigma: f32,
    roi_half_width: usize,
    max_jump_px: f32,
    max_gap_scans: usize,
    min_score: f32,
    min_width: f32,
    max_width: f32,
    prior_weight: f32,
}

#[derive(Debug, Clone, Serialize)]
struct EdgelDto {
    x: f32,
    y: f32,
    nx: f32,
    ny: f32,
    strength: f32,
    ix: u32,
    iy: u32,
}

#[derive(Debug, Clone, Serialize)]
struct MetaEdgels {
    pre_smooth: bool,
    low_thresh: f32,
    high_thresh: f32,
    border: &'static str,
    subpix: &'static str,
    count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct GraphNodeDto {
    id: usize,
    kind: &'static str,
    x: f32,
    y: f32,
    degree: usize,
}

#[derive(Debug, Clone, Serialize)]
struct GraphEdgeDto {
    id: usize,
    a: usize,
    b: usize,
    is_loop: bool,
    points: Vec<[f32; 2]>,
}

#[derive(Debug, Clone, Serialize)]
struct GraphDto {
    width: usize,
    height: usize,
    nodes: Vec<GraphNodeDto>,
    edges: Vec<GraphEdgeDto>,
}

#[derive(Debug, Clone, Serialize)]
struct MetaContour {
    edge_low_thresh: f32,
    edge_high_thresh: f32,
    connectivity: &'static str,
    min_component_size: usize,
    node_count: usize,
    edge_count: usize,
    junctions: usize,
    ends: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Command::Morphology(args) => run_morphology(args),
        Command::Pyramid(args) => run_pyramid(args),
        Command::Edge1d(args) => run_edge1d(args),
        Command::LaserRows(args) => run_laser_rows(args),
        Command::LaserCols(args) => run_laser_cols(args),
        Command::Edgels2d(args) => run_edgels2d(args),
        Command::ContourGraph(args) => run_contour_graph(args),
    }
}

fn run_morphology(args: MorphologyArgs) -> Result<()> {
    let (case_dir, truth) = prepare_case(&args.common, "morphology")?;
    let img = load_input_u8(&args.common.input)?;
    validate_dims(&truth, &img)?;

    let opened = open3x3_binary_u8(&img.as_view());
    let closed = close3x3_binary_u8(&img.as_view());

    save_u8_image(case_dir.join("open.png"), &opened)?;
    save_u8_image(case_dir.join("close.png"), &closed)?;

    write_json(
        case_dir.join("meta.json"),
        &MetaMorphology {
            operation: "open+close",
            structuring_element: "square",
            se_size: 3,
            pixel_rule: "binary pixel set iff value > 0",
        },
    )?;

    Ok(())
}

fn run_pyramid(args: PyramidArgs) -> Result<()> {
    let (case_dir, truth) = prepare_case(&args.common, "pyramid")?;
    let img = load_input_u8(&args.common.input)?;
    validate_dims(&truth, &img)?;

    let mut pyr = PyramidF32::new();
    pyr.build_from_u8(&img.as_view(), args.levels);

    let mut sizes = Vec::new();
    for i in 0..pyr.num_levels() {
        let level = pyr.level(i).expect("level index validated by loop");
        sizes.push([level.width(), level.height()]);
        let vis = f32_to_u8_vis(level.data(), level.width(), level.height());
        save_luma_raw(
            case_dir.join(format!("level_{i}.png")),
            level.width(),
            level.height(),
            vis,
        )?;
    }

    write_json(
        case_dir.join("meta.json"),
        &MetaPyramid {
            requested_levels: args.levels,
            built_levels: pyr.num_levels(),
            level_sizes: sizes,
            policy: "2x2 mean downsample with drop-odd dimensions",
        },
    )?;

    Ok(())
}

fn run_edge1d(args: Edge1dArgs) -> Result<()> {
    let (case_dir, truth) = prepare_case(&args.common, "edge1d")?;
    let img = load_input_u8(&args.common.input)?;
    validate_dims(&truth, &img)?;

    if img.height() != 1 {
        bail!(
            "edge1d fixture must be a 1-row image (height == 1), got height={}.",
            img.height()
        );
    }

    let signal_u8 = img.as_view().row(0).to_vec();

    let cfg = Edge1DConfig {
        sigma: args.sigma,
        border: BorderMode::Clamp,
        pos_thresh: args.pos_thresh,
        neg_thresh: args.neg_thresh,
        refine: SubpixRefine::Parabolic3,
    };

    let mut det = Edge1DDetector::new(args.sigma);
    let peaks = det.detect_in_u8(&signal_u8, &cfg);

    let pair_cfg = vm_edge::EdgePairConfig {
        min_width: args.min_width,
        max_width: args.max_width,
        prefer_bright_on_dark: true,
    };
    let best_pair = vm_edge::best_edge_pair(&peaks, &pair_cfg);

    let signal_f32: Vec<f32> = signal_u8.iter().map(|&v| v as f32).collect();
    let kernel = DoGKernel1D::new(args.sigma);
    let mut response = vec![0.0f32; signal_f32.len()];
    vm_edge::conv1d::convolve_f32(
        &signal_f32,
        &kernel.dg,
        kernel.radius,
        cfg.border.clone(),
        &mut response,
    );

    write_csv(case_dir.join("signal.csv"), &signal_f32)?;
    write_csv(case_dir.join("response.csv"), &response)?;

    let result = if let Some(p) = best_pair {
        Edge1dResult {
            left_x: Some(p.left.x),
            right_x: Some(p.right.x),
            center_x: Some(p.center_x),
            width: Some(p.width),
            score: Some(p.score),
        }
    } else {
        Edge1dResult {
            left_x: None,
            right_x: None,
            center_x: None,
            width: None,
            score: None,
        }
    };

    write_json(case_dir.join("result.json"), &result)?;
    write_json(
        case_dir.join("meta.json"),
        &MetaEdge1d {
            sigma: args.sigma,
            border: "Clamp",
            pos_thresh: args.pos_thresh,
            neg_thresh: args.neg_thresh,
            refine: "Parabolic3",
        },
    )?;

    Ok(())
}

fn run_laser_rows(args: LaserArgs) -> Result<()> {
    let (case_dir, truth) = prepare_case(&args.common, "laser_rows")?;
    let img = load_input_u8(&args.common.input)?;
    validate_dims(&truth, &img)?;
    validate_laser_truth(&truth, "rows", img.height())?;

    let mut cfg = LaserExtractConfig {
        axis: ScanAxis::Rows,
        roi_half_width: args.roi_half_width,
        max_jump_px: args.max_jump_px,
        max_gap_scans: args.max_gap_scans,
        min_score: args.min_score,
        min_width: args.min_width,
        max_width: args.max_width,
        prior_weight: args.prior_weight,
        ..LaserExtractConfig::default()
    };
    cfg.edge_cfg.sigma = args.sigma;
    cfg.edge_cfg.refine = SubpixRefine::Parabolic3;

    let mut extractor = LaserExtractor::new(args.sigma);
    let line = extractor.extract_line_u8(&img.as_view(), 0..img.height(), &cfg, None);

    write_json(case_dir.join("line.json"), &laser_line_dto(&line, "rows"))?;
    write_json(
        case_dir.join("meta.json"),
        &MetaLaser {
            axis: "rows",
            sigma: args.sigma,
            roi_half_width: args.roi_half_width,
            max_jump_px: args.max_jump_px,
            max_gap_scans: args.max_gap_scans,
            min_score: args.min_score,
            min_width: args.min_width,
            max_width: args.max_width,
            prior_weight: args.prior_weight,
        },
    )?;

    let overlay = render_laser_overlay(&img, &line, true);
    overlay
        .save(case_dir.join("overlay.png"))
        .context("writing laser_rows overlay.png")?;

    Ok(())
}

fn run_laser_cols(args: LaserArgs) -> Result<()> {
    let (case_dir, truth) = prepare_case(&args.common, "laser_cols")?;
    let img = load_input_u8(&args.common.input)?;
    validate_dims(&truth, &img)?;
    validate_laser_truth(&truth, "cols", img.width())?;

    let mut cfg = LaserExtractConfig {
        axis: ScanAxis::Cols {
            access: ColAccess::Gather,
        },
        roi_half_width: args.roi_half_width,
        max_jump_px: args.max_jump_px,
        max_gap_scans: args.max_gap_scans,
        min_score: args.min_score,
        min_width: args.min_width,
        max_width: args.max_width,
        prior_weight: args.prior_weight,
        ..LaserExtractConfig::default()
    };
    cfg.edge_cfg.sigma = args.sigma;
    cfg.edge_cfg.refine = SubpixRefine::Parabolic3;

    let mut extractor = LaserExtractor::new(args.sigma);
    let line = extractor.extract_line_u8(&img.as_view(), 0..img.width(), &cfg, None);

    write_json(case_dir.join("line.json"), &laser_line_dto(&line, "cols"))?;
    write_json(
        case_dir.join("meta.json"),
        &MetaLaser {
            axis: "cols",
            sigma: args.sigma,
            roi_half_width: args.roi_half_width,
            max_jump_px: args.max_jump_px,
            max_gap_scans: args.max_gap_scans,
            min_score: args.min_score,
            min_width: args.min_width,
            max_width: args.max_width,
            prior_weight: args.prior_weight,
        },
    )?;

    let overlay = render_laser_overlay(&img, &line, false);
    overlay
        .save(case_dir.join("overlay.png"))
        .context("writing laser_cols overlay.png")?;

    Ok(())
}

fn run_edgels2d(args: Edgels2dArgs) -> Result<()> {
    let (case_dir, truth) = prepare_case(&args.common, "edgels2d")?;
    let img = load_input_u8(&args.common.input)?;
    validate_dims(&truth, &img)?;

    let cfg = Edge2DConfig {
        pre_smooth: args.pre_smooth,
        low_thresh: args.low_thresh,
        high_thresh: args.high_thresh,
        border: BorderMode::Clamp,
        subpix: Subpix2D::ParabolicAlongNormal,
        ..Edge2DConfig::default()
    };

    let mut det = Edge2DDetector::new();
    let edgels = det.detect_u8(&img.as_view(), &cfg);
    let out: Vec<EdgelDto> = edgels
        .iter()
        .map(|e| EdgelDto {
            x: e.p.x,
            y: e.p.y,
            nx: e.n.x,
            ny: e.n.y,
            strength: e.strength,
            ix: e.idx.0 as u32,
            iy: e.idx.1 as u32,
        })
        .collect();

    write_json(case_dir.join("edgels.json"), &out)?;
    write_json(
        case_dir.join("meta.json"),
        &MetaEdgels {
            pre_smooth: cfg.pre_smooth,
            low_thresh: cfg.low_thresh,
            high_thresh: cfg.high_thresh,
            border: "Clamp",
            subpix: "ParabolicAlongNormal",
            count: out.len(),
        },
    )?;

    Ok(())
}

fn run_contour_graph(args: ContourArgs) -> Result<()> {
    let (case_dir, truth) = prepare_case(&args.common, "contour_graph")?;
    let img = load_input_u8(&args.common.input)?;
    validate_dims(&truth, &img)?;

    let edge_cfg = Edge2DConfig {
        low_thresh: args.low_thresh,
        high_thresh: args.high_thresh,
        border: BorderMode::Clamp,
        subpix: Subpix2D::ParabolicAlongNormal,
        ..Edge2DConfig::default()
    };
    let contour_cfg = ContourBuildConfig {
        connectivity: Connectivity::C8,
        min_component_size: args.min_component_size,
        record_strengths: false,
    };

    let mut edge_det = Edge2DDetector::new();
    let graph =
        build_graph_from_detector_output(&img.as_view(), &mut edge_det, &edge_cfg, &contour_cfg);

    let nodes = graph
        .nodes
        .iter()
        .map(|n| GraphNodeDto {
            id: n.id,
            kind: node_kind_name(n.kind),
            x: n.p.x,
            y: n.p.y,
            degree: n.degree,
        })
        .collect::<Vec<_>>();

    let edges = graph
        .edges
        .iter()
        .map(|e| GraphEdgeDto {
            id: e.id,
            a: e.a,
            b: e.b,
            is_loop: e.is_loop,
            points: e.points.iter().map(|p| [p.x, p.y]).collect(),
        })
        .collect::<Vec<_>>();

    write_json(
        case_dir.join("graph.json"),
        &GraphDto {
            width: graph.width,
            height: graph.height,
            nodes,
            edges,
        },
    )?;

    write_json(
        case_dir.join("meta.json"),
        &MetaContour {
            edge_low_thresh: edge_cfg.low_thresh,
            edge_high_thresh: edge_cfg.high_thresh,
            connectivity: "C8",
            min_component_size: contour_cfg.min_component_size,
            node_count: graph.nodes.len(),
            edge_count: graph.edges.len(),
            junctions: graph.num_junctions(),
            ends: graph.num_ends(),
        },
    )?;

    Ok(())
}

fn prepare_case(common: &CommonArgs, case_name: &str) -> Result<(PathBuf, TruthEnvelope)> {
    ensure_file_exists(&common.input, "input")?;
    ensure_file_exists(&common.truth, "truth")?;

    let truth: TruthEnvelope = read_json(&common.truth)
        .with_context(|| format!("reading truth json at {}", common.truth.display()))?;

    if truth.case != case_name {
        bail!(
            "truth case mismatch: expected '{}', got '{}'.",
            case_name,
            truth.case
        );
    }

    let case_dir = common.out.join(case_name);
    fs::create_dir_all(&case_dir)
        .with_context(|| format!("creating output directory {}", case_dir.display()))?;

    fs::copy(&common.input, case_dir.join("input.png")).with_context(|| {
        format!(
            "copying input {} -> {}",
            common.input.display(),
            case_dir.join("input.png").display()
        )
    })?;
    fs::copy(&common.truth, case_dir.join("truth.json")).with_context(|| {
        format!(
            "copying truth {} -> {}",
            common.truth.display(),
            case_dir.join("truth.json").display()
        )
    })?;

    Ok((case_dir, truth))
}

fn load_input_u8(path: &Path) -> Result<Image<u8>> {
    let dyn_img =
        image::open(path).with_context(|| format!("opening input image {}", path.display()))?;
    let luma = dyn_img.to_luma8();
    let (w, h) = luma.dimensions();
    let data = luma.into_raw();

    Image::from_vec(w as usize, h as usize, data)
        .with_context(|| format!("constructing vm-core image from {}", path.display()))
}

fn validate_dims(truth: &TruthEnvelope, img: &Image<u8>) -> Result<()> {
    if truth.width != img.width() || truth.height != img.height() {
        bail!(
            "truth dimensions ({}, {}) do not match input dimensions ({}, {}).",
            truth.width,
            truth.height,
            img.width(),
            img.height()
        );
    }
    Ok(())
}

fn validate_laser_truth(
    truth: &TruthEnvelope,
    expected_axis: &str,
    expected_len: usize,
) -> Result<()> {
    let payload: LaserTruthPayload =
        serde_json::from_value(truth.truth.clone()).with_context(|| {
            format!(
                "parsing laser truth payload for case '{}' as axis/center arrays",
                truth.case
            )
        })?;

    if payload.axis != expected_axis {
        bail!(
            "laser truth axis mismatch: expected '{}', got '{}'.",
            expected_axis,
            payload.axis
        );
    }
    if payload.true_center.len() != expected_len {
        bail!(
            "laser truth length mismatch: expected {}, got {}.",
            expected_len,
            payload.true_center.len()
        );
    }

    if let Some(widths) = payload.true_width
        && widths.len() != expected_len
    {
        bail!(
            "laser truth true_width length mismatch: expected {}, got {}.",
            expected_len,
            widths.len()
        );
    }

    Ok(())
}

fn save_u8_image(path: PathBuf, img: &Image<u8>) -> Result<()> {
    save_luma_raw(path, img.width(), img.height(), img.data().to_vec())
}

fn save_luma_raw(path: PathBuf, width: usize, height: usize, data: Vec<u8>) -> Result<()> {
    let gray = GrayImage::from_raw(width as u32, height as u32, data)
        .context("constructing GrayImage from raw bytes")?;
    gray.save(&path)
        .with_context(|| format!("saving image {}", path.display()))
}

fn f32_to_u8_vis(data: &[f32], width: usize, height: usize) -> Vec<u8> {
    let _ = (width, height);
    if data.is_empty() {
        return Vec::new();
    }

    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in data {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }

    if (max_v - min_v).abs() < 1e-12 {
        return vec![0u8; data.len()];
    }

    let scale = 255.0 / (max_v - min_v);
    data.iter()
        .map(|&v| ((v - min_v) * scale).round().clamp(0.0, 255.0) as u8)
        .collect()
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("serializing json")?;
    fs::write(&path, bytes).with_context(|| format!("writing json {}", path.display()))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let data = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&data).with_context(|| format!("parsing json {}", path.display()))
}

fn write_csv(path: PathBuf, values: &[f32]) -> Result<()> {
    let mut file =
        fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    writeln!(file, "index,value").context("writing csv header")?;
    for (i, v) in values.iter().enumerate() {
        writeln!(file, "{i},{v}").context("writing csv row")?;
    }
    Ok(())
}

fn laser_line_dto(line: &vm_laser::LaserLine, axis: &'static str) -> LaserLineDto {
    let samples = line
        .samples
        .iter()
        .map(|s| LaserSampleDto {
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

    LaserLineDto {
        axis,
        samples,
        points,
    }
}

fn render_laser_overlay(input: &Image<u8>, line: &vm_laser::LaserLine, is_rows: bool) -> RgbImage {
    let gray = GrayImage::from_raw(
        input.width() as u32,
        input.height() as u32,
        input.data().to_vec(),
    )
    .expect("dimensions and data length must match");
    let mut rgb = image::DynamicImage::ImageLuma8(gray).to_rgb8();

    for s in &line.samples {
        if !s.valid {
            continue;
        }

        let (x, y) = if is_rows {
            (s.center, s.scan_i as f32)
        } else {
            (s.scan_i as f32, s.center)
        };
        draw_dot(&mut rgb, x, y, Rgb([255, 64, 64]));
    }

    rgb
}

fn draw_dot(img: &mut RgbImage, x: f32, y: f32, color: Rgb<u8>) {
    let xi = x.round() as i32;
    let yi = y.round() as i32;

    for dy in -1..=1 {
        for dx in -1..=1 {
            let nx = xi + dx;
            let ny = yi + dy;
            if nx < 0 || ny < 0 {
                continue;
            }
            let (ux, uy) = (nx as u32, ny as u32);
            if ux >= img.width() || uy >= img.height() {
                continue;
            }
            img.put_pixel(ux, uy, color);
        }
    }
}

fn node_kind_name(kind: vm_contour::NodeKind) -> &'static str {
    match kind {
        vm_contour::NodeKind::End => "End",
        vm_contour::NodeKind::Junction => "Junction",
        vm_contour::NodeKind::Isolated => "Isolated",
        vm_contour::NodeKind::LoopAnchor => "LoopAnchor",
    }
}

fn ensure_file_exists(path: &Path, what: &str) -> Result<()> {
    if !path.exists() {
        bail!("{} file does not exist: {}", what, path.display());
    }
    if !path.is_file() {
        bail!("{} path is not a file: {}", what, path.display());
    }
    Ok(())
}
