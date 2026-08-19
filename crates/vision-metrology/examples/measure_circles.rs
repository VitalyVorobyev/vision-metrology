//! Integration example: `measure_circles`
//!
//! Full end-to-end metrology pipeline on a synthetic image containing three
//! white circles on a black background.
//!
//! ## Pipeline
//! 1. Generate a 512×512 synthetic image with 3 circles (radii 40, 60, 80 px).
//! 2. Run `Edge2DDetector` to extract subpixel edgels.
//! 3. Build a `ContourGraph` from the edgels to extract connected components.
//! 4. For each connected component with ≥ 60 edgels, collect all polyline points
//!    from all arcs in that component and attempt `fit_circle` with RANSAC.
//! 5. Collect valid ellipses (semi-axis ratio < 1.2, plausible radius).
//! 6. Print a JSON measurement report: `[{"cx": …, "cy": …, "a": …, "b": …, "angle": …}, …]`.
//! 7. Assert that 3 ellipses were found, each centre within 0.02 px and each
//!    radius within 0.10 px of ground truth.
//!
//! ## Run
//! ```text
//! cargo run -p vision-metrology --example measure_circles
//! ```
//!
//! Output is deterministic (no RNG, no file I/O).

use vision_metrology::Circle2f;
use vision_metrology::contour::{
    Connectivity, ContourBuildConfig, ContourGraph, NodeId, build_graph_from_edgels,
};
use vision_metrology::fit::{Fit, FitConfig, RansacConfig, fit_circle};
use vision_metrology::{Edge2DConfig, Edge2DDetector};
use vision_metrology::{Image, Point2f};

// ---------------------------------------------------------------------------
// Ground-truth circles
// ---------------------------------------------------------------------------

struct CircleSpec {
    cx: f32,
    cy: f32,
    r: f32,
}

const CIRCLES: [CircleSpec; 3] = [
    CircleSpec {
        cx: 128.0,
        cy: 128.0,
        r: 40.0,
    },
    CircleSpec {
        cx: 320.0,
        cy: 160.0,
        r: 60.0,
    },
    CircleSpec {
        cx: 220.0,
        cy: 360.0,
        r: 80.0,
    },
];

// ---------------------------------------------------------------------------
// Synthetic image generation
// ---------------------------------------------------------------------------

/// Generate a 512×512 grayscale image with 3 anti-aliased white circles.
///
/// Each pixel at (x, y) is set using a smoothstep transition centered at
/// each circle's boundary, giving the edge detector a gradual 2-px ramp
/// rather than a hard 1-bit step.
fn generate_synthetic_image(width: usize, height: usize) -> Image<u8> {
    let mut data = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let px = x as f32;
            let py = y as f32;
            let mut val = 0.0f32;
            for c in &CIRCLES {
                let dx = px - c.cx;
                let dy = py - c.cy;
                let dist = (dx * dx + dy * dy).sqrt();
                // Smooth ramp: full white inside (r-2), full black outside (r+2).
                let t = ((c.r + 2.0 - dist) / 4.0).clamp(0.0, 1.0);
                let smooth = t * t * (3.0 - 2.0 * t);
                val = val.max(smooth);
            }
            data[y * width + x] = (val * 255.0).round() as u8;
        }
    }
    Image::from_vec(width, height, data).expect("valid image dimensions")
}

// ---------------------------------------------------------------------------
// ScaleAnnotatedEdgel → Edgel conversion
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Collect all polyline points from a connected component
// ---------------------------------------------------------------------------

/// Collect all polyline points from all arcs reachable from the given
/// start node via DFS over the contour graph topology.
fn collect_component_points(graph: &ContourGraph, start_node: NodeId) -> Vec<Point2f> {
    let mut visited_nodes = vec![false; graph.nodes.len()];
    let mut visited_edges = vec![false; graph.edges.len()];
    let mut stack = vec![start_node];
    let mut pts: Vec<Point2f> = Vec::new();

    while let Some(nid) = stack.pop() {
        if visited_nodes[nid] {
            continue;
        }
        visited_nodes[nid] = true;

        for &eid in &graph.nodes[nid].incident_edges {
            if visited_edges[eid] {
                continue;
            }
            visited_edges[eid] = true;
            pts.extend_from_slice(&graph.edges[eid].points);

            // Visit the other endpoint of this edge.
            let edge = &graph.edges[eid];
            let other = if edge.a == nid { edge.b } else { edge.a };
            stack.push(other);
        }
    }

    pts
}

// ---------------------------------------------------------------------------
// JSON report
// ---------------------------------------------------------------------------

fn print_json_report(circles: &[Fit<Circle2f>]) {
    println!("[");
    for (i, f) in circles.iter().enumerate() {
        let comma = if i + 1 < circles.len() { "," } else { "" };
        // rms and max_dev are what make this a measurement rather than a
        // number: a roundness check reads max_dev, a fit-quality gate reads rms.
        println!(
            "  {{\"cx\": {:.4}, \"cy\": {:.4}, \"r\": {:.4}, \
             \"rms\": {:.4}, \"max_dev\": {:.4}, \"n\": {}}}{}",
            f.model.center.x, f.model.center.y, f.model.radius, f.rms, f.max_dev, f.n_used, comma
        );
    }
    println!("]");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let (w, h) = (512usize, 512usize);

    // --- Step 1: generate synthetic image ---
    println!("Generating 512x512 synthetic image with 3 circles (r=40, 60, 80)...");
    let img = generate_synthetic_image(w, h);

    // --- Step 2: subpixel edge detection ---
    println!("Running Edge2DDetector...");
    let mut det = Edge2DDetector::new();
    let edgels = det.detect(&img.as_view(), &Edge2DConfig::default());
    println!("  Detected {} edgels.", edgels.len());

    // --- Step 3: build contour graph ---
    println!("Building ContourGraph...");
    let contour_cfg = ContourBuildConfig {
        connectivity: Connectivity::C8,
        min_component_size: 5,
        record_strengths: false,
        record_geometry: false,
        ..Default::default()
    };
    let graph = build_graph_from_edgels(w, h, &edgels, &contour_cfg);
    println!(
        "  ContourGraph: {} nodes, {} edges.",
        graph.nodes.len(),
        graph.edges.len()
    );

    // --- Step 4 + 5: fit ellipses to connected components ---
    println!("Fitting circles to connected components with >= 60 total edgels...");
    // The targets are circles, so fit circles: three parameters instead of
    // five, and the fit reports the residuals that qualify the measurement.
    let fit_cfg = FitConfig {
        ransac: Some(RansacConfig {
            iters: 500,
            inlier_tol: 2.0,
            min_inliers: 30,
            seed: 42,
        }),
        ..FitConfig::default()
    };
    let mut circles: Vec<Fit<Circle2f>> = Vec::new();

    // Walk over all nodes that have not been visited yet as component roots.
    let mut node_visited = vec![false; graph.nodes.len()];

    for start_nid in 0..graph.nodes.len() {
        if node_visited[start_nid] {
            continue;
        }

        let pts = collect_component_points(&graph, start_nid);

        // Mark all nodes in this component as visited.
        // (We do a second DFS just to mark visited flags.)
        mark_component_visited(&graph, start_nid, &mut node_visited);

        if pts.len() < 60 {
            continue;
        }

        let result = fit_circle(&pts, &fit_cfg);
        let Ok(fit) = result else { continue };

        // Reject implausible radii and off-image centres.
        if !(10.0..=200.0).contains(&fit.model.radius) {
            continue;
        }
        if fit.model.center.x < 0.0
            || fit.model.center.x > w as f32
            || fit.model.center.y < 0.0
            || fit.model.center.y > h as f32
        {
            continue;
        }
        // A fit whose points do not actually lie on a circle is not a circle.
        // This gate is only possible because the fit reports its residual.
        if fit.rms > 1.0 {
            continue;
        }

        circles.push(fit);
    }

    // Deduplicate: remove near-duplicate ellipses (centres within 15 px).
    let mut deduped: Vec<Fit<Circle2f>> = Vec::new();
    for f in &circles {
        let is_dup = deduped
            .iter()
            .any(|e| (e.model.center - f.model.center).norm() < 15.0);
        if !is_dup {
            deduped.push(f.clone());
        }
    }

    println!("  Found {} distinct circle(s).", deduped.len());

    // --- Step 6: print JSON report ---
    println!("\nMeasurement report:");
    print_json_report(&deduped);

    // --- Step 7: assert correctness ---
    assert!(
        deduped.len() >= 3,
        "Expected >= 3 circles, found {}. \
        ContourGraph had {} edges; longest component had {} pts.",
        deduped.len(),
        graph.edges.len(),
        graph
            .iter_edges()
            .map(|e| e.points.len())
            .max()
            .unwrap_or(0)
    );

    // Match each detected ellipse to the nearest ground-truth circle by centre.
    for c in &CIRCLES {
        let nearest = deduped
            .iter()
            .min_by(|a, b| {
                let da = (a.model.center - Point2f::new(c.cx, c.cy)).norm();
                let db = (b.model.center - Point2f::new(c.cx, c.cy)).norm();
                da.partial_cmp(&db).expect("finite")
            })
            .expect("at least one circle");

        let cx_err = (nearest.model.center.x - c.cx).abs();
        let cy_err = (nearest.model.center.y - c.cy).abs();
        let r_fit = nearest.model.radius;
        let r_err = (r_fit - c.r).abs();

        // Tolerances are tight on purpose. A synthetic, noise-free, perfectly
        // round target measured with subpixel edges + a RANSAC ellipse fit
        // should land on the centre essentially exactly; anything above a
        // hundredth of a pixel here means a systematic bias, not noise. The
        // previous 5 px / 1.5 px bounds passed happily while multi-scale edgel
        // merging was dragging every centre 0.07-0.10 px negative in both axes
        // (a missing (2^l - 1)/2 half-pixel term). Do not loosen these.
        assert!(
            cx_err <= 0.02,
            "Circle r={}: centre_x error {cx_err:.4} > 0.02 px",
            c.r
        );
        assert!(
            cy_err <= 0.02,
            "Circle r={}: centre_y error {cy_err:.4} > 0.02 px",
            c.r
        );
        assert!(
            r_err <= 0.10,
            "Circle r={}: radius error {r_err:.4} > 0.10 px (r_fit={r_fit:.2})",
            c.r
        );

        println!(
            "  r={:.0}: measured r={:.2}, cx={:.2}, cy={:.2}  \
            [errors: r={:.2}, cx={:.2}, cy={:.2}]",
            c.r, r_fit, nearest.model.center.x, nearest.model.center.y, r_err, cx_err, cy_err
        );
    }

    println!("\nAll assertions passed.");
}

/// Mark all nodes in the connected component reachable from `start` as visited.
fn mark_component_visited(graph: &ContourGraph, start: NodeId, visited: &mut [bool]) {
    let mut stack = vec![start];
    while let Some(nid) = stack.pop() {
        if visited[nid] {
            continue;
        }
        visited[nid] = true;
        for &eid in &graph.nodes[nid].incident_edges {
            let edge = &graph.edges[eid];
            let other = if edge.a == nid { edge.b } else { edge.a };
            if !visited[other] {
                stack.push(other);
            }
        }
    }
}
