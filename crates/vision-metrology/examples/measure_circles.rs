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
//!    from all arcs in that component and attempt `ConicFitter::fit_ellipse_ransac`.
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

use vision_metrology::{ConicFitConfig, ConicFitter, Ellipse2f};
use vision_metrology::{
    Connectivity, ContourBuildConfig, ContourGraph, NodeId, build_graph_from_edgels,
};
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

fn print_json_report(ellipses: &[Ellipse2f]) {
    println!("[");
    for (i, ell) in ellipses.iter().enumerate() {
        let comma = if i + 1 < ellipses.len() { "," } else { "" };
        println!(
            "  {{\"cx\": {:.3}, \"cy\": {:.3}, \"a\": {:.3}, \"b\": {:.3}, \"angle\": {:.4}}}{}",
            ell.center.x,
            ell.center.y,
            ell.semi_major(),
            ell.semi_minor(),
            ell.angle,
            comma
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
    println!("Fitting ellipses to connected components with >= 60 total edgels...");
    let fit_cfg = ConicFitConfig {
        use_bookstein: false,
        ransac_iters: 500,
        inlier_tol: 2.0,
        min_inliers: 30,
        rng_seed: 42,
    };
    let mut fitter = ConicFitter::new();
    let mut ellipses: Vec<Ellipse2f> = Vec::new();

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

        let result = fitter.fit_ellipse_ransac(&pts, &fit_cfg);
        let Ok(ell) = result else { continue };

        // Reject degenerate ellipses: semi-axis ratio must be < 1.2
        // (true circles have ratio = 1; allow small fitting noise).
        let ratio = ell.semi_major() / ell.semi_minor().max(1e-6);
        if ratio > 1.2 {
            continue;
        }

        // Reject ellipses with unreasonably small or large mean radius.
        let r = (ell.semi_major() + ell.semi_minor()) * 0.5;
        if !(10.0..=200.0).contains(&r) {
            continue;
        }

        // Reject ellipses whose centre is outside the image.
        if ell.center.x < 0.0
            || ell.center.x > w as f32
            || ell.center.y < 0.0
            || ell.center.y > h as f32
        {
            continue;
        }

        ellipses.push(ell);
    }

    // Deduplicate: remove near-duplicate ellipses (centres within 15 px).
    let mut deduped: Vec<Ellipse2f> = Vec::new();
    for ell in &ellipses {
        let is_dup = deduped.iter().any(|e| {
            let dx = e.center.x - ell.center.x;
            let dy = e.center.y - ell.center.y;
            (dx * dx + dy * dy).sqrt() < 15.0
        });
        if !is_dup {
            deduped.push(ell.clone());
        }
    }

    println!("  Found {} distinct ellipse(s).", deduped.len());

    // --- Step 6: print JSON report ---
    println!("\nMeasurement report:");
    print_json_report(&deduped);

    // --- Step 7: assert correctness ---
    assert!(
        deduped.len() >= 3,
        "Expected >= 3 ellipses, found {}. \
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
                let da = ((a.center.x - c.cx).powi(2) + (a.center.y - c.cy).powi(2)).sqrt();
                let db = ((b.center.x - c.cx).powi(2) + (b.center.y - c.cy).powi(2)).sqrt();
                da.partial_cmp(&db).expect("finite")
            })
            .expect("at least one ellipse");

        let cx_err = (nearest.center.x - c.cx).abs();
        let cy_err = (nearest.center.y - c.cy).abs();
        let r_fit = (nearest.semi_major() + nearest.semi_minor()) * 0.5;
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
            c.r, r_fit, nearest.center.x, nearest.center.y, r_err, cx_err, cy_err
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
