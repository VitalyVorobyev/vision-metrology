//! Example: ContourGraph + smooth_polyline — build contour graph from a synthetic circle.
//!
//! ## Pipeline
//! 1. Create a 128×128 image with a ring: pixels whose distance to center is in [38, 42] are
//!    set to 255; all others are 0.
//! 2. Detect edges with `Edge2DDetector`.
//! 3. Build a `ContourGraph` with `build_graph_from_edgels`.
//! 4. Print node count and edge count.
//! 5. Take the longest graph-edge's points, smooth with `smooth_polyline(pts, 5)`, and
//!    print point count before and after.
//!
//! ## Run
//! ```text
//! cargo run -p vision-metrology --example contour_graph
//! ```

use vision_metrology::Image;
use vision_metrology::{
    Connectivity, ContourBuildConfig, NodeKind, build_graph_from_edgels, smooth_polyline,
};
use vision_metrology::{Edge2DConfig, Edge2DDetector};

fn main() {
    let (w, h) = (128usize, 128usize);
    let cx = 64.0f32;
    let cy = 64.0f32;

    // --- Step 1: generate ring image ---
    println!("Generating 128x128 ring image (radius 40, ring width 4 px)...");
    let data: Vec<u8> = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if (38.0..=42.0).contains(&dist) {
                    255u8
                } else {
                    0u8
                }
            })
        })
        .collect();
    let img = Image::from_vec(w, h, data).expect("valid image");

    // --- Step 2: detect edgels ---
    println!("Running Edge2DDetector...");
    let mut det = Edge2DDetector::new();
    let cfg = Edge2DConfig::default();
    let edgels = det.detect(&img.as_view(), &cfg);
    println!("  Detected {} edgel(s).", edgels.len());

    // --- Step 3: build contour graph ---
    println!("Building ContourGraph (C8, min_component_size=5)...");
    let contour_cfg = ContourBuildConfig {
        connectivity: Connectivity::C8,
        min_component_size: 5,
        record_strengths: false,
        record_geometry: false,
        ..Default::default()
    };
    let graph = build_graph_from_edgels(w, h, &edgels, &contour_cfg);

    // --- Step 4: print node and edge counts ---
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Edges: {}", graph.edges.len());

    // A ring has exactly two boundaries -- the inner and the outer circle --
    // and neither of them branches, so the graph must be two closed loops with
    // no junctions. Asserting only "at least one edge" would also accept the
    // shattered 288-node / 432-edge output produced before the edgel mask was
    // thinned.
    assert_eq!(
        graph.edges.len(),
        2,
        "a ring must trace as exactly two closed contours, got {}",
        graph.edges.len()
    );
    assert!(
        graph.nodes.iter().all(|n| n.kind == NodeKind::LoopAnchor),
        "a ring has no junctions or endpoints; every node must be a loop anchor"
    );

    // --- Step 5: smooth the longest edge's polyline ---
    let longest_edge = graph
        .iter_edges_by_length()
        .next()
        .expect("at least one edge exists");

    let pts_before = longest_edge.points.len();
    let smoothed = smooth_polyline(&longest_edge.points, 5.0);
    let pts_after = smoothed.len();

    println!(
        "\nLongest contour edge: {pts_before} points before smoothing, {pts_after} after smoothing (sigma=5.0 px)."
    );

    // The outer boundary has circumference ~2*pi*42 = 264 px, so a correctly
    // traced contour is a couple of hundred points -- not the 15 that the
    // over-fragmented graph used to yield.
    assert!(
        pts_before >= 200,
        "Expected the longest edge to trace most of a circle, got {pts_before} points"
    );
    assert_eq!(
        pts_before, pts_after,
        "smooth_polyline should preserve the point count"
    );

    // The smoothed points should stay close to radius 40 (within a few px).
    let max_dist_err = smoothed
        .iter()
        .map(|p| {
            let dx = p.x - cx;
            let dy = p.y - cy;
            let d = (dx * dx + dy * dy).sqrt();
            (d - 40.0).abs()
        })
        .fold(0.0f32, f32::max);
    println!("Max distance from circle (r=40) after smoothing: {max_dist_err:.2} px");

    println!("\nAll assertions passed.");
}
