//! Example: LsdDetector — detect line segments in a synthetic image with step edges.
//!
//! LSD (Line Segment Detector) operates on gradient-coherent regions. It works best
//! with **step edges** rather than 1-pixel-wide pen strokes. This example draws two
//! step-edge rectangles to create long, clean horizontal and vertical edges.
//!
//! ## Pipeline
//! 1. Create a 128×128 image (background 0) with two bright rectangles (value 200):
//!    - Rect A: rows 10–40, cols 20–110  → top/bottom horizontal edges, left/right vertical edges
//!    - Rect B: rows 55–90, cols 15–60   → additional horizontal and vertical edges
//! 2. Run `LsdDetector` with `scale = 1.0` (no downscaling) and a relaxed `density_th`.
//! 3. Print the detected segment count and each segment's endpoints, length, and NFA.
//! 4. Assert at least 1 segment is detected.
//!
//! ## Run
//! ```text
//! cargo run -p vision-metrology --example line_segments
//! ```

use vision_metrology::Image;
use vision_metrology::shape::{LsdConfig, LsdDetector};

fn main() {
    let (w, h) = (128usize, 128usize);

    // --- Step 1: create image with bright rectangles on dark background ---
    // Step edges at every rectangle boundary generate gradient-coherent regions
    // that LSD can detect reliably.
    println!("Generating 128x128 image with two bright rectangles on dark background...");
    let mut data = vec![0u8; w * h];

    // Rectangle A: rows 10–40, cols 20–110
    for y in 10..=40 {
        for x in 20..=110 {
            data[y * w + x] = 200;
        }
    }
    // Rectangle B: rows 55–90, cols 15–60
    for y in 55..=90 {
        for x in 15..=60 {
            data[y * w + x] = 200;
        }
    }

    let img = Image::from_vec(w, h, data).expect("valid image");

    // --- Step 2: run LSD at full resolution ---
    // downscale_levels=0 keeps 1-px sampling; density_th lowered slightly for
    // clean synthetic edges.
    println!("Running LsdDetector (full resolution, density_th=0.5)...");
    let mut det = LsdDetector::new();
    let cfg = LsdConfig {
        downscale_levels: 0,
        density_th: 0.5,
        ..LsdConfig::default()
    };
    let segs = det.detect(&img.as_view(), &cfg);

    // --- Step 3: print results ---
    println!("Detected {} line segment(s):", segs.len());
    for (i, s) in segs.iter().enumerate() {
        println!(
            "  [{i}] p1=({:.1},{:.1}) p2=({:.1},{:.1}) len={:.1} nfa={:.2}",
            s.p1.x, s.p1.y, s.p2.x, s.p2.y, s.length, s.nfa
        );
    }

    // --- Step 4: assert at least 1 segment detected ---
    assert!(
        !segs.is_empty(),
        "Expected at least 1 line segment, but found none"
    );

    // All accepted segments should have negative NFA (log10(NFA) < 0 = significant).
    let all_negative_nfa = segs.iter().all(|s| s.nfa < 0.0);
    println!("\nAll segments have negative NFA (statistically significant): {all_negative_nfa}");

    // All segments should have length >= the configured min_length (3.0 px).
    let all_long_enough = segs.iter().all(|s| s.length >= 3.0);
    assert!(all_long_enough, "All segments should have length >= 3.0 px");

    println!("\nAll assertions passed.");
}
