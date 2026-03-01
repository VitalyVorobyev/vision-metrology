//! Example: MultiScaleEdgeDetector — 3-level detection on a synthetic step-edge image.
//!
//! ## Pipeline
//! 1. Create a 256×256 u8 image: left half 0, right half 200.
//! 2. Run `MultiScaleEdgeDetector` with `num_levels: 3`.
//! 3. Print total edgel count.
//! 4. Group edgels by pyramid level and print count per level.
//! 5. Assert total > 0.
//!
//! ## Run
//! ```text
//! cargo run -p vision-metrology --example multiscale_edges
//! ```

use vm_core::Image;
use vm_multiscale::{MultiScaleConfig, MultiScaleEdgeDetector};

fn main() {
    let (w, h) = (256usize, 256usize);

    // --- Step 1: synthetic step-edge image ---
    println!("Generating 256x256 step-edge image (left=0, right=200)...");
    let data: Vec<u8> = (0..h)
        .flat_map(|_y| (0..w).map(|x| if x < 128 { 0u8 } else { 200u8 }))
        .collect();
    let img = Image::from_vec(w, h, data).expect("valid image");

    // --- Step 2: multi-scale detection ---
    println!("Running MultiScaleEdgeDetector (3 levels)...");
    let mut det = MultiScaleEdgeDetector::new();
    let cfg = MultiScaleConfig {
        num_levels: 3,
        ..MultiScaleConfig::default()
    };
    let edgels = det.detect_u8(&img.as_view(), &cfg);

    // --- Step 3: print total count ---
    println!("Total scale-annotated edgels detected: {}", edgels.len());

    // --- Step 4: group by pyramid level ---
    let max_level = edgels.iter().map(|e| e.level).max().unwrap_or(0);
    println!("\nEdgels per pyramid level:");
    for lvl in 0..=max_level {
        let count = edgels.iter().filter(|e| e.level == lvl).count();
        let scale_sigma = edgels
            .iter()
            .filter(|e| e.level == lvl)
            .map(|e| e.scale)
            .next()
            .unwrap_or(0.0);
        println!("  Level {lvl}: {count} edgels  (sigma ≈ {scale_sigma:.2})");
    }

    // Print a few sample edgels to show the level-0 coordinate mapping.
    let show = edgels.len().min(3);
    if show > 0 {
        println!("\nFirst {show} edgel(s):");
        for e in &edgels[..show] {
            println!(
                "  p=({:.1},{:.1}), level={}, scale={:.2}, strength={:.1}",
                e.p.x, e.p.y, e.level, e.scale, e.strength
            );
        }
    }

    // --- Step 5: assertions ---
    assert!(
        !edgels.is_empty(),
        "Expected edgels on the step edge, got none"
    );

    // All edgels should have level-0 x-coordinates near 128 (the step position).
    let near_step = edgels
        .iter()
        .filter(|e| (e.p.x - 128.0).abs() <= 8.0)
        .count();
    assert!(
        near_step > 0,
        "Expected at least some edgels near x=128, got {near_step}"
    );
    println!("\n{near_step} edgel(s) within ±8 px of the step at x=128.");

    println!("\nAll assertions passed.");
}
