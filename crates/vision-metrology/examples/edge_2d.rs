//! Example: Edge2DDetector — detect edges on a synthetic step-edge image.
//!
//! ## Pipeline
//! 1. Create a 64×64 u8 image: left half 0, right half 200 (vertical step edge at x=32).
//! 2. Detect edgels with `Edge2DDetector::detect_u8` using the default config.
//! 3. Print the total edgel count and stats for the first few edgels.
//! 4. Assert that at least some edgels are detected near x=32.
//!
//! ## Run
//! ```text
//! cargo run -p vision-metrology --example edge_2d
//! ```

use vm_core::Image;
use vm_edge::edge2d::{Edge2DConfig, Edge2DDetector};

fn main() {
    let (w, h) = (64usize, 64usize);

    // --- Step 1: synthetic step-edge image ---
    // Left half (x < 32) = 0 (dark); right half (x >= 32) = 200 (bright).
    println!("Creating 64x64 step-edge image: x<32 → 0, x>=32 → 200...");
    let data: Vec<u8> = (0..h)
        .flat_map(|_y| (0..w).map(|x| if x < 32 { 0u8 } else { 200u8 }))
        .collect();
    let img = Image::from_vec(w, h, data).expect("valid image");

    // --- Step 2: detect edgels ---
    println!("Running Edge2DDetector with default config...");
    let mut det = Edge2DDetector::new();
    let cfg = Edge2DConfig::default();
    let edgels = det.detect_u8(&img.as_view(), &cfg);

    // --- Step 3: print stats ---
    println!("Detected {} edgel(s).", edgels.len());
    let show = edgels.len().min(5);
    if show > 0 {
        println!("First {} edgel(s):", show);
        for e in &edgels[..show] {
            println!(
                "  p = ({:.2}, {:.2}), n = ({:.2}, {:.2}), strength = {:.2}",
                e.p.x, e.p.y, e.n.x, e.n.y, e.strength
            );
        }
    }

    // Mean x-position of all edgels — should be close to 32.
    let mean_x = edgels.iter().map(|e| e.p.x).sum::<f32>() / edgels.len() as f32;
    println!("Mean edgel x-position: {mean_x:.2} (expected near 32)");

    // --- Step 4: assertions ---
    assert!(
        !edgels.is_empty(),
        "Expected edgels on the step edge, but got none"
    );

    // At least some edgels should be within ±3 px of x=32.
    let near_edge = edgels
        .iter()
        .filter(|e| (e.p.x - 32.0).abs() <= 3.0)
        .count();
    assert!(
        near_edge > 0,
        "Expected edgels near x=32, but found {near_edge} within ±3 px"
    );
    println!("{near_edge} edgel(s) found within ±3 px of x=32.");

    // Majority of edgels should have positive nx (dark-to-bright = pointing right).
    let pos_nx = edgels.iter().filter(|e| e.n.x > 0.0).count();
    assert!(
        pos_nx * 2 >= edgels.len(),
        "Expected majority of normals to point right (+x), got {pos_nx}/{} with n.x > 0",
        edgels.len()
    );

    println!("\nAll assertions passed.");
}
