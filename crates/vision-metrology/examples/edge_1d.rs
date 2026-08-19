//! Example: Edge1DDetector — detect edges on a synthetic 1D signal.
//!
//! ## Pipeline
//! 1. Create a synthetic signal: 64 f32 samples, first 32 at 0.0, last 32 at 100.0.
//!    This represents a rising step edge at position 32.
//! 2. Detect edges with `Edge1DDetector::new(sigma)` and a low threshold.
//! 3. Print the detected edge positions and polarities.
//! 4. Assert that exactly one rising edge is found near position 32.
//!
//! ## Run
//! ```text
//! cargo run -p vision-metrology --example edge_1d
//! ```

use vision_metrology::edge::edge1d::{Edge1DConfig, Edge1DDetector, EdgePolarity};

fn main() {
    // --- Step 1: create a 64-sample step-edge signal ---
    println!("Creating 64-sample step-edge signal (0.0 for i<32, 100.0 for i>=32)...");
    let signal: Vec<f32> = (0..64)
        .map(|i| if i < 32 { 0.0f32 } else { 100.0 })
        .collect();

    // --- Step 2: detect edges ---
    // Use a low positive/negative threshold so the single step edge is found.
    let sigma = 1.2f32;
    let mut det = Edge1DDetector::new(sigma);
    let cfg = Edge1DConfig {
        sigma,
        pos_thresh: 1.0,
        neg_thresh: 1.0,
        ..Edge1DConfig::default()
    };

    let peaks = det.detect_in(&signal, &cfg);

    // --- Step 3: print results ---
    println!("Detected {} edge peak(s):", peaks.len());
    for p in &peaks {
        let polarity_str = match p.polarity {
            EdgePolarity::Rising => "Rising",
            EdgePolarity::Falling => "Falling",
        };
        println!(
            "  position = {:.3}, strength = {:.3}, polarity = {}",
            p.x, p.strength, polarity_str
        );
    }

    // --- Step 4: assert a rising edge near position 32 ---
    let rising_near_32 = peaks
        .iter()
        .any(|p| p.polarity == EdgePolarity::Rising && (p.x - 32.0).abs() <= 2.0);

    assert!(
        rising_near_32,
        "Expected a rising edge near position 32; detected peaks: {:?}",
        peaks
    );

    println!("\nAll assertions passed.");
}
