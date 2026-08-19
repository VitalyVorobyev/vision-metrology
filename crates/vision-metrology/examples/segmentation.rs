//! Example: Otsu threshold, CCL, and component_stats on a synthetic bimodal image.
//!
//! ## Pipeline
//! 1. Create a 64×64 u8 image (dark background = 30) with two bright squares (value 200):
//!    - Square A: rows 10–20, cols 10–20
//!    - Square B: rows 40–55, cols 40–55
//! 2. Run `otsu_threshold_u8` and print the computed threshold.
//! 3. Threshold to binary and apply `label_connected_components_u8` with C8.
//! 4. Print number of components found.
//! 5. Run `component_stats` and print centroid and pixel_count for each component.
//! 6. Assert exactly 2 components are found.
//!
//! ## Run
//! ```text
//! cargo run -p vision-metrology --example segmentation
//! ```

use vision_metrology::Image;
use vision_metrology::contour::Connectivity;
use vision_metrology::segment::{
    component_stats, label_connected_components_u8, otsu_threshold_u8,
};

fn main() {
    let (w, h) = (64usize, 64usize);

    // --- Step 1: create bimodal image ---
    println!("Generating 64x64 bimodal image (background=30, squares=200)...");
    let mut data = vec![30u8; w * h];

    // Square A: rows 10–20, cols 10–20
    for y in 10..=20 {
        for x in 10..=20 {
            data[y * w + x] = 200;
        }
    }
    // Square B: rows 40–55, cols 40–55
    for y in 40..=55 {
        for x in 40..=55 {
            data[y * w + x] = 200;
        }
    }

    let img = Image::from_vec(w, h, data).expect("valid image");

    // --- Step 2: Otsu thresholding ---
    let threshold = otsu_threshold_u8(&img.as_view());
    println!("Otsu threshold: {threshold}");

    // Otsu maximises between-class variance.  For a very unequal bimodal
    // histogram (many dark pixels, few bright ones) the optimal cut is at the
    // dark-class value itself: pixels *strictly greater than* the threshold
    // become foreground.  Acceptable range: 30 ≤ threshold < 200.
    assert!(
        (30..200).contains(&threshold),
        "Otsu threshold {threshold} should be in [30, 200)"
    );

    // --- Step 3: binarise and CCL ---
    let binary_data: Vec<u8> = img
        .data()
        .iter()
        .map(|&v| if v > threshold { 255u8 } else { 0u8 })
        .collect();
    let binary = Image::from_vec(w, h, binary_data).expect("valid binary image");

    println!("Running label_connected_components_u8 (C8)...");
    let ccl = label_connected_components_u8(&binary.as_view(), Connectivity::C8);

    // --- Step 4: print component count ---
    println!("Number of connected components: {}", ccl.num_labels);

    // --- Step 5: component_stats ---
    let stats = component_stats(&ccl, 1);
    println!("\nComponent statistics:");
    for s in &stats {
        println!(
            "  Label {}: pixel_count={}, centroid=({:.1}, {:.1}), bbox=({:.0},{:.0})–({:.0},{:.0})",
            s.label,
            s.pixel_count,
            s.centroid.x,
            s.centroid.y,
            s.bbox.x,
            s.bbox.y,
            s.bbox.x + s.bbox.width,
            s.bbox.y + s.bbox.height,
        );
    }

    // --- Step 6: assertions ---
    assert_eq!(
        ccl.num_labels, 2,
        "Expected exactly 2 connected components, found {}",
        ccl.num_labels
    );
    assert_eq!(
        stats.len(),
        2,
        "component_stats should return 2 entries, got {}",
        stats.len()
    );

    // Square A (11×11 = 121 px) centroid ~ (15, 15)
    // Square B (16×16 = 256 px) centroid ~ (47.5, 47.5)
    // Labels are assigned in raster order, so label 1 is Square A, label 2 is Square B.
    let s_a = stats.iter().find(|s| s.label == 1).unwrap();
    let s_b = stats.iter().find(|s| s.label == 2).unwrap();

    assert!(
        (s_a.centroid.x - 15.0).abs() < 1.0 && (s_a.centroid.y - 15.0).abs() < 1.0,
        "Square A centroid expected near (15,15), got ({:.1},{:.1})",
        s_a.centroid.x,
        s_a.centroid.y
    );
    assert!(
        (s_b.centroid.x - 47.5).abs() < 1.0 && (s_b.centroid.y - 47.5).abs() < 1.0,
        "Square B centroid expected near (47.5,47.5), got ({:.1},{:.1})",
        s_b.centroid.x,
        s_b.centroid.y
    );

    println!("\nAll assertions passed.");
}
