//! Example: morphology operations on a synthetic binary cross.
//!
//! ## Pipeline
//! 1. Create a 21×21 image with a cross: horizontal bar rows 9–11, vertical bar cols 9–11.
//! 2. Erode with `StructuringElement::Square(1)` — print foreground pixel count.
//! 3. Dilate the original with `StructuringElement::Square(2)` — print foreground pixel count.
//! 4. Compute `chamfer_distance_u8` on the cross — find the pixel with maximum distance.
//! 5. Run `thin_binary_u8` — print remaining foreground pixels.
//!
//! ## Run
//! ```text
//! cargo run -p vision-metrology --example morphology
//! ```

use vision_metrology::Image;
use vision_metrology::vm_primitives::morph::{
    StructuringElement, chamfer_distance_u8, dilate_binary_u8, erode_binary_u8, thin_binary_u8,
};

fn count_fg(data: &[u8]) -> usize {
    data.iter().filter(|&&v| v > 0).count()
}

fn main() {
    let (w, h) = (21usize, 21usize);

    // --- Step 1: create cross image ---
    // Horizontal bar: rows 9, 10, 11 (all columns)
    // Vertical bar:   cols 9, 10, 11 (all rows)
    println!("Creating 21x21 binary cross image...");
    let mut data = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let in_horiz = (9..=11).contains(&y);
            let in_vert = (9..=11).contains(&x);
            if in_horiz || in_vert {
                data[y * w + x] = 255;
            }
        }
    }
    let cross = Image::from_vec(w, h, data).expect("valid image");
    let original_fg = count_fg(cross.data());
    println!("  Original foreground pixels: {original_fg}");

    // --- Step 2: erode with Square(1) (3×3 SE) ---
    let eroded = erode_binary_u8(&cross.as_view(), &StructuringElement::Square(1));
    let eroded_fg = count_fg(eroded.data());
    println!("Eroded (Square(1)): {eroded_fg} foreground pixels remain.");

    assert!(
        eroded_fg < original_fg,
        "Erosion should reduce foreground pixel count (got {eroded_fg} >= {original_fg})"
    );

    // --- Step 3: dilate the original with Square(2) (5×5 SE) ---
    let dilated = dilate_binary_u8(&cross.as_view(), &StructuringElement::Square(2));
    let dilated_fg = count_fg(dilated.data());
    println!("Dilated (Square(2)): {dilated_fg} foreground pixels.");

    assert!(
        dilated_fg > original_fg,
        "Dilation should increase foreground pixel count (got {dilated_fg} <= {original_fg})"
    );

    // --- Step 4: chamfer distance ---
    let dist_map = chamfer_distance_u8(&cross.as_view());
    let (max_idx, max_dist) = dist_map
        .data()
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a): &(usize, f32), (_, b): &(usize, f32)| {
            a.partial_cmp(b).expect("finite compare")
        })
        .expect("non-empty image");
    let max_x = max_idx % w;
    let max_y = max_idx / w;
    println!("Max chamfer distance: {max_dist:.2} px at pixel ({max_x}, {max_y}).");

    // The maximum distance in a 21×21 cross must be at a corner of the image.
    assert!(
        max_dist > 0.0,
        "Expected non-zero max chamfer distance, got {max_dist}"
    );

    // Corner pixels (e.g. (0,0)) should be farthest from the cross.
    // Verify the pixel with max distance is in the background.
    assert_eq!(
        *cross.as_view().get(max_x, max_y).unwrap(),
        0,
        "Pixel with max chamfer distance ({max_x},{max_y}) should be background"
    );

    // --- Step 5: thinning ---
    let thinned = thin_binary_u8(&cross.as_view());
    let thinned_fg = count_fg(thinned.data());
    println!("Thinned: {thinned_fg} foreground pixels (skeleton).");

    assert!(
        thinned_fg > 0,
        "Thinning should retain at least the skeleton centre"
    );
    assert!(
        thinned_fg <= original_fg,
        "Thinning should not add foreground pixels"
    );

    println!("\nAll assertions passed.");
}
