//! Example: PyramidF32 — build a 4-level pyramid from a synthetic gradient image.
//!
//! ## Pipeline
//! 1. Generate a 128×128 grayscale image where `pixel = (x + y) % 256`.
//! 2. Build a 4-level pyramid with `PyramidF32::build_from_u8`.
//! 3. Print each level's index, (width, height), and mean pixel value.
//! 4. Assert that each level's dimensions are half those of the previous level.
//!
//! ## Run
//! ```text
//! cargo run -p vision-metrology --example pyramid
//! ```

use vm_core::Image;
use vm_pyr::PyramidF32;

fn main() {
    let (w, h) = (128usize, 128usize);

    // --- Step 1: generate synthetic gradient image ---
    println!("Generating 128x128 gradient image: pixel = (x + y) % 256 ...");
    let data: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x + y) % 256) as u8))
        .collect();
    let img = Image::from_vec(w, h, data).expect("valid image dimensions");

    // --- Step 2: build 4-level pyramid ---
    println!("Building 4-level PyramidF32...");
    let mut pyr = PyramidF32::new();
    pyr.build_from_u8(&img.as_view(), 4);

    println!("  Built {} levels.", pyr.num_levels());
    assert_eq!(pyr.num_levels(), 4, "expected exactly 4 levels");

    // --- Step 3: print level info ---
    println!("\nLevel breakdown:");
    let mut prev_w = w;
    let mut prev_h = h;

    for i in 0..pyr.num_levels() {
        let level = pyr.level(i).expect("level should exist");
        let lw = level.width();
        let lh = level.height();
        let data = level.data();
        let mean = data.iter().copied().sum::<f32>() / data.len() as f32;

        println!("  Level {i}: {}x{} pixels, mean value = {mean:.2}", lw, lh);

        // --- Step 4: assert dimensions halve each time (after level 0) ---
        if i > 0 {
            assert_eq!(
                lw,
                prev_w / 2,
                "Level {i} width {lw} != prev_w/2 = {}",
                prev_w / 2
            );
            assert_eq!(
                lh,
                prev_h / 2,
                "Level {i} height {lh} != prev_h/2 = {}",
                prev_h / 2
            );
        }

        prev_w = lw;
        prev_h = lh;
    }

    // Verify known dimensions: 128 → 64 → 32 → 16
    let dims: Vec<(usize, usize)> = (0..pyr.num_levels())
        .map(|i| {
            let l = pyr.level(i).unwrap();
            (l.width(), l.height())
        })
        .collect();
    assert_eq!(
        dims,
        vec![(128, 128), (64, 64), (32, 32), (16, 16)],
        "unexpected pyramid dimensions"
    );

    // Level-0 mean: for pixel = (x + y) % 256 over a 128×128 image the mean is ≈ 127.5
    let l0 = pyr.level(0).unwrap();
    let mean0 = l0.data().iter().copied().sum::<f32>() / l0.data().len() as f32;
    assert!(
        (mean0 - 127.5).abs() < 1.0,
        "Level 0 mean should be near 127.5, got {mean0:.2}"
    );

    println!("\nAll assertions passed.");
}
