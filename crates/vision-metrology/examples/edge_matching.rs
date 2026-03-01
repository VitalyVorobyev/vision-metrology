//! Example: EdgeModel + RigidEdgeMatcher — embed a rectangle model in a scene and verify match.
//!
//! ## Pipeline
//! 1. Build a 30×15 rectangle perimeter of edgels at origin using the same approach as
//!    the matcher tests (`rect_edgels`).
//! 2. Create `EdgeModel::from_edgels(model_edgels, 5)`.
//! 3. Create scene edgels: same rectangle translated to (80, 60), with integer `.idx` fields.
//! 4. Configure `RigidMatchConfig` with a position search covering the scene.
//! 5. Run `matcher.match_model(...)` and print tx, ty, angle, score.
//! 6. Assert result is `Some` and score >= 0.3.
//!
//! ## Run
//! ```text
//! cargo run -p vision-metrology --example edge_matching
//! ```

use vm_core::{Point2f, Rect2f, Vec2f};
use vm_edge::edge2d::Edgel;
use vm_match::{EdgeModel, RigidEdgeMatcher, RigidMatchConfig};

/// Build a rectangular perimeter of edgels at (ox, oy) with dimensions w×h.
/// Normals: top edge → (0,-1), right → (1,0), bottom → (0,1), left → (-1,0).
fn rect_edgels(ox: f32, oy: f32, w: f32, h: f32) -> Vec<Edgel> {
    let mut v = Vec::new();
    let steps = ((w + h) * 2.0) as usize;
    for i in 0..steps {
        let t = i as f32 / steps as f32;
        let (x, y, nx, ny): (f32, f32, f32, f32) = if t < 0.25 {
            // Top edge: left → right, normal pointing up
            (ox + t * 4.0 * w, oy, 0.0, -1.0)
        } else if t < 0.5 {
            // Right edge: top → bottom, normal pointing right
            (ox + w, oy + (t - 0.25) * 4.0 * h, 1.0, 0.0)
        } else if t < 0.75 {
            // Bottom edge: right → left, normal pointing down
            (ox + w - (t - 0.5) * 4.0 * w, oy + h, 0.0, 1.0)
        } else {
            // Left edge: bottom → top, normal pointing left
            (ox, oy + h - (t - 0.75) * 4.0 * h, -1.0, 0.0)
        };
        let nn = (nx * nx + ny * ny).sqrt();
        v.push(Edgel {
            p: Point2f { x, y },
            n: Vec2f {
                x: nx / nn,
                y: ny / nn,
            },
            strength: 1.0,
            idx: (x.round() as usize, y.round() as usize),
        });
    }
    v
}

fn main() {
    // --- Step 1 + 2: build model from a 30×15 rectangle at origin ---
    println!("Building 30x15 rectangle model at origin...");
    let model_raw = rect_edgels(0.0, 0.0, 30.0, 15.0);
    println!(
        "  Model edgels before EdgeModel::from_edgels: {}",
        model_raw.len()
    );
    let model = EdgeModel::from_edgels(model_raw, 5);
    println!(
        "  Model centroid: ({:.1}, {:.1}), chamfer map: {}x{}",
        model.centroid.x, model.centroid.y, model.chamfer_w, model.chamfer_h
    );

    // --- Step 3: scene edgels — same rectangle translated to (80, 60) ---
    println!("\nBuilding scene edgels (rectangle at (80, 60))...");
    let scene_raw = rect_edgels(80.0, 60.0, 30.0, 15.0);
    // Ensure idx matches the integer pixel position in the scene.
    let scene_edgels: Vec<Edgel> = scene_raw
        .iter()
        .map(|e| Edgel {
            p: e.p,
            n: e.n,
            strength: e.strength,
            idx: (e.p.x.round() as usize, e.p.y.round() as usize),
        })
        .collect();
    println!("  Scene edgels: {}", scene_edgels.len());

    // --- Step 4: configure matcher ---
    // Position search covers the region where the rectangle can appear.
    let cfg = RigidMatchConfig {
        angle_range: (-0.1, 0.1),
        angle_step: 0.05,
        position_search: Rect2f {
            x: 0.0,
            y: 0.0,
            width: 200.0,
            height: 150.0,
        },
        chamfer_threshold: 5.0,
        min_score: 0.3,
        refine_icp: true,
        top_k: 5,
        ..RigidMatchConfig::default()
    };

    // --- Step 5: match ---
    println!("\nRunning RigidEdgeMatcher...");
    let mut matcher = RigidEdgeMatcher::new();
    let result = matcher.match_model(&model, &scene_edgels, &cfg);

    match &result {
        Some(r) => {
            let (tx, ty) = r.translation();
            println!("Match found:");
            println!("  tx={tx:.2}, ty={ty:.2}");
            println!("  angle={:.4} rad", r.angle());
            println!("  scale={:.4}", r.scale());
            println!("  score={:.3} (inlier fraction)", r.score);
            println!("  inlier_count={}", r.inlier_count);
            println!("  chamfer_mean={:.3} px", r.chamfer_mean);
        }
        None => {
            println!("No match found.");
        }
    }

    // --- Step 6: assertions ---
    assert!(result.is_some(), "Expected a match result, got None");
    let r = result.unwrap();
    assert!(r.score >= 0.3, "Expected score >= 0.3, got {:.3}", r.score);
    // Rigid match: scale should be 1.0.
    assert!(
        (r.scale() - 1.0).abs() < 1e-4,
        "Rigid match scale should be 1.0, got {:.6}",
        r.scale()
    );

    println!("\nAll assertions passed.");
}
