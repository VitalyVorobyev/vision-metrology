use criterion::{Criterion, black_box, criterion_group, criterion_main};
use vision_metrology::edge::edge2d::Edgel;
use vision_metrology::{EdgeModel, RigidEdgeMatcher, RigidMatchConfig};
use vision_metrology::{Point2f, Rect2f, Vec2f};

fn make_edgel(x: f32, y: f32, nx: f32, ny: f32) -> Edgel {
    Edgel {
        p: Point2f { x, y },
        n: Vec2f { x: nx, y: ny },
        strength: 1.0,
        idx: (x as usize, y as usize),
    }
}

/// 20-edgel rectangular model.
fn build_model() -> EdgeModel {
    let n = 20usize;
    let edgels: Vec<Edgel> = (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let (x, y, nx, ny) = if t < 0.25 {
                (t * 4.0 * 20.0, 0.0, 0.0, -1.0f32)
            } else if t < 0.5 {
                (20.0, (t - 0.25) * 4.0 * 10.0, 1.0, 0.0)
            } else if t < 0.75 {
                (20.0 - (t - 0.5) * 4.0 * 20.0, 10.0, 0.0, 1.0)
            } else {
                (0.0, 10.0 - (t - 0.75) * 4.0 * 10.0, -1.0, 0.0)
            };
            make_edgel(x, y, nx, ny)
        })
        .collect();
    EdgeModel::from_edgels(edgels, 5)
}

/// Scatter ~200 scene edgels across a 1280×1024 region.
fn build_scene_edgels(w: usize, h: usize) -> Vec<Edgel> {
    let mut v = Vec::new();
    // Sparse grid of edgels.
    let step = 32usize;
    for y in (0..h).step_by(step) {
        for x in (0..w).step_by(step) {
            v.push(make_edgel(x as f32, y as f32, 1.0, 0.0));
        }
    }
    v
}

fn bench_rigid_match(c: &mut Criterion) {
    let model = build_model();
    let scene = build_scene_edgels(1280, 1024);
    let cfg = RigidMatchConfig {
        angle_range: (-0.5, 0.5),
        angle_step: 0.1,
        position_search: Rect2f {
            x: 0.0,
            y: 0.0,
            width: 200.0,
            height: 200.0,
        },
        chamfer_threshold: 5.0,
        min_score: 0.3,
        refine_icp: false,
        top_k: 5,
        resolution_factor: 1.0,
        ..RigidMatchConfig::default()
    };

    let mut matcher = RigidEdgeMatcher::new();
    c.bench_function("rigid_match_1280x1024_20edgel_model", |b| {
        b.iter(|| matcher.match_model(black_box(&model), black_box(&scene), black_box(&cfg)))
    });
}

criterion_group!(benches, bench_rigid_match);
criterion_main!(benches);
