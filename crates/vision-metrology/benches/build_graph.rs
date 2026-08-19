use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vision_metrology::Edgel;
use vision_metrology::contour::{
    Connectivity, ContourBuildConfig, build_graph_from_edgels, smooth_polyline,
};
use vision_metrology::{Point2f, Vec2f};

fn synthetic_edgels(width: usize, height: usize) -> Vec<Edgel> {
    let mut out = Vec::with_capacity(52_000);

    for y in (16..height.saturating_sub(16)).step_by(20) {
        for x in 32..width.saturating_sub(32) {
            out.push(Edgel {
                p: Point2f::new(x as f32, y as f32),
                n: Vec2f::new(1.0, 0.0),
                strength: 1.0,
                idx: (x, y),
            });
        }
    }

    for x in (64..width.saturating_sub(64)).step_by(80) {
        for y in 64..height.saturating_sub(64) {
            if y % 8 == 0 {
                out.push(Edgel {
                    p: Point2f::new(x as f32, y as f32),
                    n: Vec2f::new(0.0, 1.0),
                    strength: 0.8,
                    idx: (x, y),
                });
            }
        }
    }

    out
}

fn bench_build_graph(c: &mut Criterion) {
    let width = 1280;
    let height = 1024;
    let edgels = synthetic_edgels(width, height);

    let cfg = ContourBuildConfig {
        connectivity: Connectivity::C8,
        min_component_size: 2,
        record_strengths: false,
        record_geometry: false,
        ..Default::default()
    };

    c.bench_function("contour_build_graph_50k", |b| {
        b.iter(|| {
            let g = build_graph_from_edgels(width, height, black_box(&edgels), black_box(&cfg));
            black_box((g.nodes.len(), g.edges.len()));
        });
    });
}

fn bench_smooth_polyline(c: &mut Criterion) {
    // A 5k-point wiggly contour, the size a full-resolution part outline
    // reaches on a 1280x1024 frame.
    let points: Vec<Point2f> = (0..5000)
        .map(|i| {
            let t = i as f32 * 0.02;
            Point2f::new(
                640.0 + (300.0 + 5.0 * (13.0 * t).sin()) * t.cos(),
                512.0 + (300.0 + 5.0 * (13.0 * t).sin()) * t.sin(),
            )
        })
        .collect();

    c.bench_function("contour_smooth_polyline_5k_sigma2", |b| {
        b.iter(|| {
            let out = smooth_polyline(black_box(&points), black_box(2.0));
            black_box(out.len());
        });
    });
}

criterion_group!(benches, bench_build_graph, bench_smooth_polyline);
criterion_main!(benches);
