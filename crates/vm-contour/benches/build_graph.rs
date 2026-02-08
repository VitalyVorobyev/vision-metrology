use criterion::{Criterion, black_box, criterion_group, criterion_main};
use vm_contour::{Connectivity, ContourBuildConfig, build_graph_from_edgels};
use vm_core::{Point2f, Vec2f};
use vm_edge::edge2d::Edgel;

fn synthetic_edgels(width: usize, height: usize) -> Vec<Edgel> {
    let mut out = Vec::with_capacity(52_000);

    for y in (16..height.saturating_sub(16)).step_by(20) {
        for x in 32..width.saturating_sub(32) {
            out.push(Edgel {
                p: Point2f {
                    x: x as f32,
                    y: y as f32,
                },
                n: Vec2f { x: 1.0, y: 0.0 },
                strength: 1.0,
                idx: (x, y),
            });
        }
    }

    for x in (64..width.saturating_sub(64)).step_by(80) {
        for y in 64..height.saturating_sub(64) {
            if y % 8 == 0 {
                out.push(Edgel {
                    p: Point2f {
                        x: x as f32,
                        y: y as f32,
                    },
                    n: Vec2f { x: 0.0, y: 1.0 },
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
    };

    c.bench_function("vm_contour_build_graph_50k", |b| {
        b.iter(|| {
            let g = build_graph_from_edgels(width, height, black_box(&edgels), black_box(&cfg));
            black_box((g.nodes.len(), g.edges.len()));
        });
    });
}

criterion_group!(benches, bench_build_graph);
criterion_main!(benches);
