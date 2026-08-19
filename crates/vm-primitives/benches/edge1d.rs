use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vm_primitives::{BorderMode, Edge1DConfig, Edge1DDetector, SubpixRefine};

/// A row of Gaussian-blurred stripes, the signal shape the laser path feeds
/// this detector per scan line.
fn stripe_row(len: usize) -> Vec<u8> {
    let mut out = vec![20u8; len];
    for k in 0..8 {
        let center = 80.0 + 150.0 * k as f32;
        for (i, v) in out.iter_mut().enumerate() {
            let d = (i as f32 - center) / 3.0;
            let g = 200.0 * (-0.5 * d * d).exp();
            *v = (*v).max((20.0 + g) as u8);
        }
    }
    out
}

fn bench_edge1d(c: &mut Criterion) {
    let row = stripe_row(1280);
    let cfg = Edge1DConfig {
        sigma: 1.2,
        border: BorderMode::Clamp,
        pos_thresh: 1.0,
        neg_thresh: 1.0,
        refine: SubpixRefine::Parabolic3,
    };
    let mut det = Edge1DDetector::new(cfg.sigma);

    c.bench_function("edge1d_detect_u8_row1280", |b| {
        b.iter(|| {
            let peaks = det.detect_in_ref(black_box(&row), black_box(&cfg));
            black_box(peaks.len());
        });
    });
}

criterion_group!(benches, bench_edge1d);
criterion_main!(benches);
