use criterion::{Criterion, black_box, criterion_group, criterion_main};
use vision_metrology::{ConicFitConfig, ConicFitter, Ellipse2f, LsdConfig, LsdDetector};
use vision_metrology::{Image, Point2f, Vec2f};

// ---------------------------------------------------------------------------
// Synthetic image helpers
// ---------------------------------------------------------------------------

/// 1280×1024 image with 4 step edges (horizontal + vertical).
fn synthetic_multiline_image() -> Image<u8> {
    let w = 1280usize;
    let h = 1024usize;
    let mut data = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let v = if y > h / 4 && y < 3 * h / 4 && x > w / 4 && x < 3 * w / 4 {
                200u8
            } else {
                50u8
            };
            data[y * w + x] = v;
        }
    }
    Image::from_vec(w, h, data).expect("valid image")
}

/// Generate 1000 noisy points around a known ellipse.
fn ellipse_point_set() -> Vec<Point2f> {
    use core::f32::consts::PI;
    let ell = Ellipse2f {
        center: Point2f { x: 300.0, y: 200.0 },
        semi_axes: Vec2f { x: 100.0, y: 50.0 },
        angle: PI / 6.0,
    };
    // Inliers
    let mut pts: Vec<Point2f> = (0..800)
        .map(|i| {
            let t = 2.0 * PI * i as f32 / 800.0;
            ell.point_at(t)
        })
        .collect();
    // Outliers (20% of 1000)
    for i in 0..200usize {
        pts.push(Point2f {
            x: (i * 37 % 600) as f32,
            y: (i * 53 % 400) as f32,
        });
    }
    pts
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_lsd_detect_u8_1280x1024(c: &mut Criterion) {
    let img = synthetic_multiline_image();
    let mut det = LsdDetector::new();
    let cfg = LsdConfig::default();

    // Warm up scratch buffers
    let _ = det.detect(&img.as_view(), &cfg);

    c.bench_function("lsd_detect_u8_1280x1024", |b| {
        b.iter(|| {
            let segs = det.detect(black_box(&img.as_view()), black_box(&cfg));
            black_box(segs)
        })
    });
}

fn bench_lsd_detect_u8_512x512(c: &mut Criterion) {
    let w = 512usize;
    let h = 512usize;
    let data: Vec<u8> = (0..w * h)
        .map(|i| if i / w > h / 3 { 200u8 } else { 50u8 })
        .collect();
    let img = Image::from_vec(w, h, data).expect("valid");
    let mut det = LsdDetector::new();
    let cfg = LsdConfig::default();
    let _ = det.detect(&img.as_view(), &cfg);

    c.bench_function("lsd_detect_u8_512x512", |b| {
        b.iter(|| {
            let segs = det.detect(black_box(&img.as_view()), black_box(&cfg));
            black_box(segs)
        })
    });
}

fn bench_conic_ransac_1000pts(c: &mut Criterion) {
    let pts = ellipse_point_set();
    let mut fitter = ConicFitter::new();
    let cfg = ConicFitConfig {
        use_bookstein: false,
        ransac_iters: 200,
        inlier_tol: 2.0,
        min_inliers: 600,
        rng_seed: 42,
    };

    // Warm up
    let _ = fitter.fit_ellipse_ransac(&pts, &cfg);

    c.bench_function("conic_ransac_1000pts", |b| {
        b.iter(|| {
            let r = fitter.fit_ellipse_ransac(black_box(&pts), black_box(&cfg));
            black_box(r)
        })
    });
}

fn bench_conic_direct_fit(c: &mut Criterion) {
    use core::f32::consts::PI;
    let ell = Ellipse2f {
        center: Point2f { x: 50.0, y: 40.0 },
        semi_axes: Vec2f { x: 20.0, y: 10.0 },
        angle: 0.0,
    };
    let pts: Vec<Point2f> = (0..100)
        .map(|i| ell.point_at(2.0 * PI * i as f32 / 100.0))
        .collect();
    let mut fitter = ConicFitter::new();
    let cfg = ConicFitConfig::default();

    c.bench_function("conic_direct_bookstein_100pts", |b| {
        b.iter(|| {
            let r = fitter.fit(black_box(&pts), black_box(&cfg));
            black_box(r)
        })
    });
}

criterion_group!(
    benches,
    bench_lsd_detect_u8_1280x1024,
    bench_lsd_detect_u8_512x512,
    bench_conic_ransac_1000pts,
    bench_conic_direct_fit,
);
criterion_main!(benches);
