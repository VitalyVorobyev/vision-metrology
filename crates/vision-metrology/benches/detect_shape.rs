use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vision_metrology::fit::{
    FitConfig, RansacConfig, RobustLoss, fit_circle, fit_ellipse, fit_line,
};
use vision_metrology::{Circle2f, Ellipse2f, LsdConfig, LsdDetector};
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
        center: Point2f::new(300.0, 200.0),
        semi_axes: Vec2f::new(100.0, 50.0),
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
        pts.push(Point2f::new((i * 37 % 600) as f32, (i * 53 % 400) as f32));
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

fn bench_ellipse_ransac_1000pts(c: &mut Criterion) {
    let pts = ellipse_point_set();
    let cfg = FitConfig {
        ransac: Some(RansacConfig {
            iters: 200,
            inlier_tol: 2.0,
            min_inliers: 600,
            seed: 42,
        }),
        ..FitConfig::default()
    };
    let _ = fit_ellipse(&pts, &cfg); // warm up

    c.bench_function("fit_ellipse_ransac_1000pts", |b| {
        b.iter(|| black_box(fit_ellipse(black_box(&pts), black_box(&cfg))))
    });
}

fn bench_ellipse_direct(c: &mut Criterion) {
    use core::f32::consts::PI;
    let ell = Ellipse2f {
        center: Point2f::new(50.0, 40.0),
        semi_axes: Vec2f::new(20.0, 10.0),
        angle: 0.0,
    };
    let pts: Vec<Point2f> = (0..100)
        .map(|i| ell.point_at(2.0 * PI * i as f32 / 100.0))
        .collect();
    let cfg = FitConfig::default();

    c.bench_function("fit_ellipse_100pts", |b| {
        b.iter(|| black_box(fit_ellipse(black_box(&pts), black_box(&cfg))))
    });
}

/// The two fits this crate previously had no implementation of at all.
fn bench_circle_and_line(c: &mut Criterion) {
    use core::f32::consts::TAU;
    let truth = Circle2f {
        center: Point2f::new(300.0, 200.0),
        radius: 120.0,
    };
    let circle_pts: Vec<Point2f> = (0..500)
        .map(|i| truth.point_at(i as f32 * TAU / 500.0))
        .collect();
    let plain = FitConfig::default();
    let robust = FitConfig {
        loss: RobustLoss::Tukey { c: 2.0 },
        ..FitConfig::default()
    };

    c.bench_function("fit_circle_500pts", |b| {
        b.iter(|| black_box(fit_circle(black_box(&circle_pts), black_box(&plain))))
    });
    c.bench_function("fit_circle_500pts_tukey", |b| {
        b.iter(|| black_box(fit_circle(black_box(&circle_pts), black_box(&robust))))
    });

    let line_pts: Vec<Point2f> = (0..500)
        .map(|i| Point2f::new(i as f32 * 0.7, 40.0 + (i % 3) as f32 * 0.1))
        .collect();
    c.bench_function("fit_line_500pts", |b| {
        b.iter(|| black_box(fit_line(black_box(&line_pts), black_box(&plain))))
    });
    c.bench_function("fit_line_500pts_tukey", |b| {
        b.iter(|| black_box(fit_line(black_box(&line_pts), black_box(&robust))))
    });
}

criterion_group!(
    benches,
    bench_lsd_detect_u8_1280x1024,
    bench_lsd_detect_u8_512x512,
    bench_ellipse_ransac_1000pts,
    bench_ellipse_direct,
    bench_circle_and_line,
);
criterion_main!(benches);
