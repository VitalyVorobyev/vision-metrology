use criterion::{Criterion, black_box, criterion_group, criterion_main};
use vm_core::Image;
use vm_laser::{ColAccess, LaserExtractConfig, LaserExtractor, ScanAxis};

fn build_vertical_stripe(width: usize, height: usize, x_l: usize, x_r: usize) -> Image<u8> {
    let mut data = vec![0u8; width * height];
    for y in 0..height {
        for x in x_l..x_r {
            data[y * width + x] = 255;
        }
    }
    Image::from_vec(width, height, data).expect("valid image")
}

fn build_horizontal_stripe(width: usize, height: usize, y_l: usize, y_r: usize) -> Image<u8> {
    let mut data = vec![0u8; width * height];
    for y in y_l..y_r {
        for x in 0..width {
            data[y * width + x] = 255;
        }
    }
    Image::from_vec(width, height, data).expect("valid image")
}

fn bench_rows(c: &mut Criterion) {
    let img = build_vertical_stripe(1280, 512, 600, 606);
    let view = img.as_view();
    let mut ext = LaserExtractor::new(1.2);
    let cfg = LaserExtractConfig {
        axis: ScanAxis::Rows,
        ..LaserExtractConfig::default()
    };

    c.bench_function("vm_laser_rows_1280x512", |b| {
        b.iter(|| {
            let line = ext.extract_line_u8(black_box(&view), 0..512, black_box(&cfg), None);
            black_box(line.points.len());
        });
    });
}

fn bench_cols_gather(c: &mut Criterion) {
    let img = build_horizontal_stripe(512, 1280, 600, 606);
    let view = img.as_view();
    let mut ext = LaserExtractor::new(1.2);
    let cfg = LaserExtractConfig {
        axis: ScanAxis::Cols {
            access: ColAccess::Gather,
        },
        ..LaserExtractConfig::default()
    };

    c.bench_function("vm_laser_cols_gather_512x1280", |b| {
        b.iter(|| {
            let line = ext.extract_line_u8(black_box(&view), 0..512, black_box(&cfg), None);
            black_box(line.points.len());
        });
    });
}

criterion_group!(benches, bench_rows, bench_cols_gather);
criterion_main!(benches);
