//! Per-stage timing probe for the shape matcher.
//!
//! Prints the cost of the pyramid build, each direction-field level, and the
//! full `find_u8` on a clean and on a cluttered 1280x1024 fixture, plus a
//! small config sweep on the cluttered scene. Build with
//! `--features trace-cands` to also see per-level candidate counts and stage
//! times from inside the matcher on stderr.
use std::num::NonZeroUsize;

use std::time::Instant;
use vision_metrology::matching::{
    ShapeMatcher, ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig, ShapeSearchTuning,
};
use vision_metrology::{Image, Pyramid, Rect2f};
use vm_primitives::{DirectionField, SmoothKind};

fn main() {
    // same fixture as benches/match_shape.rs cluttered_scene, abbreviated
    let (w, h) = (1280usize, 1024usize);
    let mut data = vec![40u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - 700.0) / 1.0;
            let dy = (y as f32 - 470.0) / 1.0;
            let (sn, cs) = 0.9f32.sin_cos();
            let mx = cs * dx + sn * dy;
            let my = -sn * dx + cs * dy;
            let inside = ((-90.0..90.0).contains(&mx) && (-90.0..-30.0).contains(&my))
                || ((-90.0..-30.0).contains(&mx) && (-30.0..90.0).contains(&my));
            if inside {
                data[y * w + x] = 210;
            }
        }
    }
    let mut state = 0x1234_5678_9abc_def0u64;
    for v in data.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = ((state >> 33) & 0x1f) as i32 - 16;
        *v = (i32::from(*v) + noise).clamp(0, 255) as u8;
    }
    let scene = Image::from_vec(w, h, data).unwrap();

    let reference = {
        let mut d = vec![40u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let mx = x as f32 - 640.0;
                let my = y as f32 - 512.0;
                let inside = ((-90.0..90.0).contains(&mx) && (-90.0..-30.0).contains(&my))
                    || ((-90.0..-30.0).contains(&mx) && (-30.0..90.0).contains(&my));
                if inside {
                    d[y * w + x] = 210;
                }
            }
        }
        Image::from_vec(w, h, d).unwrap()
    };
    let roi = Rect2f {
        x: 520.0,
        y: 392.0,
        width: 240.0,
        height: 240.0,
    };
    let cfg = ShapeModelConfig {
        max_points: NonZeroUsize::new(800),
        ..Default::default()
    };
    let model = ShapeModelBuilder::new()
        .build(&reference.as_view(), roi, &cfg)
        .unwrap();
    let n_lv = model.num_levels();
    println!("model levels: {n_lv}");

    // Stage 1: pyramid
    let mut pyr = Pyramid::new();
    pyr.build(&scene.as_view(), n_lv); // warm
    let t = Instant::now();
    for _ in 0..50 {
        pyr.build(&scene.as_view(), n_lv);
    }
    println!(
        "pyramid build      : {:8.3} ms",
        t.elapsed().as_secs_f64() * 1e3 / 50.0
    );

    // Stage 2: fields per level
    let mut fields: Vec<DirectionField> = (0..n_lv).map(|_| DirectionField::new()).collect();
    for (l, f) in fields.iter_mut().enumerate() {
        f.build_image_f32(pyr.level(l).unwrap(), SmoothKind::Binomial3, 10.0);
    }
    for (l, field) in fields.iter_mut().enumerate() {
        let img = pyr.level(l).unwrap();
        let t = Instant::now();
        for _ in 0..50 {
            field.build_image_f32(img, SmoothKind::Binomial3, 10.0);
        }
        println!(
            "field level {l} ({:4}x{:4}): {:8.3} ms",
            img.width(),
            img.height(),
            t.elapsed().as_secs_f64() * 1e3 / 50.0
        );
    }

    // Full find for reference
    let mut matcher = ShapeMatcher::new();
    let scfg = ShapeSearchConfig::default();
    let _ = matcher.find(&scene.as_view(), &model, &scfg);
    let t = Instant::now();
    let mut found = 0usize;
    for _ in 0..50 {
        found += matcher.find(&scene.as_view(), &model, &scfg).len();
    }
    println!(
        "find_u8 total      : {:8.3} ms   (found {} / 50)",
        t.elapsed().as_secs_f64() * 1e3 / 50.0,
        found
    );

    // Cluttered scene: texture + distractors, as in the clutter bench.
    let mut cdata = scene.data().to_vec();
    let mut state = 0x1234_5678_9abc_def0u64;
    for v in cdata.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = ((state >> 33) & 0x1f) as i32 - 16;
        *v = (i32::from(*v) + noise).clamp(0, 255) as u8;
    }
    // Shade ramp + distractor rectangles, replicating the bench fixture.
    for y in 0..h {
        for x in 0..w {
            let shade = (x as f32 * 0.01) as i32;
            let v = i32::from(cdata[y * w + x]) + shade;
            cdata[y * w + x] = v.clamp(0, 255) as u8;
        }
    }
    let mut st = 0xfeed_beef_dead_c0deu64;
    for _ in 0..40 {
        let mut next = |m: usize| {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 33) as usize) % m
        };
        let (rx, ry) = (next(w - 80), next(h - 80));
        if (rx as f32 - 700.0).abs() < 260.0 && (ry as f32 - 470.0).abs() < 260.0 {
            continue;
        }
        let (rw, rh) = (20 + next(50), 20 + next(50));
        let bright = 80 + next(120) as i32;
        for y in ry..(ry + rh).min(h) {
            for x in rx..(rx + rw).min(w) {
                cdata[y * w + x] = bright as u8;
            }
        }
    }
    let clutter = Image::from_vec(w, h, cdata).unwrap();
    let cases = [
        ("clutter default", ShapeSearchConfig::default()),
        (
            "clutter min_score 0.7",
            ShapeSearchConfig {
                min_score: 0.7,
                ..Default::default()
            },
        ),
        (
            "clutter max_cand 32",
            ShapeSearchConfig {
                tuning: ShapeSearchTuning {
                    max_candidates: 32,
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "clutter last_level 1",
            ShapeSearchConfig {
                tuning: ShapeSearchTuning {
                    last_level: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "clutter greediness 1.0",
            ShapeSearchConfig {
                tuning: ShapeSearchTuning {
                    greediness: 1.0,
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
    ];
    for (name, cfg) in cases {
        eprintln!("=== case: {name}");
        let _ = matcher.find(&clutter.as_view(), &model, &cfg);
        let t = Instant::now();
        let mut n = 0usize;
        for _ in 0..30 {
            n += matcher.find(&clutter.as_view(), &model, &cfg).len();
        }
        println!(
            "{name:24}: {:8.3} ms  (found {n}/30, truncated {})",
            t.elapsed().as_secs_f64() * 1e3 / 30.0,
            matcher.truncated()
        );
    }
}
