//! `corr` module: wrapper-level correctness (position, angle, error paths)
//! independent of the coordinate-convention cross-check in
//! `tests/corrmatch_bridge.rs`.

use core::num::NonZeroUsize;

use vision_metrology::corr::{
    CorrConfig, CorrTemplate, CorrTemplateConfig, DisplacementConfig, Refine, displacement, find,
    find_topk,
};
use vision_metrology::{Error, Image, Rect2f};

const W: usize = 320;
const H: usize = 240;

/// Integer-hash pseudo-random value in `[0, 1)`; deterministic, no crate.
fn hash01(ix: i32, iy: i32) -> f32 {
    let mut h = (ix as i64)
        .wrapping_mul(374_761_393)
        .wrapping_add((iy as i64).wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    ((h & 0xFFFF) as f32) / 65535.0
}

/// Smooth value noise (bilinear-interpolated integer-grid hash, Hermite
/// eased) at grid spacing `cell`. Aperiodic over any finite domain — unlike
/// a sum of sinusoids, it has no repeating false ZNCC peak.
fn value_noise(x: f32, y: f32, cell: f32) -> f32 {
    let gx = x / cell;
    let gy = y / cell;
    let x0 = gx.floor() as i32;
    let y0 = gy.floor() as i32;
    let fx = gx - x0 as f32;
    let fy = gy - y0 as f32;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let a = hash01(x0, y0) * (1.0 - sx) + hash01(x0 + 1, y0) * sx;
    let b = hash01(x0, y0 + 1) * (1.0 - sx) + hash01(x0 + 1, y0 + 1) * sx;
    a * (1.0 - sy) + b * sy
}

/// Two-octave noise pattern, rigidly rotated by `angle` about `(cx, cy)`:
/// enough local structure for ZNCC to have one sharp, unambiguous peak (a
/// periodic pattern would give it several near-equal ones), and non-flat
/// down to a coarse pyramid level (corrmatch rejects zero-variance patches
/// as degenerate).
fn textured(w: usize, h: usize, cx: f32, cy: f32, angle: f32) -> Image<u8> {
    let (sn, cs) = angle.sin_cos();
    let mut data = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let mx = cs * dx + sn * dy;
            let my = -sn * dx + cs * dy;
            let v = 128.0
                + 90.0 * (value_noise(mx, my, 9.0) - 0.5)
                + 40.0 * (value_noise(mx, my, 2.5) - 0.5);
            data[y * w + x] = v.clamp(0.0, 255.0) as u8;
        }
    }
    Image::from_vec(w, h, data).expect("valid image")
}

fn rect() -> Rect2f {
    Rect2f {
        x: 100.0,
        y: 70.0,
        width: 48.0,
        height: 48.0,
    }
}

/// A shallow pyramid: enough for a 48x48 template searched near identity
/// shift, and avoids a coarse level (a handful of pixels after four 2x2
/// box-downsamples) where this fixture's periodic texture can alias down to
/// near-zero variance and corrmatch's own `DegenerateTemplate` check trips.
fn tpl_cfg(rotation: bool) -> CorrTemplateConfig {
    CorrTemplateConfig {
        rotation,
        max_levels: NonZeroUsize::new(3),
        ..Default::default()
    }
}

#[test]
fn find_recovers_a_pure_translation() {
    let reference = textured(W, H, W as f32 / 2.0, H as f32 / 2.0, 0.0);
    let r = rect();
    let template = CorrTemplate::from_image(&reference.as_view(), r, &tpl_cfg(true))
        .expect("template compiles");
    assert_eq!(template.width(), 48);
    assert_eq!(template.height(), 48);

    let (dx, dy) = (7.0, -4.0);
    let scene = textured(W, H, W as f32 / 2.0 + dx, H as f32 / 2.0 + dy, 0.0);
    let m = find(&template, &scene.as_view(), &CorrConfig::default()).expect("find succeeds");

    // `find` reports corrmatch's own quadratic-peak subpixel refinement, not
    // this crate's Lucas-Kanade stage (that only exists for `displacement`)
    // — 1 px is a loose but appropriate bound for that.
    let expected = r.center() + nalgebra::Vector2::new(dx, dy);
    assert!(
        (m.position - expected).norm() < 1.0,
        "position {:?} vs expected {:?}",
        m.position,
        expected
    );
    assert!(m.score > 0.9, "score {}", m.score);
}

#[test]
fn find_recovers_rotation_when_enabled() {
    let reference = textured(W, H, W as f32 / 2.0, H as f32 / 2.0, 0.0);
    let r = rect();
    let template = CorrTemplate::from_image(&reference.as_view(), r, &tpl_cfg(true))
        .expect("template compiles");
    assert!(template.is_rotated());

    let truth_angle = 12f32.to_radians();
    let scene = textured(W, H, W as f32 / 2.0, H as f32 / 2.0, truth_angle);
    let search = CorrConfig {
        rotation: true,
        ..Default::default()
    };
    let m = find(&template, &scene.as_view(), &search).expect("find succeeds");

    assert!(
        (m.angle - truth_angle).abs() < 0.03,
        "angle {} vs truth {}",
        m.angle,
        truth_angle
    );
}

#[test]
fn rotation_search_without_a_rotated_template_is_an_error() {
    let reference = textured(W, H, W as f32 / 2.0, H as f32 / 2.0, 0.0);
    let template = CorrTemplate::from_image(&reference.as_view(), rect(), &tpl_cfg(false))
        .expect("template compiles");
    assert!(!template.is_rotated());

    let scene = textured(W, H, W as f32 / 2.0, H as f32 / 2.0, 0.0);
    let search = CorrConfig {
        rotation: true,
        ..Default::default()
    };
    let err = find(&template, &scene.as_view(), &search).unwrap_err();
    assert!(matches!(err, Error::InvalidConfig(_)));
}

#[test]
fn find_topk_returns_scores_in_descending_order() {
    let reference = textured(W, H, W as f32 / 2.0, H as f32 / 2.0, 0.0);
    let template = CorrTemplate::from_image(&reference.as_view(), rect(), &tpl_cfg(true))
        .expect("template compiles");
    let scene = textured(W, H, W as f32 / 2.0 + 2.0, H as f32 / 2.0, 0.0);
    let ms = find_topk(&template, &scene.as_view(), 3, &CorrConfig::default())
        .expect("find_topk succeeds");
    assert!(!ms.is_empty());
    for pair in ms.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
}

#[test]
fn displacement_recovers_a_subpixel_shift_with_lk() {
    let prev = textured(W, H, W as f32 / 2.0, H as f32 / 2.0, 0.0);
    let (dx, dy) = (2.3f32, -1.6f32);
    let curr = textured(W, H, W as f32 / 2.0 + dx, H as f32 / 2.0 + dy, 0.0);

    let cfg = DisplacementConfig {
        window: Rect2f {
            x: 100.0,
            y: 70.0,
            width: 100.0,
            height: 80.0,
        },
        search: (8, 8),
        refine: Refine::LucasKanade { iters: 5 },
        min_score: 0.5,
    };
    let d = displacement(&prev.as_view(), &curr.as_view(), &cfg).expect("displacement succeeds");
    assert!(
        (d.shift.x - dx).abs() < 0.15,
        "shift.x {} vs {dx}",
        d.shift.x
    );
    assert!(
        (d.shift.y - dy).abs() < 0.15,
        "shift.y {} vs {dy}",
        d.shift.y
    );
    assert!(d.score > 0.5);
}

#[test]
fn displacement_window_must_overlap_prev() {
    let prev = textured(W, H, W as f32 / 2.0, H as f32 / 2.0, 0.0);
    let curr = textured(W, H, W as f32 / 2.0, H as f32 / 2.0, 0.0);
    let cfg = DisplacementConfig {
        window: Rect2f {
            x: 10_000.0,
            y: 10_000.0,
            width: 40.0,
            height: 40.0,
        },
        ..Default::default()
    };
    let err = displacement(&prev.as_view(), &curr.as_view(), &cfg).unwrap_err();
    assert!(matches!(err, Error::InvalidConfig(_)));
}

#[test]
fn displacement_below_min_score_is_degenerate() {
    let prev = textured(W, H, W as f32 / 2.0, H as f32 / 2.0, 0.0);
    // Unrelated pattern: stage-1 ZNCC should score poorly.
    let mut curr_data = vec![0u8; W * H];
    for (i, p) in curr_data.iter_mut().enumerate() {
        *p = ((i * 2654435761u64 as usize) % 256) as u8;
    }
    let curr = Image::from_vec(W, H, curr_data).expect("valid image");

    let cfg = DisplacementConfig {
        window: rect(),
        min_score: 0.9,
        ..Default::default()
    };
    let err = displacement(&prev.as_view(), &curr.as_view(), &cfg).unwrap_err();
    assert!(matches!(err, Error::Degenerate(_)));
}
