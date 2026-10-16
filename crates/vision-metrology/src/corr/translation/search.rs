use super::TranslationStatus;
use super::plane::Plane;

#[derive(Clone, Copy)]
pub(super) struct Peak {
    pub shift: [i32; 2],
    pub score: f64,
    pub support: usize,
    pub margin: f64,
}

pub(super) fn search(
    a: &Plane,
    b: &Plane,
    bounds: [i32; 4],
    min_support: usize,
    scores: &mut Vec<([i32; 2], f64, usize)>,
) -> Result<Peak, TranslationStatus> {
    scores.clear();
    let mut best: Option<Peak> = None;
    let mut support = 0;
    for dy in bounds[2]..=bounds[3] {
        for dx in bounds[0]..=bounds[1] {
            let (score, n) = correlation(a, b, [dx, dy]);
            support = support.max(n);
            if n < min_support {
                continue;
            }
            let Some(score) = score else {
                continue;
            };
            scores.push(([dx, dy], score, n));
            // Scan order (y, x) is the explicit tie break.
            if best.is_none_or(|p| score > p.score) {
                best = Some(Peak {
                    shift: [dx, dy],
                    score,
                    support: n,
                    margin: 0.0,
                });
            }
        }
    }
    let mut best = best.ok_or(if support < min_support {
        TranslationStatus::InsufficientSupport
    } else {
        TranslationStatus::LowTexture
    })?;
    let second = scores
        .iter()
        .filter(|(p, _, _)| {
            let dx = p[0] - best.shift[0];
            let dy = p[1] - best.shift[1];
            dx * dx + dy * dy >= 4
        })
        .map(|(_, s, _)| *s)
        .fold(-1.0, f64::max);
    best.margin = best.score - second;
    Ok(best)
}

fn correlation(a: &Plane, b: &Plane, shift: [i32; 2]) -> (Option<f64>, usize) {
    let [dx, dy] = shift;
    let x0 = (-dx).max(0) as usize;
    let y0 = (-dy).max(0) as usize;
    let x1 = (b.w as i32 - dx).min(a.w as i32).max(0) as usize;
    let y1 = (b.h as i32 - dy).min(a.h as i32).max(0) as usize;
    let (mut sa, mut sb, mut aa, mut bb, mut ab, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0usize);
    for y in y0..y1 {
        for x in x0..x1 {
            let i = y * a.w + x;
            let j = (y as i32 + dy) as usize * b.w + (x as i32 + dx) as usize;
            if !a.valid[i] || !b.valid[j] {
                continue;
            }
            let (u, v) = (a.data[i] as f64, b.data[j] as f64);
            sa += u;
            sb += v;
            aa += u * u;
            bb += v * v;
            ab += u * v;
            n += 1;
        }
    }
    if n < 2 {
        return (None, n);
    }
    let nf = n as f64;
    let (va, vb) = (aa - sa * sa / nf, bb - sb * sb / nf);
    if va <= 1e-12 || vb <= 1e-12 {
        return (None, n);
    }
    (
        Some(((ab - sa * sb / nf) / (va * vb).sqrt()).clamp(-1.0, 1.0)),
        n,
    )
}
