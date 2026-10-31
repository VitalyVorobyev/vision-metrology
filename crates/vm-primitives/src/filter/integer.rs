//! Exact integer box mean and binomial filter on `u8` images (ADR-0023, Phase 2).
//!
//! All arithmetic is integer, so the output is bit-identical to the naive reference on every
//! target. The horizontal pass pads each row once through the border, the vertical pass
//! keeps running column sums (box) or weighted row sums (binomial), and the rounding divide
//! happens once per output pixel. Box divides by `(2r+1)²`, using a float reciprocal with an
//! exact integer correction; binomial divides by `4^order`, which is a shift.
//!
//! The per-row loops are written as plain slice code and compiled for each CPU level by
//! `multiversion` (safe, no `unsafe`; ADR-0023 decision 4).

use multiversion::multiversion;

use crate::core::{BorderMode, map_index};

/// Where sample `i` comes from: itself when in bounds, otherwise whatever `border` says.
/// `None` means "use the border constant" (`map_index` returns `None` only for `Constant`).
#[inline(always)]
fn resolve(i: isize, len: usize, border: &BorderMode<u8>) -> Option<usize> {
    if i >= 0 && (i as usize) < len {
        Some(i as usize)
    } else {
        map_index(i, len, border)
    }
}

/// The value a `Constant` border supplies; unused (0) for the other modes.
#[inline(always)]
fn fill_value(border: &BorderMode<u8>) -> u32 {
    match border {
        BorderMode::Constant(c) => *c as u32,
        _ => 0,
    }
}

/// Reusable scratch for the integer filters. It grows to the largest size seen.
#[derive(Default, Debug)]
pub(crate) struct IntScratch {
    /// Horizontally filtered image, `width · height`.
    hsum: Vec<u32>,
    /// One padded source row, `width + 2r`.
    pad: Vec<u32>,
    /// Vertical accumulator for one output row, `width`.
    acc: Vec<u32>,
    /// Horizontal result for a row that lies entirely in a `Constant` border, `width`.
    konst: Vec<u32>,
}

impl IntScratch {
    fn ensure(&mut self, width: usize, height: usize, r: usize) {
        grow(&mut self.hsum, width * height);
        grow(&mut self.pad, width + 2 * r);
        grow(&mut self.acc, width);
        grow(&mut self.konst, width);
    }
}

fn grow(v: &mut Vec<u32>, len: usize) {
    if v.len() < len {
        v.resize(len, 0);
    }
}

/// Copy `row` into `pad` with `r` samples of border on each side: `pad[k]` is the sample at
/// `k - r`. The interior is a straight copy; only the `2r` border entries call `resolve`.
#[inline(always)]
fn pad_row(row: &[u8], r: usize, border: &BorderMode<u8>, fill: u32, pad: &mut [u32]) {
    let w = row.len();
    for (p, &v) in pad[r..r + w].iter_mut().zip(row) {
        *p = v as u32;
    }
    for k in 0..r {
        let left = k as isize - r as isize;
        pad[k] = match resolve(left, w, border) {
            Some(i) => row[i] as u32,
            None => fill,
        };
        let right = (w + k) as isize;
        pad[r + w + k] = match resolve(right, w, border) {
            Some(i) => row[i] as u32,
            None => fill,
        };
    }
}

/// The row of horizontal sums at image row `i`, resolved by `border`. A row outside the
/// image in a `Constant` border is the constant row `konst`.
#[inline(always)]
fn row_of<'a>(
    hsum: &'a [u32],
    konst: &'a [u32],
    w: usize,
    h: usize,
    i: isize,
    border: &BorderMode<u8>,
) -> &'a [u32] {
    match resolve(i, h, border) {
        Some(j) => &hsum[j * w..(j + 1) * w],
        None => konst,
    }
}

/// `round_half_up(s / d)`, computed as `floor((2s + d) / 2d)` in exact integer arithmetic.
///
/// The quotient comes from a float reciprocal, which can be one too high or low. The
/// remainder then decides the correction, so the result is always exact. The bounds hold
/// for `d < 2²²` (see [`box_mean_u8`]): `2s + d < 2³¹` and the float error is below one.
#[inline(always)]
fn round_div(s: u32, d: u32, inv: f32) -> u8 {
    let x = (2 * s + d) as i32;
    let den = (2 * d) as i32;
    let q0 = (x as f32 * inv) as i32;
    let rem = x - q0 * den;
    (q0 + (rem >= den) as i32 - (rem < 0) as i32) as u8
}

/// Up to this window width the horizontal pass sums taps directly (vectorises), beyond it
/// it switches to a running sum (O(1) per pixel).
const DIRECT_WINDOW: usize = 9;

/// Box body: horizontal running sums, vertical running sums across rows, one divide per pixel.
#[inline(always)]
fn box_body(
    src: &[u8],
    w: usize,
    h: usize,
    r: usize,
    border: &BorderMode<u8>,
    s: &mut IntScratch,
    dst: &mut [u8],
) {
    let win = 2 * r + 1;
    let d = (win * win) as u32;
    let inv = 1.0f32 / (2 * d) as f32;
    let fill = fill_value(border);
    let IntScratch {
        hsum,
        pad,
        acc,
        konst,
    } = s;

    for y in 0..h {
        let pad = &mut pad[..w + 2 * r];
        pad_row(&src[y * w..(y + 1) * w], r, border, fill, pad);
        let out = &mut hsum[y * w..(y + 1) * w];
        if win <= DIRECT_WINDOW {
            // Sum the window tap by tap: each pass is a straight vector add over the row.
            out.fill(0);
            for j in 0..win {
                for (o, &p) in out.iter_mut().zip(&pad[j..j + w]) {
                    *o += p;
                }
            }
        } else {
            let mut run: u32 = pad[..win].iter().sum();
            out[0] = run;
            for x in 1..w {
                run = run + pad[x + 2 * r] - pad[x - 1];
                out[x] = run;
            }
        }
    }
    let konst = &mut konst[..w];
    konst.fill(win as u32 * fill);
    let hsum = &hsum[..w * h];
    let konst: &[u32] = konst;
    let acc = &mut acc[..w];

    // The window at row 0 is rows -r..=r.
    acc.fill(0);
    for i in -(r as isize)..=(r as isize) {
        let row = row_of(hsum, konst, w, h, i, border);
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += v;
        }
    }
    for y in 0..h {
        if y > 0 {
            // Slide the window down: add row y+r, remove row y-1-r. Both are resolved the
            // same way, so the window is always the multiset of resolved rows.
            let add = row_of(hsum, konst, w, h, (y + r) as isize, border);
            let sub = row_of(hsum, konst, w, h, y as isize - r as isize - 1, border);
            for ((a, &p), &m) in acc.iter_mut().zip(add).zip(sub) {
                *a = *a + p - m;
            }
        }
        for (o, &v) in dst[y * w..(y + 1) * w].iter_mut().zip(acc.iter()) {
            *o = round_div(v, d, inv);
        }
    }
}

/// Exact box mean of width `2·radius + 1` on `u8`, rounded half up, with `border` for
/// out-of-image samples in both directions. `dst` gets `width · height` pixels.
///
/// # Panics
/// Panics when the slices do not match `width · height`, or when `(2·radius + 1)² ≥ 2²²`
/// (the rounding divide is exact only below that bound).
pub(crate) fn box_mean_u8(
    src: &[u8],
    width: usize,
    height: usize,
    radius: usize,
    border: &BorderMode<u8>,
    scratch: &mut IntScratch,
    dst: &mut [u8],
) {
    check_sizes(src, width, height, dst);
    let win = 2 * radius + 1;
    assert!(
        win * win < 1 << 22,
        "box window too large for exact rounding"
    );
    if width == 0 || height == 0 {
        return;
    }
    scratch.ensure(width, height, radius);
    box_dispatch(src, width, height, radius, border, scratch, dst);
}

#[multiversion(targets = "simd")]
fn box_dispatch(
    src: &[u8],
    width: usize,
    height: usize,
    radius: usize,
    border: &BorderMode<u8>,
    scratch: &mut IntScratch,
    dst: &mut [u8],
) {
    box_body(src, width, height, radius, border, scratch, dst);
}

/// Binomial body: integer weights `C(order, k)`, separable, divided by `4^order` by a shift.
#[inline(always)]
fn binomial_body(
    src: &[u8],
    w: usize,
    h: usize,
    order: usize,
    border: &BorderMode<u8>,
    s: &mut IntScratch,
    dst: &mut [u8],
) {
    let r = order / 2;
    let mut coef = [0u32; MAX_ORDER + 1];
    binomial_row(order, &mut coef);
    let coef = &coef[..=order];
    let shift = 2 * order as u32;
    let half = 1u32 << (shift - 1);
    let fill = fill_value(border) * (1u32 << order);
    let IntScratch {
        hsum,
        pad,
        acc,
        konst,
    } = s;

    for y in 0..h {
        let pad = &mut pad[..w + 2 * r];
        pad_row(&src[y * w..(y + 1) * w], r, border, fill_value(border), pad);
        let out = &mut hsum[y * w..(y + 1) * w];
        out.fill(0);
        for (k, &c) in coef.iter().enumerate() {
            for (o, &p) in out.iter_mut().zip(&pad[k..k + w]) {
                *o += c * p;
            }
        }
    }
    let konst = &mut konst[..w];
    konst.fill(fill);
    let hsum = &hsum[..w * h];
    let konst: &[u32] = konst;
    let acc = &mut acc[..w];

    for y in 0..h {
        acc.fill(0);
        for (k, &c) in coef.iter().enumerate() {
            let row = row_of(hsum, konst, w, h, (y + k) as isize - r as isize, border);
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += c * v;
            }
        }
        for (o, &v) in dst[y * w..(y + 1) * w].iter_mut().zip(acc.iter()) {
            *o = ((v + half) >> shift) as u8;
        }
    }
}

/// Largest binomial order handled: the intermediate sums stay below `255 · 4^order < 2³²`.
const MAX_ORDER: usize = 12;

/// Pascal's row `C(order, 0..=order)` into `out`.
fn binomial_row(order: usize, out: &mut [u32]) {
    out[0] = 1;
    for k in 1..=order {
        out[k] = out[k - 1] * (order - k + 1) as u32 / k as u32;
    }
}

/// Binomial smoothing of order `order` (even, `2 ≤ order ≤ 12`) on `u8`: the separable
/// kernel `C(order, k) / 2^order` in each direction, rounded half up to `u8`. The kernel
/// has radius `order / 2`, and `border` resolves out-of-image samples.
///
/// # Panics
/// Panics when the slices do not match `width · height`, or `order` is odd or out of range.
pub(crate) fn binomial_u8(
    src: &[u8],
    width: usize,
    height: usize,
    order: usize,
    border: &BorderMode<u8>,
    scratch: &mut IntScratch,
    dst: &mut [u8],
) {
    check_sizes(src, width, height, dst);
    assert!(
        order.is_multiple_of(2) && (2..=MAX_ORDER).contains(&order),
        "binomial order must be even and in 2..=12, got {order}"
    );
    if width == 0 || height == 0 {
        return;
    }
    scratch.ensure(width, height, order / 2);
    binomial_dispatch(src, width, height, order, border, scratch, dst);
}

#[multiversion(targets = "simd")]
fn binomial_dispatch(
    src: &[u8],
    width: usize,
    height: usize,
    order: usize,
    border: &BorderMode<u8>,
    scratch: &mut IntScratch,
    dst: &mut [u8],
) {
    binomial_body(src, width, height, order, border, scratch, dst);
}

fn check_sizes(src: &[u8], width: usize, height: usize, dst: &[u8]) {
    assert_eq!(src.len(), width * height, "src must be width · height");
    assert_eq!(dst.len(), width * height, "dst must be width · height");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic 8-bit fixture from a 64-bit LCG (no RNG crate).
    fn pattern(w: usize, h: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..w * h)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 56) as u8
            })
            .collect()
    }

    /// Border rule written out independently of `map_index`: `None` is the constant.
    fn naive_index(i: isize, len: usize, border: &BorderMode<u8>) -> Option<usize> {
        match border {
            BorderMode::Constant(_) => (i >= 0 && (i as usize) < len).then_some(i as usize),
            BorderMode::Clamp => Some(i.clamp(0, len as isize - 1) as usize),
            BorderMode::Reflect101 => {
                if len == 1 {
                    return Some(0);
                }
                let period = 2 * (len as isize - 1);
                let mut j = i.rem_euclid(period);
                if j >= len as isize {
                    j = period - j;
                }
                Some(j as usize)
            }
        }
    }

    fn sample(src: &[u8], w: usize, h: usize, x: isize, y: isize, border: &BorderMode<u8>) -> u32 {
        let fill = match border {
            BorderMode::Constant(c) => *c as u32,
            _ => 0,
        };
        match (naive_index(x, w, border), naive_index(y, h, border)) {
            (Some(i), Some(j)) => src[j * w + i] as u32,
            _ => fill,
        }
    }

    /// Naive box: the exact integer sum over the window, divided with integer arithmetic.
    fn naive_box(src: &[u8], w: usize, h: usize, r: usize, border: &BorderMode<u8>) -> Vec<u8> {
        let win = (2 * r + 1) as u32;
        let d = win * win;
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0u32;
                for dy in -(r as isize)..=(r as isize) {
                    for dx in -(r as isize)..=(r as isize) {
                        s += sample(src, w, h, x as isize + dx, y as isize + dy, border);
                    }
                }
                out[y * w + x] = ((2 * s + d) / (2 * d)) as u8;
            }
        }
        out
    }

    /// Naive binomial: the 2-D weighted sum `C(n,a)·C(n,b)` over the window, divided by `4^n`.
    fn naive_binomial(
        src: &[u8],
        w: usize,
        h: usize,
        n: usize,
        border: &BorderMode<u8>,
    ) -> Vec<u8> {
        let r = n / 2;
        let mut c = [0u32; MAX_ORDER + 1];
        binomial_row(n, &mut c);
        let den = 1u32 << (2 * n);
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0u32;
                for a in 0..=n {
                    for b in 0..=n {
                        let dx = a as isize - r as isize;
                        let dy = b as isize - r as isize;
                        s += c[a]
                            * c[b]
                            * sample(src, w, h, x as isize + dx, y as isize + dy, border);
                    }
                }
                out[y * w + x] = ((2 * s + den) / (2 * den)) as u8;
            }
        }
        out
    }

    fn borders() -> Vec<BorderMode<u8>> {
        vec![
            BorderMode::Clamp,
            BorderMode::Reflect101,
            BorderMode::Constant(7),
        ]
    }

    #[test]
    fn round_div_matches_integer_division_for_small_windows() {
        for r in [0usize, 1, 2, 4, 7, 12, 30] {
            let d = ((2 * r + 1) * (2 * r + 1)) as u32;
            let inv = 1.0f32 / (2 * d) as f32;
            for s in 0..=255 * d {
                assert_eq!(
                    round_div(s, d, inv) as u32,
                    (2 * s + d) / (2 * d),
                    "s={s} d={d}"
                );
            }
        }
    }

    #[test]
    fn round_div_is_exact_at_the_largest_window() {
        // The largest window the bound allows: (2·1023+1)² < 2²².
        let d = 2047u32 * 2047;
        let inv = 1.0f32 / (2 * d) as f32;
        for k in 0..=255u32 {
            for t in [0, d / 2, d - 1] {
                let s = k * d + t;
                assert_eq!(
                    round_div(s.min(255 * d), d, inv) as u32,
                    (2 * s.min(255 * d) + d) / (2 * d)
                );
            }
        }
    }

    #[test]
    fn box_matches_naive_exact_on_every_border_and_size() {
        let sizes = [(24usize, 17usize), (100, 40), (3, 5), (1, 9), (9, 1)];
        for (w, h) in sizes {
            let src = pattern(w, h, 0x9E37_79B9_7F4A_7C15 ^ (w * 131 + h) as u64);
            for r in [0usize, 1, 2, 5, 9, 20] {
                for border in borders() {
                    let mut scratch = IntScratch::default();
                    let mut got = vec![0u8; w * h];
                    box_mean_u8(&src, w, h, r, &border, &mut scratch, &mut got);
                    assert_eq!(
                        got,
                        naive_box(&src, w, h, r, &border),
                        "{w}x{h} r={r} {border:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn binomial_matches_naive_exact_on_every_border_and_size() {
        let sizes = [(24usize, 17usize), (100, 40), (3, 5), (1, 9)];
        for (w, h) in sizes {
            let src = pattern(w, h, 0xD1B5_4A32_D192_ED03 ^ (w * 17 + h) as u64);
            for n in [2usize, 4, 6, 8] {
                for border in borders() {
                    let mut scratch = IntScratch::default();
                    let mut got = vec![0u8; w * h];
                    binomial_u8(&src, w, h, n, &border, &mut scratch, &mut got);
                    assert_eq!(
                        got,
                        naive_binomial(&src, w, h, n, &border),
                        "{w}x{h} n={n} {border:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn constant_image_is_preserved_when_border_matches() {
        let (w, h) = (31usize, 13usize);
        let src = vec![42u8; w * h];
        for border in [
            BorderMode::Clamp,
            BorderMode::Reflect101,
            BorderMode::Constant(42),
        ] {
            let mut scratch = IntScratch::default();
            let mut out = vec![0u8; w * h];
            box_mean_u8(&src, w, h, 4, &border, &mut scratch, &mut out);
            assert!(out.iter().all(|&v| v == 42), "box {border:?}");
            binomial_u8(&src, w, h, 6, &border, &mut scratch, &mut out);
            assert!(out.iter().all(|&v| v == 42), "binomial {border:?}");
        }
    }

    /// Best-of-25 after 3 warm-ups, the same protocol as `tools/filter_bench`.
    fn best_ms(mut f: impl FnMut()) -> f64 {
        for _ in 0..3 {
            f();
        }
        (0..25)
            .map(|_| {
                let t = std::time::Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e3
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Run with `cargo test --release -p vm-primitives -- --ignored --nocapture time_`.
    #[test]
    #[ignore = "timing only"]
    fn time_box_and_binomial_dispatched_and_portable() {
        let (w, h) = (1280usize, 1024usize);
        let src = pattern(w, h, 0x9E37_79B9_7F4A_7C15);
        let mut out = vec![0u8; w * h];
        let mut scratch = IntScratch::default();
        for r in [1usize, 4, 8] {
            let dispatched = best_ms(|| {
                box_mean_u8(
                    std::hint::black_box(&src),
                    w,
                    h,
                    r,
                    &BorderMode::Clamp,
                    &mut scratch,
                    &mut out,
                );
                std::hint::black_box(&out);
            });
            let portable = best_ms(|| {
                scratch.ensure(w, h, r);
                box_body(
                    std::hint::black_box(&src),
                    w,
                    h,
                    r,
                    &BorderMode::Clamp,
                    &mut scratch,
                    &mut out,
                );
                std::hint::black_box(&out);
            });
            println!("box u8 r={r}: dispatched {dispatched:.3} ms, portable {portable:.3} ms");
        }
        for n in [2usize, 4, 6] {
            let dispatched = best_ms(|| {
                binomial_u8(
                    std::hint::black_box(&src),
                    w,
                    h,
                    n,
                    &BorderMode::Clamp,
                    &mut scratch,
                    &mut out,
                );
                std::hint::black_box(&out);
            });
            println!("binomial u8 order={n}: dispatched {dispatched:.3} ms");
        }
    }

    #[test]
    #[should_panic(expected = "even and in 2..=12")]
    fn odd_binomial_order_is_rejected() {
        let src = vec![0u8; 4];
        let mut out = vec![0u8; 4];
        binomial_u8(
            &src,
            2,
            2,
            3,
            &BorderMode::Clamp,
            &mut IntScratch::default(),
            &mut out,
        );
    }
}
