//! Exact separable FIR on `f32` images: register-blocked, with every `BorderMode`.
//!
//! The horizontal pass pads each source row once through `resolve`, so the inner loops
//! never branch on the border. The vertical pass reads whole rows and keeps each output
//! chunk in a fixed-size local accumulator across all taps (ADR-0023, Phase 0: this is what
//! brings the FIR to parity with libblur). Accumulation is `f32`, measured against an f64
//! reference at about 6e-5 on 0..255 data (ADR-0023).

use super::kernel::Kernel1D;
use crate::core::{BorderMode, map_index};

/// Where sample `i` comes from: itself when in bounds, otherwise whatever `border` says.
/// `None` means "use the border constant". `map_index` is only called out of bounds, because
/// it returns `None` for every index under `Constant`, in bounds or not.
#[inline]
fn resolve(i: isize, len: usize, border: &BorderMode<f32>) -> Option<usize> {
    if i >= 0 && (i as usize) < len {
        Some(i as usize)
    } else {
        map_index(i, len, border)
    }
}

/// Output columns per register block.
const CH: usize = 32;

/// Reusable scratch for [`separable_fir_f32`]. It grows to the largest size seen and is
/// then reused, so repeated calls on same-sized images allocate nothing.
#[derive(Default, Debug)]
pub(crate) struct FirScratch {
    /// Horizontally filtered image, `width · height`.
    tmp: Vec<f32>,
    /// One padded source row, `width + 2r`.
    pad: Vec<f32>,
    /// Two rows of `width`: a constant input row and its horizontally filtered result,
    /// used only for `Constant` borders.
    konst: Vec<f32>,
}

impl FirScratch {
    fn ensure(&mut self, width: usize, height: usize, r: usize) {
        grow(&mut self.tmp, width * height);
        grow(&mut self.pad, width + 2 * r);
        grow(&mut self.konst, 2 * width);
    }
}

fn grow(v: &mut Vec<f32>, len: usize) {
    if v.len() < len {
        v.resize(len, 0.0);
    }
}

/// Separable convolution of a `width × height` image with the symmetric `kernel`.
///
/// Out-of-bounds samples are resolved by `border` in both passes. For `Constant(c)` a
/// sample outside the image is `c`, so the filtered constant row is `c · Σkernel`, which
/// is exactly what the horizontal pass produces for it.
///
/// # Panics
/// Panics when `src` or `dst` do not hold `width · height` values.
pub(crate) fn separable_fir_f32(
    src: &[f32],
    width: usize,
    height: usize,
    kernel: &Kernel1D,
    border: BorderMode<f32>,
    scratch: &mut FirScratch,
    dst: &mut [f32],
) {
    assert_eq!(
        src.len(),
        width * height,
        "src must hold width * height values"
    );
    assert_eq!(
        dst.len(),
        width * height,
        "dst must hold width * height values"
    );
    if width == 0 || height == 0 {
        return;
    }
    let r = kernel.radius();
    let taps = kernel.taps();
    scratch.ensure(width, height, r);
    let FirScratch { tmp, pad, konst } = scratch;
    let w = width;

    // Horizontal pass over every row.
    for y in 0..height {
        let out = &mut tmp[y * w..(y + 1) * w];
        hpass_row(&src[y * w..(y + 1) * w], out, taps, &border, pad);
    }

    // The horizontally filtered constant row, for `Constant` borders only.
    let (flat, konst_out) = konst.split_at_mut(w);
    if let BorderMode::Constant(c) = border {
        flat.fill(c);
        hpass_row(flat, &mut konst_out[..w], taps, &border, pad);
    }
    let konst_row: &[f32] = &konst_out[..w];

    // Vertical pass, one output row at a time.
    for y in 0..height {
        let out = &mut dst[y * w..(y + 1) * w];
        vpass_row(&tmp[..], konst_row, w, height, y, taps, &border, out);
    }
}

/// Horizontal FIR of one row: pad through `border`, then register-blocked taps.
fn hpass_row(
    row: &[f32],
    out: &mut [f32],
    taps: &[f32],
    border: &BorderMode<f32>,
    pad: &mut [f32],
) {
    let w = row.len();
    let r = taps.len() - 1;
    let pad = &mut pad[..w + 2 * r];
    let fill = match border {
        BorderMode::Constant(c) => *c,
        _ => 0.0,
    };
    for (p, slot) in pad.iter_mut().enumerate() {
        *slot = match resolve(p as isize - r as isize, w, border) {
            Some(i) => row[i],
            None => fill,
        };
    }
    let pad: &[f32] = pad;

    let mut x0 = 0;
    while x0 + CH <= w {
        let mut acc = [0f32; CH];
        let centre = &pad[r + x0..r + x0 + CH];
        for (a, &v) in acc.iter_mut().zip(centre) {
            *a = taps[0] * v;
        }
        for (j, &kj) in taps.iter().enumerate().skip(1) {
            let left = &pad[r + x0 - j..r + x0 - j + CH];
            let right = &pad[r + x0 + j..r + x0 + j + CH];
            for i in 0..CH {
                acc[i] += kj * (left[i] + right[i]);
            }
        }
        out[x0..x0 + CH].copy_from_slice(&acc);
        x0 += CH;
    }
    for x in x0..w {
        let mut a = taps[0] * pad[r + x];
        for (j, &kj) in taps.iter().enumerate().skip(1) {
            a += kj * (pad[r + x - j] + pad[r + x + j]);
        }
        out[x] = a;
    }
}

/// Vertical FIR for output row `y`. Rows outside the image come from `border`: a mapped
/// row of `tmp`, or the filtered constant row when the border is `Constant`.
#[allow(clippy::too_many_arguments)]
fn vpass_row(
    tmp: &[f32],
    konst_row: &[f32],
    w: usize,
    height: usize,
    y: usize,
    taps: &[f32],
    border: &BorderMode<f32>,
    out: &mut [f32],
) {
    let row_at = |offset: isize| -> &[f32] {
        match resolve(y as isize + offset, height, border) {
            Some(yy) => &tmp[yy * w..(yy + 1) * w],
            None => konst_row,
        }
    };
    let centre = row_at(0);
    let mut x0 = 0;
    while x0 + CH <= w {
        let mut acc = [0f32; CH];
        for (a, &v) in acc.iter_mut().zip(&centre[x0..x0 + CH]) {
            *a = taps[0] * v;
        }
        for (j, &kj) in taps.iter().enumerate().skip(1) {
            let up = &row_at(-(j as isize))[x0..x0 + CH];
            let down = &row_at(j as isize)[x0..x0 + CH];
            for i in 0..CH {
                acc[i] += kj * (up[i] + down[i]);
            }
        }
        out[x0..x0 + CH].copy_from_slice(&acc);
        x0 += CH;
    }
    for x in x0..w {
        let mut a = taps[0] * centre[x];
        for (j, &kj) in taps.iter().enumerate().skip(1) {
            a += kj * (row_at(-(j as isize))[x] + row_at(j as isize)[x]);
        }
        out[x] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Border rule, written out independently of the crate: `Some(index)` for an in-bounds or
    /// border-mapped sample, `None` for the constant. Reflect-101 is the period-`2(n-1)`
    /// mirror, with no repeated edge pixel.
    fn border_index(i: isize, n: usize, border: BorderMode<f32>) -> Option<usize> {
        let n = n as isize;
        if (0..n).contains(&i) {
            return Some(i as usize);
        }
        match border {
            BorderMode::Constant(_) => None,
            BorderMode::Clamp => Some(i.clamp(0, n - 1) as usize),
            BorderMode::Reflect101 => {
                if n == 1 {
                    return Some(0);
                }
                let period = 2 * (n - 1);
                let m = i.rem_euclid(period);
                Some(if m < n { m } else { period - m } as usize)
            }
        }
    }

    /// Naive 2-D reference in f64: the direct double sum over the (2r+1)² window.
    fn reference(
        src: &[f32],
        w: usize,
        h: usize,
        kernel: &Kernel1D,
        border: BorderMode<f32>,
    ) -> Vec<f64> {
        let r = kernel.radius() as isize;
        let taps: Vec<f64> = kernel.taps().iter().map(|&t| t as f64).collect();
        let tap = |d: isize| taps[d.unsigned_abs()];
        let c = match border {
            BorderMode::Constant(c) => c as f64,
            _ => 0.0,
        };
        let mut out = vec![0.0f64; w * h];
        for y in 0..h as isize {
            for x in 0..w as isize {
                let mut s = 0.0;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sample = match (
                            border_index(x + dx, w, border),
                            border_index(y + dy, h, border),
                        ) {
                            (Some(xx), Some(yy)) => src[yy * w + xx] as f64,
                            _ => c,
                        };
                        s += tap(dy) * tap(dx) * sample;
                    }
                }
                out[y as usize * w + x as usize] = s;
            }
        }
        out
    }

    /// Deterministic 0..255 test pattern (a fixed LCG, no RNG crate).
    fn pattern(w: usize, h: usize) -> Vec<f32> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        (0..w * h)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 56) & 0xFF) as f32
            })
            .collect()
    }

    fn check(w: usize, h: usize, kernel: &Kernel1D, border: BorderMode<f32>) {
        let src = pattern(w, h);
        let mut scratch = FirScratch::default();
        let mut got = vec![0f32; w * h];
        separable_fir_f32(&src, w, h, kernel, border, &mut scratch, &mut got);
        let want = reference(&src, w, h, kernel, border);
        for (i, (g, x)) in got.iter().zip(&want).enumerate() {
            assert!(
                (*g as f64 - x).abs() < 2e-3,
                "{w}x{h} r={} {border:?}: pixel {i}: fir {g} vs reference {x}",
                kernel.radius()
            );
        }
    }

    const BORDERS: [BorderMode<f32>; 3] = [
        BorderMode::Clamp,
        BorderMode::Reflect101,
        BorderMode::Constant(7.0),
    ];

    #[test]
    fn matches_the_reference_for_every_border_on_a_small_image() {
        for border in BORDERS {
            check(24, 17, &Kernel1D::gaussian(1.5), border);
        }
    }

    #[test]
    fn matches_the_reference_across_full_and_tail_blocks() {
        // Width 100 is three full 32-column blocks plus a 4-column tail.
        for border in BORDERS {
            check(100, 40, &Kernel1D::gaussian(2.0), border);
        }
    }

    #[test]
    fn matches_the_reference_when_the_radius_exceeds_the_image() {
        // Radius 4 on a 3-pixel-wide image: every sample is a border sample.
        for border in BORDERS {
            check(3, 50, &Kernel1D::box_mean(4), border);
            check(50, 2, &Kernel1D::box_mean(4), border);
        }
    }

    #[test]
    fn radius_zero_is_the_identity() {
        let k = Kernel1D::from_folded(&[1.0]);
        let src = pattern(9, 5);
        let mut out = vec![0f32; 45];
        separable_fir_f32(
            &src,
            9,
            5,
            &k,
            BorderMode::Clamp,
            &mut FirScratch::default(),
            &mut out,
        );
        assert_eq!(out, src);
    }

    #[test]
    fn a_constant_image_stays_constant_when_the_border_matches_it() {
        let (w, h) = (70, 33);
        let flat = vec![42.0f32; w * h];
        for border in [
            BorderMode::Clamp,
            BorderMode::Reflect101,
            BorderMode::Constant(42.0),
        ] {
            let mut out = vec![0f32; w * h];
            separable_fir_f32(
                &flat,
                w,
                h,
                &Kernel1D::gaussian(2.5),
                border,
                &mut FirScratch::default(),
                &mut out,
            );
            for &v in &out {
                assert!((v - 42.0).abs() < 1e-4, "{border:?}: {v}");
            }
        }
    }

    #[test]
    fn a_foreign_constant_border_only_reaches_the_edge() {
        // Gaussian sigma 2.5 has radius 8: pixels at least 8 away from every edge never see
        // the border, so they keep the image value; the corner sees the border.
        let (w, h) = (70, 33);
        let flat = vec![42.0f32; w * h];
        let mut out = vec![0f32; w * h];
        let k = Kernel1D::gaussian(2.5);
        separable_fir_f32(
            &flat,
            w,
            h,
            &k,
            BorderMode::Constant(7.0),
            &mut FirScratch::default(),
            &mut out,
        );
        assert!(
            (out[16 * w + 35] - 42.0).abs() < 1e-4,
            "interior {}",
            out[16 * w + 35]
        );
        let corner = out[0];
        assert!(
            corner > 7.0 + 1.0 && corner < 42.0 - 1.0,
            "corner {corner} should be a mix"
        );
    }

    #[test]
    fn scratch_is_reused_without_growing() {
        let (w, h) = (64, 48);
        let src = pattern(w, h);
        let k = Kernel1D::gaussian(2.0);
        let mut scratch = FirScratch::default();
        let mut out = vec![0f32; w * h];
        separable_fir_f32(&src, w, h, &k, BorderMode::Clamp, &mut scratch, &mut out);
        let caps = (
            scratch.tmp.capacity(),
            scratch.pad.capacity(),
            scratch.konst.capacity(),
        );
        for _ in 0..3 {
            separable_fir_f32(&src, w, h, &k, BorderMode::Clamp, &mut scratch, &mut out);
        }
        assert_eq!(
            caps,
            (
                scratch.tmp.capacity(),
                scratch.pad.capacity(),
                scratch.konst.capacity()
            )
        );
    }

    #[test]
    #[should_panic(expected = "src must hold")]
    fn size_mismatch_is_rejected() {
        let mut out = vec![0f32; 4];
        separable_fir_f32(
            &[0.0; 3],
            2,
            2,
            &Kernel1D::box_mean(1),
            BorderMode::Clamp,
            &mut FirScratch::default(),
            &mut out,
        );
    }
}
