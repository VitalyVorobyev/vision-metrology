// The index loops below do offset arithmetic over several arrays at once; iterator forms
// would hide that. The lint is allowed for this prototype only, not for library code.
#![allow(clippy::needless_range_loop)]

//! Phase 0 prototype of the filter kernels, plus the f64 reference they are checked against.
//!
//! Prototype, not library code: the library's `filter` module (ADR-0023) is designed after
//! the headroom this crate measures. Border handling is `Clamp` everywhere (invariant 11).

/// Clamped index into `0..n`.
#[inline]
fn clampi(i: isize, n: usize) -> usize {
    i.clamp(0, n as isize - 1) as usize
}

/// Reusable scratch for [`BoxU8`]: allocated once, so a timed call allocates nothing.
pub struct BoxU8 {
    /// Horizontal window sums (u32), one per source pixel.
    hsum: Vec<u32>,
    /// Running vertical sums, one per column.
    colsum: Vec<u32>,
    /// Row-sized padded copy of the source row, clamped at the borders.
    pad: Vec<u32>,
}

impl BoxU8 {
    pub fn new(width: usize, height: usize, radius: usize) -> Self {
        let _ = height;
        Self {
            hsum: vec![0; width * height],
            colsum: vec![0; width],
            // One slot beyond the window: the running sum reads `x + 2r + 1` at x = w - 1.
            pad: vec![0; width + 2 * radius + 1],
        }
    }

    /// Exact `(2r+1)^2` box mean of `src` into `dst`, rounded half up, with a single
    /// integer reciprocal per pixel. Exact for `n <= 1023` (checked by [`box_u8_naive`]).
    pub fn apply(&mut self, src: &[u8], w: usize, h: usize, r: usize, dst: &mut [u8]) {
        let n = 2 * r + 1;
        let d = (n * n) as u64;
        // ceil(2^40 / d): floor((x + d/2) * m / 2^40) is exact for x < 2^20 / d-ish; see docs.
        let m = (1u64 << 40).div_ceil(d);
        let half = d / 2;

        // Horizontal pass: window sums per row via a running sum over the clamped row.
        for y in 0..h {
            let row = &src[y * w..(y + 1) * w];
            for (i, p) in self.pad.iter_mut().enumerate() {
                *p = row[clampi(i as isize - r as isize, w)] as u32;
            }
            let mut s: u32 = self.pad[..2 * r + 1].iter().sum();
            let out = &mut self.hsum[y * w..(y + 1) * w];
            for x in 0..w {
                out[x] = s;
                s += self.pad[x + 2 * r + 1] - self.pad[x];
            }
        }

        // Vertical pass: running column sums across the rows of `hsum`, vectorised over x.
        let hs = &self.hsum;
        let row_of = |y: isize| {
            let yy = clampi(y, h);
            &hs[yy * w..(yy + 1) * w]
        };
        self.colsum.iter_mut().for_each(|c| *c = 0);
        for k in -(r as isize)..=(r as isize) {
            for (c, &v) in self.colsum.iter_mut().zip(row_of(k)) {
                *c += v;
            }
        }
        for y in 0..h {
            let out = &mut dst[y * w..(y + 1) * w];
            for (o, &c) in out.iter_mut().zip(self.colsum.iter()) {
                let q = (((c as u64) + half) * m) >> 40;
                *o = q as u8;
            }
            if y + 1 < h {
                let add = row_of((y + 1 + r) as isize);
                let sub = row_of(y as isize - r as isize);
                for ((c, &a), &s) in self.colsum.iter_mut().zip(add).zip(sub) {
                    *c = *c + a - s;
                }
            }
        }
    }
}

/// Normalised Gaussian taps `k[0..=r]` (symmetric, folded), computed in f64.
pub fn gaussian_taps(sigma: f64, r: usize) -> Vec<f64> {
    let g: Vec<f64> = (0..=r)
        .map(|j| (-(j as f64).powi(2) / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f64 = g[0] + 2.0 * g[1..].iter().sum::<f64>();
    g.iter().map(|v| v / total).collect()
}

/// One tap update, `a + k·s`: separate multiply and add, or one fused `mul_add`. Rust never
/// contracts the former into the latter, so the choice is explicit.
#[inline(always)]
fn madd<const FUSED: bool>(k: f32, s: f32, a: f32) -> f32 {
    if FUSED { k.mul_add(s, a) } else { a + k * s }
}

/// Reusable scratch for [`GaussF32`].
pub struct GaussF32 {
    tmp: Vec<f32>,
    pad: Vec<f32>,
    taps: Vec<f32>,
    r: usize,
}

impl GaussF32 {
    pub fn new(width: usize, height: usize, sigma: f64) -> Self {
        // Truncate at 3 sigma, the same radius the reference uses.
        let r = (3.0 * sigma).ceil() as usize;
        let taps = gaussian_taps(sigma, r).iter().map(|&v| v as f32).collect();
        Self {
            tmp: vec![0.0; width * height],
            pad: vec![0.0; width + 2 * r],
            taps,
            r,
        }
    }

    pub fn radius(&self) -> usize {
        self.r
    }

    /// Register-blocked separable Gaussian: each output chunk of `CH` columns keeps its
    /// accumulator in a fixed-size local array across all taps, so the loads and stores
    /// per tap are the two taps' inputs only, not the whole row.
    pub fn apply(&mut self, src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
        self.apply_generic::<false>(src, w, h, dst)
    }

    /// Same as [`apply`](Self::apply), with each tap update as one fused `mul_add`.
    pub fn apply_fused(&mut self, src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
        self.apply_generic::<true>(src, w, h, dst)
    }

    fn apply_generic<const FUSED: bool>(&mut self, src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
        const CH: usize = 32;
        let r = self.r;
        let k = &self.taps;
        // Horizontal, chunked over x; the padded row is clamped once per row.
        for y in 0..h {
            let row = &src[y * w..(y + 1) * w];
            for (i, p) in self.pad.iter_mut().enumerate() {
                *p = row[clampi(i as isize - r as isize, w)];
            }
            let pad = &self.pad;
            let out = &mut self.tmp[y * w..(y + 1) * w];
            let mut x0 = 0;
            while x0 + CH <= w {
                let mut acc = [0f32; CH];
                let c = &pad[r + x0..r + x0 + CH];
                for (a, &v) in acc.iter_mut().zip(c) {
                    *a = k[0] * v;
                }
                for j in 1..=r {
                    let l = &pad[r + x0 - j..r + x0 - j + CH];
                    let rr = &pad[r + x0 + j..r + x0 + j + CH];
                    let kj = k[j];
                    for i in 0..CH {
                        acc[i] = madd::<FUSED>(kj, l[i] + rr[i], acc[i]);
                    }
                }
                out[x0..x0 + CH].copy_from_slice(&acc);
                x0 += CH;
            }
            // Tail columns, plain loop.
            for x in x0..w {
                let mut a = k[0] * pad[r + x];
                for j in 1..=r {
                    a += k[j] * (pad[r + x - j] + pad[r + x + j]);
                }
                out[x] = a;
            }
        }
        // Vertical, chunked over x: the rows are read, the accumulator stays in registers.
        let tmp = &self.tmp;
        for y in 0..h {
            let out = &mut dst[y * w..(y + 1) * w];
            let mut x0 = 0;
            while x0 + CH <= w {
                let mut acc = [0f32; CH];
                let c = &tmp[y * w + x0..y * w + x0 + CH];
                for (a, &v) in acc.iter_mut().zip(c) {
                    *a = k[0] * v;
                }
                for j in 1..=r {
                    let up = clampi(y as isize - j as isize, h);
                    let dn = clampi(y as isize + j as isize, h);
                    let u = &tmp[up * w + x0..up * w + x0 + CH];
                    let d = &tmp[dn * w + x0..dn * w + x0 + CH];
                    let kj = k[j];
                    for i in 0..CH {
                        acc[i] = madd::<FUSED>(kj, u[i] + d[i], acc[i]);
                    }
                }
                out[x0..x0 + CH].copy_from_slice(&acc);
                x0 += CH;
            }
            for x in x0..w {
                let mut a = k[0] * tmp[y * w + x];
                for j in 1..=r {
                    let up = clampi(y as isize - j as isize, h);
                    let dn = clampi(y as isize + j as isize, h);
                    a += k[j] * (tmp[up * w + x] + tmp[dn * w + x]);
                }
                out[x] = a;
            }
        }
    }

    /// The first cut: one row-saxpy per tap. Kept as the baseline for the blocked version.
    pub fn apply_saxpy(&mut self, src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
        let r = self.r;
        let k = &self.taps;
        // Horizontal.
        for y in 0..h {
            let row = &src[y * w..(y + 1) * w];
            for (i, p) in self.pad.iter_mut().enumerate() {
                *p = row[clampi(i as isize - r as isize, w)];
            }
            let acc = &mut self.tmp[y * w..(y + 1) * w];
            let centre = &self.pad[r..r + w];
            for (a, &c) in acc.iter_mut().zip(centre) {
                *a = k[0] * c;
            }
            for j in 1..=r {
                let left = &self.pad[r - j..r - j + w];
                let right = &self.pad[r + j..r + j + w];
                let kj = k[j];
                for ((a, &l), &rr) in acc.iter_mut().zip(left).zip(right) {
                    *a += kj * (l + rr);
                }
            }
        }
        // Vertical, row-saxpy across the whole row.
        let tmp = &self.tmp;
        for y in 0..h {
            let out = &mut dst[y * w..(y + 1) * w];
            let c = &tmp[y * w..(y + 1) * w];
            for (o, &v) in out.iter_mut().zip(c) {
                *o = k[0] * v;
            }
            for j in 1..=r {
                let up = clampi(y as isize - j as isize, h);
                let dn = clampi(y as isize + j as isize, h);
                let (u, d) = (&tmp[up * w..(up + 1) * w], &tmp[dn * w..(dn + 1) * w]);
                let kj = k[j];
                for ((o, &a), &b) in out.iter_mut().zip(u).zip(d) {
                    *o += kj * (a + b);
                }
            }
        }
    }
}

/// Naive exact box mean: direct `(2r+1)^2` loops with clamp, integer rounding half up.
pub fn box_u8_naive(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let n = (2 * r + 1) as u64;
    let d = n * n;
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0u64;
            for dy in -(r as isize)..=(r as isize) {
                for dx in -(r as isize)..=(r as isize) {
                    let yy = clampi(y as isize + dy, h);
                    let xx = clampi(x as isize + dx, w);
                    s += src[yy * w + xx] as u64;
                }
            }
            out[y * w + x] = ((s + d / 2) / d) as u8;
        }
    }
    out
}

/// f64 reference Gaussian: naive separable loops, the same truncation and `Clamp` borders.
pub fn gaussian_f64_reference(src: &[f32], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let r = (3.0 * sigma).ceil() as usize;
    let k = gaussian_taps(sigma, r);
    let tap = |j: isize| k[j.unsigned_abs()];
    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for j in -(r as isize)..=(r as isize) {
                s += tap(j) * src[y * w + clampi(x as isize + j, w)] as f64;
            }
            tmp[y * w + x] = s;
        }
    }
    let mut out = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for j in -(r as isize)..=(r as isize) {
                s += tap(j) * tmp[clampi(y as isize + j, h) * w + x];
            }
            out[y * w + x] = s;
        }
    }
    out
}

/// Deterministic test image (no RNG crate: a 64-bit LCG, so the fixture is reproducible).
pub fn test_image(w: usize, h: usize) -> Vec<u8> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..w * h)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = (state >> 59) as usize; // 0..31
            let (x, y) = (i % w, i / w);
            (((x * 7 + y * 5) % 256) ^ noise) as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 24;
    const H: usize = 17;

    /// Golden files come from `reference.py` (scipy), an implementation this crate does not
    /// share, so these tests check the f64 oracle against something other than itself.
    fn golden(text: &str) -> Vec<f64> {
        text.split_whitespace()
            .map(|t| t.parse().expect("number"))
            .collect()
    }

    fn src_u8() -> Vec<u8> {
        golden(include_str!("../golden/src.txt"))
            .iter()
            .map(|&v| v as u8)
            .collect()
    }

    #[test]
    fn oracle_gaussian_matches_scipy() {
        let src: Vec<f32> = src_u8().iter().map(|&v| v as f32).collect();
        let got = gaussian_f64_reference(&src, W, H, 1.5);
        let want = golden(include_str!("../golden/gauss_1_5.txt"));
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-9, "oracle {g} vs scipy {w}");
        }
    }

    #[test]
    fn oracle_box_matches_scipy() {
        let want: Vec<u8> = golden(include_str!("../golden/box_r1.txt"))
            .iter()
            .map(|&v| v as u8)
            .collect();
        assert_eq!(box_u8_naive(&src_u8(), W, H, 1), want);
    }

    #[test]
    fn prototype_box_matches_scipy() {
        let want: Vec<u8> = golden(include_str!("../golden/box_r1.txt"))
            .iter()
            .map(|&v| v as u8)
            .collect();
        let mut out = vec![0u8; W * H];
        BoxU8::new(W, H, 1).apply(&src_u8(), W, H, 1, &mut out);
        assert_eq!(out, want);
    }

    #[test]
    fn prototype_gaussian_matches_scipy() {
        let src: Vec<f32> = src_u8().iter().map(|&v| v as f32).collect();
        let mut out = vec![0f32; W * H];
        GaussF32::new(W, H, 1.5).apply(&src, W, H, &mut out);
        let want = golden(include_str!("../golden/gauss_1_5.txt"));
        for (g, w) in out.iter().zip(&want) {
            assert!((*g as f64 - w).abs() < 1e-3, "prototype {g} vs scipy {w}");
        }
    }

    /// Width 100 = 3 full 32-column blocks plus a 4-column tail, so both code paths run.
    #[test]
    fn blocked_gaussian_equals_saxpy_on_full_and_tail_blocks() {
        let (w, h) = (100usize, 40usize);
        let src: Vec<f32> = test_image(w, h).iter().map(|&v| v as f32).collect();
        let mut a = vec![0f32; w * h];
        let mut b = vec![0f32; w * h];
        let mut g = GaussF32::new(w, h, 2.0);
        g.apply(&src, w, h, &mut a);
        g.apply_saxpy(&src, w, h, &mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-4, "blocked {x} vs saxpy {y}");
        }
    }

    #[test]
    fn box_is_exact_on_a_wider_image() {
        let (w, h) = (100usize, 40usize);
        let src = test_image(w, h);
        for r in [1usize, 3, 8] {
            let mut out = vec![0u8; w * h];
            BoxU8::new(w, h, r).apply(&src, w, h, r, &mut out);
            assert_eq!(out, box_u8_naive(&src, w, h, r), "r={r}");
        }
    }
}
