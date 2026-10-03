use crate::core::{BorderMode, map_index};

/// Convolve a 1-D `f32` signal with a symmetric kernel.
///
/// `kernel` must have length `2 * radius + 1`. The output `out` must have the
/// same length as `signal`. `border` controls how out-of-bounds positions are
/// handled during the convolution.
///
/// # Panics
/// Panics when `out.len() != signal.len()` or `kernel.len() != 2 * radius + 1`.
pub fn convolve_f32(
    signal: &[f32],
    kernel: &[f32],
    radius: usize,
    border: BorderMode<f32>,
    out: &mut [f32],
) {
    assert_eq!(out.len(), signal.len(), "out must match signal length");
    assert_eq!(
        kernel.len(),
        2 * radius + 1,
        "kernel len must be 2*radius+1"
    );

    let n = signal.len();
    if n == 0 {
        return;
    }

    match border {
        BorderMode::Clamp => convolve_clamp(signal, kernel, radius, out),
        BorderMode::Constant(c) => convolve_constant(signal, kernel, radius, c, out),
        BorderMode::Reflect101 => convolve_reflect101(signal, kernel, radius, out),
    }
}

fn convolve_clamp(signal: &[f32], kernel: &[f32], radius: usize, out: &mut [f32]) {
    let n = signal.len();

    if n > 2 * radius {
        convolve_clamp_fast(signal, kernel, radius, out);
        return;
    }

    convolve_clamp_safe(signal, kernel, radius, out);
}

fn convolve_clamp_fast(signal: &[f32], kernel: &[f32], radius: usize, out: &mut [f32]) {
    let n = signal.len();
    let klen = kernel.len();

    // Left border.
    for (i, out_i) in out.iter_mut().take(radius.min(n)).enumerate() {
        let mut acc = 0.0f32;
        for (k, &kv) in kernel.iter().enumerate() {
            let idx = clamp_index(i as isize + radius as isize - k as isize, n);
            acc += signal[idx] * kv;
        }
        *out_i = acc;
    }

    // Interior without border checks.
    let interior_start = radius;
    let interior_end = n.saturating_sub(radius);
    if interior_start < interior_end {
        let s_ptr = signal.as_ptr();
        let k_ptr = kernel.as_ptr();

        // SAFETY:
        // - `i` in `[radius, n-radius)` guarantees full kernel footprint in bounds.
        // - `base = i-radius`, `base + (klen-1) = i+radius <= n-1`.
        // - Pointers derive from valid slices and are only offset within bounds.
        unsafe {
            for (i, out_i) in out
                .iter_mut()
                .enumerate()
                .take(interior_end)
                .skip(interior_start)
            {
                let base = i - radius;
                let mut acc = 0.0f32;
                for k in 0..klen {
                    acc += *s_ptr.add(base + k) * *k_ptr.add(klen - 1 - k);
                }
                *out_i = acc;
            }
        }
    }

    // Right border.
    for (i, out_i) in out
        .iter_mut()
        .enumerate()
        .skip(interior_end)
        .take(n - interior_end)
    {
        let mut acc = 0.0f32;
        for (k, &kv) in kernel.iter().enumerate() {
            let idx = clamp_index(i as isize + radius as isize - k as isize, n);
            acc += signal[idx] * kv;
        }
        *out_i = acc;
    }
}

fn convolve_clamp_safe(signal: &[f32], kernel: &[f32], radius: usize, out: &mut [f32]) {
    let n = signal.len();
    for (i, out_i) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, &kv) in kernel.iter().enumerate() {
            let idx = clamp_index(i as isize + radius as isize - k as isize, n);
            acc += signal[idx] * kv;
        }
        *out_i = acc;
    }
}

fn convolve_constant(signal: &[f32], kernel: &[f32], radius: usize, c: f32, out: &mut [f32]) {
    let n = signal.len() as isize;
    for (i, out_i) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, &kv) in kernel.iter().enumerate() {
            let idx = i as isize + radius as isize - k as isize;
            let v = if idx < 0 || idx >= n {
                c
            } else {
                signal[idx as usize]
            };
            acc += v * kv;
        }
        *out_i = acc;
    }
}

fn convolve_reflect101(signal: &[f32], kernel: &[f32], radius: usize, out: &mut [f32]) {
    let n = signal.len();
    for (i, out_i) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, &kv) in kernel.iter().enumerate() {
            let idx = map_index(
                i as isize + radius as isize - k as isize,
                n,
                &BorderMode::<f32>::Reflect101,
            )
            .expect("reflect101 index must map for non-empty signal");
            acc += signal[idx] * kv;
        }
        *out_i = acc;
    }
}

/// Convolve a 1-D `f32` signal with a symmetric `f64` kernel, accumulating in `f64`.
///
/// The precise counterpart of [`convolve_f32`], for operators whose output is
/// differenced before it is rounded. `out` is resized to the signal length.
pub(crate) fn convolve_f64(
    signal: &[f32],
    kernel: &[f64],
    radius: usize,
    border: BorderMode<f32>,
    out: &mut Vec<f64>,
) {
    debug_assert_eq!(
        kernel.len(),
        2 * radius + 1,
        "kernel len must be 2*radius+1"
    );
    let n = signal.len();
    out.clear();
    out.resize(n, 0.0);
    let at = |idx: isize| -> f64 {
        if (0..n as isize).contains(&idx) {
            return f64::from(signal[idx as usize]);
        }
        match (map_index(idx, n, &border), border) {
            (Some(j), _) => f64::from(signal[j]),
            (None, BorderMode::Constant(c)) => f64::from(c),
            (None, _) => 0.0,
        }
    };
    for (i, out_i) in out.iter_mut().enumerate() {
        *out_i = if i >= radius && i + radius < n {
            // The whole footprint is inside: no border lookups.
            signal[i - radius..=i + radius]
                .iter()
                .zip(kernel.iter().rev())
                .map(|(&v, &k)| f64::from(v) * k)
                .sum()
        } else {
            kernel
                .iter()
                .enumerate()
                .map(|(k, &kv)| at(i as isize + radius as isize - k as isize) * kv)
                .sum()
        };
    }
}

#[inline]
fn clamp_index(i: isize, len: usize) -> usize {
    if i < 0 { 0 } else { (i as usize).min(len - 1) }
}

#[cfg(test)]
mod tests {
    use crate::core::BorderMode;

    use super::{convolve_f32, convolve_f64};

    #[test]
    fn convolve_matches_expected_identity() {
        let signal = [1.0f32, 2.0, 3.0, 4.0];
        let kernel = [1.0f32];
        let mut out = vec![0.0f32; signal.len()];
        convolve_f32(&signal, &kernel, 0, BorderMode::Clamp, &mut out);
        assert_eq!(&out, &signal);
    }

    /// The `f64` convolution is the `f32` one, unrounded: every border mode, signals
    /// shorter and longer than the kernel.
    #[test]
    fn convolve_f64_matches_convolve_f32() {
        let kernel = [0.1f32, 0.2, 0.4, 0.2, 0.1];
        let kernel64: Vec<f64> = kernel.iter().map(|&k| f64::from(k)).collect();
        for n in [1usize, 3, 4, 9] {
            let signal: Vec<f32> = (0..n).map(|i| (i * i) as f32 * 0.5 + 1.0).collect();
            for border in [
                BorderMode::Clamp,
                BorderMode::Constant(7.0),
                BorderMode::Reflect101,
            ] {
                let mut want = vec![0.0f32; n];
                convolve_f32(&signal, &kernel, 2, border, &mut want);
                let mut got = Vec::new();
                convolve_f64(&signal, &kernel64, 2, border, &mut got);
                for (g, w) in got.iter().zip(&want) {
                    assert!(
                        (g - f64::from(*w)).abs() < 1e-5,
                        "n={n} {border:?}: {got:?} vs {want:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn convolve_constant_border() {
        let signal = [1.0f32, 2.0, 3.0];
        let kernel = [1.0f32, 1.0, 1.0];
        let mut out = vec![0.0f32; signal.len()];
        convolve_f32(&signal, &kernel, 1, BorderMode::Constant(0.0), &mut out);
        assert_eq!(out, vec![3.0, 6.0, 5.0]);
    }
}
