use vm_core::{BorderMode, map_index};

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

#[inline]
fn clamp_index(i: isize, len: usize) -> usize {
    if i < 0 { 0 } else { (i as usize).min(len - 1) }
}

#[cfg(test)]
mod tests {
    use vm_core::BorderMode;

    use crate::conv1d::convolve_f32;

    #[test]
    fn convolve_matches_expected_identity() {
        let signal = [1.0f32, 2.0, 3.0, 4.0];
        let kernel = [1.0f32];
        let mut out = vec![0.0f32; signal.len()];
        convolve_f32(&signal, &kernel, 0, BorderMode::Clamp, &mut out);
        assert_eq!(&out, &signal);
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
