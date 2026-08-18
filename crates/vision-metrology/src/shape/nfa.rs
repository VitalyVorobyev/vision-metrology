//! Number-of-False-Alarms (NFA) computation for line validation.
//!
//! The NFA for a region of `n` pixels having `k` pixels aligned within
//! angle tolerance `p` is:
//!
//! ```text
//! NFA(n, k, p) = N_T * C(n, k) * p^k * (1 - p)^(n - k)
//! ```
//!
//! where `N_T` is the number of tests (proportional to `W * H * max_segs^2`).
//!
//! All arithmetic is performed in log₁₀ domain to avoid overflow.
//! `lgamma` from the `libm` crate is used for the log-binomial coefficient.
//!
//! A segment is accepted when `log10(NFA) < log_eps` (i.e. NFA < 10^log_eps).
//! With the default `log_eps = 0.0`, this means fewer than 1 false alarm on
//! average over the full image.

/// Compute `log10 C(n, k) = log10(n!) - log10(k!) - log10((n-k)!)`
/// using Stirling-based `lgamma` from libm.
///
/// For small `n` (≤ 20) the exact integer formula avoids the function call.
#[inline]
pub(crate) fn log10_binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }
    // log10(C(n,k)) = (lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)) / ln(10)
    let ln10 = core::f64::consts::LN_10;
    let v = libm::lgamma(n as f64 + 1.0)
        - libm::lgamma(k as f64 + 1.0)
        - libm::lgamma((n - k) as f64 + 1.0);
    v / ln10
}

/// Compute `log10(NFA)` for a region with `n` total pixels, `k` aligned pixels,
/// alignment probability `p` (fraction of angle range), and `n_tests` total tests.
///
/// # Parameters
/// - `n_tests`: number of independent tests (`N_T = W * H * max_segs * (max_segs-1) / 2`,
///   typically pre-computed once per frame).
/// - `n`: region pixel count.
/// - `k`: aligned pixel count (pixels within `ang_th` of the region angle).
/// - `p`: per-pixel alignment probability = `ang_th / π` (fraction of the half-circle).
///
/// # Returns
/// `log10(NFA)`. Negative values indicate a valid (statistically significant) segment.
pub(crate) fn log10_nfa(n_tests: f64, n: u64, k: u64, p: f64) -> f64 {
    if k == 0 {
        return f64::INFINITY;
    }
    let ln10 = core::f64::consts::LN_10;
    let log10_p = p.ln() / ln10;
    let log10_q = (1.0 - p).ln() / ln10;
    let log10_c = log10_binomial(n, k);
    let log10_nt = n_tests.log10();

    log10_nt + log10_c + k as f64 * log10_p + (n - k) as f64 * log10_q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomial_small_values() {
        // C(5, 2) = 10 → log10 = 1.0
        let v = log10_binomial(5, 2);
        assert!((v - 1.0).abs() < 1e-9, "log10 C(5,2)={v}");

        // C(10, 0) = 1 → log10 = 0
        assert!((log10_binomial(10, 0)).abs() < 1e-9);

        // C(10, 10) = 1 → log10 = 0
        assert!((log10_binomial(10, 10)).abs() < 1e-9);

        // C(n, k>n) → -inf
        assert!(log10_binomial(3, 5).is_infinite());
    }

    #[test]
    fn nfa_perfect_alignment_is_small() {
        // A region with all pixels perfectly aligned (p=1) and large k should have very
        // negative log NFA (very significant).
        // p = 22.5 / 180 ≈ 0.125; n=100, k=100 aligned
        let p = 22.5_f64 / 180.0;
        let n_tests = 1e8_f64; // W*H * max_segs^2 / 2 for 1280x1024
        let log_nfa = log10_nfa(n_tests, 100, 100, p);
        // With 100 perfect aligned pixels out of 100, NFA should be strongly negative
        assert!(log_nfa < -10.0, "log_nfa={log_nfa}");
    }

    #[test]
    fn nfa_random_alignment_is_large() {
        // If k ≈ p*n (random), NFA ≈ n_tests → log10(NFA) ≈ log10(n_tests) > 0
        let p = 22.5_f64 / 180.0;
        let n: u64 = 100;
        let k = (p * n as f64).round() as u64; // k ≈ expected by chance
        let n_tests = 1e8_f64;
        let log_nfa = log10_nfa(n_tests, n, k, p);
        // For a random alignment fraction, NFA should be >> 1 (positive log10)
        assert!(log_nfa > 0.0, "log_nfa={log_nfa}");
    }
}
