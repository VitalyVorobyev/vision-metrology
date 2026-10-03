//! Level crossings on a 1-D profile: end levels, interpolated crossings of a level, and
//! the local half-contrast edge.
//!
//! These locate an edge where the profile crosses an intensity *level*, rather than where
//! its derivative peaks. Positions are in samples (sample `i` sits at `x = i`), linearly
//! interpolated between the two samples a crossing falls between.

use std::num::NonZeroUsize;

use super::edge1d::EdgePolarity;

/// How [`LevelCrossing1D::half_contrast`] refines an edge. Distances are in samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfContrastConfig {
    /// Inner edge of each flank window, measured from the current estimate. It is also
    /// how far the next crossing may be from the estimate, and how far the result may be
    /// from the seed.
    pub flank_near: f32,
    /// Outer edge of each flank window, measured from the current estimate.
    pub flank_far: f32,
    /// The iteration stops once the estimate moves by `tol` or less.
    pub tol: f32,
    /// The most flank-and-crossing iterations; the last estimate is returned when they
    /// run out.
    pub max_iter: NonZeroUsize,
    /// Minimum `|after − before|` between the flank levels, in the profile's units.
    pub min_contrast: f32,
}

impl Default for HalfContrastConfig {
    fn default() -> Self {
        Self {
            flank_near: 3.0,
            flank_far: 8.0,
            tol: 0.01,
            max_iter: NonZeroUsize::new(5).unwrap_or(NonZeroUsize::MIN),
            min_contrast: 0.0,
        }
    }
}

/// An edge located as a level crossing.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LevelEdge {
    /// Subpixel position of the crossing, in samples.
    pub x: f32,
    /// The level on the lower-index side of the edge.
    pub before: f32,
    /// The level on the higher-index side of the edge.
    pub after: f32,
    /// The level crossed: the mean of `before` and `after`.
    pub level: f32,
    /// How many level-and-crossing evaluations produced the position.
    pub iterations: usize,
}

/// What [`LevelCrossing1D::half_contrast`] found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LevelOutcome {
    /// The edge, at a crossing of its local half-contrast level.
    Found(LevelEdge),
    /// The flank levels differ by less than
    /// [`min_contrast`](HalfContrastConfig::min_contrast).
    LowContrast {
        /// The level on the lower-index side.
        before: f32,
        /// The level on the higher-index side.
        after: f32,
    },
    /// A flank window held no samples, the level was not crossed within `flank_near` of
    /// the estimate, or the crossing moved more than `flank_near` from the seed.
    NoCrossing,
}

/// Reusable level-crossing locator. It owns its scratch, so calls do not allocate once the
/// buffers have grown to the profile length.
///
/// # Example
/// ```
/// use std::num::NonZeroUsize;
/// use vm_primitives::{EdgePolarity, LevelCrossing1D};
///
/// let s = [0.1f32, 0.1, 0.1, 0.4, 0.9, 0.9, 0.9];
/// let mut lc = LevelCrossing1D::new();
/// let (before, after) = lc.end_levels(&s, NonZeroUsize::new(3).unwrap());
/// assert_eq!((before, after), (0.1, 0.9));
/// // The mid level 0.5 is crossed a fifth of the way from sample 3 to sample 4.
/// let xs = lc.crossings(&s, 0.5, EdgePolarity::Rising);
/// assert_eq!(xs.len(), 1);
/// assert!((xs[0] - 3.2).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Default)]
pub struct LevelCrossing1D {
    /// Values whose median is being taken.
    scratch: Vec<f32>,
    /// Crossings from the last [`crossings`](Self::crossings) call.
    xs: Vec<f32>,
}

impl LevelCrossing1D {
    /// A locator with empty scratch.
    pub fn new() -> Self {
        Self::default()
    }

    /// The medians of the first and of the last `count` samples (all of `s` when it is
    /// shorter). An even count takes the mean of the two middle values, like `np.median`.
    ///
    /// # Panics
    /// Panics when `s` is empty.
    pub fn end_levels(&mut self, s: &[f32], count: NonZeroUsize) -> (f32, f32) {
        assert!(!s.is_empty(), "end levels of an empty profile");
        let k = count.get().min(s.len());
        let before = median_of(&mut self.scratch, s[..k].iter().copied());
        let after = median_of(&mut self.scratch, s[s.len() - k..].iter().copied());
        // Neither range is empty, so both medians exist.
        (before.unwrap_or(f32::NAN), after.unwrap_or(f32::NAN))
    }

    /// Every crossing of `level` with the given polarity, linearly interpolated, in
    /// increasing order. The slice is valid until the next call.
    ///
    /// Between samples `a = s[i]` and `b = s[i + 1]`, a rising crossing is
    /// `a ≤ level < b` and a falling one `a ≥ level > b`; the position is
    /// `i + (level − a) / (b − a)`. A sample exactly on the level therefore starts a
    /// crossing and does not end one, so a profile that touches the level and leaves it
    /// again on the same side has none.
    pub fn crossings(&mut self, s: &[f32], level: f32, polarity: EdgePolarity) -> &[f32] {
        self.xs.clear();
        for (i, w) in s.windows(2).enumerate() {
            let (a, b) = (w[0], w[1]);
            let crosses = match polarity {
                EdgePolarity::Rising => a <= level && level < b,
                EdgePolarity::Falling => a >= level && level > b,
            };
            if crosses {
                self.xs.push(interpolate(i, a, b, f64::from(level)) as f32);
            }
        }
        &self.xs
    }

    /// Refine `seed` to the crossing of the local half-contrast level.
    ///
    /// Each iteration takes the medians of the samples `flank_near..=flank_far` before
    /// and after the current estimate, and moves the estimate to the crossing of their
    /// mean nearest to it, of either polarity and within `flank_near`. A crossing here is
    /// a change of `s ≥ level` between two samples. Because the flanks are re-centred on
    /// the crossing they define, the result does not depend on where in `±flank_near` the
    /// search started. It stops when the estimate moves by `tol` or less, or after
    /// `max_iter` iterations.
    pub fn half_contrast(
        &mut self,
        s: &[f32],
        seed: f32,
        cfg: &HalfContrastConfig,
    ) -> LevelOutcome {
        let near = f64::from(cfg.flank_near);
        let far = f64::from(cfg.flank_far);
        let seed = f64::from(seed);
        let mut x = seed;
        let mut iterations = 0;
        loop {
            iterations += 1;
            let before = self.flank_median(s, x, -far, -near);
            let after = self.flank_median(s, x, near, far);
            let (Some(before), Some(after)) = (before, after) else {
                return LevelOutcome::NoCrossing;
            };
            if (f64::from(after) - f64::from(before)).abs() < f64::from(cfg.min_contrast) {
                return LevelOutcome::LowContrast { before, after };
            }
            let level = 0.5 * (f64::from(before) + f64::from(after));
            let Some(found) = nearest_crossing(s, level, x, near) else {
                return LevelOutcome::NoCrossing;
            };
            if (found - seed).abs() > near {
                return LevelOutcome::NoCrossing;
            }
            let moved = (found - x).abs();
            x = found;
            if moved <= f64::from(cfg.tol) || iterations >= cfg.max_iter.get() {
                return LevelOutcome::Found(LevelEdge {
                    x: x as f32,
                    before,
                    after,
                    level: level as f32,
                    iterations,
                });
            }
        }
    }

    /// The median of the samples `i` with `lo ≤ i − x ≤ hi`, or `None` when there are none.
    fn flank_median(&mut self, s: &[f32], x: f64, lo: f64, hi: f64) -> Option<f32> {
        let n = s.len();
        if n == 0 || x + hi < 0.0 || x + lo > (n - 1) as f64 {
            return None;
        }
        // A range one sample wider than needed; the predicate below decides membership.
        let first = (x + lo).floor().max(0.0) as usize;
        let last = ((x + hi).ceil() as usize).min(n - 1);
        let inside = (first..=last).filter(|&i| {
            let rel = i as f64 - x;
            rel >= lo && rel <= hi
        });
        median_of(&mut self.scratch, inside.map(|i| s[i]))
    }
}

/// `i + (level − a) / (b − a)`, in `f64`.
fn interpolate(i: usize, a: f32, b: f32, level: f64) -> f64 {
    let (a, b) = (f64::from(a), f64::from(b));
    i as f64 + (level - a) / (b - a)
}

/// The crossing of `level` nearest to `x` (equal distances: the earlier one), if it is
/// within `window` of `x`. A crossing is a change of `s ≥ level` between neighbours.
fn nearest_crossing(s: &[f32], level: f64, x: f64, window: f64) -> Option<f64> {
    if s.len() < 2 || x - window > (s.len() - 1) as f64 || x + window < 0.0 {
        return None;
    }
    // A crossing in segment `i` lies in `[i, i + 1]`, so only these segments can hold one
    // within `window` of `x`; scanning them in order keeps the earlier of equal distances.
    let first = (x - window - 1.0).floor().max(0.0) as usize;
    let last = ((x + window).floor().max(0.0) as usize).min(s.len() - 2);
    let mut best: Option<(f64, f64)> = None;
    for i in first..=last {
        let (a, b) = (s[i], s[i + 1]);
        if (f64::from(a) >= level) != (f64::from(b) >= level) {
            let at = interpolate(i, a, b, level);
            let d = (at - x).abs();
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, at));
            }
        }
    }
    best.filter(|&(d, _)| d <= window).map(|(_, at)| at)
}

/// The median of `values`, collected into `scratch`: the middle value, or the mean of the
/// two middle values for an even count. `None` when `values` is empty.
fn median_of(scratch: &mut Vec<f32>, values: impl Iterator<Item = f32>) -> Option<f32> {
    scratch.clear();
    scratch.extend(values);
    let n = scratch.len();
    if n == 0 {
        return None;
    }
    let (lower, upper, _) = scratch.select_nth_unstable_by(n / 2, f32::total_cmp);
    let upper = *upper;
    if n % 2 == 1 {
        return Some(upper);
    }
    let lower = lower.iter().copied().max_by(f32::total_cmp)?;
    Some((0.5 * (f64::from(lower) + f64::from(upper))) as f32)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        HalfContrastConfig, LevelCrossing1D, LevelOutcome, interpolate, median_of, nearest_crossing,
    };
    use crate::EdgePolarity;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("nonzero")
    }

    #[test]
    fn medians_match_np_median_for_odd_and_even_counts() {
        let mut scratch = Vec::new();
        let mut med = |v: &[f32]| median_of(&mut scratch, v.iter().copied());
        assert_eq!(med(&[5.0, 1.0, 3.0]), Some(3.0));
        assert_eq!(med(&[4.0, 1.0, 3.0, 2.0]), Some(2.5), "mean of 2 and 3");
        assert_eq!(med(&[7.0]), Some(7.0));
        assert_eq!(med(&[0.25, 0.75]), Some(0.5));
        assert_eq!(
            med(&[2.0, 2.0, 9.0, 2.0]),
            Some(2.0),
            "repeated middle values"
        );
        assert_eq!(med(&[]), None);
    }

    #[test]
    fn end_levels_take_the_first_and_last_samples() {
        let mut lc = LevelCrossing1D::new();
        let s = [3.0f32, 1.0, 2.0, 50.0, 50.0, 9.0, 7.0, 8.0, 10.0];
        assert_eq!(lc.end_levels(&s, nz(3)), (2.0, 8.0));
        // An even count: sorted (1, 2, 3, 50) and (7, 8, 9, 10).
        assert_eq!(lc.end_levels(&s, nz(4)), (2.5, 8.5));
        // A count past the length takes the whole profile at both ends.
        assert_eq!(lc.end_levels(&[1.0, 3.0], nz(5)), (2.0, 2.0));
    }

    /// The equality rules: a sample on the level starts a crossing (`a ≤ L < b`,
    /// `a ≥ L > b`) and never ends one.
    #[test]
    fn crossings_follow_the_equality_rules() {
        let mut lc = LevelCrossing1D::new();
        assert_eq!(
            lc.crossings(&[0.0, 0.5, 1.0], 0.5, EdgePolarity::Rising),
            &[1.0]
        );
        assert_eq!(
            lc.crossings(&[1.0, 0.5, 0.0], 0.5, EdgePolarity::Falling),
            &[1.0]
        );
        // A plateau on the level: the crossing is where the profile leaves it.
        assert_eq!(
            lc.crossings(&[0.0, 0.5, 0.5, 1.0], 0.5, EdgePolarity::Rising),
            &[2.0]
        );
        // Touching the level and turning back is not a crossing.
        assert!(
            lc.crossings(&[0.0, 0.5, 0.0], 0.5, EdgePolarity::Rising)
                .is_empty()
        );
        // A rising level is not crossed by a falling profile, nor the reverse.
        assert!(
            lc.crossings(&[1.0, 0.0], 0.5, EdgePolarity::Rising)
                .is_empty()
        );
        assert!(
            lc.crossings(&[0.0, 1.0], 0.5, EdgePolarity::Falling)
                .is_empty()
        );
        // Linear interpolation, and every crossing in order.
        assert_eq!(
            lc.crossings(&[0.0, 4.0, 0.0, 4.0], 1.0, EdgePolarity::Rising),
            &[0.25, 2.25]
        );
        assert_eq!(
            lc.crossings(&[0.0, 4.0, 0.0, 4.0], 1.0, EdgePolarity::Falling),
            &[1.75]
        );
    }

    /// Scanning only the segments that can hold a crossing within the window gives the
    /// same answer as scanning the whole profile, ties included.
    #[test]
    fn the_windowed_crossing_search_matches_a_full_scan() {
        let full = |s: &[f32], level: f64, x: f64, window: f64| {
            let mut best: Option<(f64, f64)> = None;
            for (i, w) in s.windows(2).enumerate() {
                if (f64::from(w[0]) >= level) != (f64::from(w[1]) >= level) {
                    let at = interpolate(i, w[0], w[1], level);
                    let d = (at - x).abs();
                    if best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, at));
                    }
                }
            }
            best.filter(|&(d, _)| d <= window).map(|(_, at)| at)
        };
        let mut state = 0x2545_f491_u32;
        let mut next = |m: u32| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) % m
        };
        for _ in 0..5000 {
            let n = 2 + next(30) as usize;
            // Quarter steps make exact ties and samples on the level common.
            let s: Vec<f32> = (0..n).map(|_| 0.25 * next(5) as f32).collect();
            let level = 0.25 * f64::from(next(5)) + if next(2) == 0 { 0.0 } else { 0.125 };
            let x = f64::from(next(4 * n as u32 + 16)) * 0.25 - 4.0;
            let window = 0.5 * f64::from(next(10));
            assert_eq!(
                nearest_crossing(&s, level, x, window),
                full(&s, level, x, window),
                "{s:?} level {level} x {x} window {window}"
            );
        }
    }

    /// A pixel-integrated step from `dark` to `bright` at `edge`, blurred by a Gaussian of
    /// σ = `sigma` samples (truncated at 4σ, mirrored at the ends), plus `shade · i`.
    fn blurred_step(
        n: usize,
        edge: f64,
        sigma: f64,
        dark: f64,
        bright: f64,
        shade: f64,
    ) -> Vec<f32> {
        let cover: Vec<f64> = (0..n)
            .map(|i| {
                let i = i as f64;
                (i + 0.5 - (i - 0.5).max(edge)).clamp(0.0, 1.0)
            })
            .collect();
        let r = (4.0 * sigma).ceil() as isize;
        let w: Vec<f64> = (-r..=r)
            .map(|k| (-0.5 * (k as f64 / sigma).powi(2)).exp())
            .collect();
        let sum: f64 = w.iter().sum();
        (0..n as isize)
            .map(|i| {
                let mut acc = 0.0;
                for (k, wk) in (-r..=r).zip(&w) {
                    let mut j = i + k;
                    if j < 0 {
                        j = -j - 1;
                    }
                    if j >= n as isize {
                        j = 2 * n as isize - j - 1;
                    }
                    acc += wk * cover[j as usize];
                }
                (dark + (bright - dark) * acc / sum + shade * i as f64) as f32
            })
            .collect()
    }

    /// The local half-contrast crossing of a blurred, pixel-integrated step lands on the
    /// step, from a seed anywhere within `±flank_near` of it.
    #[test]
    fn half_contrast_converges_on_a_pixel_integrated_step() {
        let mut lc = LevelCrossing1D::new();
        let cfg = HalfContrastConfig::default();
        for edge in [40.0, 40.3, 40.5, 40.75] {
            let s = blurred_step(100, edge, 1.2, 0.15, 0.75, 0.0);
            for seed in [-2.9, -1.0, 0.0, 1.3, 2.9] {
                let LevelOutcome::Found(e) = lc.half_contrast(&s, (edge + seed) as f32, &cfg)
                else {
                    panic!("edge {edge}, seed {seed}: no crossing")
                };
                assert!(
                    (f64::from(e.x) - edge).abs() < 0.02,
                    "edge {edge}, seed {seed}: found {}",
                    e.x
                );
                assert!((e.before - 0.15).abs() < 1e-3 && (e.after - 0.75).abs() < 1e-3);
                assert!(e.iterations >= 1 && e.iterations <= 5);
            }
        }
    }

    /// A linear shading along the profile moves both flanks equally, so the local level
    /// follows it and the crossing stays on the step. One level taken from the profile's
    /// ends would be off by the shading at the step's distance from the centre.
    #[test]
    fn half_contrast_follows_shading_that_a_global_level_misses() {
        let mut lc = LevelCrossing1D::new();
        let edge = 37.3;
        let s = blurred_step(100, edge, 1.2, 0.1, 0.8, 0.004);
        let LevelOutcome::Found(e) = lc.half_contrast(&s, 36.0, &HalfContrastConfig::default())
        else {
            panic!("no crossing")
        };
        // Linear interpolation between whole samples of this step is good to 0.019 at a
        // 0.3 phase; the flank centres sit 0.2 off the step, adding 0.004 more.
        assert!((f64::from(e.x) - edge).abs() < 0.03, "found {}", e.x);

        let (before, after) = lc.end_levels(&s, NonZeroUsize::new(3).expect("nonzero"));
        let global = lc.crossings(&s, 0.5 * (before + after), EdgePolarity::Rising)[0];
        assert!(
            (f64::from(global) - edge).abs() > 0.1,
            "the end-level crossing should miss: {global}"
        );
    }

    /// The standard normal CDF, by Abramowitz–Stegun 7.1.26 (|error| < 1.5e-7).
    fn ndtr(z: f64) -> f64 {
        let x = z.abs() / std::f64::consts::SQRT_2;
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        let erf = 1.0 - poly * (-x * x).exp();
        0.5 * (1.0 + if z < 0.0 { -erf } else { erf })
    }

    /// CaliperBench's `test_half_contrast_does_not_depend_on_where_the_search_starts`:
    /// a blurred step with a long tail on its dark side, refined from five starts.
    #[test]
    fn half_contrast_does_not_depend_on_the_seed() {
        let s: Vec<f32> = (0..120)
            .map(|i| {
                let r = f64::from(i);
                let tail = if r < 60.3 {
                    0.15 * (-(60.3 - r).max(0.0) / 4.0).exp()
                } else {
                    0.0
                };
                (0.1 + tail + 0.6 * ndtr((r - 60.3) / 1.3)) as f32
            })
            .collect();
        let mut lc = LevelCrossing1D::new();
        let cfg = HalfContrastConfig::default();
        let found: Vec<f32> = [-2.5f32, -1.0, 0.0, 1.0, 2.5]
            .iter()
            .map(|d| match lc.half_contrast(&s, 60.0 + d, &cfg) {
                LevelOutcome::Found(e) => e.x,
                other => panic!("start {d}: {other:?}"),
            })
            .collect();
        let (lo, hi) = found
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        assert!(hi - lo < 0.01, "spread {} over {found:?}", hi - lo);
    }

    #[test]
    fn half_contrast_names_each_failure() {
        let mut lc = LevelCrossing1D::new();
        let cfg = HalfContrastConfig::default();
        let step = blurred_step(60, 30.0, 1.0, 0.2, 0.6, 0.0);

        // Too little contrast between the flanks.
        let strict = HalfContrastConfig {
            min_contrast: 0.5,
            ..cfg
        };
        match lc.half_contrast(&step, 30.0, &strict) {
            LevelOutcome::LowContrast { before, after } => {
                assert!((before - 0.2).abs() < 1e-3 && (after - 0.6).abs() < 1e-3);
            }
            other => panic!("{other:?}"),
        }

        // A seed so close to the end that the flank before it is empty.
        assert_eq!(lc.half_contrast(&step, 1.0, &cfg), LevelOutcome::NoCrossing);

        // A flat profile has levels but no crossing.
        let flat = vec![0.5f32; 60];
        assert_eq!(
            lc.half_contrast(&flat, 30.0, &cfg),
            LevelOutcome::NoCrossing
        );

        // The step is 5 samples from the seed: the level is crossed beyond `flank_near`.
        assert_eq!(
            lc.half_contrast(&step, 25.0, &cfg),
            LevelOutcome::NoCrossing
        );
    }

    /// One iteration is enough when the seed already sits on the crossing; `max_iter` caps
    /// the count otherwise.
    #[test]
    fn half_contrast_counts_its_iterations() {
        let mut lc = LevelCrossing1D::new();
        let s = blurred_step(80, 40.0, 1.2, 0.0, 1.0, 0.0);
        let LevelOutcome::Found(e) = lc.half_contrast(&s, 40.0, &HalfContrastConfig::default())
        else {
            panic!("no crossing")
        };
        assert!(e.iterations <= 2, "{e:?}");
        assert!((e.x - 40.0).abs() < 0.02, "{e:?}");
        assert!(
            e.before.abs() < 1e-3 && (e.after - 1.0).abs() < 1e-3,
            "{e:?}"
        );
        assert!((e.level - 0.5 * (e.before + e.after)).abs() < 1e-7, "{e:?}");

        let once = HalfContrastConfig {
            max_iter: nz(1),
            tol: 0.0,
            ..HalfContrastConfig::default()
        };
        let LevelOutcome::Found(e) = lc.half_contrast(&s, 38.5, &once) else {
            panic!("no crossing")
        };
        assert_eq!(e.iterations, 1);
    }
}
