//! Caliper profiles sampled from the continuous derivative of Gaussian
//! ([`Derivative::Continuous`](super::Derivative::Continuous)).
//!
//! Each profile sample is the mean, over the lines across the placement, of
//! [`ProfileSampler`]'s value and derivative along the scan direction at that line's
//! point. A gradient peak found on the samples is then refined on the continuous
//! response: Newton steps on a parabola through three evaluations `0.1σ` apart along the
//! scan direction.

use vm_primitives::edge::{ProfileSample, ProfileSampler, ProfileSamplerConfig, ProfileStation};
use vm_primitives::{EdgePeak, EdgePolarity, ImageView, Pixel, Point2f, Vec2f};

use super::placement::Placement;

/// Newton steps per peak, at most.
const MAX_STEPS: usize = 4;
/// A step shorter than this, in pixels, ends the refinement.
const STEP_TOL_PX: f64 = 1e-4;

/// The sampler and the buffers of the continuous profile path.
#[derive(Debug, Clone, Default)]
pub(super) struct ContinuousProfile {
    sampler: ProfileSampler,
    stations: Vec<ProfileStation>,
    samples: Vec<Option<ProfileSample>>,
    /// The cross-averaged derivative per profile sample, in input units per sample; 0
    /// where a sample is invalid.
    pub resp: Vec<f32>,
    /// Whether every line of each profile sample was valid.
    valid: Vec<bool>,
    pub peaks: Vec<EdgePeak>,
}

/// Where one evaluation sits: placement, profile length, lines across, σ.
#[derive(Clone, Copy)]
pub(super) struct Scan {
    pub placement: Placement,
    pub n: usize,
    pub across: usize,
    pub sigma: f32,
    /// Pixels per profile sample.
    pub spacing: f32,
}

impl ContinuousProfile {
    /// Fill `profile` with the smoothed intensity and `resp` with the derivative at every
    /// profile sample. Returns whether every sample was valid.
    pub fn build<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        scan: Scan,
        profile: &mut [f32],
    ) -> bool {
        let n = profile.len();
        self.stations.clear();
        for i in 0..n {
            for j in 0..scan.across {
                self.stations.push(station(scan, i as f64, j));
            }
        }
        self.run(img, scan, &[0.0]);
        self.resp.clear();
        self.valid.clear();
        let mut all = true;
        for (i, out) in profile.iter_mut().enumerate() {
            let lines = &self.samples[i * scan.across..(i + 1) * scan.across];
            match mean(lines) {
                Some((v, d)) => {
                    *out = v as f32;
                    self.resp.push((d * f64::from(scan.spacing)) as f32);
                    self.valid.push(true);
                }
                None => {
                    *out = 0.0;
                    self.resp.push(0.0);
                    self.valid.push(false);
                    all = false;
                }
            }
        }
        all
    }

    /// Whether every sample of the last [`build`](Self::build) was valid.
    pub fn all_valid(&self) -> bool {
        self.valid.iter().all(|&v| v)
    }

    /// Whether each sample of the last [`build`](Self::build) was valid.
    pub fn valid(&self) -> &[bool] {
        &self.valid
    }

    /// The derivative peaks of `resp` above `threshold`, each refined on the continuous
    /// response, into `peaks`. A peak next to an invalid sample, or whose refinement
    /// reaches one, is dropped: it is not evidence.
    pub fn find_peaks<P: Pixel>(&mut self, img: &ImageView<'_, P>, scan: Scan, threshold: f32) {
        self.peaks.clear();
        for i in 1..self.resp.len().saturating_sub(1) {
            if !(self.valid[i - 1] && self.valid[i] && self.valid[i + 1]) {
                continue;
            }
            let (a, b, c) = (self.resp[i - 1], self.resp[i], self.resp[i + 1]);
            let polarity = if b >= a && b > c && b > threshold {
                EdgePolarity::Rising
            } else if b <= a && b < c && -b > threshold {
                EdgePolarity::Falling
            } else {
                continue;
            };
            if let Some((x, value)) = self.refine(img, scan, i, polarity) {
                self.peaks.push(EdgePeak {
                    x: x as f32,
                    idx: i,
                    value: value as f32,
                    strength: value.abs() as f32,
                    polarity,
                });
            }
        }
    }

    /// Newton steps from the parabola through samples `i − 1..=i + 1`: each fits a
    /// parabola to the response at `x − δ`, `x`, `x + δ` (`δ` = 0.1σ) and moves to its
    /// vertex, at most half a sample per step and never past the neighbouring samples.
    fn refine<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        scan: Scan,
        i: usize,
        polarity: EdgePolarity,
    ) -> Option<(f64, f64)> {
        let sign = if polarity == EdgePolarity::Rising {
            1.0
        } else {
            -1.0
        };
        let r = |k: usize| sign * f64::from(self.resp[k]);
        let (lo, hi) = ((i - 1) as f64, (i + 1) as f64);
        let mut x = i as f64 + vertex(r(i - 1), r(i), r(i + 1)).clamp(-0.5, 0.5);
        let spacing = f64::from(scan.spacing);
        let delta_px = 0.1 * f64::from(scan.sigma);
        let delta = delta_px / spacing;
        let mut value = f64::from(self.resp[i]);
        for _ in 0..MAX_STEPS {
            let [a, b, c] = self.response_at(img, scan, x, delta_px as f32)?;
            value = b;
            let step = (vertex(sign * a, sign * b, sign * c) * delta).clamp(-0.5, 0.5);
            x = (x + step).clamp(lo, hi);
            if (step * spacing).abs() < STEP_TOL_PX {
                break;
            }
        }
        Some((x, value))
    }

    /// The cross-averaged response, per profile sample, at the continuous profile
    /// position `x` and `delta_px` either side of it along the scan direction; `None` if
    /// any sample is invalid. The three share each line's station, so they are offsets of
    /// one rounded point.
    fn response_at<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        scan: Scan,
        x: f64,
        delta_px: f32,
    ) -> Option<[f64; 3]> {
        self.stations.clear();
        for j in 0..scan.across {
            self.stations.push(station(scan, x, j));
        }
        self.run(img, scan, &[-delta_px, 0.0, delta_px]);
        let mut out = [0.0; 3];
        for (k, o) in out.iter_mut().enumerate() {
            let (mut d, mut count) = (0.0f64, 0.0f64);
            for line in self.samples.as_chunks::<3>().0 {
                d += f64::from(line[k]?.derivative);
                count += 1.0;
            }
            *o = d / count.max(1.0) * f64::from(scan.spacing);
        }
        Some(out)
    }

    /// Sample every station at `offsets`. Stations a placement builds from finite
    /// geometry are valid; anything else leaves every sample invalid.
    fn run<P: Pixel>(&mut self, img: &ImageView<'_, P>, scan: Scan, offsets: &[f32]) {
        self.samples.clear();
        self.samples
            .resize(self.stations.len() * offsets.len(), None);
        let cfg = ProfileSamplerConfig {
            sigma: scan.sigma.max(1e-3),
        };
        if self
            .sampler
            .sample(img, None, &self.stations, offsets, &cfg, &mut self.samples)
            .is_err()
        {
            self.samples.iter_mut().for_each(|s| *s = None);
        }
    }
}

fn station(scan: Scan, x: f64, j: usize) -> ProfileStation {
    let (p, u) = scan.placement.station_at(x, scan.n, j, scan.across);
    ProfileStation {
        p: Point2f::new(p.0 as f32, p.1 as f32),
        n: Vec2f::new(u.0 as f32, u.1 as f32),
    }
}

/// The mean value and derivative over the lines of one sample, if all are valid.
fn mean(lines: &[Option<ProfileSample>]) -> Option<(f64, f64)> {
    let (mut v, mut d) = (0.0f64, 0.0f64);
    for s in lines {
        let s = (*s)?;
        v += f64::from(s.value);
        d += f64::from(s.derivative);
    }
    let k = lines.len().max(1) as f64;
    Some((v / k, d / k))
}

/// The vertex of the parabola through `(−1, a)`, `(0, b)`, `(1, c)`, in units of the
/// spacing; 0 unless the parabola opens downwards.
fn vertex(a: f64, b: f64, c: f64) -> f64 {
    let curvature = a - 2.0 * b + c;
    if curvature < 0.0 {
        0.5 * (a - c) / curvature
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use vm_primitives::{EdgePolarity, Image, Point2f};

    use super::Placement;
    use crate::measure::diagnostics::explain;
    use crate::measure::{
        Caliper, Derivative, Locate, MeasureArc, MeasureConfig, MeasureRadial, MeasureRect,
        MeasureStrip, OffImage, ProfileConfig, RejectReason,
    };

    /// A step `n·p = d` (`n` at `deg` from +x), 40 → 200 DN, each pixel its coverage of
    /// the bright side over an 8 × 8 grid, as `f32`.
    fn step(size: usize, deg: f32, d: f32) -> Image<f32> {
        let (s, c) = deg.to_radians().sin_cos();
        let data = (0..size * size)
            .map(|i| {
                let (x, y) = ((i % size) as f32, (i / size) as f32);
                let mut cover = 0.0;
                for k in 0..64 {
                    let (u, v) = ((k % 8) as f32 / 8.0 - 0.4375, (k / 8) as f32 / 8.0 - 0.4375);
                    cover += f32::from(u8::from(c * (x + u) + s * (y + v) >= d));
                }
                40.0 + 160.0 * cover / 64.0
            })
            .collect();
        Image::from_vec(size, size, data).expect("valid image")
    }

    fn continuous(sigma: f32) -> MeasureConfig {
        MeasureConfig {
            profile: ProfileConfig {
                sigma,
                derivative: Derivative::Continuous,
                ..ProfileConfig::default()
            },
            ..MeasureConfig::default()
        }
    }

    fn strip(start: Point2f, end: Point2f, half_width: f32, across: usize) -> MeasureStrip {
        MeasureStrip {
            start,
            end,
            half_width,
            samples: None,
            across: NonZeroUsize::new(across),
        }
    }

    #[test]
    fn every_placement_finds_the_step_on_the_continuous_response() {
        // A step at 20° through (40.3, 40.3); each placement crosses it along its normal.
        let deg = 20.0f32;
        let (s, c) = deg.to_radians().sin_cos();
        let d = 40.3 * (c + s);
        let img = step(80, deg, d);
        let off = |p: Point2f| c * p.x + s * p.y - d;
        let cfg = continuous(1.5);
        let at = |t: f32| Point2f::new(40.3 + t * c, 40.3 + t * s);
        let rect = MeasureRect {
            center: at(0.4),
            angle: deg.to_radians(),
            half_len: 12.0,
            half_width: 3.0,
        };
        // A circle of radius 300 about a far centre is the straight step to within
        // 0.01 px over this radial scan's ±1 px of arc.
        let radial = MeasureRadial {
            center: at(-300.0),
            radius: 300.3,
            angle: deg.to_radians(),
            half_len: 12.0,
            half_width: 1.0,
        };
        let mut cals = [
            Caliper::rect(rect, cfg),
            Caliper::strip(strip(at(-12.0), at(13.0), 2.0, 3), cfg),
            Caliper::radial(radial, cfg),
        ];
        for cal in &mut cals {
            let e = cal.measure(&img.as_view()).expect("the step")[0];
            assert_eq!(e.polarity, EdgePolarity::Rising);
            assert!(
                off(e.p).abs() < 0.03,
                "located {} px off the step",
                off(e.p)
            );
        }
        // An arc scans along its tangent: clockwise at angle 0, that is −y.
        let arc = MeasureArc {
            center: Point2f::new(40.0, 40.0),
            radius: 20.0,
            angle_start: 0.0,
            angle_extent: -1.0,
            half_width: 2.0,
        };
        let (p, u) = Placement::Arc(arc).station_at(0.0, 21, 0, 3);
        assert!(
            (p.0 - 58.0).abs() < 1e-6 && (p.1 - 40.0).abs() < 1e-6,
            "{p:?}"
        );
        assert!(
            u.0.abs() < 1e-12 && (u.1 + 1.0).abs() < 1e-12,
            "tangent {u:?}"
        );
    }

    #[test]
    fn amplitudes_and_level_methods_match_the_default_derivative() {
        let img = step(64, 0.0, 31.37);
        let line = strip(Point2f::new(12.0, 32.0), Point2f::new(52.0, 32.0), 0.0, 1);
        let dog = Caliper::strip(line, MeasureConfig::default())
            .measure(&img.as_view())
            .expect("the step")[0];
        let cont = Caliper::strip(line, continuous(1.0))
            .measure(&img.as_view())
            .expect("the step")[0];
        // Both responses are per profile sample and read a ramp's slope. The taper narrows
        // the kernel, so a step's peak reads higher: 1.15× in the continuous limit.
        let ratio = cont.amplitude / dog.amplitude;
        assert!((1.1..1.35).contains(&ratio), "amplitude ratio {ratio}");
        // An unblurred step at σ 1: the sampling bias reaches 0.08 px (the phase sweep).
        assert!((cont.p.x - 31.37).abs() < 0.1, "edge at {}", cont.p.x);

        for locate in [
            Locate::MidpointCrossing {
                endpoint_samples: NonZeroUsize::new(3).expect("nonzero"),
                min_contrast: 5.0,
            },
            Locate::HalfContrast {
                flank_near_px: 3.0,
                flank_far_px: 8.0,
                tol_px: 0.01,
                max_iter: NonZeroUsize::new(5).expect("nonzero"),
                min_contrast: 5.0,
            },
        ] {
            let cfg = MeasureConfig {
                locate,
                ..continuous(1.0)
            };
            let mut cal = Caliper::strip(line, cfg);
            let e = cal.measure(&img.as_view()).expect("the step")[0];
            assert!((e.p.x - 31.37).abs() < 0.1, "{locate:?}: edge at {}", e.p.x);
            // The level methods read the smoothed profile as it is.
            let trace = explain(&mut cal, &img.as_view());
            assert_eq!(trace.profile, trace.smoothed);
        }
    }

    #[test]
    fn support_off_the_image_is_not_evidence() {
        // The strip starts 2 px from the left border; with σ 1 the first samples' support
        // leaves the image. The step at x = 20 is far from it.
        let img = step(40, 0.0, 20.0);
        let line = strip(Point2f::new(2.0, 20.0), Point2f::new(30.0, 20.0), 0.0, 1);
        let mut cal = Caliper::strip(line, continuous(1.0));
        let e = cal
            .measure(&img.as_view())
            .expect("the step, on the valid part")[0];
        assert!((e.p.x - 20.0).abs() < 0.05, "edge at {}", e.p.x);
        let trace = explain(&mut cal, &img.as_view());
        assert_eq!(
            trace.response[0], 0.0,
            "an invalid sample carries no response"
        );
        assert_eq!(trace.profile[0], 0.0);

        let mut reject = continuous(1.0);
        reject.profile.off_image = OffImage::Reject;
        assert_eq!(
            Caliper::strip(line, reject).measure(&img.as_view()),
            Err(RejectReason::OffImage)
        );
        let midpoint = MeasureConfig {
            locate: Locate::MidpointCrossing {
                endpoint_samples: NonZeroUsize::new(3).expect("nonzero"),
                min_contrast: 5.0,
            },
            ..continuous(1.0)
        };
        assert_eq!(
            Caliper::strip(line, midpoint).measure(&img.as_view()),
            Err(RejectReason::OffImage),
            "a level method cannot read levels off border fill"
        );
        // An edge whose support needs pixels outside the image is not found at all.
        let at_border = step(40, 0.0, 1.5);
        let near = strip(Point2f::new(0.0, 20.0), Point2f::new(20.0, 20.0), 0.0, 1);
        assert_eq!(
            Caliper::strip(near, continuous(1.0)).measure(&at_border.as_view()),
            Err(RejectReason::OffImage)
        );
    }
}
