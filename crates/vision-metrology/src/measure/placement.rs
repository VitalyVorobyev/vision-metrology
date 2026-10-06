//! Caliper placements: where a caliper sits, and how a profile index maps to the image.

use std::num::NonZeroUsize;

use vm_primitives::{Point2f, Vec2f, Vec2fExt};

/// A rectangular measurement region.
///
/// The caliper scans **along** its own x-axis (the direction `angle` points in)
/// and averages **across** it. Averaging is what buys the sub-pixel repeatability:
/// each profile sample is the mean of `2·half_width + 1` interpolated pixels, so
/// noise falls as `1/√n` while a straight edge perpendicular to the scan stays
/// exactly as sharp.
///
/// That last part is the constraint worth remembering: `half_width` may only be
/// increased while the edge stays parallel to the averaging direction. On a
/// curved edge a wide caliper smears the very transition it is measuring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureRect {
    /// Centre of the rectangle, in image coordinates.
    pub center: Point2f,
    /// Direction of the scan axis, in radians.
    pub angle: f32,
    /// Half-length along the scan axis, in pixels. The profile is
    /// `2·half_len + 1` samples long at `step = 1`.
    pub half_len: f32,
    /// Half-width across the scan axis, in pixels. `0.0` samples a single line.
    pub half_width: f32,
}

/// An annular measurement region: scans **along** a circular arc, averaging
/// **radially**.
///
/// Use this to find features that cross a circular path — gear teeth, slots
/// around a bore, the tab on a can end. To measure the circle *itself*, use
/// [`MeasureRadial`]; that is what
/// [`MetrologyShape::Circle`](super::MetrologyShape::Circle) does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureArc {
    /// Centre of the arc.
    pub center: Point2f,
    /// Radius of the scan path, in pixels.
    pub radius: f32,
    /// Start angle, in radians.
    pub angle_start: f32,
    /// Signed angular extent, in radians. Negative sweeps clockwise.
    pub angle_extent: f32,
    /// Half-width of the radial averaging band, in pixels.
    pub half_width: f32,
}

/// A caliper that scans **radially** and averages **along the arc**.
///
/// This is the geometry to measure a circular edge with, and it exists because
/// a [`MeasureRect`] cannot do it without bias. A rect averages along a
/// *chord*: on a circle of radius 40, samples 5 px to either side of the
/// caliper sit at radius 40.31, outside the edge, so the averaged profile is
/// contaminated by the wrong side of the transition and the measured radius
/// comes out low. Measured on a synthetic disc: **−0.12 px** at
/// `half_width = 5`, growing with width and shrinking with radius.
///
/// Averaging along the arc puts every averaged sample at the *same* radius, so
/// a circular edge stays perfectly sharp however wide the caliper is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureRadial {
    /// Centre of the circle being measured.
    pub center: Point2f,
    /// Nominal radius the caliper is centred on, in pixels.
    pub radius: f32,
    /// Angular position of the caliper on the circle, in radians.
    pub angle: f32,
    /// Half-length of the radial search, in pixels.
    pub half_len: f32,
    /// Half-width of the arc-following average, in pixels of **arc length**.
    pub half_width: f32,
}

/// A straight strip from `start` to `end`: scans along it, averages across it.
///
/// Unlike [`MeasureRect`], a strip is anchored at its endpoints and can fix its
/// sample counts, so a profile can be specified sample for sample:
///
/// - `samples` points along the strip, **both endpoints included**, spaced
///   `length / (samples − 1)` apart;
/// - `across` lines spread evenly over `±half_width` (one line sits on the centre line);
/// - an edge's `t` is its distance from `start`.
///
/// `None` counts fall back to the same rule as a rect: one sample per
/// [`ProfileConfig::step`](super::ProfileConfig::step) along the strip and about one
/// line per pixel across it.
///
/// # Example
/// ```
/// use std::num::NonZeroUsize;
/// use vision_metrology::measure::{Caliper, MeasureConfig, MeasureStrip};
/// use vision_metrology::{Image, Point2f};
///
/// // A bright bar on columns 16..48, scanned right to left.
/// let data: Vec<f32> = (0..9 * 64)
///     .map(|i| if (16..48).contains(&(i % 64)) { 1.0 } else { 0.0 })
///     .collect();
/// let img = Image::from_vec(64, 9, data).unwrap();
/// let strip = MeasureStrip {
///     start: Point2f::new(63.0, 4.0),
///     end: Point2f::new(0.0, 4.0),
///     half_width: 0.0,
///     samples: NonZeroUsize::new(64),
///     across: NonZeroUsize::new(1),
/// };
/// let cfg = MeasureConfig { threshold: 0.01, ..MeasureConfig::default() };
/// let mut cal = Caliper::strip(strip, cfg);
/// let edges = cal.measure(&img.as_view()).expect("two edges");
/// // Distances from `start` = (63, 4): the edges at x = 47.5 and x = 15.5.
/// assert!((edges[0].t - 15.5).abs() < 1e-3 && (edges[1].t - 47.5).abs() < 1e-3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MeasureStrip {
    /// First end of the scan, in image coordinates.
    pub start: Point2f,
    /// Last end of the scan, in image coordinates.
    pub end: Point2f,
    /// Half-width across the scan, in pixels.
    pub half_width: f32,
    /// Number of samples along the strip, endpoints included.
    pub samples: Option<NonZeroUsize>,
    /// Number of lines averaged across the strip.
    pub across: Option<NonZeroUsize>,
}

impl MeasureStrip {
    /// Length of the strip, in pixels.
    pub fn length(&self) -> f32 {
        self.geometry().length as f32
    }

    /// Unit scan direction and its left normal, and the length — in `f64`, the way
    /// every strip coordinate is computed.
    pub(crate) fn geometry(&self) -> StripGeometry {
        let (dx, dy) = (
            f64::from(self.end.x) - f64::from(self.start.x),
            f64::from(self.end.y) - f64::from(self.start.y),
        );
        let length = dx.hypot(dy);
        let (ux, uy) = if length > 0.0 {
            (dx / length, dy / length)
        } else {
            (1.0, 0.0)
        };
        StripGeometry {
            start: (f64::from(self.start.x), f64::from(self.start.y)),
            u: (ux, uy),
            normal: (-uy, ux),
            length,
        }
    }
}

/// [`MeasureStrip`] geometry in `f64`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StripGeometry {
    pub start: (f64, f64),
    pub u: (f64, f64),
    pub normal: (f64, f64),
    pub length: f64,
}

impl StripGeometry {
    /// Distance from `start` of sample `i` of `n`: `i · length / (n − 1)`, the last one
    /// exactly `length` (numpy's `linspace`).
    pub(crate) fn along(&self, i: usize, n: usize) -> f64 {
        if n < 2 {
            0.0
        } else if i + 1 == n {
            self.length
        } else {
            i as f64 * (self.length / (n - 1) as f64)
        }
    }

    /// Offset of line `j` of `a` across the strip, evenly over `±half_width` with the
    /// last one exactly `+half_width`; a single line sits on the centre line.
    pub(crate) fn across(half_width: f64, j: usize, a: usize) -> f64 {
        if a < 2 {
            0.0
        } else if j + 1 == a {
            half_width
        } else {
            j as f64 * (2.0 * half_width / (a - 1) as f64) - half_width
        }
    }

    /// The image point `t` along and `o` across.
    pub(crate) fn point(&self, t: f64, o: f64) -> (f64, f64) {
        (
            self.start.0 + t * self.u.0 + o * self.normal.0,
            self.start.1 + t * self.u.1 + o * self.normal.1,
        )
    }
}

/// The geometry a caliper is currently placed on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Placement {
    Rect(MeasureRect),
    Arc(MeasureArc),
    Radial(MeasureRadial),
    Strip(MeasureStrip),
}

impl Placement {
    /// Number of profile samples this placement produces at `step` pixels.
    pub(crate) fn profile_len(&self, step: f32) -> usize {
        let step = step.max(1e-3);
        if let Placement::Strip(s) = *self {
            let length = s.geometry().length;
            if length <= 0.0 {
                return 1;
            }
            return s
                .samples
                .map_or((length / f64::from(step)) as usize + 1, NonZeroUsize::get);
        }
        let span = match *self {
            Placement::Rect(r) => 2.0 * r.half_len,
            Placement::Radial(r) => 2.0 * r.half_len,
            // One sample per pixel of arc length keeps the profile's units
            // comparable to a rect's, so `sigma` and `threshold` mean the same
            // thing for both.
            Placement::Arc(a) => a.angle_extent.abs() * a.radius,
            Placement::Strip(_) => unreachable!("handled above"),
        };
        (span.max(0.0) / step) as usize + 1
    }

    /// Distance between profile samples, in pixels, for a profile of `n` samples: the
    /// scan's extent over `n − 1`, the rate at which [`point_at`](Self::point_at)'s `t`
    /// advances per sample. `step` when `n < 2`.
    pub(crate) fn spacing(&self, step: f32, n: usize) -> f32 {
        if n < 2 {
            return step.max(1e-3);
        }
        let denom = (n - 1) as f32;
        match *self {
            Placement::Rect(r) => 2.0 * r.half_len / denom,
            Placement::Radial(r) => 2.0 * r.half_len / denom,
            Placement::Arc(a) => a.angle_extent.abs() * a.radius / denom,
            Placement::Strip(s) => (s.geometry().length / (n - 1) as f64) as f32,
        }
    }

    /// The spacing the pixel-unit settings (σ, the derivative kernel radius, the
    /// half-contrast flank distances and tolerance) are converted to samples with.
    ///
    /// A strip uses its real [`spacing`](Self::spacing); rect, arc and radial placements
    /// use the nominal `step`, which differs from the real spacing when the extent is not
    /// a whole number of steps.
    pub(crate) fn sigma_spacing(&self, step: f32, n: usize) -> f32 {
        match *self {
            Placement::Strip(_) => self.spacing(step, n),
            _ => step.max(1e-3),
        }
    }

    /// Number of lines averaged across the scan.
    pub(crate) fn across_count(&self) -> usize {
        match *self {
            Placement::Strip(MeasureStrip {
                across: Some(a), ..
            }) => a.get(),
            _ => (2.0 * self.half_width()).max(0.0) as usize + 1,
        }
    }

    /// Half-width of the cross-scan average, in pixels.
    pub(crate) fn half_width(&self) -> f32 {
        match *self {
            Placement::Rect(r) => r.half_width,
            Placement::Arc(a) => a.half_width,
            Placement::Radial(r) => r.half_width,
            Placement::Strip(s) => s.half_width,
        }
    }

    /// Unit scan direction at profile position `x` — the direction an edge
    /// position is measured along, used for the obliquity check.
    pub(crate) fn scan_dir(&self, x: f32, denom: f32) -> Vec2f {
        match *self {
            Placement::Rect(r) => {
                let (sa, ca) = r.angle.sin_cos();
                Vec2f::new(ca, sa)
            }
            Placement::Radial(r) => {
                let (sa, ca) = r.angle.sin_cos();
                Vec2f::new(ca, sa)
            }
            Placement::Arc(a) => {
                // Tangent to the arc at this position.
                let phi = a.angle_start + a.angle_extent * x / denom;
                let (sp, cp) = phi.sin_cos();
                let t = Vec2f::new(-sp, cp);
                if a.angle_extent < 0.0 { -t } else { t }
            }
            Placement::Strip(s) => {
                let g = s.geometry();
                Vec2f::new(g.u.0 as f32, g.u.1 as f32)
            }
        }
    }

    /// Map a subpixel profile index back to image coordinates and the scan coordinate `t`.
    pub(crate) fn point_at(&self, x: f32, denom: f32) -> (Point2f, f32) {
        match *self {
            Placement::Rect(r) => {
                let (s, c) = r.angle.sin_cos();
                let t = -r.half_len + 2.0 * r.half_len * x / denom;
                (r.center + Vec2f::new(c, s) * t, t)
            }
            Placement::Arc(a) => {
                let phi = a.angle_start + a.angle_extent * x / denom;
                let (s, c) = phi.sin_cos();
                (
                    a.center + Vec2f::new(c, s) * a.radius,
                    (phi - a.angle_start) * a.radius,
                )
            }
            Placement::Radial(r) => {
                let (s, c) = r.angle.sin_cos();
                let t = -r.half_len + 2.0 * r.half_len * x / denom;
                (r.center + Vec2f::new(c, s) * (r.radius + t), t)
            }
            Placement::Strip(st) => {
                let g = st.geometry();
                let t = f64::from(x) * (g.length / f64::from(denom));
                let (px, py) = g.point(t, 0.0);
                (Point2f::new(px as f32, py as f32), t as f32)
            }
        }
    }
}

/// `(n − 1)` as the divisor of a profile index, at least 1.
#[inline]
fn index_denom(n: usize) -> f32 {
    (n.saturating_sub(1)).max(1) as f32
}

impl MeasureRect {
    /// Position of profile sample `i` of `n`, offset `s` across the scan.
    #[inline]
    pub(crate) fn sample_point(&self, i: usize, n: usize, s: f32) -> Point2f {
        let (sa, ca) = self.angle.sin_cos();
        let u = Vec2f::new(ca, sa);
        let t = -self.half_len + 2.0 * self.half_len * i as f32 / index_denom(n);
        self.center + u * t + u.perp() * s
    }
}

impl MeasureArc {
    /// Position of profile sample `i` of `n`, offset `s` radially.
    #[inline]
    pub(crate) fn sample_point(&self, i: usize, n: usize, s: f32) -> Point2f {
        let phi = self.angle_start + self.angle_extent * i as f32 / index_denom(n);
        let (sp, cp) = phi.sin_cos();
        let radial = Vec2f::new(cp, sp);
        self.center + radial * (self.radius + s)
    }
}

impl MeasureRadial {
    /// Position of profile sample `i` of `n`, offset `s` **along the arc** at the
    /// sample's own radius rather than along a straight chord — the reason the
    /// placement exists.
    #[inline]
    pub(crate) fn sample_point(&self, i: usize, n: usize, s: f32) -> Point2f {
        let rad = self.radius - self.half_len + 2.0 * self.half_len * i as f32 / index_denom(n);
        // Arc length `s` at radius `rad` is an angle of `s / rad`.
        let dphi = if rad.abs() > 1e-3 { s / rad } else { 0.0 };
        let (sp, cp) = (self.angle + dphi).sin_cos();
        self.center + Vec2f::new(cp, sp) * rad
    }
}
