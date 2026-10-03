//! Caliper placements: where a caliper sits, and how a profile index maps to the image.

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
    /// `2·half_len + 1` samples long.
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

/// The geometry a caliper is currently placed on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Placement {
    Rect(MeasureRect),
    Arc(MeasureArc),
    Radial(MeasureRadial),
}

impl Placement {
    /// Number of profile samples this placement produces at `step` pixels.
    pub(crate) fn profile_len(&self, step: f32) -> usize {
        let step = step.max(1e-3);
        let span = match *self {
            Placement::Rect(r) => 2.0 * r.half_len,
            Placement::Radial(r) => 2.0 * r.half_len,
            // One sample per pixel of arc length keeps the profile's units
            // comparable to a rect's, so `sigma` and `threshold` mean the same
            // thing for both.
            Placement::Arc(a) => a.angle_extent.abs() * a.radius,
        };
        (span.max(0.0) / step) as usize + 1
    }

    /// Half-width of the cross-scan average, in pixels.
    pub(crate) fn half_width(&self) -> f32 {
        match *self {
            Placement::Rect(r) => r.half_width,
            Placement::Arc(a) => a.half_width,
            Placement::Radial(r) => r.half_width,
        }
    }

    /// Absolute position of profile sample `i` of `n`, offset `s` across the scan.
    ///
    /// For [`Placement::Radial`] the cross-offset is applied **along the arc**
    /// at the sample's own radius rather than along a straight chord — that is
    /// the whole reason the variant exists.
    pub(crate) fn sample_point(&self, i: usize, n: usize, s: f32) -> Point2f {
        let denom = (n.saturating_sub(1)).max(1) as f32;
        match *self {
            Placement::Rect(r) => {
                let (sa, ca) = r.angle.sin_cos();
                let u = Vec2f::new(ca, sa);
                let t = -r.half_len + 2.0 * r.half_len * i as f32 / denom;
                r.center + u * t + u.perp() * s
            }
            Placement::Arc(a) => {
                let phi = a.angle_start + a.angle_extent * i as f32 / denom;
                let (sp, cp) = phi.sin_cos();
                let radial = Vec2f::new(cp, sp);
                a.center + radial * (a.radius + s)
            }
            Placement::Radial(r) => {
                let rad = r.radius - r.half_len + 2.0 * r.half_len * i as f32 / denom;
                // Arc length `s` at radius `rad` is an angle of `s / rad`.
                let dphi = if rad.abs() > 1e-3 { s / rad } else { 0.0 };
                let (sp, cp) = (r.angle + dphi).sin_cos();
                r.center + Vec2f::new(cp, sp) * rad
            }
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
        }
    }

    /// Recover a profile index from an edge's scan coordinate `t`.
    pub(crate) fn index_of(&self, t: f32, denom: f32) -> f32 {
        // For rect and radial, `t` is a signed distance from the caliper centre
        // along the scan axis, so the index is a plain affine map back.
        let linear = |half_len: f32| {
            if half_len.abs() < 1e-6 {
                0.0
            } else {
                (t + half_len) * denom / (2.0 * half_len)
            }
        };
        match *self {
            Placement::Rect(r) => linear(r.half_len),
            Placement::Radial(r) => linear(r.half_len),
            Placement::Arc(a) => {
                if a.radius.abs() < 1e-6 {
                    0.0
                } else {
                    t / a.radius / a.angle_extent * denom
                }
            }
        }
    }
}
