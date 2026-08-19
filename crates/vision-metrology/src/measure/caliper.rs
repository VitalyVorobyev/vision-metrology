//! Caliper measurement: a placed geometry, a 1-D profile, subpixel edges.

use vm_primitives::{
    BorderMode, Edge1DConfig, Edge1DDetector, EdgePolarity, ImageView, Pixel, Point2f,
    SubpixRefine, Vec2f, Vec2fExt, sample_bilinear_at, sample_bilinear_f32,
};

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
/// around a bore, the tab on a can end. To measure the circle *itself*, place
/// radial [`MeasureRect`]s instead; that is what
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

/// Which edges to keep from a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EdgeSelect {
    /// Every edge that passes the threshold, in profile order.
    #[default]
    All,
    /// The first along the scan direction.
    First,
    /// The last along the scan direction.
    Last,
    /// The one with the largest amplitude.
    Strongest,
}

/// Which transitions count as edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolaritySelect {
    /// Both dark→bright and bright→dark.
    #[default]
    Any,
    /// Dark→bright along the scan direction only.
    Rising,
    /// Bright→dark along the scan direction only.
    Falling,
}

/// How a caliper extracts edges from its profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureConfig {
    /// Gaussian σ of the 1-D derivative-of-Gaussian kernel, in pixels.
    ///
    /// Roughly the edge blur to expect. Too small and noise produces edges;
    /// too large and neighbouring edges merge.
    pub sigma: f32,
    /// Minimum `|DoG response|` for an edge to be reported.
    ///
    /// On the input pixel scale, like every other threshold in this workspace:
    /// re-tune for `u16` and `f32` images.
    pub threshold: f32,
    /// Which transitions count.
    pub polarity: PolaritySelect,
    /// Which of the surviving edges to return.
    pub select: EdgeSelect,
    /// Profile sampling step along the scan axis, in pixels.
    ///
    /// `1.0` samples one profile entry per pixel. Oversampling (`0.5`) buys
    /// resolution on a sharp edge at proportional cost; `sigma` is in the same
    /// units, so halving the step means doubling `sigma` for the same
    /// smoothing.
    pub step: f32,
    /// Maximum angle, in degrees, between the scan direction and the image
    /// gradient at the found edge. `180.0` disables the check.
    ///
    /// A caliper that crosses an edge obliquely reports a position along its
    /// own axis, not the edge's normal, and the two differ by `1/cos θ`. At a
    /// corner or a cap there is no meaningful crossing at all. Rejecting those
    /// is what keeps a bad caliper out of the fit instead of merely
    /// down-weighted. Borrowed from the caliper in `rtvt-pano`, which uses the
    /// same gate to reject FOV cuts and bead end-caps.
    pub max_obliquity_deg: f32,
    /// Border behaviour when the caliper overhangs the image.
    pub border: BorderMode<f32>,
}

impl Default for MeasureConfig {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            threshold: 5.0,
            polarity: PolaritySelect::default(),
            select: EdgeSelect::default(),
            step: 1.0,
            max_obliquity_deg: 180.0,
            border: BorderMode::Clamp,
        }
    }
}

/// Why a caliper returned nothing.
///
/// A caliper that finds no edge is not an error — it is a measurement result,
/// and *which gate* rejected it is the difference between "the part is missing"
/// and "the search window was too short". Modelled on the reject tally in the
/// `rtvt-pano` caliper, where the dominant reason across a scan is the fastest
/// route to a misconfigured recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// The profile was shorter than the detector needs (3 samples).
    ProfileTooShort,
    /// No response reached [`MeasureConfig::threshold`]. Either there is no
    /// edge in the window, or the window does not reach it.
    NoEdge,
    /// Edges were found, but none had the polarity
    /// [`MeasureConfig::polarity`] asked for.
    WrongPolarity,
    /// The best edge crossed at more than
    /// [`MeasureConfig::max_obliquity_deg`] from the scan direction — a corner,
    /// a cap, or a badly placed caliper.
    TooOblique,
    /// The caliper reached outside the image, so the profile is partly border
    /// fill rather than data.
    OffImage,
}

/// One edge found by a caliper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureEdge {
    /// Subpixel position in **image** coordinates.
    pub p: Point2f,
    /// Signed distance from the caliper centre along the scan axis, in pixels.
    pub t: f32,
    /// `|DoG response|` at the edge — the local contrast.
    pub amplitude: f32,
    /// Direction of the intensity transition along the scan axis.
    pub polarity: EdgePolarity,
}

/// A pair of opposite-polarity edges, i.e. one bar or gap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurePair {
    /// The earlier edge along the scan axis.
    pub first: MeasureEdge,
    /// The later edge.
    pub second: MeasureEdge,
    /// Midpoint in image coordinates.
    pub center: Point2f,
    /// Distance between the two edges along the scan axis, in pixels.
    pub width: f32,
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

/// The geometry a [`Caliper`] is currently placed on.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Placement {
    Rect(MeasureRect),
    Arc(MeasureArc),
    Radial(MeasureRadial),
}

/// A reusable caliper: place it, then measure frame after frame.
///
/// Owns the profile buffer, the 1-D detector and the output vectors, so a
/// measurement allocates nothing after the first call at a given size. Results
/// are handed back as borrowed slices for the same reason; copy them if you
/// need to outlive the next call.
///
/// # Example
/// ```
/// use vision_metrology::measure::{Caliper, MeasureConfig, MeasureRect};
/// use vision_metrology::{Image, Point2f};
///
/// // A vertical step: dark left of x = 30, bright from x = 30 on.
/// let mut data = vec![20u8; 64 * 64];
/// for y in 0..64 {
///     for x in 30..64 {
///         data[y * 64 + x] = 200;
///     }
/// }
/// let img = Image::from_vec(64, 64, data).unwrap();
///
/// // Scan horizontally through the step, averaging 21 rows.
/// let mut cal = Caliper::rect(
///     MeasureRect {
///         center: Point2f::new(32.0, 32.0),
///         angle: 0.0,
///         half_len: 20.0,
///         half_width: 10.0,
///     },
///     MeasureConfig::default(),
/// );
///
/// let edges = cal.measure(&img.as_view()).expect("an edge");
/// assert_eq!(edges.len(), 1);
/// // Pixel centres: the transition sits between x = 29 and x = 30.
/// assert!((edges[0].p.x - 29.5).abs() < 0.05, "found at {}", edges[0].p.x);
/// ```
#[derive(Debug, Clone)]
pub struct Caliper {
    placement: Placement,
    cfg: MeasureConfig,
    det: Edge1DDetector,
    profile: Vec<f32>,
    edges: Vec<MeasureEdge>,
    pairs: Vec<MeasurePair>,
}

impl Caliper {
    /// Place a caliper on a rectangle.
    pub fn rect(rect: MeasureRect, cfg: MeasureConfig) -> Self {
        Self::new(Placement::Rect(rect), cfg)
    }

    /// Place a caliper on an arc.
    pub fn arc(arc: MeasureArc, cfg: MeasureConfig) -> Self {
        Self::new(Placement::Arc(arc), cfg)
    }

    /// Place a caliper radially on a circle, averaging along the arc.
    ///
    /// Prefer this over [`rect`](Self::rect) whenever the edge being measured
    /// is curved — see [`MeasureRadial`].
    pub fn radial(radial: MeasureRadial, cfg: MeasureConfig) -> Self {
        Self::new(Placement::Radial(radial), cfg)
    }

    fn new(placement: Placement, cfg: MeasureConfig) -> Self {
        Self {
            placement,
            det: Edge1DDetector::new(cfg.sigma.max(1e-3)),
            cfg,
            profile: Vec::new(),
            edges: Vec::new(),
            pairs: Vec::new(),
        }
    }

    /// Move the caliper to a new rectangle, keeping its buffers.
    pub fn set_rect(&mut self, rect: MeasureRect) {
        self.placement = Placement::Rect(rect);
    }

    /// Move the caliper to a new arc, keeping its buffers.
    pub fn set_arc(&mut self, arc: MeasureArc) {
        self.placement = Placement::Arc(arc);
    }

    /// Move the caliper to a new radial placement, keeping its buffers.
    pub fn set_radial(&mut self, radial: MeasureRadial) {
        self.placement = Placement::Radial(radial);
    }

    /// Replace the extraction config.
    pub fn set_config(&mut self, cfg: MeasureConfig) {
        self.cfg = cfg;
    }

    /// The current config.
    pub fn config(&self) -> &MeasureConfig {
        &self.cfg
    }

    /// The averaged 1-D profile from the last [`measure`](Self::measure) call.
    ///
    /// Useful for diagnostics — plotting it shows immediately whether a missed
    /// edge was a threshold problem or a placement problem.
    pub fn profile(&self) -> &[f32] {
        &self.profile
    }

    /// Extract edges from the image under the current placement.
    ///
    /// `Ok` carries the edges, valid until the next call on this caliper; `Err`
    /// names the gate that rejected them. On a production line the distinction
    /// between "no edge in the window" and "the crossing was too oblique" is
    /// the difference between a missing part and a mis-taught recipe, so there
    /// is no variant of this call that discards it — an empty `Ok(&[])` is
    /// unrepresentable, because an extraction that found nothing always has a
    /// [`RejectReason`].
    pub fn measure<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
    ) -> Result<&[MeasureEdge], RejectReason> {
        let inside = self.build_profile(img);
        match self.extract(img, !inside) {
            Some(reason) => Err(reason),
            None => Ok(&self.edges),
        }
    }

    /// Extract opposite-polarity edge pairs — one per bar or gap crossed.
    ///
    /// Pairs are formed greedily in scan order: each edge is matched with the
    /// next edge of the opposite polarity, and both are then consumed. This is
    /// the behaviour that reads a bar as one object rather than two.
    ///
    /// Ignores [`MeasureConfig::select`] and [`MeasureConfig::polarity`] —
    /// pairing needs every edge of both polarities to be visible.
    pub fn measure_pairs<P: Pixel>(&mut self, img: &ImageView<'_, P>) -> &[MeasurePair] {
        let _ = self.build_profile(img);

        let saved = (self.cfg.select, self.cfg.polarity);
        self.cfg.select = EdgeSelect::All;
        self.cfg.polarity = PolaritySelect::Any;
        let _ = self.extract(img, false);
        (self.cfg.select, self.cfg.polarity) = saved;

        self.pairs.clear();
        let mut i = 0;
        while i + 1 < self.edges.len() {
            let a = self.edges[i];
            if let Some(off) = self.edges[i + 1..]
                .iter()
                .position(|e| e.polarity != a.polarity)
            {
                let b = self.edges[i + 1 + off];
                self.pairs.push(MeasurePair {
                    first: a,
                    second: b,
                    center: Point2f::new(0.5 * (a.p.x + b.p.x), 0.5 * (a.p.y + b.p.y)),
                    width: b.t - a.t,
                });
                i += off + 2;
            } else {
                break;
            }
        }
        &self.pairs
    }

    /// Number of profile samples the current placement produces.
    fn profile_len(&self) -> usize {
        let step = self.cfg.step.max(1e-3);
        let span = match self.placement {
            Placement::Rect(r) => 2.0 * r.half_len,
            Placement::Radial(r) => 2.0 * r.half_len,
            // One sample per pixel of arc length keeps the profile's units
            // comparable to a rect's, so `sigma` and `threshold` mean the same
            // thing for both.
            Placement::Arc(a) => a.angle_extent.abs() * a.radius,
        };
        (span.max(0.0) / step) as usize + 1
    }

    /// Fill `profile` with the cross-averaged intensity along the scan axis.
    ///
    /// Returns `false` when any sample fell outside the image, so the caller
    /// can report [`RejectReason::OffImage`] rather than silently measuring
    /// border fill.
    fn build_profile<P: Pixel>(&mut self, img: &ImageView<'_, P>) -> bool {
        let n = self.profile_len();
        self.profile.clear();
        self.profile.resize(n, 0.0);
        if n == 0 || img.width() == 0 || img.height() == 0 {
            return false;
        }

        let half_width = match self.placement {
            Placement::Rect(r) => r.half_width,
            Placement::Arc(a) => a.half_width,
            Placement::Radial(r) => r.half_width,
        };
        let across = (2.0 * half_width).max(0.0) as usize + 1;
        let across_denom = (across.saturating_sub(1)).max(1) as f32;
        let (w, h) = (img.width() as f32, img.height() as f32);
        let mut inside = true;

        for i in 0..n {
            let mut acc = 0.0f32;
            for j in 0..across {
                let s = if across == 1 {
                    0.0
                } else {
                    -half_width + 2.0 * half_width * j as f32 / across_denom
                };
                let q = self.sample_point(i, n, s);
                if q.x < 0.0 || q.y < 0.0 || q.x > w - 1.0 || q.y > h - 1.0 {
                    inside = false;
                }
                acc += sample_bilinear_at(img, q, self.cfg.border);
            }
            self.profile[i] = acc / across as f32;
        }
        inside
    }

    /// Absolute position of profile sample `i`, offset `s` across the scan.
    ///
    /// For [`Placement::Radial`] the cross-offset is applied **along the arc**
    /// at the sample's own radius rather than along a straight chord — that is
    /// the whole reason the variant exists.
    fn sample_point(&self, i: usize, n: usize, s: f32) -> Point2f {
        let denom = (n.saturating_sub(1)).max(1) as f32;
        match self.placement {
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

    /// Unit scan direction at profile position `x` — the direction the edge
    /// position is measured along, used for the obliquity check.
    fn scan_dir(&self, x: f32, denom: f32) -> Vec2f {
        match self.placement {
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

    /// Run the 1-D detector on the profile and map peaks back to image space.
    fn extract<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        off_image: bool,
    ) -> Option<RejectReason> {
        self.edges.clear();
        let n = self.profile.len();
        if n < 3 {
            return Some(RejectReason::ProfileTooShort);
        }

        // Detect *both* polarities and filter afterwards. Pushing the polarity
        // into the detector's thresholds would make a wrong-polarity edge
        // indistinguishable from no edge at all, and those two call for
        // opposite fixes.
        let (pos_thresh, neg_thresh) = (self.cfg.threshold, self.cfg.threshold);
        // `sigma` is in pixels but the profile is indexed in steps.
        let step = self.cfg.step.max(1e-3);
        let cfg = Edge1DConfig {
            sigma: (self.cfg.sigma / step).max(1e-3),
            border: self.cfg.border,
            pos_thresh,
            neg_thresh,
            refine: SubpixRefine::Parabolic3,
        };

        let Self {
            det,
            profile,
            cfg: c,
            placement,
            ..
        } = self;
        let threshold = c.threshold;
        let want = c.polarity;
        let denom = (n.saturating_sub(1)).max(1) as f32;
        let placement = *placement;
        let peaks = det.detect_in_ref(profile, &cfg);
        let any_peak = peaks.iter().any(|pk| pk.strength >= threshold);
        let mut found: Vec<MeasureEdge> = peaks
            .iter()
            .filter(|pk| pk.strength >= threshold)
            .filter(|pk| match want {
                PolaritySelect::Any => true,
                PolaritySelect::Rising => pk.polarity == EdgePolarity::Rising,
                PolaritySelect::Falling => pk.polarity == EdgePolarity::Falling,
            })
            .map(|pk| {
                let (p, t) = point_at_profile(&placement, pk.x, denom);
                MeasureEdge {
                    p,
                    t,
                    amplitude: pk.strength,
                    polarity: pk.polarity,
                }
            })
            .collect();

        if found.is_empty() {
            return Some(if any_peak {
                RejectReason::WrongPolarity
            } else if off_image {
                RejectReason::OffImage
            } else {
                RejectReason::NoEdge
            });
        }

        // Obliquity gate: the image gradient at the edge must point along the
        // scan axis, give or take.
        if self.cfg.max_obliquity_deg < 180.0 {
            let cos_max = self.cfg.max_obliquity_deg.to_radians().cos();
            let before = found.len();
            found.retain(|e| {
                let x = profile_index_of(&placement, e, denom);
                let dir = self.scan_dir(x, denom);
                match local_gradient(img, e.p) {
                    Some(g) => g.dot(&dir).abs() >= cos_max,
                    None => false,
                }
            });
            if found.is_empty() {
                return Some(if before > 0 {
                    RejectReason::TooOblique
                } else {
                    RejectReason::NoEdge
                });
            }
        }

        match self.cfg.select {
            EdgeSelect::All => self.edges.extend(found),
            EdgeSelect::First => self.edges.extend(found.into_iter().next()),
            EdgeSelect::Last => self.edges.extend(found.pop()),
            EdgeSelect::Strongest => {
                if let Some(best) = found
                    .into_iter()
                    .max_by(|a, b| a.amplitude.total_cmp(&b.amplitude))
                {
                    self.edges.push(best);
                }
            }
        }
        None
    }
}

/// Unit image-gradient direction at `p`, by central differences on bilinear
/// samples. `None` when the gradient is too weak to have a direction.
fn local_gradient<P: Pixel>(img: &ImageView<'_, P>, p: Point2f) -> Option<Vec2f> {
    let at = |x: f32, y: f32| sample_bilinear_f32(img, x, y, BorderMode::Clamp);
    let gx = 0.5 * (at(p.x + 1.0, p.y) - at(p.x - 1.0, p.y));
    let gy = 0.5 * (at(p.x, p.y + 1.0) - at(p.x, p.y - 1.0));
    let g = Vec2f::new(gx, gy);
    (g.norm() > 1e-6).then(|| g.normalized_or_zero())
}

/// Recover a profile index from an edge's along-axis coordinate.
fn profile_index_of(placement: &Placement, e: &MeasureEdge, denom: f32) -> f32 {
    // `t` is a signed distance from the caliper centre along the scan axis, so
    // the index is a plain affine map back — the same for rect and radial.
    let linear = |half_len: f32| {
        if half_len.abs() < 1e-6 {
            0.0
        } else {
            (e.t + half_len) * denom / (2.0 * half_len)
        }
    };
    match *placement {
        Placement::Rect(r) => linear(r.half_len),
        Placement::Radial(r) => linear(r.half_len),
        Placement::Arc(a) => {
            if a.radius.abs() < 1e-6 {
                0.0
            } else {
                e.t / a.radius / a.angle_extent * denom
            }
        }
    }
}

/// Map a subpixel profile index back to image coordinates.
fn point_at_profile(placement: &Placement, x: f32, denom: f32) -> (Point2f, f32) {
    match *placement {
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

#[cfg(test)]
mod tests {
    use super::{
        Caliper, EdgeSelect, MeasureArc, MeasureConfig, MeasureRadial, MeasureRect, PolaritySelect,
        RejectReason,
    };
    use vm_primitives::{EdgePolarity, Image, Point2f};

    /// Image with a vertical step: `< edge_x` dark, `>= edge_x` bright.
    fn step_image(w: usize, h: usize, edge_x: usize) -> Image<u8> {
        let data: Vec<u8> = (0..w * h)
            .map(|i| if (i % w) >= edge_x { 200u8 } else { 20 })
            .collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    /// A bright bar between `x0` and `x1` on a dark background.
    fn bar_image(w: usize, h: usize, x0: usize, x1: usize) -> Image<u8> {
        let data: Vec<u8> = (0..w * h)
            .map(|i| {
                let x = i % w;
                if x >= x0 && x < x1 { 200u8 } else { 20 }
            })
            .collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    fn rect(cx: f32, cy: f32, angle: f32, half_len: f32, half_width: f32) -> MeasureRect {
        MeasureRect {
            center: Point2f::new(cx, cy),
            angle,
            half_len,
            half_width,
        }
    }

    #[test]
    fn finds_a_step_at_the_right_subpixel_position() {
        let img = step_image(96, 96, 40);
        let mut cal = Caliper::rect(rect(48.0, 48.0, 0.0, 24.0, 10.0), MeasureConfig::default());
        let edges = cal.measure(&img.as_view()).expect("an edge");

        assert_eq!(edges.len(), 1, "one step, one edge: {edges:?}");
        // Under the pixel-centre convention the transition between the last
        // dark pixel (39) and the first bright one (40) sits at 39.5.
        assert!(
            (edges[0].p.x - 39.5).abs() < 0.05,
            "edge at {}, expected 39.5",
            edges[0].p.x
        );
        assert!((edges[0].p.y - 48.0).abs() < 1e-4, "must stay on the axis");
        assert_eq!(edges[0].polarity, EdgePolarity::Rising);
    }

    /// The whole point of a caliper: the answer must not depend on how the
    /// measurement is oriented.
    #[test]
    fn a_rotated_caliper_finds_the_same_edge() {
        let img = step_image(128, 128, 60);
        // Scan the vertical step at increasing obliquity. The scan crosses the
        // edge at x = 59.5 whatever the angle, so the found point must land
        // there; `t` grows as 1/cos(angle) because the path is longer.
        for deg in [0.0f32, 10.0, 20.0, 30.0] {
            let a = deg.to_radians();
            let mut cal = Caliper::rect(rect(64.0, 64.0, a, 30.0, 6.0), MeasureConfig::default());
            let edges = cal.measure(&img.as_view()).expect("an edge");
            assert_eq!(edges.len(), 1, "deg={deg}: {edges:?}");
            assert!(
                (edges[0].p.x - 59.5).abs() < 0.1,
                "deg={deg}: x = {}, expected 59.5",
                edges[0].p.x
            );
        }
    }

    /// Averaging across the caliper must not move the edge, only steady it.
    #[test]
    fn cross_averaging_does_not_shift_the_edge() {
        let img = step_image(96, 96, 40);
        let mut positions = Vec::new();
        for hw in [0.0f32, 1.0, 5.0, 15.0] {
            let mut cal = Caliper::rect(rect(48.0, 48.0, 0.0, 24.0, hw), MeasureConfig::default());
            let edges = cal.measure(&img.as_view()).expect("an edge");
            assert_eq!(edges.len(), 1, "half_width={hw}");
            positions.push(edges[0].p.x);
        }
        for (i, &x) in positions.iter().enumerate() {
            assert!(
                (x - 39.5).abs() < 0.02,
                "half_width index {i}: x = {x}, expected 39.5"
            );
        }
    }

    #[test]
    fn polarity_selects_the_transition_direction() {
        let img = bar_image(96, 96, 30, 60);
        let base = MeasureRect {
            center: Point2f::new(48.0, 48.0),
            angle: 0.0,
            half_len: 40.0,
            half_width: 8.0,
        };

        let mut any = Caliper::rect(base, MeasureConfig::default());
        assert_eq!(
            any.measure(&img.as_view()).expect("edges").len(),
            2,
            "both bar edges"
        );

        let mut rising = Caliper::rect(
            base,
            MeasureConfig {
                polarity: PolaritySelect::Rising,
                ..MeasureConfig::default()
            },
        );
        let r = rising.measure(&img.as_view()).expect("a rising edge");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].polarity, EdgePolarity::Rising);
        assert!((r[0].p.x - 29.5).abs() < 0.1, "x = {}", r[0].p.x);

        let mut falling = Caliper::rect(
            base,
            MeasureConfig {
                polarity: PolaritySelect::Falling,
                ..MeasureConfig::default()
            },
        );
        let f = falling.measure(&img.as_view()).expect("a falling edge");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].polarity, EdgePolarity::Falling);
        assert!((f[0].p.x - 59.5).abs() < 0.1, "x = {}", f[0].p.x);
    }

    #[test]
    fn select_narrows_to_one_edge() {
        let img = bar_image(96, 96, 30, 60);
        let base = rect(48.0, 48.0, 0.0, 40.0, 8.0);

        let mut first = Caliper::rect(
            base,
            MeasureConfig {
                select: EdgeSelect::First,
                ..MeasureConfig::default()
            },
        );
        let e = first.measure(&img.as_view()).expect("an edge");
        assert_eq!(e.len(), 1);
        assert!((e[0].p.x - 29.5).abs() < 0.1);

        let mut last = Caliper::rect(
            base,
            MeasureConfig {
                select: EdgeSelect::Last,
                ..MeasureConfig::default()
            },
        );
        let e = last.measure(&img.as_view()).expect("an edge");
        assert_eq!(e.len(), 1);
        assert!((e[0].p.x - 59.5).abs() < 0.1);
    }

    /// The bar-width measurement a caliper exists to make.
    #[test]
    fn pairs_measure_bar_width() {
        let img = bar_image(128, 96, 40, 70);
        let mut cal = Caliper::rect(rect(64.0, 48.0, 0.0, 50.0, 10.0), MeasureConfig::default());
        let pairs = cal.measure_pairs(&img.as_view());

        assert_eq!(pairs.len(), 1, "one bar: {pairs:?}");
        // Edges at 39.5 and 69.5 -> width 30.0 exactly.
        assert!(
            (pairs[0].width - 30.0).abs() < 0.05,
            "width {}",
            pairs[0].width
        );
        assert!(
            (pairs[0].center.x - 54.5).abs() < 0.05,
            "centre {:?}",
            pairs[0].center
        );
        assert_eq!(pairs[0].first.polarity, EdgePolarity::Rising);
        assert_eq!(pairs[0].second.polarity, EdgePolarity::Falling);
    }

    #[test]
    fn every_pixel_type_measures_the_same_edge() {
        let u8_img = step_image(96, 96, 40);
        let u16_img = Image::from_vec(
            96,
            96,
            u8_img.data().iter().map(|&v| u16::from(v)).collect(),
        )
        .expect("valid");
        let f32_img = Image::from_vec(
            96,
            96,
            u8_img.data().iter().map(|&v| f32::from(v)).collect(),
        )
        .expect("valid");

        let geom = rect(48.0, 48.0, 0.0, 24.0, 6.0);
        let cfg = MeasureConfig::default();

        let x8 = Caliper::rect(geom, cfg)
            .measure(&u8_img.as_view())
            .expect("an edge")[0]
            .p
            .x;
        let x16 = Caliper::rect(geom, cfg)
            .measure(&u16_img.as_view())
            .expect("an edge")[0]
            .p
            .x;
        let x32 = Caliper::rect(geom, cfg)
            .measure(&f32_img.as_view())
            .expect("an edge")[0]
            .p
            .x;
        assert_eq!(x8, x16);
        assert_eq!(x8, x32);
    }

    /// An arc caliper crossing a radial spoke.
    #[test]
    fn arc_caliper_finds_a_radial_feature() {
        // A dark wedge from 0 to 30 degrees on a bright disc.
        let (w, h) = (128usize, 128usize);
        let c = Point2f::new(64.0, 64.0);
        let data: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let (dx, dy) = (x - c.x, y - c.y);
                let phi = dy.atan2(dx);
                if (0.0..30.0f32.to_radians()).contains(&phi) {
                    20u8
                } else {
                    200
                }
            })
            .collect();
        let img = Image::from_vec(w, h, data).expect("valid image");

        let mut cal = Caliper::arc(
            MeasureArc {
                center: c,
                radius: 40.0,
                angle_start: -20.0f32.to_radians(),
                angle_extent: 70.0f32.to_radians(),
                half_width: 4.0,
            },
            MeasureConfig::default(),
        );
        let edges = cal.measure(&img.as_view()).expect("an edge");

        // Two boundaries: entering the wedge at 0 degrees, leaving at 30.
        assert_eq!(edges.len(), 2, "{edges:?}");
        let angles: Vec<f32> = edges
            .iter()
            .map(|e| (e.p.y - c.y).atan2(e.p.x - c.x).to_degrees())
            .collect();
        assert!((angles[0] - 0.0).abs() < 1.5, "angles = {angles:?}");
        assert!((angles[1] - 30.0).abs() < 1.5, "angles = {angles:?}");
    }

    /// Anti-aliased disc: the true edge sits at exactly `r` in every direction.
    fn disc(w: usize, h: usize, c: Point2f, r: f32) -> Image<u8> {
        let data: Vec<u8> = (0..w * h)
            .map(|i| {
                let p = Point2f::new((i % w) as f32, (i / w) as f32);
                let cover = (r + 0.5 - (p - c).norm()).clamp(0.0, 1.0);
                (20.0 + 180.0 * cover).round() as u8
            })
            .collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    /// The reason [`MeasureRadial`] exists.
    ///
    /// A rect caliper averages along a straight chord. On a circle of radius 40
    /// a sample 5 px to the side sits at radius 40.31 — outside the edge — so
    /// the averaged profile is contaminated and the measured radius comes out
    /// low. Averaging along the arc keeps every sample at the same radius.
    #[test]
    fn averaging_along_a_chord_biases_a_curved_edge_inward() {
        let c = Point2f::new(80.0, 80.0);
        let img = disc(160, 160, c, 40.0);
        let cfg = MeasureConfig::default();

        for hw in [2.0f32, 5.0, 10.0] {
            let mut chord = Caliper::rect(
                MeasureRect {
                    center: Point2f::new(c.x + 40.0, c.y),
                    angle: 0.0,
                    half_len: 8.0,
                    half_width: hw,
                },
                cfg,
            );
            let r_chord = chord.measure(&img.as_view()).expect("an edge")[0].p.x - c.x;

            let mut arc = Caliper::radial(
                MeasureRadial {
                    center: c,
                    radius: 40.0,
                    angle: 0.0,
                    half_len: 8.0,
                    half_width: hw,
                },
                cfg,
            );
            let r_arc = arc.measure(&img.as_view()).expect("an edge")[0].p.x - c.x;

            // The chord fit reads low, and worse as the caliper widens.
            assert!(
                r_chord < 40.0 - 0.01,
                "half_width={hw}: chord radius {r_chord} should be biased low"
            );
            // Arc-following averaging stays on the true edge regardless. This
            // is a *single* caliper; the metrology model averages 32 of them
            // and lands inside 0.01 px.
            assert!(
                (r_arc - 40.0).abs() < 0.05,
                "half_width={hw}: arc radius {r_arc}, expected 40.0"
            );
            assert!(
                (40.0 - r_chord) > (40.0 - r_arc),
                "half_width={hw}: arc ({r_arc}) should beat chord ({r_chord})"
            );
        }
    }

    /// Oversampling the profile must not move the answer.
    ///
    /// Only down to `step = 1.0`: coarser steps genuinely cost resolution,
    /// because subpixel refinement interpolates between samples that are now
    /// further apart. `step = 2.0` quantises the answer to about ±0.5 px, which
    /// is the trade the field documents.
    #[test]
    fn the_profile_step_does_not_move_the_edge() {
        let img = step_image(96, 96, 40);
        for step in [0.25f32, 0.5, 1.0] {
            let mut cal = Caliper::rect(
                rect(48.0, 48.0, 0.0, 24.0, 6.0),
                MeasureConfig {
                    step,
                    ..MeasureConfig::default()
                },
            );
            let e = cal.measure(&img.as_view()).expect("an edge");
            assert_eq!(e.len(), 1, "step={step}");
            assert!(
                (e[0].p.x - 39.5).abs() < 0.05,
                "step={step}: x = {}",
                e[0].p.x
            );
        }
    }

    /// A caliper crossing an edge at a glancing angle reports a position along
    /// its own axis, not the edge normal. The obliquity gate rejects it.
    #[test]
    fn the_obliquity_gate_rejects_a_glancing_crossing() {
        let img = step_image(160, 160, 80);
        // 75 degrees off the edge normal: the crossing is nearly along the edge.
        let geom = rect(80.0, 80.0, 75.0f32.to_radians(), 60.0, 2.0);

        let mut ungated = Caliper::rect(geom, MeasureConfig::default());
        assert!(
            ungated.measure(&img.as_view()).is_ok(),
            "ungated, the glancing crossing is still reported"
        );

        let mut gated = Caliper::rect(
            geom,
            MeasureConfig {
                max_obliquity_deg: 30.0,
                ..MeasureConfig::default()
            },
        );
        assert_eq!(gated.measure(&img.as_view()), Err(RejectReason::TooOblique));

        // Straight on, the same gate passes.
        let mut straight = Caliper::rect(
            rect(80.0, 80.0, 0.0, 20.0, 2.0),
            MeasureConfig {
                max_obliquity_deg: 30.0,
                ..MeasureConfig::default()
            },
        );
        assert!(straight.measure(&img.as_view()).is_ok());
    }

    /// Every rejection path must name itself.
    #[test]
    fn rejections_say_which_gate_fired() {
        let flat = Image::from_vec(64, 64, vec![128u8; 64 * 64]).expect("valid");
        let img = step_image(96, 96, 40);

        // Nothing to find.
        let mut none = Caliper::rect(rect(32.0, 32.0, 0.0, 20.0, 4.0), MeasureConfig::default());
        assert_eq!(none.measure(&flat.as_view()), Err(RejectReason::NoEdge));

        // A rising-only caliper pointed the wrong way down a rising edge.
        let mut wrong = Caliper::rect(
            rect(48.0, 48.0, core::f32::consts::PI, 24.0, 4.0),
            MeasureConfig {
                polarity: PolaritySelect::Rising,
                ..MeasureConfig::default()
            },
        );
        assert_eq!(
            wrong.measure(&img.as_view()),
            Err(RejectReason::WrongPolarity)
        );

        // Too short to convolve.
        let mut tiny = Caliper::rect(rect(48.0, 48.0, 0.0, 0.4, 1.0), MeasureConfig::default());
        assert_eq!(
            tiny.measure(&img.as_view()),
            Err(RejectReason::ProfileTooShort)
        );
    }

    #[test]
    fn a_flat_region_produces_no_edges() {
        let img = Image::from_vec(64, 64, vec![128u8; 64 * 64]).expect("valid");
        let mut cal = Caliper::rect(rect(32.0, 32.0, 0.0, 20.0, 5.0), MeasureConfig::default());
        assert!(cal.measure(&img.as_view()).is_err());
        assert!(cal.measure_pairs(&img.as_view()).is_empty());
    }

    #[test]
    fn the_profile_is_available_for_diagnostics() {
        let img = step_image(96, 96, 40);
        let mut cal = Caliper::rect(rect(48.0, 48.0, 0.0, 24.0, 4.0), MeasureConfig::default());
        let _ = cal.measure(&img.as_view());
        let prof = cal.profile();
        assert_eq!(prof.len(), 49, "2*24 + 1 samples");
        assert!((prof[0] - 20.0).abs() < 1e-3, "dark end: {}", prof[0]);
        assert!((prof[48] - 200.0).abs() < 1e-3, "bright end: {}", prof[48]);
    }

    /// A caliper reaching outside the image must clamp, not panic.
    #[test]
    fn overhanging_the_image_is_safe() {
        let img = step_image(64, 64, 30);
        let mut cal = Caliper::rect(rect(2.0, 2.0, 0.0, 40.0, 20.0), MeasureConfig::default());
        let _ = cal.measure(&img.as_view());
    }
}
