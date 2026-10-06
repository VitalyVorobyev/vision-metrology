//! The reusable caliper: a placed geometry, a 1-D profile, subpixel edges.

use std::num::NonZeroUsize;

use vm_primitives::{
    BorderMode, Derivative1D, Edge1DConfig, Edge1DDetector, EdgePolarity, HalfContrastConfig,
    ImageView, LevelCrossing1D, LevelEdge, LevelOutcome, Pixel, Point2f, SubpixRefine, Vec2f,
    Vec2fExt, sample_bilinear_at, sample_bilinear_f32,
};

use super::config::{
    Derivative, EdgeSelect, Locate, MeasureConfig, OffImage, PolaritySelect, RejectReason,
};
use super::placement::{
    MeasureArc, MeasureRadial, MeasureRect, MeasureStrip, Placement, StripGeometry,
};
use super::select::{Candidate, MeasureEdge, MeasurePair, pair_edges, select_edges};

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
    levels_finder: LevelCrossing1D,
    profile: Vec<f32>,
    /// Edges that passed threshold, polarity and the obliquity gate, before `select`.
    cands: Vec<Candidate>,
    /// The candidates `select` kept (refined, for `Locate::HalfContrast`).
    chosen: Vec<Candidate>,
    /// The level crossings behind the last call's edges.
    levels: Vec<LevelEdge>,
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

    /// Place a caliper on a strip between two points.
    ///
    /// The strip can fix its sample counts and reports `t` as the distance from
    /// `start` — see [`MeasureStrip`].
    pub fn strip(strip: MeasureStrip, cfg: MeasureConfig) -> Self {
        Self::new(Placement::Strip(strip), cfg)
    }

    fn new(placement: Placement, cfg: MeasureConfig) -> Self {
        Self {
            placement,
            det: Edge1DDetector::new(cfg.profile.sigma.max(1e-3)),
            levels_finder: LevelCrossing1D::new(),
            cfg,
            profile: Vec::new(),
            cands: Vec::new(),
            chosen: Vec::new(),
            levels: Vec::new(),
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

    /// Move the caliper to a new strip, keeping its buffers.
    pub fn set_strip(&mut self, strip: MeasureStrip) {
        self.placement = Placement::Strip(strip);
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

    /// Distance between profile samples, in pixels, at the current placement and config.
    ///
    /// The scan's extent divided by `samples − 1`, where the extent is `2·half_len` for a
    /// rect or radial placement, `|angle_extent|·radius` of arc length for an arc and the
    /// length for a strip. It differs from
    /// [`ProfileConfig::step`](super::ProfileConfig::step) whenever the extent is not a
    /// whole number of steps; a profile shorter than two samples reports `step`.
    ///
    /// It converts a profile index, such as [`LevelEdge::x`], to pixels along the scan:
    /// index `x` sits `x · spacing` from the first sample.
    pub fn spacing(&self) -> f32 {
        let step = self.cfg.profile.step;
        self.placement
            .spacing(step, self.placement.profile_len(step))
    }

    /// Number of lines averaged across the scan.
    pub(crate) fn across(&self) -> usize {
        self.placement.across_count()
    }

    /// The edges that passed threshold, polarity and the obliquity gate on the last call,
    /// before `select`.
    pub(crate) fn candidates(&self) -> impl ExactSizeIterator<Item = MeasureEdge> + '_ {
        self.cands.iter().map(|c| c.edge)
    }

    /// The smoothed profile and the derivative response of the last profile, recomputed
    /// with the current config. Off the hot path: it allocates both, and leaves every
    /// result of the last call as it was.
    pub(crate) fn smoothed_and_response(&mut self) -> (Vec<f32>, Vec<f32>) {
        let step = self.cfg.profile.step;
        let spacing = self
            .placement
            .sigma_spacing(step, self.placement.profile_len(step));
        let det_cfg = self.detector_config(spacing, SubpixRefine::Parabolic3);
        let smoothed = self.det.smooth_in_ref(&self.profile, &det_cfg).to_vec();
        let _ = self.det.detect_in_ref(&self.profile, &det_cfg);
        (smoothed, self.det.response().to_vec())
    }

    /// The level crossings behind the edges of the last [`measure`](Self::measure) call:
    /// the one crossing of [`Locate::MidpointCrossing`], one per refined edge of
    /// [`Locate::HalfContrast`], none for [`Locate::GradientPeak`].
    ///
    /// Positions are in profile samples, like the [`profile`](Self::profile) index
    /// ([`spacing`](Self::spacing) converts them to pixels); the levels are on the input
    /// pixel scale. A rejected call keeps the crossings it found
    /// before the gate that rejected it.
    pub fn levels(&self) -> &[LevelEdge] {
        &self.levels
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
        if !inside && self.cfg.profile.off_image == OffImage::Reject {
            self.clear_results();
            return Err(RejectReason::OffImage);
        }
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
    /// [`MeasureConfig::locate`] still applies; [`Locate::MidpointCrossing`] finds a
    /// single edge, so it never forms a pair.
    pub fn measure_pairs<P: Pixel>(&mut self, img: &ImageView<'_, P>) -> &[MeasurePair] {
        let _ = self.build_profile(img);

        let saved = (self.cfg.select, self.cfg.polarity);
        self.cfg.select = EdgeSelect::All;
        self.cfg.polarity = PolaritySelect::Any;
        let _ = self.extract(img, false);
        (self.cfg.select, self.cfg.polarity) = saved;

        pair_edges(&self.edges, &mut self.pairs);
        &self.pairs
    }

    /// Fill `profile` with the cross-averaged intensity along the scan axis.
    ///
    /// Returns `false` when any sample fell outside the image, so the caller
    /// can report [`RejectReason::OffImage`] rather than silently measuring
    /// border fill.
    fn build_profile<P: Pixel>(&mut self, img: &ImageView<'_, P>) -> bool {
        let n = self.placement.profile_len(self.cfg.profile.step);
        self.profile.clear();
        self.profile.resize(n, 0.0);
        if n == 0 || img.width() == 0 || img.height() == 0 {
            return false;
        }
        let half_width = self.placement.half_width();
        let across = self.placement.across_count();
        let border = self.cfg.profile.border;
        let profile = &mut self.profile;
        // One loop per placement, so the per-sample geometry is not dispatched in it.
        match self.placement {
            Placement::Rect(r) => fill_profile(profile, img, half_width, across, border, |i, s| {
                r.sample_point(i, n, s)
            }),
            Placement::Arc(a) => fill_profile(profile, img, half_width, across, border, |i, s| {
                a.sample_point(i, n, s)
            }),
            Placement::Radial(r) => {
                fill_profile(profile, img, half_width, across, border, |i, s| {
                    r.sample_point(i, n, s)
                })
            }
            Placement::Strip(st) => {
                fill_strip_profile(profile, img, st.geometry(), half_width, across, border)
            }
        }
    }

    fn clear_results(&mut self) {
        self.cands.clear();
        self.chosen.clear();
        self.levels.clear();
        self.edges.clear();
    }

    /// Locate edges on the profile, gate and select them, and fill `edges`.
    fn extract<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        off_image: bool,
    ) -> Option<RejectReason> {
        self.clear_results();
        let n = self.profile.len();
        if n < 3 {
            return Some(RejectReason::ProfileTooShort);
        }
        // `sigma` and the level methods' distances are in pixels, the profile in samples.
        let spacing = self.placement.sigma_spacing(self.cfg.profile.step, n);
        let found = match self.cfg.locate {
            Locate::GradientPeak { refine } => self
                .gradient_candidates(spacing, refine)
                .and_then(|()| self.gate_and_select(img)),
            Locate::MidpointCrossing {
                endpoint_samples,
                min_contrast,
            } => self
                .midpoint_candidate(spacing, endpoint_samples, min_contrast)
                .and_then(|()| self.gate_and_select(img)),
            Locate::HalfContrast {
                flank_near_px,
                flank_far_px,
                tol_px,
                max_iter,
                min_contrast,
            } => {
                let hc = HalfContrastConfig {
                    flank_near: flank_near_px / spacing,
                    flank_far: flank_far_px / spacing,
                    tol: tol_px / spacing,
                    max_iter,
                    min_contrast,
                };
                self.gradient_candidates(spacing, SubpixRefine::Parabolic3)
                    .and_then(|()| self.gate_and_select(img))
                    .and_then(|()| self.refine_half_contrast(spacing, &hc))
            }
        };
        match found {
            Ok(()) => {
                self.edges.extend(self.chosen.iter().map(|c| c.edge));
                None
            }
            // A profile partly made of border fill that held nothing reports that.
            Err(RejectReason::NoEdge | RejectReason::LowContrast | RejectReason::NoCrossing)
                if off_image =>
            {
                Some(RejectReason::OffImage)
            }
            Err(reason) => Some(reason),
        }
    }

    /// The 1-D detector configuration for a profile sampled every `spacing` pixels.
    fn detector_config(&self, spacing: f32, refine: SubpixRefine) -> Edge1DConfig {
        Edge1DConfig {
            sigma: (self.cfg.profile.sigma / spacing).max(1e-3),
            derivative: derivative_in_samples(self.cfg.profile.derivative, spacing),
            border: self.cfg.profile.border,
            pos_thresh: self.cfg.threshold,
            neg_thresh: self.cfg.threshold,
            refine,
        }
    }

    /// Fill `cands` with the derivative peaks that pass threshold and polarity, mapped to
    /// image space.
    fn gradient_candidates(
        &mut self,
        spacing: f32,
        refine: SubpixRefine,
    ) -> Result<(), RejectReason> {
        // Detect *both* polarities and filter afterwards. Pushing the polarity
        // into the detector's thresholds would make a wrong-polarity edge
        // indistinguishable from no edge at all, and those two call for
        // opposite fixes.
        let threshold = self.cfg.threshold;
        let det_cfg = self.detector_config(spacing, refine);
        let want = self.cfg.polarity;
        let denom = profile_denom(self.profile.len());
        let placement = self.placement;
        let peaks = self.det.detect_in_ref(&self.profile, &det_cfg);
        let mut any_peak = false;
        for pk in peaks.iter().filter(|pk| pk.strength >= threshold) {
            any_peak = true;
            if want.admits(pk.polarity) {
                let (p, t) = placement.point_at(pk.x, denom);
                self.cands.push(Candidate {
                    x: pk.x,
                    edge: MeasureEdge {
                        p,
                        t,
                        amplitude: pk.strength,
                        polarity: pk.polarity,
                    },
                });
            }
        }
        match (self.cands.is_empty(), any_peak) {
            (false, _) => Ok(()),
            (true, true) => Err(RejectReason::WrongPolarity),
            (true, false) => Err(RejectReason::NoEdge),
        }
    }

    /// Put the one midpoint crossing in `cands`, checking contrast, then polarity, then
    /// that a crossing exists.
    fn midpoint_candidate(
        &mut self,
        spacing: f32,
        endpoint_samples: NonZeroUsize,
        min_contrast: f32,
    ) -> Result<(), RejectReason> {
        let det_cfg = self.detector_config(spacing, SubpixRefine::Parabolic3);
        let n = self.profile.len();
        let smooth = self.det.smooth_in_ref(&self.profile, &det_cfg);
        let (before, after) = self.levels_finder.end_levels(smooth, endpoint_samples);
        let contrast = (f64::from(after) - f64::from(before)).abs();
        if contrast < f64::from(min_contrast) {
            return Err(RejectReason::LowContrast);
        }
        let polarity = if after > before {
            EdgePolarity::Rising
        } else {
            EdgePolarity::Falling
        };
        let first_admits = match self.cfg.select {
            EdgeSelect::StrongestInOrder(seq) => seq.first.admits(polarity),
            _ => true,
        };
        if !(self.cfg.polarity.admits(polarity) && first_admits) {
            return Err(RejectReason::WrongPolarity);
        }
        let level = (0.5 * (f64::from(before) + f64::from(after))) as f32;
        let middle = (n - 1) as f64 / 2.0;
        let off_middle = |x: f32| (f64::from(x) - middle).abs();
        let x = self
            .levels_finder
            .crossings(smooth, level, polarity)
            .iter()
            .copied()
            .reduce(|best, x| {
                if off_middle(x) < off_middle(best) {
                    x
                } else {
                    best
                }
            })
            .ok_or(RejectReason::NoCrossing)?;
        let (p, t) = self.placement.point_at(x, profile_denom(n));
        self.cands.push(Candidate {
            x,
            edge: MeasureEdge {
                p,
                t,
                amplitude: contrast as f32,
                polarity,
            },
        });
        self.levels.push(LevelEdge {
            x,
            before,
            after,
            level,
            iterations: 1,
        });
        Ok(())
    }

    /// Obliquity gate on `cands`, then `select` into `chosen`.
    fn gate_and_select<P: Pixel>(&mut self, img: &ImageView<'_, P>) -> Result<(), RejectReason> {
        // The image gradient at the edge must point along the scan axis, give or take.
        if self.cfg.max_obliquity_deg < 180.0 {
            let cos_max = self.cfg.max_obliquity_deg.to_radians().cos();
            let denom = profile_denom(self.profile.len());
            let placement = self.placement;
            self.cands.retain(|c| {
                let dir = placement.scan_dir(c.x, denom);
                match local_gradient(img, c.edge.p) {
                    Some(g) => g.dot(&dir).abs() >= cos_max,
                    None => false,
                }
            });
            if self.cands.is_empty() {
                return Err(RejectReason::TooOblique);
            }
        }
        select_edges(&self.cands, self.cfg.select, &mut self.chosen)
    }

    /// Move each chosen edge to its local half-contrast crossing.
    fn refine_half_contrast(
        &mut self,
        spacing: f32,
        hc: &HalfContrastConfig,
    ) -> Result<(), RejectReason> {
        let det_cfg = self.detector_config(spacing, SubpixRefine::Parabolic3);
        let denom = profile_denom(self.profile.len());
        let placement = self.placement;
        let smooth = self.det.smooth_in_ref(&self.profile, &det_cfg);
        for c in &mut self.chosen {
            let level = match self.levels_finder.half_contrast(smooth, c.x, hc) {
                LevelOutcome::Found(level) => level,
                LevelOutcome::LowContrast { .. } => return Err(RejectReason::LowContrast),
                LevelOutcome::NoCrossing => return Err(RejectReason::NoCrossing),
            };
            let (p, t) = placement.point_at(level.x, denom);
            c.x = level.x;
            c.edge.p = p;
            c.edge.t = t;
            c.edge.amplitude = (f64::from(level.after) - f64::from(level.before)).abs() as f32;
            self.levels.push(level);
        }
        if self.chosen.windows(2).any(|w| w[0].x >= w[1].x) {
            return Err(RejectReason::IncompleteSequence);
        }
        Ok(())
    }
}

/// `(n − 1)` as the divisor that maps a profile index to its fraction of the scan, at
/// least 1.
fn profile_denom(n: usize) -> f32 {
    (n.saturating_sub(1)).max(1) as f32
}

/// The detector's derivative operator for a profile sampled every `spacing` pixels.
fn derivative_in_samples(d: Derivative, spacing: f32) -> Derivative1D {
    match d {
        Derivative::DerivativeOfGaussian => Derivative1D::DerivativeOfGaussian,
        Derivative::SmoothThenCentral { radius_px } => Derivative1D::SmoothThenCentral {
            radius: NonZeroUsize::new((radius_px / spacing).round().max(1.0) as usize)
                .unwrap_or(NonZeroUsize::MIN),
        },
    }
}

/// Fill `profile` with the mean of `across` points `point(i, s)`, `s` spread evenly over
/// `±half_width`, sampled bilinearly. Returns `false` when any point fell outside the image.
fn fill_profile<P: Pixel>(
    profile: &mut [f32],
    img: &ImageView<'_, P>,
    half_width: f32,
    across: usize,
    border: BorderMode<f32>,
    point: impl Fn(usize, f32) -> Point2f,
) -> bool {
    let across_denom = (across.saturating_sub(1)).max(1) as f32;
    let (w, h) = (img.width() as f32, img.height() as f32);
    let mut inside = true;
    for (i, out) in profile.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for j in 0..across {
            let s = if across == 1 {
                0.0
            } else {
                -half_width + 2.0 * half_width * j as f32 / across_denom
            };
            let q = point(i, s);
            if q.x < 0.0 || q.y < 0.0 || q.x > w - 1.0 || q.y > h - 1.0 {
                inside = false;
            }
            acc += sample_bilinear_at(img, q, border);
        }
        *out = acc / across as f32;
    }
    inside
}

/// [`fill_profile`] for a strip: points and bilinear weights in `f64`, the mean over the
/// lines across accumulated in `f64` and stored as `f32`.
fn fill_strip_profile<P: Pixel>(
    profile: &mut [f32],
    img: &ImageView<'_, P>,
    g: StripGeometry,
    half_width: f32,
    across: usize,
    border: BorderMode<f32>,
) -> bool {
    let n = profile.len();
    let (wf, hf) = ((img.width() - 1) as f64, (img.height() - 1) as f64);
    let mut inside = true;
    for (i, out) in profile.iter_mut().enumerate() {
        let t = g.along(i, n);
        let mut acc = 0.0f64;
        for j in 0..across {
            let o = StripGeometry::across(f64::from(half_width), j, across);
            let (x, y) = g.point(t, o);
            acc += if x >= 0.0 && y >= 0.0 && x <= wf && y <= hf {
                bilinear_inside(img, x, y)
            } else {
                inside = false;
                f64::from(sample_bilinear_f32(img, x as f32, y as f32, border))
            };
        }
        *out = (acc / across as f64) as f32;
    }
    inside
}

/// Bilinear interpolation at a point inside `[0, w − 1] × [0, h − 1]`, in `f64`; the
/// right and bottom neighbours are clamped to the last column and row.
fn bilinear_inside<P: Pixel>(img: &ImageView<'_, P>, x: f64, y: f64) -> f64 {
    let (w, h) = (img.width(), img.height());
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (dx, dy) = (x - x0 as f64, y - y0 as f64);
    let at = |xx: usize, yy: usize| f64::from(img.row(yy)[xx].to_f32());
    (1.0 - dx) * (1.0 - dy) * at(x0, y0)
        + dx * (1.0 - dy) * at(x1, y0)
        + (1.0 - dx) * dy * at(x0, y1)
        + dx * dy * at(x1, y1)
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

#[cfg(test)]
mod tests {
    use super::Caliper;
    use crate::measure::{
        EdgeSelect, MeasureArc, MeasureConfig, MeasureRadial, MeasureRect, PolaritySelect,
        ProfileConfig, RejectReason,
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

    /// The textbook operator and the log-parabola refinement find the same ideal step.
    #[test]
    fn every_derivative_and_refinement_finds_the_step() {
        use crate::measure::{Derivative, Locate};
        use vm_primitives::SubpixRefine;
        let img = step_image(96, 96, 40);
        for derivative in [
            Derivative::DerivativeOfGaussian,
            Derivative::SmoothThenCentral { radius_px: 3.0 },
        ] {
            for refine in [SubpixRefine::Parabolic3, SubpixRefine::Gaussian3] {
                let cfg = MeasureConfig {
                    locate: Locate::GradientPeak { refine },
                    profile: ProfileConfig {
                        derivative,
                        step: 0.5,
                        ..ProfileConfig::default()
                    },
                    ..MeasureConfig::default()
                };
                let mut cal = Caliper::rect(rect(48.0, 48.0, 0.0, 24.0, 4.0), cfg);
                let e = cal.measure(&img.as_view()).expect("an edge");
                assert_eq!(e.len(), 1, "{derivative:?} {refine:?}");
                assert!(
                    (e[0].p.x - 39.5).abs() < 0.02,
                    "{derivative:?} {refine:?}: x = {}",
                    e[0].p.x
                );
            }
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
                    profile: ProfileConfig {
                        step,
                        ..ProfileConfig::default()
                    },
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

    /// An ordered sequence reads a bar as its rising edge, then its falling edge.
    #[test]
    fn strongest_in_order_reads_a_bar_in_scan_order() {
        use crate::measure::EdgeSequence;
        let img = bar_image(96, 96, 30, 60);
        let (rising, falling) = (PolaritySelect::Rising, PolaritySelect::Falling);
        let measure = |first, second, polarity| {
            let cfg = MeasureConfig {
                polarity,
                select: EdgeSelect::StrongestInOrder(EdgeSequence { first, second }),
                ..MeasureConfig::default()
            };
            Caliper::rect(rect(48.0, 48.0, 0.0, 40.0, 8.0), cfg)
                .measure(&img.as_view())
                .map(|e| e.iter().map(|e| e.p.x).collect::<Vec<_>>())
        };

        let bar = measure(rising, Some(falling), PolaritySelect::Any).expect("both edges");
        assert_eq!(bar.len(), 2);
        assert!(
            (bar[0] - 29.5).abs() < 0.1 && (bar[1] - 59.5).abs() < 0.1,
            "{bar:?}"
        );

        // Falling, then rising: nothing rises after x = 59.5.
        assert_eq!(
            measure(falling, Some(rising), PolaritySelect::Any),
            Err(RejectReason::IncompleteSequence)
        );
        // `polarity` filters first: only the falling edge is left for a rising entry.
        assert_eq!(
            measure(rising, Some(falling), PolaritySelect::Falling),
            Err(RejectReason::WrongPolarity)
        );
    }

    /// "After" is along the profile, not along `t`: an arc swept clockwise has `t`
    /// decreasing, and the sequence still follows the scan.
    #[test]
    fn strongest_in_order_follows_the_profile_on_a_clockwise_arc() {
        use crate::measure::EdgeSequence;
        let (w, h) = (128usize, 128usize);
        let c = Point2f::new(64.0, 64.0);
        // A dark wedge from 0 to 30 degrees on a bright disc.
        let data: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - c.x, (i / w) as f32 - c.y);
                if (0.0..30.0f32.to_radians()).contains(&y.atan2(x)) {
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
                angle_start: 50.0f32.to_radians(),
                angle_extent: -70.0f32.to_radians(),
                half_width: 4.0,
            },
            MeasureConfig {
                select: EdgeSelect::StrongestInOrder(EdgeSequence {
                    first: PolaritySelect::Falling,
                    second: Some(PolaritySelect::Rising),
                }),
                ..MeasureConfig::default()
            },
        );
        let edges = cal.measure(&img.as_view()).expect("both wedge edges");
        let angles: Vec<f32> = edges
            .iter()
            .map(|e| (e.p.y - c.y).atan2(e.p.x - c.x).to_degrees())
            .collect();
        // Sweeping from 50 down to -20 degrees, the scan enters the wedge at 30
        // degrees and leaves it at 0, where `t` is smaller.
        assert!((angles[0] - 30.0).abs() < 1.5, "{angles:?}");
        assert!((angles[1] - 0.0).abs() < 1.5, "{angles:?}");
        assert!(edges[0].t > edges[1].t, "t runs backwards: {edges:?}");
    }

    mod strip {
        use std::num::NonZeroUsize;

        use super::super::Caliper;
        use crate::measure::{
            Derivative, MeasureConfig, MeasureStrip, OffImage, ProfileConfig, RejectReason,
        };
        use vm_primitives::{EdgePolarity, Image, Point2f};

        fn nz(n: usize) -> Option<NonZeroUsize> {
            NonZeroUsize::new(n)
        }

        fn strip(
            start: (f32, f32),
            end: (f32, f32),
            half_width: f32,
            n: usize,
            a: usize,
        ) -> MeasureStrip {
            MeasureStrip {
                start: Point2f::new(start.0, start.1),
                end: Point2f::new(end.0, end.1),
                half_width,
                samples: nz(n),
                across: nz(a),
            }
        }

        /// The textbook settings: σ = 1 sample, a radius-3 Gaussian, central
        /// differences, strict bounds, a response floor of 0.01 on a [0, 1] image.
        fn textbook(spacing: f32) -> MeasureConfig {
            MeasureConfig {
                threshold: 0.01,
                profile: ProfileConfig {
                    sigma: spacing,
                    derivative: Derivative::SmoothThenCentral {
                        radius_px: 3.0 * spacing,
                    },
                    off_image: OffImage::Reject,
                    ..ProfileConfig::default()
                },
                ..MeasureConfig::default()
            }
        }

        /// A bright bar on columns 16..48 of a 9 × 64 image in [0, 1].
        fn bar() -> Image<f32> {
            let data = (0..9 * 64)
                .map(|i| {
                    if (16..48).contains(&(i % 64)) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            Image::from_vec(64, 9, data).expect("valid image")
        }

        /// On the plane `2x + 3y` every sample of a straight strip is exact, so the
        /// profile of a diagonal strip 3 wide is a straight ramp from 25 to 100.
        #[test]
        fn the_strip_profile_is_bilinear_and_averaged_across() {
            let data = (0..30 * 30)
                .map(|i| 2.0 * (i % 30) as f32 + 3.0 * (i / 30) as f32)
                .collect();
            let img: Image<f32> = Image::from_vec(30, 30, data).expect("valid image");
            let mut cal =
                Caliper::strip(strip((5.0, 5.0), (20.0, 20.0), 1.0, 16, 3), textbook(1.0));
            let _ = cal.measure(&img.as_view());
            let prof = cal.profile();
            assert_eq!(prof.len(), 16);
            for (i, &v) in prof.iter().enumerate() {
                let want = 25.0 + 5.0 * i as f32;
                assert!(
                    (v - want).abs() < 1e-4,
                    "profile[{i}] = {v}, expected {want}"
                );
            }
        }

        /// Positions are distances from `start` in image pixels, whatever the
        /// direction or the number of samples.
        #[test]
        fn edges_are_distances_from_the_start() {
            let img = bar();
            for (start, end, n) in [
                ((0.0, 4.0), (63.0, 4.0), 64),
                ((63.0, 4.0), (0.0, 4.0), 64),
                ((0.0, 4.0), (63.0, 4.0), 127),
            ] {
                let spacing = 63.0 / (n - 1) as f32;
                let mut cal = Caliper::strip(strip(start, end, 0.0, n, 1), textbook(spacing));
                let edges = cal.measure(&img.as_view()).expect("two edges");
                let ts: Vec<f32> = edges.iter().map(|e| e.t).collect();
                assert_eq!(edges.len(), 2, "{start:?} -> {end:?}, n={n}: {ts:?}");
                assert!((ts[0] - 15.5).abs() < 1e-4, "{start:?}, n={n}: {ts:?}");
                assert!((ts[1] - 47.5).abs() < 1e-4, "{start:?}, n={n}: {ts:?}");
                assert_eq!(edges[0].polarity, EdgePolarity::Rising);
                assert_eq!(edges[1].polarity, EdgePolarity::Falling);
            }
        }

        /// A strip ending exactly on the last pixel centre is inside; one row lower
        /// than the image is not, and `Reject` says so before looking for edges.
        #[test]
        fn strict_bounds_accept_the_last_pixel_centre_and_reject_beyond() {
            let img = bar();
            let mut inside =
                Caliper::strip(strip((0.0, 8.0), (63.0, 8.0), 0.0, 64, 1), textbook(1.0));
            assert!(inside.measure(&img.as_view()).is_ok());

            let short: Image<f32> = Image::from_vec(64, 3, vec![0.0; 64 * 3]).expect("valid");
            let mut out = Caliper::strip(strip((0.0, 4.0), (63.0, 4.0), 0.0, 64, 1), textbook(1.0));
            assert_eq!(out.measure(&short.as_view()), Err(RejectReason::OffImage));

            // A wide strip whose outer line leaves the image is rejected too.
            let mut wide =
                Caliper::strip(strip((0.0, 7.0), (63.0, 7.0), 2.0, 64, 5), textbook(1.0));
            assert_eq!(wide.measure(&img.as_view()), Err(RejectReason::OffImage));
        }

        /// Without explicit counts a strip samples like a rect: one sample per step
        /// along, about one line per pixel across.
        #[test]
        fn unset_counts_follow_the_step() {
            let img = bar();
            let mut cal = Caliper::strip(
                MeasureStrip {
                    samples: None,
                    across: None,
                    ..strip((0.0, 4.0), (63.0, 4.0), 1.5, 2, 2)
                },
                MeasureConfig::default(),
            );
            let _ = cal.measure(&img.as_view());
            assert_eq!(cal.profile().len(), 64, "floor(63 / 1) + 1");
        }

        #[test]
        fn a_zero_length_strip_is_too_short() {
            let img = bar();
            let mut cal = Caliper::strip(strip((5.0, 4.0), (5.0, 4.0), 0.0, 16, 1), textbook(1.0));
            assert_eq!(
                cal.measure(&img.as_view()),
                Err(RejectReason::ProfileTooShort)
            );
        }
    }

    /// The level methods: `Locate::MidpointCrossing` and `Locate::HalfContrast`.
    mod levels {
        use std::num::NonZeroUsize;

        use super::super::Caliper;
        use crate::measure::{
            Derivative, EdgeSelect, EdgeSequence, Locate, MeasureConfig, MeasureStrip, OffImage,
            PolaritySelect, ProfileConfig, RejectReason,
        };
        use vm_primitives::{EdgePolarity, Image, Point2f};

        fn nz(n: usize) -> NonZeroUsize {
            NonZeroUsize::new(n).expect("nonzero")
        }

        /// A 3-row f32 image whose every row is `row`.
        fn rows(row: &[f32]) -> Image<f32> {
            let data = (0..3).flat_map(|_| row.iter().copied()).collect();
            Image::from_vec(row.len(), 3, data).expect("valid image")
        }

        /// A one-line strip along row 1 sampling every column once.
        fn along(w: usize) -> MeasureStrip {
            MeasureStrip {
                start: Point2f::new(0.0, 1.0),
                end: Point2f::new((w - 1) as f32, 1.0),
                half_width: 0.0,
                samples: NonZeroUsize::new(w),
                across: NonZeroUsize::new(1),
            }
        }

        fn midpoint(min_contrast: f32) -> Locate {
            Locate::MidpointCrossing {
                endpoint_samples: nz(3),
                min_contrast,
            }
        }

        fn half_contrast(min_contrast: f32) -> Locate {
            Locate::HalfContrast {
                flank_near_px: 3.0,
                flank_far_px: 8.0,
                tol_px: 0.01,
                max_iter: nz(5),
                min_contrast,
            }
        }

        /// CaliperBench's textbook settings (σ of one sample, a radius-3 Gaussian, strict
        /// bounds) with the given locate method.
        fn textbook(locate: Locate) -> MeasureConfig {
            MeasureConfig {
                threshold: 0.01,
                locate,
                profile: ProfileConfig {
                    derivative: Derivative::SmoothThenCentral { radius_px: 3.0 },
                    off_image: OffImage::Reject,
                    ..ProfileConfig::default()
                },
                ..MeasureConfig::default()
            }
        }

        /// A smoothing σ so small that the Gaussian is `[0, 1, 0]` in `f32`: the level
        /// methods then read the profile exactly as sampled.
        fn unsmoothed(locate: Locate) -> MeasureConfig {
            MeasureConfig {
                locate,
                profile: ProfileConfig {
                    sigma: 1e-3,
                    off_image: OffImage::Reject,
                    ..ProfileConfig::default()
                },
                ..MeasureConfig::default()
            }
        }

        fn measure(
            row: &[f32],
            cfg: MeasureConfig,
        ) -> Result<Vec<(f32, EdgePolarity)>, RejectReason> {
            Caliper::strip(along(row.len()), cfg)
                .measure(&rows(row).as_view())
                .map(|e| e.iter().map(|e| (e.t, e.polarity)).collect())
        }

        /// A unit step at column 16 (CaliperBench's `test_textbook_method_comparison`):
        /// the smoothed step is antisymmetric about 15.5, so the mid level crosses there.
        #[test]
        fn midpoint_finds_the_middle_of_a_step() {
            let step: Vec<f32> = (0..64).map(|x| if x >= 16 { 1.0 } else { 0.0 }).collect();
            let mut cal = Caliper::strip(along(64), textbook(midpoint(0.05)));
            let edges = cal
                .measure(&rows(&step).as_view())
                .expect("one edge")
                .to_vec();
            assert_eq!(edges.len(), 1);
            assert!((edges[0].t - 15.5).abs() < 1e-5, "t = {}", edges[0].t);
            assert_eq!(edges[0].polarity, EdgePolarity::Rising);
            assert!((edges[0].amplitude - 1.0).abs() < 1e-6);
            let levels = cal.levels();
            assert_eq!(levels.len(), 1);
            // The normalised kernel sums to 1 within an `f32` ulp.
            let (l, eps) = (levels[0], 1e-6);
            assert!(l.before.abs() < eps && (l.after - 1.0).abs() < eps, "{l:?}");
            assert!((l.level - 0.5).abs() < eps, "{l:?}");
            assert_eq!(l.x, edges[0].t, "one sample per pixel from x = 0");

            // Two entries in the sequence: the midpoint gives one edge, so the second is
            // missing.
            let pair = MeasureConfig {
                select: EdgeSelect::StrongestInOrder(EdgeSequence {
                    first: PolaritySelect::Rising,
                    second: Some(PolaritySelect::Falling),
                }),
                ..textbook(midpoint(0.05))
            };
            assert_eq!(measure(&step, pair), Err(RejectReason::IncompleteSequence));
        }

        /// By hand: the end levels are the medians of (0, 0.3, 0.1) and (1, 0.7, 0.9),
        /// so the level is 0.5, crossed rising only between 0.2 and 0.6, at
        /// `4 + 0.3 / 0.4 = 4.75`.
        #[test]
        fn midpoint_on_a_hand_computed_profile() {
            let row = [0.0, 0.3, 0.1, 0.1, 0.2, 0.6, 0.9, 1.0, 0.8, 1.0, 0.7, 0.9];
            let mut cal = Caliper::strip(along(row.len()), unsmoothed(midpoint(0.05)));
            let edges = cal
                .measure(&rows(&row).as_view())
                .expect("one edge")
                .to_vec();
            assert_eq!(cal.profile(), &row[..], "the profile is the row itself");
            assert!((edges[0].t - 4.75).abs() < 1e-6, "t = {}", edges[0].t);
            assert!((edges[0].amplitude - 0.8).abs() < 1e-6);
            let l = cal.levels()[0];
            assert_eq!((l.before, l.after), (0.1, 0.9));
            assert!((l.level - 0.5).abs() < 1e-7);
        }

        /// Two rising crossings 11 samples either side of the middle (31.5): the earlier
        /// one wins the tie. Each crosses `0.5` halfway between 0.25 and 0.75.
        #[test]
        fn midpoint_ties_go_to_the_earlier_crossing() {
            let mut row = vec![0.0f32; 64];
            row[20] = 0.25;
            row[21] = 0.75;
            row[22..=30].fill(1.0);
            row[42] = 0.25;
            row[43] = 0.75;
            row[44..].fill(1.0);
            let edges = measure(&row, unsmoothed(midpoint(0.05))).expect("one edge");
            assert_eq!(edges, vec![(20.5, EdgePolarity::Rising)]);
        }

        /// CaliperBench's order: contrast, then polarity, then the crossing.
        #[test]
        fn midpoint_checks_contrast_then_polarity_then_crossing() {
            let falling: Vec<f32> = (0..32).map(|x| if x < 16 { 0.8 } else { 0.2 }).collect();
            let rising_only = |locate| MeasureConfig {
                polarity: PolaritySelect::Rising,
                ..unsmoothed(locate)
            };
            // Too little contrast for 0.7: that is reported before the wrong polarity.
            assert_eq!(
                measure(&falling, rising_only(midpoint(0.7))),
                Err(RejectReason::LowContrast)
            );
            assert_eq!(
                measure(&falling, rising_only(midpoint(0.05))),
                Err(RejectReason::WrongPolarity)
            );
            // The first entry of a sequence filters the polarity too.
            let sequence = MeasureConfig {
                select: EdgeSelect::StrongestInOrder(EdgeSequence {
                    first: PolaritySelect::Rising,
                    second: None,
                }),
                ..unsmoothed(midpoint(0.05))
            };
            assert_eq!(
                measure(&falling, sequence),
                Err(RejectReason::WrongPolarity)
            );
            assert_eq!(
                measure(&falling, unsmoothed(midpoint(0.05))),
                Ok(vec![(15.5, EdgePolarity::Falling)])
            );
            // Equal end levels pass a zero contrast floor; a flat profile is never crossed.
            assert_eq!(
                measure(&[0.5; 32], unsmoothed(midpoint(0.0))),
                Err(RejectReason::NoCrossing)
            );
            // A bar returns to its start level: no contrast between the ends.
            let mut bar = vec![0.0f32; 64];
            bar[16..48].fill(1.0);
            assert_eq!(
                measure(&bar, textbook(midpoint(0.05))),
                Err(RejectReason::LowContrast)
            );
        }

        /// A pixel-integrated step from `dark` to `bright` at `edge`, blurred by a Gaussian
        /// of σ = 1.2 px, plus `shade · x`.
        fn shaded_step(w: usize, edge: f64, dark: f64, bright: f64, shade: f64) -> Vec<f32> {
            let cover = |x: f64| (x + 0.5 - (x - 0.5).max(edge)).clamp(0.0, 1.0);
            (0..w)
                .map(|x| {
                    let x = x as f64;
                    let (mut acc, mut sum) = (0.0, 0.0);
                    for k in -6..=6 {
                        let wk = (-0.5 * (f64::from(k) / 1.2).powi(2)).exp();
                        acc += wk * cover(x + f64::from(k));
                        sum += wk;
                    }
                    (dark + (bright - dark) * acc / sum + shade * x) as f32
                })
                .collect()
        }

        /// On a symmetric bar the half-contrast edges are the gradient ones, refined onto
        /// the mid level of each side.
        #[test]
        fn half_contrast_refines_each_selected_edge() {
            let mut bar = vec![0.0f32; 64];
            bar[16..48].fill(1.0);
            let mut cal = Caliper::strip(along(64), textbook(half_contrast(0.0)));
            let edges = cal
                .measure(&rows(&bar).as_view())
                .expect("two edges")
                .to_vec();
            assert_eq!(edges.len(), 2);
            assert!((edges[0].t - 15.5).abs() < 1e-4, "{edges:?}");
            assert!((edges[1].t - 47.5).abs() < 1e-4, "{edges:?}");
            assert_eq!(edges[0].polarity, EdgePolarity::Rising);
            assert_eq!(edges[1].polarity, EdgePolarity::Falling);
            let levels = cal.levels();
            assert_eq!(levels.len(), 2);
            let eps = 1e-6;
            assert!(levels[0].before.abs() < eps && (levels[0].after - 1.0).abs() < eps);
            assert!((levels[1].before - 1.0).abs() < eps && levels[1].after.abs() < eps);
            assert!((edges[0].amplitude - 1.0).abs() < eps);

            // A gradient peak leaves no level crossings behind.
            cal.set_config(textbook(Locate::default()));
            let _ = cal.measure(&rows(&bar).as_view());
            assert!(cal.levels().is_empty());
        }

        /// Shading along the scan moves the end levels but not the local ones: the
        /// half-contrast edge stays on the step, the midpoint does not.
        #[test]
        fn half_contrast_reads_local_levels_where_the_midpoint_reads_the_ends() {
            let edge = 37.3;
            let row = shaded_step(100, edge, 0.1, 0.8, 0.004);
            let found = |locate| measure(&row, textbook(locate)).expect("an edge")[0].0;
            let local = found(half_contrast(0.05));
            let ends = found(midpoint(0.05));
            assert!(
                (f64::from(local) - edge).abs() < 0.03,
                "half contrast at {local}"
            );
            assert!((f64::from(ends) - edge).abs() > 0.1, "midpoint at {ends}");
        }

        /// A staircase 0 → ½ → 1 with steps 4 px apart has two gradient peaks but one
        /// half-contrast edge: both seeds converge onto the crossing between the steps,
        /// so the refined edges are no longer in increasing order.
        #[test]
        fn half_contrast_merging_two_seeds_is_an_incomplete_sequence() {
            let row: Vec<f32> = (0..64)
                .map(|x| match x {
                    ..30 => 0.0,
                    30..34 => 0.5,
                    _ => 1.0,
                })
                .collect();
            let gradient = measure(&row, textbook(Locate::default())).expect("two peaks");
            assert_eq!(gradient.len(), 2, "{gradient:?}");
            assert_eq!(
                measure(&row, textbook(half_contrast(0.0))),
                Err(RejectReason::IncompleteSequence)
            );
            // One seed alone refines onto the middle of the staircase.
            let strongest = MeasureConfig {
                select: EdgeSelect::First,
                ..textbook(half_contrast(0.0))
            };
            let one = measure(&row, strongest).expect("one edge");
            assert!((one[0].0 - 31.5).abs() < 0.05, "{one:?}");
        }

        #[test]
        fn half_contrast_names_low_contrast_and_a_missing_crossing() {
            let row = shaded_step(64, 30.0, 0.2, 0.5, 0.0);
            assert_eq!(
                measure(&row, textbook(half_contrast(0.5))),
                Err(RejectReason::LowContrast)
            );
            // The step 2 px from the start: the flank 3–8 px before it is off the profile.
            let near_start = shaded_step(64, 2.0, 0.2, 0.8, 0.0);
            assert_eq!(
                measure(&near_start, textbook(half_contrast(0.0))),
                Err(RejectReason::NoCrossing)
            );
        }
    }

    /// `spacing()` on placements whose extent is not a whole number of steps: the real
    /// distance between samples, the rate at which an edge's `t` advances per profile index.
    mod spacing {
        use std::num::NonZeroUsize;

        use super::super::Caliper;
        use crate::measure::{Locate, MeasureArc, MeasureConfig, MeasureRect};
        use vm_primitives::{Image, Point2f};

        fn midpoint() -> MeasureConfig {
            MeasureConfig {
                locate: Locate::MidpointCrossing {
                    endpoint_samples: NonZeroUsize::new(3).expect("nonzero"),
                    min_contrast: 10.0,
                },
                ..MeasureConfig::default()
            }
        }

        /// A rect of `half_len = 10.3` at `step = 1` takes 21 samples over 20.6 px: they
        /// are 1.03 px apart, not 1.
        #[test]
        fn a_rect_reports_the_real_sample_distance() {
            let img = super::step_image(96, 96, 40);
            let rect = MeasureRect {
                center: Point2f::new(41.3, 48.0),
                angle: 0.0,
                half_len: 10.3,
                half_width: 4.0,
            };
            let mut cal = Caliper::rect(rect, midpoint());
            let edges = cal.measure(&img.as_view()).expect("an edge").to_vec();
            let n = cal.profile().len();
            assert_eq!(n, 21);
            let spacing = cal.spacing();
            assert!((spacing - 20.6 / 20.0).abs() < 1e-6, "spacing {spacing}");
            let gap = (rect.sample_point(1, n, 0.0) - rect.sample_point(0, n, 0.0)).norm();
            assert!(
                (spacing - gap).abs() < 1e-5,
                "spacing {spacing}, samples {gap} apart"
            );

            // The crossing at x = 39.5 is `level.x · spacing` from the first sample.
            assert!(
                (edges[0].p.x - 39.5).abs() < 0.05,
                "edge at {}",
                edges[0].p.x
            );
            let level = cal.levels()[0];
            let t = level.x * spacing - rect.half_len;
            assert!(
                (t - edges[0].t).abs() < 1e-4,
                "t {t}, edge t {}",
                edges[0].t
            );
            assert!(
                (rect.center.x + t - edges[0].p.x).abs() < 1e-4,
                "x {}",
                rect.center.x + t
            );
        }

        /// An arc of radius 30 over 0.75 rad is 22.5 px of arc length in 23 samples.
        #[test]
        fn an_arc_reports_the_real_arc_length_between_samples() {
            // Dark above row 64, bright from row 64 on: the arc crosses it at y = 63.5.
            let data: Vec<u8> = (0..128 * 128)
                .map(|i| if i / 128 >= 64 { 200 } else { 20 })
                .collect();
            let img = Image::from_vec(128, 128, data).expect("valid image");
            let arc = MeasureArc {
                center: Point2f::new(64.0, 64.0),
                radius: 30.0,
                angle_start: -0.375,
                angle_extent: 0.75,
                half_width: 2.0,
            };
            let mut cal = Caliper::arc(arc, midpoint());
            let edges = cal.measure(&img.as_view()).expect("an edge").to_vec();
            let n = cal.profile().len();
            assert_eq!(n, 23);
            let spacing = cal.spacing();
            assert!((spacing - 22.5 / 22.0).abs() < 1e-6, "spacing {spacing}");

            assert!(
                (edges[0].p.y - 63.5).abs() < 0.05,
                "edge at {}",
                edges[0].p.y
            );
            let level = cal.levels()[0];
            let t = level.x * spacing;
            assert!(
                (t - edges[0].t).abs() < 1e-4,
                "t {t}, edge t {}",
                edges[0].t
            );
            // Arc length from the start to the crossing, from the crossing's own angle.
            let phi = ((edges[0].p.y - 64.0) / 30.0).asin();
            let expected = (phi - arc.angle_start) * arc.radius;
            assert!((t - expected).abs() < 1e-3, "t {t}, expected {expected}");
        }
    }
}
