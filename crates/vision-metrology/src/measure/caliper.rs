//! The reusable caliper: a placed geometry, a 1-D profile, subpixel edges.

use std::num::NonZeroUsize;

use vm_primitives::{
    BorderMode, Derivative1D, Edge1DConfig, Edge1DDetector, EdgePolarity, ImageView, Pixel,
    Point2f, Vec2f, Vec2fExt, sample_bilinear_at, sample_bilinear_f32,
};

use super::config::{Derivative, EdgeSelect, Locate, MeasureConfig, PolaritySelect, RejectReason};
use super::placement::{MeasureArc, MeasureRadial, MeasureRect, Placement};
use super::select::{MeasureEdge, MeasurePair, pair_edges, select_edges};

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
    /// Edges that passed threshold and polarity, before `select`.
    cands: Vec<MeasureEdge>,
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
            det: Edge1DDetector::new(cfg.profile.sigma.max(1e-3)),
            cfg,
            profile: Vec::new(),
            cands: Vec::new(),
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
        let across = (2.0 * half_width).max(0.0) as usize + 1;
        let across_denom = (across.saturating_sub(1)).max(1) as f32;
        let (w, h) = (img.width() as f32, img.height() as f32);
        let border = self.cfg.profile.border;
        let mut inside = true;

        for i in 0..n {
            let mut acc = 0.0f32;
            for j in 0..across {
                let s = if across == 1 {
                    0.0
                } else {
                    -half_width + 2.0 * half_width * j as f32 / across_denom
                };
                let q = self.placement.sample_point(i, n, s);
                if q.x < 0.0 || q.y < 0.0 || q.x > w - 1.0 || q.y > h - 1.0 {
                    inside = false;
                }
                acc += sample_bilinear_at(img, q, border);
            }
            self.profile[i] = acc / across as f32;
        }
        inside
    }

    /// Run the 1-D detector on the profile and map peaks back to image space.
    fn extract<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        off_image: bool,
    ) -> Option<RejectReason> {
        self.edges.clear();
        self.cands.clear();
        let n = self.profile.len();
        if n < 3 {
            return Some(RejectReason::ProfileTooShort);
        }

        // Detect *both* polarities and filter afterwards. Pushing the polarity
        // into the detector's thresholds would make a wrong-polarity edge
        // indistinguishable from no edge at all, and those two call for
        // opposite fixes.
        let threshold = self.cfg.threshold;
        // `sigma` is in pixels but the profile is indexed in steps.
        let step = self.cfg.profile.step.max(1e-3);
        let Locate::GradientPeak { refine } = self.cfg.locate;
        let det_cfg = Edge1DConfig {
            sigma: (self.cfg.profile.sigma / step).max(1e-3),
            derivative: derivative_in_samples(self.cfg.profile.derivative, step),
            border: self.cfg.profile.border,
            pos_thresh: threshold,
            neg_thresh: threshold,
            refine,
        };

        let want = self.cfg.polarity;
        let denom = (n.saturating_sub(1)).max(1) as f32;
        let placement = self.placement;
        let peaks = self.det.detect_in_ref(&self.profile, &det_cfg);
        let mut any_peak = false;
        for pk in peaks.iter().filter(|pk| pk.strength >= threshold) {
            any_peak = true;
            let keep = match want {
                PolaritySelect::Any => true,
                PolaritySelect::Rising => pk.polarity == EdgePolarity::Rising,
                PolaritySelect::Falling => pk.polarity == EdgePolarity::Falling,
            };
            if keep {
                let (p, t) = placement.point_at(pk.x, denom);
                self.cands.push(MeasureEdge {
                    p,
                    t,
                    amplitude: pk.strength,
                    polarity: pk.polarity,
                });
            }
        }

        if self.cands.is_empty() {
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
            self.cands.retain(|e| {
                let x = placement.index_of(e.t, denom);
                let dir = placement.scan_dir(x, denom);
                match local_gradient(img, e.p) {
                    Some(g) => g.dot(&dir).abs() >= cos_max,
                    None => false,
                }
            });
            if self.cands.is_empty() {
                return Some(RejectReason::TooOblique);
            }
        }

        select_edges(&self.cands, self.cfg.select, &mut self.edges);
        None
    }
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
}
