//! Metrology model: nominal geometry, calipers, robust fit, at a fixture pose.

use vm_primitives::{Circle2f, Error, ImageView, Line2f, Pixel, Point2f, Similarity2f, Vec2fExt};

use crate::fit::{Fit, FitConfig, fit_circle, fit_line};

use super::caliper::{Caliper, EdgeSelect, MeasureConfig, MeasureEdge, MeasureRadial, MeasureRect};

/// A nominal primitive to measure.
///
/// Coordinates are in **model space** — the frame the part was taught in.
/// [`MetrologyModel::apply`] maps them into the image with the fixture pose, so
/// a model taught once follows the part wherever it lands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetrologyShape {
    /// A line segment from `a` to `b`. Calipers are placed along it, scanning
    /// perpendicular.
    Line {
        /// Start of the nominal segment.
        a: Point2f,
        /// End of the nominal segment.
        b: Point2f,
    },
    /// A circle or arc. Calipers are placed around it, scanning radially.
    Circle {
        /// Nominal centre.
        center: Point2f,
        /// Nominal radius.
        radius: f32,
        /// Angular range `(start, extent)` in radians. `None` measures the
        /// full circle.
        arc: Option<(f32, f32)>,
    },
}

/// What was measured for one object.
#[derive(Debug, Clone, PartialEq)]
pub enum MetrologyResult {
    /// A fitted line, with residuals.
    Line(Fit<Line2f>),
    /// A fitted circle, with residuals.
    Circle(Fit<Circle2f>),
}

impl MetrologyResult {
    /// RMS deviation of the measured points from the fitted primitive, in pixels.
    pub fn rms(&self) -> f32 {
        match self {
            Self::Line(f) => f.rms,
            Self::Circle(f) => f.rms,
        }
    }

    /// Largest deviation, in pixels — the number a form tolerance is checked
    /// against.
    pub fn max_dev(&self) -> f32 {
        match self {
            Self::Line(f) => f.max_dev,
            Self::Circle(f) => f.max_dev,
        }
    }

    /// How many caliper points the fit used.
    pub fn n_used(&self) -> usize {
        match self {
            Self::Line(f) => f.n_used,
            Self::Circle(f) => f.n_used,
        }
    }
}

/// One nominal primitive plus how to measure it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetrologyObject {
    /// The nominal geometry, in model space.
    pub shape: MetrologyShape,
    /// How many calipers to distribute along it.
    pub n_calipers: usize,
    /// Half-length of each caliper, i.e. how far either side of nominal to
    /// search. Make it comfortably larger than the expected deviation and
    /// comfortably smaller than the distance to the next edge.
    pub caliper_len: f32,
    /// Half-width of each caliper — how much to average along the edge.
    pub caliper_width: f32,
    /// Edge extraction parameters.
    pub measure: MeasureConfig,
    /// Fitting parameters. A robust loss here is what makes a single bad
    /// caliper — a scratch, a highlight — not move the result.
    pub fit: FitConfig,
}

impl MetrologyObject {
    /// A sensible object for `shape`: 32 calipers, ±10 px search, ±5 px averaging.
    pub fn new(shape: MetrologyShape) -> Self {
        Self {
            shape,
            n_calipers: 32,
            caliper_len: 10.0,
            caliper_width: 5.0,
            // A caliper on a nominal edge should return that edge, not every
            // edge it crosses; `Strongest` is the sane default for a model
            // whose geometry is already approximately right.
            measure: MeasureConfig {
                select: EdgeSelect::Strongest,
                ..MeasureConfig::default()
            },
            fit: FitConfig::default(),
        }
    }
}

/// A set of nominal primitives measured together at a fixture pose.
///
/// This is the piece that turns detection into inspection: a
/// [`ShapeMatcher`](crate::ShapeMatcher) says *where the part is*, and its
/// [`ShapeMatch::pose`](crate::ShapeMatch) is the fixture this model is applied
/// at. The model itself is taught once, in the part's own frame.
///
/// # Example
/// ```
/// use vision_metrology::measure::{MetrologyModel, MetrologyObject, MetrologyResult, MetrologyShape};
/// use vision_metrology::{Image, Point2f, Similarity2f, Vec2f};
///
/// // A bright disc of radius 30 centred at (64, 64).
/// let (w, h) = (128usize, 128usize);
/// let data: Vec<u8> = (0..w * h)
///     .map(|i| {
///         let (x, y) = ((i % w) as f32 - 64.0, (i / w) as f32 - 64.0);
///         // Anti-aliased, so the true edge is exactly at r = 30.
///         let cover = (30.5 - (x * x + y * y).sqrt()).clamp(0.0, 1.0);
///         (20.0 + 180.0 * cover).round() as u8
///     })
///     .collect();
/// let img = Image::from_vec(w, h, data).unwrap();
///
/// let mut model = MetrologyModel::new();
/// model.add(MetrologyObject::new(MetrologyShape::Circle {
///     center: Point2f::new(64.0, 64.0),
///     radius: 30.0,
///     arc: None,
/// }));
///
/// // Identity fixture: the model is already in image coordinates.
/// let results = model.apply(&img.as_view(), &Similarity2f::identity());
/// let MetrologyResult::Circle(fit) = &results[0] else { panic!("a circle") };
/// assert!((fit.model.radius - 30.0).abs() < 0.05, "r = {}", fit.model.radius);
/// assert!(fit.rms < 0.05, "rms = {}", fit.rms);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MetrologyModel {
    objects: Vec<MetrologyObject>,
    caliper: Option<Caliper>,
    points: Vec<Point2f>,
    results: Vec<MetrologyResult>,
    /// Per-object caliper hits from the last `apply`, for diagnostics.
    hits: Vec<Vec<MeasureEdge>>,
}

impl MetrologyModel {
    /// An empty model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an object. Returns its index, which indexes into
    /// [`apply`](Self::apply)'s result.
    pub fn add(&mut self, object: MetrologyObject) -> usize {
        self.objects.push(object);
        self.objects.len() - 1
    }

    /// The objects, in the order they will be measured.
    pub fn objects(&self) -> &[MetrologyObject] {
        &self.objects
    }

    /// The caliper hits behind each result from the last [`apply`](Self::apply).
    ///
    /// One entry per object, in the same order. Overlaying these on the image
    /// is the fastest way to see *why* a fit came out the way it did — which is
    /// usually one caliper that found the wrong edge.
    pub fn hits(&self) -> &[Vec<MeasureEdge>] {
        &self.hits
    }

    /// Measure every object, with the model's nominal geometry mapped through
    /// `fixture`.
    ///
    /// `fixture` is normally [`ShapeMatch::pose`](crate::ShapeMatch) from a
    /// shape match. Objects whose fit fails are **skipped**, so the result may
    /// be shorter than [`objects`](Self::objects) — use
    /// [`apply_checked`](Self::apply_checked) when you need to know which.
    pub fn apply<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        fixture: &Similarity2f,
    ) -> &[MetrologyResult] {
        self.run(img, fixture);
        &self.results
    }

    /// Like [`apply`](Self::apply), but reports a per-object `Result` so a
    /// failed measurement is visible rather than missing.
    pub fn apply_checked<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        fixture: &Similarity2f,
    ) -> Vec<Result<MetrologyResult, Error>> {
        let objects = core::mem::take(&mut self.objects);
        let mut out = Vec::with_capacity(objects.len());
        self.hits.clear();
        for obj in &objects {
            let mut hits = Vec::new();
            out.push(self.measure_one(img, fixture, obj, &mut hits));
            self.hits.push(hits);
        }
        self.objects = objects;
        out
    }

    fn run<P: Pixel>(&mut self, img: &ImageView<'_, P>, fixture: &Similarity2f) {
        let objects = core::mem::take(&mut self.objects);
        self.results.clear();
        self.hits.clear();
        for obj in &objects {
            let mut hits = Vec::new();
            if let Ok(r) = self.measure_one(img, fixture, obj, &mut hits) {
                self.results.push(r);
            }
            self.hits.push(hits);
        }
        self.objects = objects;
    }

    fn measure_one<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        fixture: &Similarity2f,
        obj: &MetrologyObject,
        hits: &mut Vec<MeasureEdge>,
    ) -> Result<MetrologyResult, Error> {
        if obj.n_calipers < 2 {
            return Err(Error::InvalidConfig("metrology object needs ≥2 calipers"));
        }
        // The fixture may scale, and a caliper's reach must scale with the part
        // or it will fall short of the edge it is looking for.
        let scale = fixture.scaling();
        let cal = self
            .caliper
            .get_or_insert_with(|| Caliper::rect(placeholder_rect(), obj.measure));
        cal.set_config(obj.measure);

        self.points.clear();
        hits.clear();

        match obj.shape {
            MetrologyShape::Line { a, b } => {
                let (a, b) = (fixture * a, fixture * b);
                let along = (b - a).normalized_or_zero();
                if along.norm() < 0.5 {
                    return Err(Error::Degenerate("metrology line has zero length"));
                }
                let normal = along.perp();
                for i in 0..obj.n_calipers {
                    let f = i as f32 / (obj.n_calipers - 1) as f32;
                    let base = a + (b - a) * f;
                    cal.set_rect(MeasureRect {
                        center: base,
                        angle: normal.y.atan2(normal.x),
                        half_len: obj.caliper_len * scale,
                        half_width: obj.caliper_width * scale,
                    });
                    if let Some(e) = cal.measure(img).first() {
                        self.points.push(e.p);
                        hits.push(*e);
                    }
                }
                let fit = fit_line(&self.points, &obj.fit)?;
                Ok(MetrologyResult::Line(fit))
            }
            MetrologyShape::Circle {
                center,
                radius,
                arc,
            } => {
                let center = fixture * center;
                let radius = radius * scale;
                let (start, extent) = arc.unwrap_or((0.0, core::f32::consts::TAU));
                // A rotating fixture rotates the arc with the part.
                let start = start + fixture.isometry.rotation.angle();
                // A full circle wraps, so the last caliper would repeat the
                // first; an arc has two distinct ends worth measuring.
                let closed = (extent.abs() - core::f32::consts::TAU).abs() < 1e-3;
                let denom = if closed {
                    obj.n_calipers as f32
                } else {
                    (obj.n_calipers - 1) as f32
                };
                for i in 0..obj.n_calipers {
                    let phi = start + extent * i as f32 / denom;
                    // Radial, not rect: a rect averages along a chord, which on
                    // a curved edge samples the wrong side of the transition
                    // and biases the radius inward (−0.12 px at width 5,
                    // radius 40). `MeasureRadial` averages along the arc, so
                    // every averaged sample sits at the same radius.
                    cal.set_radial(MeasureRadial {
                        center,
                        radius,
                        angle: phi,
                        half_len: obj.caliper_len * scale,
                        half_width: obj.caliper_width * scale,
                    });
                    if let Some(e) = cal.measure(img).first() {
                        self.points.push(e.p);
                        hits.push(*e);
                    }
                }
                let fit = fit_circle(&self.points, &obj.fit)?;
                Ok(MetrologyResult::Circle(fit))
            }
        }
    }
}

fn placeholder_rect() -> MeasureRect {
    MeasureRect {
        center: Point2f::origin(),
        angle: 0.0,
        half_len: 1.0,
        half_width: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::{MetrologyModel, MetrologyObject, MetrologyResult, MetrologyShape};
    use crate::fit::{FitConfig, RobustLoss};
    use vm_primitives::{Image, Point2f, Similarity2f, Vec2f};

    /// Bright disc of radius `r` centred at `c`, **anti-aliased**.
    ///
    /// Coverage ramps linearly across one pixel at the boundary, so the true
    /// subpixel edge lies at exactly `r` in every direction. A hard-thresholded
    /// disc would instead put it wherever the pixel grid happened to fall —
    /// between 0.0 and 0.5 px inside `r`, varying with the centre's sub-pixel
    /// offset — which makes it useless for asserting subpixel accuracy.
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

    /// Bright half-plane below `edge_y`, anti-aliased for the same reason.
    fn half_plane(w: usize, h: usize, edge_y: f32) -> Image<u8> {
        let data: Vec<u8> = (0..w * h)
            .map(|i| {
                let y = (i / w) as f32;
                let cover = (y - edge_y + 0.5).clamp(0.0, 1.0);
                (20.0 + 180.0 * cover).round() as u8
            })
            .collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    #[test]
    fn measures_a_circle_at_the_identity_fixture() {
        let c = Point2f::new(80.0, 70.0);
        let img = disc(160, 140, c, 40.0);

        let mut model = MetrologyModel::new();
        model.add(MetrologyObject::new(MetrologyShape::Circle {
            center: c,
            radius: 40.0,
            arc: None,
        }));

        let results = model.apply(&img.as_view(), &Similarity2f::identity());
        assert_eq!(results.len(), 1);
        let MetrologyResult::Circle(fit) = &results[0] else {
            panic!("expected a circle")
        };
        // The fixture is anti-aliased and the calipers average along the arc,
        // so the measured radius should land on the nominal one, not near it.
        assert!(
            (fit.model.radius - 40.0).abs() < 0.03,
            "r = {}",
            fit.model.radius
        );
        assert!(
            (fit.model.center - c).norm() < 0.02,
            "c = {:?}",
            fit.model.center
        );
        assert!(fit.rms < 0.03, "rms = {}", fit.rms);
        assert_eq!(fit.n_used, 32);
    }

    /// The point of a fixture: teach the model once, measure the part wherever
    /// it lands.
    #[test]
    fn a_fixture_pose_follows_the_part() {
        // Taught at the origin of model space...
        let taught_center = Point2f::new(0.0, 0.0);
        let mut model = MetrologyModel::new();
        model.add(MetrologyObject::new(MetrologyShape::Circle {
            center: taught_center,
            radius: 30.0,
            arc: None,
        }));

        // ...and the part actually appears here, rotated.
        for (tx, ty, rot) in [(90.0f32, 80.0f32, 0.0f32), (70.0, 100.0, 0.9)] {
            let actual = Point2f::new(tx, ty);
            let img = disc(192, 192, actual, 30.0);
            let fixture = Similarity2f::new(Vec2f::new(tx, ty), rot, 1.0);

            let results = model.apply(&img.as_view(), &fixture);
            let MetrologyResult::Circle(fit) = &results[0] else {
                panic!("expected a circle")
            };
            assert!(
                (fit.model.center - actual).norm() < 0.03,
                "rot={rot}: centre {:?}, expected {actual:?}",
                fit.model.center
            );
            assert!(
                (fit.model.radius - 30.0).abs() < 0.03,
                "rot={rot}: r = {}",
                fit.model.radius
            );
        }
    }

    /// A scaling fixture must scale the caliper reach too, or the calipers
    /// land nowhere near the edge.
    #[test]
    fn a_scaling_fixture_scales_the_measurement() {
        let mut model = MetrologyModel::new();
        model.add(MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(0.0, 0.0),
            radius: 20.0,
            arc: None,
        }));

        let actual = Point2f::new(96.0, 96.0);
        let img = disc(192, 192, actual, 40.0); // twice the taught radius
        let fixture = Similarity2f::new(Vec2f::new(96.0, 96.0), 0.0, 2.0);

        let results = model.apply(&img.as_view(), &fixture);
        let MetrologyResult::Circle(fit) = &results[0] else {
            panic!("expected a circle")
        };
        assert!(
            (fit.model.radius - 40.0).abs() < 0.3,
            "r = {}",
            fit.model.radius
        );
    }

    #[test]
    fn measures_a_line() {
        let img = half_plane(160, 120, 70.0);

        let mut model = MetrologyModel::new();
        model.add(MetrologyObject::new(MetrologyShape::Line {
            a: Point2f::new(30.0, 70.0),
            b: Point2f::new(130.0, 70.0),
        }));

        let results = model.apply(&img.as_view(), &Similarity2f::identity());
        let MetrologyResult::Line(fit) = &results[0] else {
            panic!("expected a line")
        };
        assert!(fit.model.dir.y.abs() < 1e-3, "should be horizontal");
        assert!((fit.model.p.y - 70.0).abs() < 0.05, "y = {}", fit.model.p.y);
        assert!(fit.rms < 0.05, "rms = {}", fit.rms);
    }

    /// A local defect must move `max_dev` — that is the number an inspection
    /// gates on — while a robust loss keeps it out of the fitted geometry.
    #[test]
    fn a_defect_shows_up_in_max_dev_without_moving_the_fit() {
        let c = Point2f::new(80.0, 80.0);
        let (w, h) = (160usize, 160usize);
        // A disc with a flat chord bitten out of its right side.
        let data: Vec<u8> = (0..w * h)
            .map(|i| {
                let p = Point2f::new((i % w) as f32, (i / w) as f32);
                let inside = (p - c).norm() < 40.0 && p.x < c.x + 34.0;
                if inside { 200u8 } else { 20 }
            })
            .collect();
        let img = Image::from_vec(w, h, data).expect("valid image");

        let mut model = MetrologyModel::new();
        let mut obj = MetrologyObject::new(MetrologyShape::Circle {
            center: c,
            radius: 40.0,
            arc: None,
        });
        obj.caliper_len = 12.0;
        obj.fit = FitConfig {
            loss: RobustLoss::Tukey { c: 1.0 },
            ..FitConfig::default()
        };
        model.add(obj);

        let results = model.apply(&img.as_view(), &Similarity2f::identity());
        let MetrologyResult::Circle(fit) = &results[0] else {
            panic!("expected a circle")
        };
        // The good part of the boundary still gives the right circle...
        assert!(
            (fit.model.radius - 40.0).abs() < 0.3,
            "r = {}",
            fit.model.radius
        );
        // ...and the bitten calipers were excluded from the fit.
        assert!(
            fit.n_used < 32,
            "the flat should cost some calipers, kept {}",
            fit.n_used
        );
    }

    #[test]
    fn hits_expose_the_points_behind_a_fit() {
        let c = Point2f::new(60.0, 60.0);
        let img = disc(128, 128, c, 25.0);
        let mut model = MetrologyModel::new();
        model.add(MetrologyObject::new(MetrologyShape::Circle {
            center: c,
            radius: 25.0,
            arc: None,
        }));
        model.apply(&img.as_view(), &Similarity2f::identity());

        assert_eq!(model.hits().len(), 1);
        assert_eq!(model.hits()[0].len(), 32);
        for e in &model.hits()[0] {
            assert!(
                ((e.p - c).norm() - 25.0).abs() < 0.2,
                "caliper hit off the boundary: {:?}",
                e.p
            );
        }
    }

    #[test]
    fn a_failed_object_is_visible_through_apply_checked() {
        // Nothing to find: a flat field.
        let img = Image::from_vec(64, 64, vec![128u8; 64 * 64]).expect("valid");
        let mut model = MetrologyModel::new();
        model.add(MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(32.0, 32.0),
            radius: 20.0,
            arc: None,
        }));

        assert!(
            model
                .apply(&img.as_view(), &Similarity2f::identity())
                .is_empty()
        );
        let checked = model.apply_checked(&img.as_view(), &Similarity2f::identity());
        assert_eq!(checked.len(), 1);
        assert!(checked[0].is_err(), "a flat field cannot yield a circle");
    }

    #[test]
    fn too_few_calipers_is_a_config_error() {
        let img = disc(64, 64, Point2f::new(32.0, 32.0), 20.0);
        let mut model = MetrologyModel::new();
        let mut obj = MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(32.0, 32.0),
            radius: 20.0,
            arc: None,
        });
        obj.n_calipers = 1;
        model.add(obj);
        let checked = model.apply_checked(&img.as_view(), &Similarity2f::identity());
        assert!(checked[0].is_err());
    }
}
