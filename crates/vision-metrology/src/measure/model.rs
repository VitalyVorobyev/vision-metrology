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

/// The fitted primitive behind a [`MetrologyResult`].
#[derive(Debug, Clone, PartialEq)]
pub enum MetrologyFit {
    /// A fitted line, with residuals.
    Line(Fit<Line2f>),
    /// A fitted circle, with residuals.
    Circle(Fit<Circle2f>),
}

/// What was measured for one object: the fit, and the caliper edges it came
/// from.
///
/// `hits` used to be a separate `MetrologyModel::hits()` accessor holding the
/// last call's edges parallel to the results — a second array the caller had to
/// keep aligned by hand, and one that a second `apply` silently invalidated.
/// Overlaying the hits on the image is the fastest way to see *why* a fit came
/// out the way it did (usually one caliper that found the wrong edge), so they
/// belong to the result that used them.
#[derive(Debug, Clone, PartialEq)]
pub struct MetrologyResult {
    /// The fitted primitive with its residual statistics.
    pub fit: MetrologyFit,
    /// The caliper edges the fit was computed from, in caliper order.
    ///
    /// Shorter than the object's `n_calipers` when some calipers found nothing.
    pub hits: Vec<MeasureEdge>,
}

impl MetrologyFit {
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

impl MetrologyResult {
    /// RMS deviation of the measured points from the fitted primitive, in pixels.
    pub fn rms(&self) -> f32 {
        self.fit.rms()
    }

    /// Largest deviation, in pixels — the number a form tolerance is checked
    /// against.
    pub fn max_dev(&self) -> f32 {
        self.fit.max_dev()
    }

    /// How many caliper points the fit used.
    pub fn n_used(&self) -> usize {
        self.fit.n_used()
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
/// [`ShapeMatcher`](crate::matching::ShapeMatcher) says *where the part is*, and its
/// [`ShapeMatch::pose`](crate::matching::ShapeMatch) is the fixture this model is applied
/// at. The model itself is taught once, in the part's own frame.
///
/// # Example
/// ```
/// use vision_metrology::measure::{
///     MetrologyFit, MetrologyModel, MetrologyObject, MetrologyShape,
/// };
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
/// let MetrologyFit::Circle(fit) = &results[0].as_ref().expect("measured").fit else { panic!("a circle") };
/// assert!((fit.model.radius - 30.0).abs() < 0.05, "r = {}", fit.model.radius);
/// assert!(fit.rms < 0.05, "rms = {}", fit.rms);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MetrologyModel {
    objects: Vec<MetrologyObject>,
    caliper: Option<Caliper>,
    points: Vec<Point2f>,
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

    /// Measure every object, with the model's nominal geometry mapped through
    /// `fixture`.
    ///
    /// `fixture` is normally [`ShapeMatch::pose`](crate::matching::ShapeMatch)
    /// from a shape match. One entry per object, in
    /// [`objects`](Self::objects) order, so index `i` is always object `i` —
    /// there is no variant that drops the failures and renumbers the rest.
    /// A `MetrologyObject` that could not be measured names its `Error`.
    pub fn apply<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        fixture: &Similarity2f,
    ) -> Vec<Result<MetrologyResult, Error>> {
        let objects = core::mem::take(&mut self.objects);
        let mut out = Vec::with_capacity(objects.len());
        for obj in &objects {
            out.push(self.measure_one(img, fixture, obj));
        }
        self.objects = objects;
        out
    }

    fn measure_one<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        fixture: &Similarity2f,
        obj: &MetrologyObject,
    ) -> Result<MetrologyResult, Error> {
        let placements = caliper_placements(obj, fixture)?;
        let mut hits: Vec<MeasureEdge> = Vec::new();
        let cal = self
            .caliper
            .get_or_insert_with(|| Caliper::rect(placeholder_rect(), obj.measure));
        cal.set_config(obj.measure);

        self.points.clear();
        for placement in &placements {
            match *placement {
                CaliperShape::Rect(r) => cal.set_rect(r),
                CaliperShape::Radial(r) => cal.set_radial(r),
            }
            if let Ok(&[e, ..]) = cal.measure(img) {
                self.points.push(e.p);
                hits.push(e);
            }
        }

        match obj.shape {
            MetrologyShape::Line { .. } => {
                let fit = fit_line(&self.points, &obj.fit)?;
                Ok(MetrologyResult {
                    fit: MetrologyFit::Line(fit),
                    hits,
                })
            }
            MetrologyShape::Circle { .. } => {
                let fit = fit_circle(&self.points, &obj.fit)?;
                Ok(MetrologyResult {
                    fit: MetrologyFit::Circle(fit),
                    hits,
                })
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

/// Where one caliper of a [`MetrologyObject`] sits, once placed — either a
/// [`MeasureRect`] (line objects) or a [`MeasureRadial`] (circle objects).
///
/// This is exactly what [`MetrologyModel::apply`] hands a [`Caliper`] via
/// `set_rect`/`set_radial`; [`diagnostics::layout`](super::diagnostics::layout)
/// exposes the same values without measuring, for overlay drawing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CaliperShape {
    /// A rectangular caliper — see [`MeasureRect`].
    Rect(MeasureRect),
    /// A radial caliper on a circle — see [`MeasureRadial`]. Note that
    /// `MeasureRadial::center` is the *circle's* centre, not the caliper's
    /// own position on the circle — the field name is inherited from the
    /// measurement geometry, not an overlay-box convention.
    Radial(MeasureRadial),
}

/// Compute every caliper placement for `obj` at `fixture`, without measuring.
///
/// The one place this geometry is written — [`MetrologyModel::measure_one`]
/// and [`diagnostics::layout`](super::diagnostics::layout) both call this
/// rather than each re-deriving it, so an overlay can never show a caliper
/// somewhere the actual measurement did not look.
pub(crate) fn caliper_placements(
    obj: &MetrologyObject,
    fixture: &Similarity2f,
) -> Result<Vec<CaliperShape>, Error> {
    if obj.n_calipers < 2 {
        return Err(Error::InvalidConfig("metrology object needs ≥2 calipers"));
    }
    // The fixture may scale, and a caliper's reach must scale with the part
    // or it will fall short of the edge it is looking for.
    let scale = fixture.scaling();
    let mut out = Vec::with_capacity(obj.n_calipers);

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
                let center = a + (b - a) * f;
                out.push(CaliperShape::Rect(MeasureRect {
                    center,
                    angle: normal.y.atan2(normal.x),
                    half_len: obj.caliper_len * scale,
                    half_width: obj.caliper_width * scale,
                }));
            }
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
                out.push(CaliperShape::Radial(MeasureRadial {
                    center,
                    radius,
                    angle: phi,
                    half_len: obj.caliper_len * scale,
                    half_width: obj.caliper_width * scale,
                }));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{MetrologyFit, MetrologyModel, MetrologyObject, MetrologyShape};
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
        let MetrologyFit::Circle(fit) = &results[0].as_ref().expect("measured").fit else {
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
            let MetrologyFit::Circle(fit) = &results[0].as_ref().expect("measured").fit else {
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
        let MetrologyFit::Circle(fit) = &results[0].as_ref().expect("measured").fit else {
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
        let MetrologyFit::Line(fit) = &results[0].as_ref().expect("measured").fit else {
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
        let MetrologyFit::Circle(fit) = &results[0].as_ref().expect("measured").fit else {
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
        let results = model.apply(&img.as_view(), &Similarity2f::identity());

        assert_eq!(results.len(), 1);
        let hits = &results[0].as_ref().expect("measured").hits;
        assert_eq!(hits.len(), 32);
        for e in hits {
            assert!(
                ((e.p - c).norm() - 25.0).abs() < 0.2,
                "caliper hit off the boundary: {:?}",
                e.p
            );
        }
    }

    #[test]
    fn a_failed_object_is_visible_rather_than_missing() {
        // Nothing to find: a flat field.
        let img = Image::from_vec(64, 64, vec![128u8; 64 * 64]).expect("valid");
        let mut model = MetrologyModel::new();
        model.add(MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(32.0, 32.0),
            radius: 20.0,
            arc: None,
        }));

        // The slot is still there — index i is object i, always.
        let checked = model.apply(&img.as_view(), &Similarity2f::identity());
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
        let checked = model.apply(&img.as_view(), &Similarity2f::identity());
        assert!(checked[0].is_err());
    }
}
