//! Python bindings for `measure`: calipers and the metrology model.
//!
//! ## `RejectReason`
//!
//! `Caliper::measure` returns `Result<&[MeasureEdge], RejectReason>` on the
//! Rust side — see invariant 21 and the module's own docs on why an empty
//! result is unrepresentable. The Python mirror raises [`MeasureRejected`],
//! a plain exception whose single argument is one of the five reason strings
//! (`"profile_too_short"`, `"no_edge"`, `"wrong_polarity"`, `"too_oblique"`,
//! `"off_image"`): `except vm.MeasureRejected as e: reason = e.args[0]`. This
//! is the ordinary Python idiom for "this call has a well-defined failure
//! mode", and it composes with `try`/`except` instead of asking every caller
//! to unwrap a tagged result by hand.
//!
//! `MetrologyModel.apply` is different: it always returns one entry per
//! object, in order, and an exception on the first failure would discard the
//! others (that discarding is exactly what the Rust side's `Vec<Result<_>>`
//! return type was chosen to prevent). Its Python mirror instead returns a
//! list where each entry is either a [`MetrologyResult`] or a
//! [`MetrologyError`] carrying the failure message — a small tagged union in
//! place of an exception, because the caller needs all the entries, not just
//! the first problem.

use pyo3::prelude::*;
use pyo3::{create_exception, exceptions::PyException};

use vision_metrology::measure::diagnostics::{
    CaliperPlacement as NativeCaliperPlacement, CaliperShape as NativeCaliperShape,
    layout as native_layout,
};
use vision_metrology::measure::{
    Caliper as NativeCaliper, MetrologyModel as NativeMetrologyModel,
    MetrologyObject as NativeMetrologyObject, MetrologyShape as NativeMetrologyShape,
    RejectReason as NativeRejectReason,
};
use vm_primitives::{Point2f, Similarity2f, Vec2f, similarity_from_parts, wrap_angle};

use crate::config::{FitConfig, MeasureConfig};
use crate::convert::{any_image_from_numpy, with_any_image};
use crate::types::{Circle, Line, MeasureEdge, MeasurePair};

create_exception!(
    vision_metrology,
    MeasureRejected,
    PyException,
    "A caliper found no edge; `args[0]` names which gate rejected it."
);

/// `Translation(position) ∘ sR ∘ Translation(−origin)`, as a similarity.
///
/// Mirrors `vision_metrology::matching::matcher::pose_from` (private to that
/// crate) — this is `ShapeMatch::pose`'s own construction, so a caller who
/// hands this binding a `ShapeMatch`'s `(x, y, angle, scale)` plus the taught
/// model's `origin` gets exactly the fixture `ShapeMatch::pose` would build.
fn pose_from(position: Point2f, angle: f32, scale: f32, origin: Point2f) -> Similarity2f {
    let (sn, cs) = wrap_angle(angle).sin_cos();
    let t = Vec2f::new(
        position.x - scale * (cs * origin.x - sn * origin.y),
        position.y - scale * (sn * origin.x + cs * origin.y),
    );
    similarity_from_parts(t, wrap_angle(angle), scale)
}

fn reject_reason_str(r: NativeRejectReason) -> &'static str {
    match r {
        NativeRejectReason::ProfileTooShort => "profile_too_short",
        NativeRejectReason::NoEdge => "no_edge",
        NativeRejectReason::WrongPolarity => "wrong_polarity",
        NativeRejectReason::TooOblique => "too_oblique",
        NativeRejectReason::OffImage => "off_image",
    }
}

/// A reusable caliper: place it on a rectangle, arc or radial path, then
/// measure frame after frame.
///
/// Construct with [`rect`](Self::rect), [`arc`](Self::arc) or
/// [`radial`](Self::radial); `move_to_*` variants reposition an existing
/// caliper, keeping its scratch buffers.
#[pyclass]
pub struct Caliper {
    inner: NativeCaliper,
}

#[pymethods]
impl Caliper {
    /// A rectangular caliper: scans along `angle`, averages across it.
    #[staticmethod]
    #[pyo3(signature = (center, angle, half_len, half_width, config=None))]
    pub fn rect(
        center: (f32, f32),
        angle: f32,
        half_len: f32,
        half_width: f32,
        config: Option<MeasureConfig>,
    ) -> Self {
        Self {
            inner: NativeCaliper::rect(
                vision_metrology::measure::MeasureRect {
                    center: Point2f::new(center.0, center.1),
                    angle,
                    half_len,
                    half_width,
                },
                config.unwrap_or_default().to_native(),
            ),
        }
    }

    /// An annular caliper: scans along a circular arc, averages radially.
    #[staticmethod]
    #[pyo3(signature = (center, radius, angle_start, angle_extent, half_width, config=None))]
    pub fn arc(
        center: (f32, f32),
        radius: f32,
        angle_start: f32,
        angle_extent: f32,
        half_width: f32,
        config: Option<MeasureConfig>,
    ) -> Self {
        Self {
            inner: NativeCaliper::arc(
                vision_metrology::measure::MeasureArc {
                    center: Point2f::new(center.0, center.1),
                    radius,
                    angle_start,
                    angle_extent,
                    half_width,
                },
                config.unwrap_or_default().to_native(),
            ),
        }
    }

    /// A radial caliper: scans outward from `center`, averaging along the
    /// arc — the geometry to measure a circular edge without bias.
    #[staticmethod]
    #[pyo3(signature = (center, radius, angle, half_len, half_width, config=None))]
    pub fn radial(
        center: (f32, f32),
        radius: f32,
        angle: f32,
        half_len: f32,
        half_width: f32,
        config: Option<MeasureConfig>,
    ) -> Self {
        Self {
            inner: NativeCaliper::radial(
                vision_metrology::measure::MeasureRadial {
                    center: Point2f::new(center.0, center.1),
                    radius,
                    angle,
                    half_len,
                    half_width,
                },
                config.unwrap_or_default().to_native(),
            ),
        }
    }

    /// Extract edges under the current placement.
    ///
    /// Raises [`MeasureRejected`] naming the gate that fired when nothing
    /// was found — see the module docstring.
    pub fn measure(
        &mut self,
        py: Python<'_>,
        img: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<MeasureEdge>> {
        let any = any_image_from_numpy(py, img)?;
        with_any_image!(any, view => {
            self.inner
                .measure(&view)
                .map(|edges| edges.iter().copied().map(MeasureEdge::from).collect())
                .map_err(|r| MeasureRejected::new_err(reject_reason_str(r)))
        })
    }

    /// Extract opposite-polarity edge pairs — one per bar or gap crossed.
    pub fn measure_pairs(
        &mut self,
        py: Python<'_>,
        img: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<MeasurePair>> {
        let any = any_image_from_numpy(py, img)?;
        with_any_image!(any, view => {
            Ok(self
                .inner
                .measure_pairs(&view)
                .iter()
                .copied()
                .map(MeasurePair::from)
                .collect())
        })
    }

    /// The averaged 1-D profile from the last `measure` call.
    pub fn profile(&self) -> Vec<f32> {
        self.inner.profile().to_vec()
    }
}

/// A nominal primitive to measure, in model space.
///
/// Constructed with the static [`line`](Self::line) / [`circle`](Self::circle)
/// methods, matching `vision_metrology::measure::MetrologyShape`.
#[pyclass(get_all, from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct MetrologyShape {
    kind: &'static str,
    a: (f32, f32),
    b: (f32, f32),
    center: (f32, f32),
    radius: f32,
    arc: Option<(f32, f32)>,
}

#[pymethods]
impl MetrologyShape {
    /// A line segment from `a` to `b`; calipers scan perpendicular to it.
    #[staticmethod]
    pub fn line(a: (f32, f32), b: (f32, f32)) -> Self {
        Self {
            kind: "line",
            a,
            b,
            center: (0.0, 0.0),
            radius: 0.0,
            arc: None,
        }
    }

    /// A circle (or arc, when `arc = (start, extent)` in radians is given);
    /// calipers scan radially.
    #[staticmethod]
    #[pyo3(signature = (center, radius, arc=None))]
    pub fn circle(center: (f32, f32), radius: f32, arc: Option<(f32, f32)>) -> Self {
        Self {
            kind: "circle",
            a: (0.0, 0.0),
            b: (0.0, 0.0),
            center,
            radius,
            arc,
        }
    }

    fn __repr__(&self) -> String {
        match self.kind {
            "line" => format!("MetrologyShape.line({:?}, {:?})", self.a, self.b),
            _ => format!(
                "MetrologyShape.circle({:?}, {:.3}, arc={:?})",
                self.center, self.radius, self.arc
            ),
        }
    }
}

impl MetrologyShape {
    fn to_native(self) -> NativeMetrologyShape {
        match self.kind {
            "line" => NativeMetrologyShape::Line {
                a: Point2f::new(self.a.0, self.a.1),
                b: Point2f::new(self.b.0, self.b.1),
            },
            _ => NativeMetrologyShape::Circle {
                center: Point2f::new(self.center.0, self.center.1),
                radius: self.radius,
                arc: self.arc,
            },
        }
    }
}

/// One nominal primitive plus how to measure it — mirrors
/// `vision_metrology::measure::MetrologyObject`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Clone)]
pub struct MetrologyObject {
    pub shape: MetrologyShape,
    pub n_calipers: usize,
    pub caliper_len: f32,
    pub caliper_width: f32,
    pub measure: MeasureConfig,
    pub fit: FitConfig,
}

#[pymethods]
impl MetrologyObject {
    /// A sensible default for `shape`: 32 calipers, +/-10 px search, +/-5 px
    /// averaging, strongest edge per caliper.
    #[new]
    #[pyo3(signature = (shape, n_calipers=None, caliper_len=None, caliper_width=None, measure=None, fit=None))]
    pub fn new(
        shape: MetrologyShape,
        n_calipers: Option<usize>,
        caliper_len: Option<f32>,
        caliper_width: Option<f32>,
        measure: Option<MeasureConfig>,
        fit: Option<FitConfig>,
    ) -> Self {
        Self {
            shape,
            n_calipers: n_calipers.unwrap_or(32),
            caliper_len: caliper_len.unwrap_or(10.0),
            caliper_width: caliper_width.unwrap_or(5.0),
            measure: measure.unwrap_or_else(|| MeasureConfig {
                select: "strongest".to_string(),
                ..MeasureConfig::default()
            }),
            fit: fit.unwrap_or_default(),
        }
    }
}

impl MetrologyObject {
    fn to_native(&self) -> PyResult<NativeMetrologyObject> {
        Ok(NativeMetrologyObject {
            shape: self.shape.to_native(),
            n_calipers: self.n_calipers,
            caliper_len: self.caliper_len,
            caliper_width: self.caliper_width,
            measure: self.measure.to_native(),
            fit: self.fit.to_native()?,
        })
    }
}

/// The fit plus the caliper edges it came from — mirrors
/// `vision_metrology::measure::MetrologyResult`.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct MetrologyResult {
    /// `"line"` or `"circle"`.
    pub kind: &'static str,
    pub line: Option<Line>,
    pub circle: Option<Circle>,
    pub rms: f32,
    pub max_dev: f32,
    pub n_used: usize,
    pub hits: Vec<MeasureEdge>,
}

#[pymethods]
impl MetrologyResult {
    fn __repr__(&self) -> String {
        format!(
            "MetrologyResult(kind='{}', rms={:.4}, max_dev={:.4}, n_used={})",
            self.kind, self.rms, self.max_dev, self.n_used
        )
    }
}

impl From<vision_metrology::measure::MetrologyResult> for MetrologyResult {
    fn from(r: vision_metrology::measure::MetrologyResult) -> Self {
        let rms = r.rms();
        let max_dev = r.max_dev();
        let n_used = r.n_used();
        let hits = r.hits.into_iter().map(MeasureEdge::from).collect();
        match r.fit {
            vision_metrology::measure::MetrologyFit::Line(f) => Self {
                kind: "line",
                line: Some(Line {
                    px: f.model.p.x,
                    py: f.model.p.y,
                    dx: f.model.dir.x,
                    dy: f.model.dir.y,
                    rms: f.rms,
                    max_dev: f.max_dev,
                    n_used: f.n_used,
                }),
                circle: None,
                rms,
                max_dev,
                n_used,
                hits,
            },
            vision_metrology::measure::MetrologyFit::Circle(f) => Self {
                kind: "circle",
                line: None,
                circle: Some(Circle {
                    cx: f.model.center.x,
                    cy: f.model.center.y,
                    r: f.model.radius,
                    rms: f.rms,
                    max_dev: f.max_dev,
                    n_used: f.n_used,
                }),
                rms,
                max_dev,
                n_used,
                hits,
            },
        }
    }
}

/// A `MetrologyObject` that could not be measured; carries the native
/// `Error`'s message.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct MetrologyError {
    pub message: String,
}

#[pymethods]
impl MetrologyError {
    fn __repr__(&self) -> String {
        format!("MetrologyError({:?})", self.message)
    }
}

/// Where one caliper of a [`MetrologyModel`] sits at a fixture pose, without
/// measuring — mirrors `vision_metrology::measure::diagnostics::CaliperPlacement`.
///
/// `kind` is `"rect"` (line objects) or `"radial"` (circle objects); `radius`
/// is only set for `"radial"` placements, where `center` is the *circle's*
/// own centre (not the caliper's own position on the circle — see the Rust
/// `CaliperShape::Radial` docs). Built to be handed straight to
/// [`Caliper.rect`](Caliper::rect) / [`Caliper.radial`](Caliper::radial):
/// `Caliper.rect(p.center, p.angle, p.half_len, p.half_width, config)` or
/// `Caliper.radial(p.center, p.radius, p.angle, p.half_len, p.half_width,
/// config)`.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaliperPlacement {
    pub object_index: usize,
    pub caliper_index: usize,
    pub kind: &'static str,
    pub center: (f32, f32),
    pub angle: f32,
    pub half_len: f32,
    pub half_width: f32,
    pub radius: Option<f32>,
}

#[pymethods]
impl CaliperPlacement {
    fn __repr__(&self) -> String {
        format!(
            "CaliperPlacement(object_index={}, caliper_index={}, kind='{}', center={:?}, angle={:.4})",
            self.object_index, self.caliper_index, self.kind, self.center, self.angle
        )
    }
}

impl From<NativeCaliperPlacement> for CaliperPlacement {
    fn from(p: NativeCaliperPlacement) -> Self {
        match p.shape {
            NativeCaliperShape::Rect(r) => Self {
                object_index: p.object_index,
                caliper_index: p.caliper_index,
                kind: "rect",
                center: (r.center.x, r.center.y),
                angle: r.angle,
                half_len: r.half_len,
                half_width: r.half_width,
                radius: None,
            },
            NativeCaliperShape::Radial(r) => Self {
                object_index: p.object_index,
                caliper_index: p.caliper_index,
                kind: "radial",
                center: (r.center.x, r.center.y),
                angle: r.angle,
                half_len: r.half_len,
                half_width: r.half_width,
                radius: Some(r.radius),
            },
        }
    }
}

/// A set of nominal primitives measured together at a fixture pose.
///
/// Mirrors `vision_metrology::measure::MetrologyModel`: `add` a
/// [`MetrologyObject`] per nominal primitive, then [`apply`](Self::apply) at
/// the fixture pose (typically a [`ShapeMatch`](crate::types::ShapeMatch)'s
/// `x`, `y`, `angle`, `scale`).
#[pyclass]
#[derive(Default)]
pub struct MetrologyModel {
    inner: NativeMetrologyModel,
}

#[pymethods]
impl MetrologyModel {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an object. Returns its index, which indexes into
    /// [`apply`](Self::apply)'s result.
    pub fn add(&mut self, object: MetrologyObject) -> PyResult<usize> {
        let native = object.to_native()?;
        Ok(self.inner.add(native))
    }

    /// How many objects have been added.
    #[getter]
    pub fn num_objects(&self) -> usize {
        self.inner.objects().len()
    }

    /// Measure every object with the model's nominal geometry mapped through
    /// the fixture `(x, y, angle, scale)` — typically a matched
    /// [`ShapeMatch`](crate::types::ShapeMatch)'s own fields — composed with
    /// `origin`, the model-space point that `(x, y)` names.
    ///
    /// `origin` defaults to `(0, 0)`; pass the taught [`ShapeModel`]'s
    /// `origin` (the same value handed to [`ShapeMatch.matrix`]) when it is
    /// nonzero, or the fixture is mis-posed — see the fixture-pose semantics
    /// on [`ShapeMatch::pose`](crate::matching::ShapeMatch) (the Rust
    /// `vision_metrology::matching` type this mirrors): `position +
    /// scale·R(angle)·(point − origin)`, not the origin-less `scale·R(angle)·point +
    /// (x, y)` this binding built before this fix.
    ///
    /// Returns one entry per object, in [`add`](Self::add) order: either a
    /// [`MetrologyResult`] or a [`MetrologyError`] naming why that object
    /// could not be measured. See the module docstring for why this is a
    /// list of outcomes rather than raising on the first failure.
    #[pyo3(signature = (image, x, y, angle=0.0, scale=1.0, origin=(0.0, 0.0)))]
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        x: f32,
        y: f32,
        angle: f32,
        scale: f32,
        origin: (f32, f32),
    ) -> PyResult<Vec<Py<PyAny>>> {
        let any = any_image_from_numpy(py, image)?;
        let fixture = pose_from(
            Point2f::new(x, y),
            angle,
            scale,
            Point2f::new(origin.0, origin.1),
        );
        let outcomes = with_any_image!(any, view => self.inner.apply(&view, &fixture));

        outcomes
            .into_iter()
            .map(|r| match r {
                Ok(res) => Ok(Py::new(py, MetrologyResult::from(res))?.into()),
                Err(e) => Ok(Py::new(
                    py,
                    MetrologyError {
                        message: e.to_string(),
                    },
                )?
                .into()),
            })
            .collect()
    }

    /// Where every caliper of every added object sits at the fixture pose,
    /// without measuring — same fixture semantics as [`apply`](Self::apply).
    ///
    /// This is what an overlay should call to draw caliper boxes: it needs no
    /// image, uses the same placement code `apply` does internally (so a
    /// caller can never draw a caliper somewhere the actual measurement did
    /// not look), and works even for objects that will go on to reject every
    /// caliper.
    #[pyo3(signature = (x, y, angle=0.0, scale=1.0, origin=(0.0, 0.0)))]
    pub fn layout(
        &self,
        x: f32,
        y: f32,
        angle: f32,
        scale: f32,
        origin: (f32, f32),
    ) -> Vec<CaliperPlacement> {
        let fixture = pose_from(
            Point2f::new(x, y),
            angle,
            scale,
            Point2f::new(origin.0, origin.1),
        );
        native_layout(&self.inner, &fixture)
            .into_iter()
            .map(CaliperPlacement::from)
            .collect()
    }
}
