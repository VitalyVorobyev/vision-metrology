//! `BeadTrace`, `BeadPassTrace` and `BeadStationTrace`: a bead tracker's run, explained.

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use vision_metrology::measure::diagnostics::{
    BeadPassTrace as NativeBeadPassTrace, BeadStationTrace as NativeBeadStationTrace,
    BeadTrace as NativeBeadTrace,
};

use super::bead::{TrackedBead, rows};
use super::caliper::CaliperTrace;
use crate::types::MeasurePair;

fn xy(p: impl Into<[f32; 2]>) -> (f32, f32) {
    let [x, y] = p.into();
    (x, y)
}

/// One station's measurement, explained — mirrors
/// `vision_metrology::measure::diagnostics::BeadStationTrace`.
///
/// `point`, `tangent`, `normal`, `strip_start` and `strip_end` are `(x, y)`; the strip
/// scans from `strip_start`, its −n end, to `strip_end`. `window` holds the offsets
/// `(lo, hi)`, in px along the normal, that a pair's midpoint had to fall in. `caliper` is
/// everything the strip's caliper computed. At a hit, `pair`, `offset` and `confidence`
/// describe the bead's pair and `reject` is `None`; at a rejection they are `None` and
/// `reject` names the reason.
#[pyclass(get_all, skip_from_py_object)]
pub struct BeadStationTrace {
    pub point: (f32, f32),
    pub tangent: (f32, f32),
    pub normal: (f32, f32),
    pub window: (f32, f32),
    pub strip_start: (f32, f32),
    pub strip_end: (f32, f32),
    pub caliper: Py<CaliperTrace>,
    pub pair: Option<MeasurePair>,
    pub offset: Option<f32>,
    pub confidence: Option<f32>,
    pub reject: Option<&'static str>,
}

impl BeadStationTrace {
    fn from_native(py: Python<'_>, s: NativeBeadStationTrace) -> PyResult<Self> {
        let hit = s.hit.ok();
        Ok(Self {
            point: xy(s.point.coords),
            tangent: xy(s.tangent),
            normal: xy(s.normal),
            window: s.window,
            strip_start: xy(s.strip.start.coords),
            strip_end: xy(s.strip.end.coords),
            caliper: Py::new(py, CaliperTrace::from_native(py, s.caliper))?,
            pair: hit.map(|h| MeasurePair::from(h.pair)),
            offset: hit.map(|h| h.offset),
            confidence: hit.map(|h| h.confidence),
            reject: s.hit.err().map(|r| r.as_str()),
        })
    }
}

#[pymethods]
impl BeadStationTrace {
    fn __repr__(&self) -> String {
        format!(
            "BeadStationTrace(point=({:.3}, {:.3}), offset={:?}, reject={:?})",
            self.point.0, self.point.1, self.offset, self.reject
        )
    }
}

/// One tracking pass, explained — mirrors
/// `vision_metrology::measure::diagnostics::BeadPassTrace`, as arrays.
///
/// Every array and list has one entry per station, as the pass measured it, before it
/// moved it. `points`, `tangents`, `normals`, `strip_starts` and `strip_ends` are
/// `(N, 2)`, and `windows` is `(N, 2)` of `(lo, hi)`. `observed` is each station's pair
/// offset, NaN where it was rejected; `weights` is the weight each observation carried in
/// the pass's last solve, 0 at a rejected station; `corrections` is the correction the
/// pass applied, in px along the normal. `reject` names each rejection (`None` at a hit)
/// and `calipers` holds each strip's `CaliperTrace`. The pass's summary is the same entry
/// of `result.passes`.
#[pyclass(get_all, skip_from_py_object)]
pub struct BeadPassTrace {
    pub points: Py<PyArray2<f32>>,
    pub tangents: Py<PyArray2<f32>>,
    pub normals: Py<PyArray2<f32>>,
    pub windows: Py<PyArray2<f32>>,
    pub strip_starts: Py<PyArray2<f32>>,
    pub strip_ends: Py<PyArray2<f32>>,
    pub observed: Py<PyArray1<f32>>,
    pub weights: Py<PyArray1<f32>>,
    pub corrections: Py<PyArray1<f32>>,
    pub reject: Vec<Option<&'static str>>,
    pub calipers: Vec<Py<CaliperTrace>>,
}

impl BeadPassTrace {
    fn from_native(py: Python<'_>, p: NativeBeadPassTrace) -> PyResult<Self> {
        let n = p.stations.len();
        let mut cols: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::with_capacity(2 * n));
        let mut observed = Vec::with_capacity(n);
        let mut reject = Vec::with_capacity(n);
        let mut calipers = Vec::with_capacity(n);
        for s in p.stations {
            let pairs = [
                xy(s.point.coords),
                xy(s.tangent),
                xy(s.normal),
                s.window,
                xy(s.strip.start.coords),
                xy(s.strip.end.coords),
            ];
            for (col, (a, b)) in cols.iter_mut().zip(pairs) {
                col.extend([a, b]);
            }
            observed.push(s.hit.map_or(f32::NAN, |h| h.offset));
            reject.push(s.hit.err().map(|r| r.as_str()));
            calipers.push(Py::new(py, CaliperTrace::from_native(py, s.caliper))?);
        }
        let [points, tangents, normals, windows, strip_starts, strip_ends] = cols;
        Ok(Self {
            points: rows(py, points)?,
            tangents: rows(py, tangents)?,
            normals: rows(py, normals)?,
            windows: rows(py, windows)?,
            strip_starts: rows(py, strip_starts)?,
            strip_ends: rows(py, strip_ends)?,
            observed: observed.into_pyarray(py).unbind(),
            weights: p.weights.into_pyarray(py).unbind(),
            corrections: p.corrections.into_pyarray(py).unbind(),
            reject,
            calipers,
        })
    }
}

#[pymethods]
impl BeadPassTrace {
    fn __repr__(&self) -> String {
        let rejected = self.reject.iter().filter(|r| r.is_some()).count();
        format!(
            "BeadPassTrace(stations={}, rejected={rejected})",
            self.reject.len()
        )
    }
}

/// A bead tracker's run, explained — mirrors
/// `vision_metrology::measure::diagnostics::BeadTrace`.
///
/// `result` is what `track` returns for the same config, image and prior. `passes` has
/// one `BeadPassTrace` per tracking pass, parallel to `result.passes`; `measure` has one
/// `BeadStationTrace` per station of the final stage, parallel to `result`'s arrays.
#[pyclass(get_all, skip_from_py_object)]
pub struct BeadTrace {
    pub result: Py<TrackedBead>,
    pub passes: Vec<Py<BeadPassTrace>>,
    pub measure: Vec<Py<BeadStationTrace>>,
}

impl BeadTrace {
    pub(super) fn from_native(py: Python<'_>, t: NativeBeadTrace) -> PyResult<Self> {
        let passes = t
            .passes
            .into_iter()
            .map(|p| Py::new(py, BeadPassTrace::from_native(py, p)?))
            .collect::<PyResult<_>>()?;
        let measure = t
            .measure
            .into_iter()
            .map(|s| Py::new(py, BeadStationTrace::from_native(py, s)?))
            .collect::<PyResult<_>>()?;
        Ok(Self {
            result: Py::new(py, TrackedBead::from_native(py, t.result)?)?,
            passes,
            measure,
        })
    }
}

#[pymethods]
impl BeadTrace {
    fn __repr__(&self) -> String {
        format!(
            "BeadTrace(passes={}, stations={})",
            self.passes.len(),
            self.measure.len()
        )
    }
}
