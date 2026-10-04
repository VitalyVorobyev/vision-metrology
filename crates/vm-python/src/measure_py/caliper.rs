//! `Caliper` and `CaliperTrace`.

use pyo3::prelude::*;

use numpy::{IntoPyArray, PyArray1};
use vision_metrology::measure::Caliper as NativeCaliper;
use vision_metrology::measure::diagnostics::{
    CaliperTrace as NativeCaliperTrace, explain as native_explain,
};

use super::{MeasureRejected, arc_from, radial_from, rect_from, reject_reason_str, strip_from};
use crate::config::MeasureConfig;
use crate::convert::{any_image_from_numpy, with_any_image};
use crate::types::{LevelEdge, MeasureEdge, MeasurePair};

/// A reusable caliper: place it on a rectangle, arc, radial path or strip, then
/// measure frame after frame.
///
/// Construct with [`rect`](Self::rect), [`arc`](Self::arc),
/// [`radial`](Self::radial) or [`strip`](Self::strip); the matching
/// `move_to_rect`, `move_to_arc`, `move_to_radial` and `move_to_strip` reposition the
/// caliper, keeping its config and scratch buffers.
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
    ) -> PyResult<Self> {
        Ok(Self {
            inner: NativeCaliper::rect(
                rect_from(center, angle, half_len, half_width),
                config.unwrap_or_default().to_native()?,
            ),
        })
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
    ) -> PyResult<Self> {
        Ok(Self {
            inner: NativeCaliper::arc(
                arc_from(center, radius, angle_start, angle_extent, half_width),
                config.unwrap_or_default().to_native()?,
            ),
        })
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
    ) -> PyResult<Self> {
        Ok(Self {
            inner: NativeCaliper::radial(
                radial_from(center, radius, angle, half_len, half_width),
                config.unwrap_or_default().to_native()?,
            ),
        })
    }

    /// A strip from `start` to `end`: scans along it, averages across it.
    ///
    /// `samples` points along the strip include both endpoints; `across` lines are
    /// spread evenly over `±half_width`. Left as `None`, they follow the config's
    /// `step` along and about one line per pixel across. An edge's `t` is its
    /// distance from `start`.
    #[staticmethod]
    #[pyo3(signature = (start, end, half_width=0.0, samples=None, across=None, config=None))]
    pub fn strip(
        start: (f32, f32),
        end: (f32, f32),
        half_width: f32,
        samples: Option<usize>,
        across: Option<usize>,
        config: Option<MeasureConfig>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: NativeCaliper::strip(
                strip_from(start, end, half_width, samples, across)?,
                config.unwrap_or_default().to_native()?,
            ),
        })
    }

    /// Move to a new rectangle, keeping the config and buffers.
    pub fn move_to_rect(&mut self, center: (f32, f32), angle: f32, half_len: f32, half_width: f32) {
        self.inner
            .set_rect(rect_from(center, angle, half_len, half_width));
    }

    /// Move to a new arc, keeping the config and buffers.
    pub fn move_to_arc(
        &mut self,
        center: (f32, f32),
        radius: f32,
        angle_start: f32,
        angle_extent: f32,
        half_width: f32,
    ) {
        self.inner.set_arc(arc_from(
            center,
            radius,
            angle_start,
            angle_extent,
            half_width,
        ));
    }

    /// Move to a new radial placement, keeping the config and buffers.
    pub fn move_to_radial(
        &mut self,
        center: (f32, f32),
        radius: f32,
        angle: f32,
        half_len: f32,
        half_width: f32,
    ) {
        self.inner
            .set_radial(radial_from(center, radius, angle, half_len, half_width));
    }

    /// Move to a new strip, keeping the config and buffers.
    #[pyo3(signature = (start, end, half_width=0.0, samples=None, across=None))]
    pub fn move_to_strip(
        &mut self,
        start: (f32, f32),
        end: (f32, f32),
        half_width: f32,
        samples: Option<usize>,
        across: Option<usize>,
    ) -> PyResult<()> {
        self.inner
            .set_strip(strip_from(start, end, half_width, samples, across)?);
        Ok(())
    }

    /// Extract edges under the current placement.
    ///
    /// Raises `MeasureRejected` when nothing was found; its `args[0]` names
    /// the gate that fired.
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

    /// Distance between profile samples, in pixels, at the current placement: the
    /// scan's extent divided by `samples - 1`, which differs from the config's `step`
    /// whenever the extent is not a whole number of steps. It converts a `LevelEdge.x`
    /// to pixels along the scan: index `x` sits `x * spacing` from the first sample.
    pub fn spacing(&self) -> f32 {
        self.inner.spacing()
    }

    /// Measure and keep every intermediate — see [`CaliperTrace`]. Never raises
    /// `MeasureRejected`: a rejection is the trace's `reject`.
    pub fn explain(&mut self, py: Python<'_>, img: &Bound<'_, PyAny>) -> PyResult<CaliperTrace> {
        let any = any_image_from_numpy(py, img)?;
        let trace = with_any_image!(any, view => native_explain(&mut self.inner, &view));
        Ok(CaliperTrace::from_native(py, trace))
    }

    /// The level crossings behind the last `measure` call's edges: one for
    /// `Locate.midpoint_crossing`, one per edge for `Locate.half_contrast`, none for
    /// `Locate.gradient_peak`. `x` is in profile samples.
    pub fn levels(&self) -> Vec<LevelEdge> {
        self.inner
            .levels()
            .iter()
            .copied()
            .map(LevelEdge::from)
            .collect()
    }
}

/// Everything one caliper measurement computed — mirrors
/// `vision_metrology::measure::diagnostics::CaliperTrace`.
///
/// `profile`, `smoothed` and `response` are `float32` arrays of `samples` entries;
/// `candidates` are the edges that passed threshold, polarity and the obliquity gate,
/// before `select`; `edges` is what `measure` returned (empty on a rejection) and
/// `reject` the reason string `measure` would have raised, or `None`.
#[pyclass(get_all)]
pub struct CaliperTrace {
    /// Distance between profile samples, in pixels, as `Caliper.spacing()` reports it.
    pub spacing: f32,
    pub samples: usize,
    pub across: usize,
    pub threshold: f32,
    pub profile: Py<PyArray1<f32>>,
    pub smoothed: Py<PyArray1<f32>>,
    pub response: Py<PyArray1<f32>>,
    pub candidates: Vec<MeasureEdge>,
    pub levels: Vec<LevelEdge>,
    pub edges: Vec<MeasureEdge>,
    pub reject: Option<&'static str>,
}

impl CaliperTrace {
    pub(super) fn from_native(py: Python<'_>, trace: NativeCaliperTrace) -> Self {
        Self {
            spacing: trace.spacing,
            samples: trace.samples,
            across: trace.across,
            threshold: trace.threshold,
            profile: trace.profile.into_pyarray(py).unbind(),
            smoothed: trace.smoothed.into_pyarray(py).unbind(),
            response: trace.response.into_pyarray(py).unbind(),
            candidates: trace
                .candidates
                .into_iter()
                .map(MeasureEdge::from)
                .collect(),
            levels: trace.levels.into_iter().map(LevelEdge::from).collect(),
            edges: trace.edges.into_iter().map(MeasureEdge::from).collect(),
            reject: trace.reject.map(reject_reason_str),
        }
    }
}

#[pymethods]
impl CaliperTrace {
    fn __repr__(&self) -> String {
        format!(
            "CaliperTrace(samples={}, candidates={}, edges={}, reject={:?})",
            self.samples,
            self.candidates.len(),
            self.edges.len(),
            self.reject
        )
    }
}
