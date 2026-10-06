//! Python bindings for `measure`: calipers and the metrology model.
//!
//! ## `RejectReason`
//!
//! `Caliper::measure` returns `Result<&[MeasureEdge], RejectReason>` on the
//! Rust side; its docs explain why an empty result is unrepresentable. The Python mirror raises [`MeasureRejected`],
//! a plain exception whose single argument is one of the eight reason strings
//! (`"profile_too_short"`, `"no_edge"`, `"wrong_polarity"`, `"too_oblique"`,
//! `"off_image"`, `"incomplete_sequence"`, `"low_contrast"`, `"no_crossing"`):
//! `except vm.MeasureRejected as e: reason = e.args[0]`. This is the ordinary
//! Python idiom for "this call has a well-defined failure mode", and it composes
//! with `try`/`except` instead of asking every caller to unwrap a tagged result
//! by hand.
//!
//! `MetrologyModel.apply` is different: it always returns one entry per
//! object, in order, and an exception on the first failure would discard the
//! others (that discarding is exactly what the Rust side's `Vec<Result<_>>`
//! return type was chosen to prevent). Its Python mirror instead returns a
//! list where each entry is either a [`MetrologyResult`] or a
//! [`MetrologyError`] carrying the failure message — a small tagged union in
//! place of an exception, because the caller needs all the entries, not just
//! the first problem.

// Reported residuals and rejections: invariant 21.

use std::num::NonZeroUsize;

use pyo3::prelude::*;
use pyo3::{
    create_exception,
    exceptions::{PyException, PyValueError},
};
use vision_metrology::measure::{
    MeasureArc, MeasureRadial, MeasureRect, MeasureStrip, RejectReason as NativeRejectReason,
};
use vm_primitives::{Point2f, Similarity2f, Vec2f, similarity_from_parts, wrap_angle};

mod caliper;
mod model;

pub use caliper::{Caliper, CaliperTrace};
pub use model::{
    CaliperPlacement, MetrologyError, MetrologyModel, MetrologyObject, MetrologyResult,
    MetrologyShape, ObjectTrace,
};

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
        NativeRejectReason::IncompleteSequence => "incomplete_sequence",
        NativeRejectReason::LowContrast => "low_contrast",
        NativeRejectReason::NoCrossing => "no_crossing",
    }
}

fn rect_from(center: (f32, f32), angle: f32, half_len: f32, half_width: f32) -> MeasureRect {
    MeasureRect {
        center: Point2f::new(center.0, center.1),
        angle,
        half_len,
        half_width,
    }
}

fn arc_from(
    center: (f32, f32),
    radius: f32,
    angle_start: f32,
    angle_extent: f32,
    half_width: f32,
) -> MeasureArc {
    MeasureArc {
        center: Point2f::new(center.0, center.1),
        radius,
        angle_start,
        angle_extent,
        half_width,
    }
}

fn radial_from(
    center: (f32, f32),
    radius: f32,
    angle: f32,
    half_len: f32,
    half_width: f32,
) -> MeasureRadial {
    MeasureRadial {
        center: Point2f::new(center.0, center.1),
        radius,
        angle,
        half_len,
        half_width,
    }
}

/// A strip; `samples` and `across` must be at least 1 when given.
fn strip_from(
    start: (f32, f32),
    end: (f32, f32),
    half_width: f32,
    samples: Option<usize>,
    across: Option<usize>,
) -> PyResult<MeasureStrip> {
    let count = |name: &str, v: Option<usize>| match v {
        None => Ok(None),
        Some(n) => NonZeroUsize::new(n)
            .map(Some)
            .ok_or_else(|| PyValueError::new_err(format!("{name} must be at least 1"))),
    };
    Ok(MeasureStrip {
        start: Point2f::new(start.0, start.1),
        end: Point2f::new(end.0, end.1),
        half_width,
        samples: count("samples", samples)?,
        across: count("across", across)?,
    })
}
