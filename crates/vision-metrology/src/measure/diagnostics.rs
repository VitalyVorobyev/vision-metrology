//! Caliper layout — where a [`MetrologyModel`]'s calipers sit at a fixture pose,
//! without measuring.
//!
//! An overlay (the lab's Measure tab, or any other UI drawing caliper boxes over
//! an image) needs exactly the geometry [`MetrologyModel::apply`] computes
//! internally to place each [`Caliper`](super::Caliper) — and needs it to draw a
//! caliper *before* an image is even available, or for calipers that never found
//! an edge. Duplicating that placement math at the call site (as the lab's Python
//! backend used to) risks silently drifting from the actual measurement whenever
//! [`MetrologyModel`]'s own placement changes. [`layout`] is the shared source:
//! [`MetrologyModel::apply`] and this function both call the same private
//! placement code (`model::caliper_placements`), so the two can never disagree.

use vm_primitives::Similarity2f;

pub use super::model::CaliperShape;
use super::model::caliper_placements;
use super::{MetrologyModel, MetrologyObject};

/// One caliper's placement, addressed by which object and which caliper within
/// it produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaliperPlacement {
    /// Index into [`MetrologyModel::objects`] — matches [`MetrologyModel::apply`]'s
    /// result order.
    pub object_index: usize,
    /// Index of this caliper within its object, `0..object.n_calipers`.
    pub caliper_index: usize,
    /// The placed geometry — a rectangle (line objects) or a radial caliper
    /// (circle objects).
    pub shape: CaliperShape,
}

/// Every caliper placement for `model`'s objects, mapped through `fixture`.
///
/// `fixture` is normally [`ShapeMatch::pose`](crate::matching::ShapeMatch) —
/// the same value [`MetrologyModel::apply`] takes. An object whose placement
/// cannot be computed (fewer than 2 calipers, or a degenerate zero-length
/// line) contributes no entries rather than failing the whole call — same
/// reasoning as [`MetrologyModel::apply`] reporting per-object results, just
/// without an error channel here since layout has nothing to attach one to.
///
/// # Example
/// ```
/// use vision_metrology::measure::diagnostics::{layout, CaliperShape};
/// use vision_metrology::measure::{MetrologyModel, MetrologyObject, MetrologyShape};
/// use vision_metrology::{Point2f, Similarity2f};
///
/// let mut model = MetrologyModel::new();
/// model.add(MetrologyObject::new(MetrologyShape::Circle {
///     center: Point2f::new(0.0, 0.0),
///     radius: 20.0,
///     arc: None,
/// }));
///
/// let placements = layout(&model, &Similarity2f::identity());
/// assert_eq!(placements.len(), 32); // MetrologyObject::new's default n_calipers
/// assert!(matches!(placements[0].shape, CaliperShape::Radial(_)));
/// ```
pub fn layout(model: &MetrologyModel, fixture: &Similarity2f) -> Vec<CaliperPlacement> {
    let mut out = Vec::new();
    for (object_index, obj) in model.objects().iter().enumerate() {
        if let Ok(placements) = caliper_placements(obj, fixture) {
            out.extend(
                placements
                    .into_iter()
                    .enumerate()
                    .map(|(caliper_index, shape)| CaliperPlacement {
                        object_index,
                        caliper_index,
                        shape,
                    }),
            );
        }
    }
    out
}

/// [`layout`] for a single object not yet added to a [`MetrologyModel`] — the
/// lab calls this while a model is still being edited, before any object has
/// been committed.
pub fn layout_object(obj: &MetrologyObject, fixture: &Similarity2f) -> Vec<CaliperShape> {
    caliper_placements(obj, fixture).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{CaliperShape, layout};
    use crate::measure::{MetrologyModel, MetrologyObject, MetrologyShape};
    use vm_primitives::{Point2f, Similarity2f, Vec2f};

    #[test]
    fn line_object_places_rect_calipers_along_the_segment() {
        let mut model = MetrologyModel::new();
        let mut obj = MetrologyObject::new(MetrologyShape::Line {
            a: Point2f::new(0.0, 0.0),
            b: Point2f::new(10.0, 0.0),
        });
        obj.n_calipers = 3;
        model.add(obj);

        let placements = layout(&model, &Similarity2f::identity());
        assert_eq!(placements.len(), 3);
        for (i, p) in placements.iter().enumerate() {
            assert_eq!(p.object_index, 0);
            assert_eq!(p.caliper_index, i);
            let CaliperShape::Rect(r) = p.shape else {
                panic!("expected a rect placement")
            };
            // Hand-computed: calipers at f = 0, 0.5, 1 along (0,0)-(10,0).
            let expected_x = 5.0 * i as f32;
            assert!((r.center.x - expected_x).abs() < 1e-4, "x = {}", r.center.x);
            assert!(r.center.y.abs() < 1e-4);
            // Scan axis is perpendicular to the segment: angle = +-pi/2.
            assert!((r.angle.abs() - core::f32::consts::FRAC_PI_2).abs() < 1e-4);
        }
    }

    #[test]
    fn circle_object_places_radial_calipers_around_it() {
        let mut model = MetrologyModel::new();
        let mut obj = MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(50.0, 50.0),
            radius: 20.0,
            arc: None,
        });
        obj.n_calipers = 4;
        model.add(obj);

        let placements = layout(&model, &Similarity2f::identity());
        assert_eq!(placements.len(), 4);
        for (i, p) in placements.iter().enumerate() {
            let CaliperShape::Radial(r) = p.shape else {
                panic!("expected a radial placement")
            };
            // `MeasureRadial::center` is the circle's own centre for every
            // caliper — the geometry the current overlay convention draws.
            assert!((r.center - Point2f::new(50.0, 50.0)).norm() < 1e-4);
            assert!((r.radius - 20.0).abs() < 1e-4);
            let expected_angle = core::f32::consts::TAU * i as f32 / 4.0;
            assert!((r.angle - expected_angle).abs() < 1e-4);
        }
    }

    #[test]
    fn a_fixture_translates_scales_and_rotates_the_layout() {
        let mut model = MetrologyModel::new();
        let mut obj = MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(0.0, 0.0),
            radius: 10.0,
            arc: None,
        });
        obj.n_calipers = 4;
        model.add(obj);

        let fixture = Similarity2f::new(Vec2f::new(100.0, 200.0), 0.0, 2.0);
        let placements = layout(&model, &fixture);
        let CaliperShape::Radial(r) = placements[0].shape else {
            panic!("expected a radial placement")
        };
        assert!((r.center - Point2f::new(100.0, 200.0)).norm() < 1e-4);
        assert!((r.radius - 20.0).abs() < 1e-4, "r = {}", r.radius);
    }

    #[test]
    fn an_unmeasurable_object_contributes_no_placements() {
        let mut model = MetrologyModel::new();
        let mut obj = MetrologyObject::new(MetrologyShape::Circle {
            center: Point2f::new(0.0, 0.0),
            radius: 10.0,
            arc: None,
        });
        obj.n_calipers = 1; // invalid: needs >= 2
        model.add(obj);

        let placements = layout(&model, &Similarity2f::identity());
        assert!(placements.is_empty());
    }
}
