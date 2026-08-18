//! Coarse-to-fine shape search.

use vm_primitives::{
    DirectionField, ImageView, Point2f, PyramidF32, Similarity2f, Vec2f, similarity_from_parts,
    wrap_angle,
};

use super::config::{Polarity, Refinement, ShapeSearchConfig};
use super::model::{ModelPoint, ShapeModel};
use super::nms::{InstanceMask, suppress};
use super::refine::{Pose, interpolate, least_squares};
use super::score::{Bound, RotPoint, rotate_into, score_at, score_pose, score_span};
use super::search::{Candidate, Grid, Span, cap_candidates, collect_local_maxima};

/// One located instance of a [`ShapeModel`].
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeMatch {
    /// Maps reference-image coordinates to scene coordinates.
    ///
    /// The model origin is baked in, so a point measured in the reference image
    /// — a fiducial, a measurement ROI corner — maps into the scene with
    /// `pose * p` and no offset bookkeeping:
    /// `Translation(position) ∘ sR ∘ Translation(−model.origin())`.
    pub pose: Similarity2f,
    /// Where the model's reference point landed, in level-0 image coordinates.
    pub position: Point2f,
    /// Similarity score in `[0, 1]`; roughly `1 − occluded_fraction`.
    pub score: f32,
    /// Model points that found any gradient at all.
    ///
    /// `support / point_count` and `score` differ when edges are present but
    /// misoriented, which is the signature of a wrong pose on a busy scene.
    pub support: usize,
    /// Pyramid level the reported score was evaluated at.
    pub level: usize,
}

impl ShapeMatch {
    /// Rotation in radians, wrapped to `(-π, π]`.
    #[inline]
    pub fn angle(&self) -> f32 {
        self.pose.isometry.rotation.angle()
    }

    /// Uniform scale factor.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.pose.scaling()
    }
}

/// Score maps for a rolling three-angle window.
#[derive(Debug, Default)]
struct MapBuffers {
    prev: Vec<f32>,
    cur: Vec<f32>,
    next: Vec<f32>,
    first: Vec<f32>,
}

impl MapBuffers {
    fn resize(&mut self, n: usize) {
        for b in [
            &mut self.prev,
            &mut self.cur,
            &mut self.next,
            &mut self.first,
        ] {
            b.clear();
            b.resize(n, f32::NEG_INFINITY);
        }
    }
}

/// Reusable coarse-to-fine shape matcher.
///
/// Owns the scene pyramid, one [`DirectionField`] per level, and every scratch
/// buffer the search needs. The only allocation per call is the returned
/// `Vec<ShapeMatch>`.
///
/// # Example
/// ```no_run
/// use vision_metrology::{Image, ShapeMatcher, ShapeModel, ShapeSearchConfig};
///
/// # fn run(model: &ShapeModel) {
/// let scene: Image<u8> = Image::new_fill(1280, 1024, 0);
/// let mut matcher = ShapeMatcher::new();
/// let cfg = ShapeSearchConfig { min_score: 0.6, max_matches: 4, ..Default::default() };
///
/// for m in matcher.find_u8(&scene.as_view(), model, &cfg) {
///     println!("{:?} at {:.1} deg, score {:.2}", m.position, m.angle().to_degrees(), m.score);
/// }
/// # }
/// ```
#[derive(Debug, Default)]
pub struct ShapeMatcher {
    pyr: PyramidF32,
    fields: Vec<DirectionField>,
    rot: Vec<RotPoint>,
    cands: Vec<Candidate>,
    next: Vec<Candidate>,
    maps: MapBuffers,
    mask: InstanceMask,
    nms_scratch: Vec<(i32, i32)>,
    truncated: bool,
}

impl ShapeMatcher {
    /// Create a matcher with empty scratch buffers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the last search hit `max_candidates` and discarded candidates.
    ///
    /// A truncated search can miss the true instance, so this distinguishes
    /// "not present" from "gave up" — worth surfacing when a search on a
    /// cluttered scene comes back empty.
    #[inline]
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// Find instances of `model` in a `u8` scene.
    pub fn find_u8(
        &mut self,
        img: &ImageView<'_, u8>,
        model: &ShapeModel,
        cfg: &ShapeSearchConfig,
    ) -> Vec<ShapeMatch> {
        self.pyr.build_from_u8(img, model.num_levels());
        self.run(model, cfg)
    }

    /// Find instances of `model` in a `u16` scene.
    ///
    /// Remember that [`ShapeSearchConfig::min_contrast`] is on the input pixel
    /// scale and needs raising for 16-bit data.
    pub fn find_u16(
        &mut self,
        img: &ImageView<'_, u16>,
        model: &ShapeModel,
        cfg: &ShapeSearchConfig,
    ) -> Vec<ShapeMatch> {
        self.pyr.build_from_u16(img, model.num_levels());
        self.run(model, cfg)
    }

    /// Find instances of `model` in an `f32` scene.
    pub fn find_f32(
        &mut self,
        img: &ImageView<'_, f32>,
        model: &ShapeModel,
        cfg: &ShapeSearchConfig,
    ) -> Vec<ShapeMatch> {
        self.pyr.build_from_f32(img, model.num_levels());
        self.run(model, cfg)
    }

    fn run(&mut self, model: &ShapeModel, cfg: &ShapeSearchConfig) -> Vec<ShapeMatch> {
        self.truncated = false;
        let n_lv = model.num_levels().min(self.pyr.num_levels());
        if n_lv == 0 || cfg.min_score > 1.0 {
            return Vec::new();
        }

        if self.fields.len() < n_lv {
            self.fields.resize_with(n_lv, DirectionField::new);
        }
        for level in 0..n_lv {
            let img = self.pyr.level(level).expect("level < n_lv");
            self.fields[level].build_image_f32(img, model.smooth(), cfg.min_contrast);
        }

        let top = n_lv - 1;
        let last = cfg.last_level.min(top);
        let angles = intersect(
            cfg.angle_range.unwrap_or(model.angle_range()),
            model.angle_range(),
        );
        let scales = intersect(
            cfg.scale_range.unwrap_or(model.scale_range()),
            model.scale_range(),
        );

        let Self {
            fields,
            rot,
            cands,
            next,
            maps,
            mask,
            nms_scratch,
            truncated,
            ..
        } = self;
        let polarity = model.polarity();

        // ---- exhaustive top level -----------------------------------------
        let field = &fields[top];
        let span = Span::for_level(cfg.roi, top, field.width(), field.height());
        if span.is_empty() {
            return Vec::new();
        }
        let lvl = model.level(top).expect("top < num_levels");
        let coarse = if top > last {
            cfg.min_score * cfg.coarse_score_factor
        } else {
            cfg.min_score
        };
        let bound = Bound::new(lvl.points.len(), cfg.greediness, coarse);
        let ang = Grid::angles(angles, step_at(cfg.angle_step, lvl.angle_step, top));
        let sca = Grid::scales(scales, step_at(cfg.scale_step, lvl.scale_step, top));

        cands.clear();
        maps.resize(span.width() * span.height());
        for si in 0..sca.len() {
            sweep_angles(
                field,
                &lvl.points,
                span,
                &ang,
                sca.scale_at(si),
                polarity,
                bound,
                coarse,
                rot,
                maps,
                cands,
            );
        }
        *truncated = cap_candidates(cands, cfg.max_candidates);

        // ---- propagate down ------------------------------------------------
        for level in (last..top).rev() {
            let field = &fields[level];
            let lvl = model.level(level).expect("level < num_levels");
            let span = Span::for_level(cfg.roi, level, field.width(), field.height());
            let thresh = if level > last {
                cfg.min_score * cfg.coarse_score_factor
            } else {
                cfg.min_score
            };
            let bound = Bound::new(lvl.points.len(), cfg.greediness, thresh);
            next.clear();
            for cand in cands.iter() {
                if let Some(best) = refine_candidate(
                    field,
                    &lvl.points,
                    span,
                    *cand,
                    step_at(cfg.angle_step, lvl.angle_step, level),
                    step_at(cfg.scale_step, lvl.scale_step, level),
                    angles,
                    scales,
                    polarity,
                    bound,
                    rot,
                ) && best.score >= thresh
                {
                    next.push(best);
                }
            }
            core::mem::swap(cands, next);
        }

        // ---- refine, score, suppress ---------------------------------------
        let field = &fields[last];
        let lvl = model.level(last).expect("last < num_levels");
        let up = (1usize << last) as f32;
        let shift = 0.5 * (up - 1.0);

        let mut out: Vec<ShapeMatch> = Vec::new();
        for cand in cands.iter() {
            let mut pose = Pose {
                x: cand.x as f32,
                y: cand.y as f32,
                angle: cand.angle,
                scale: cand.scale,
            };
            if cfg.refinement != Refinement::None {
                pose = interpolate(
                    field,
                    &lvl.points,
                    pose,
                    step_at(cfg.angle_step, lvl.angle_step, last),
                    step_at(cfg.scale_step, lvl.scale_step, last),
                    polarity,
                    rot,
                );
            }
            if cfg.refinement == Refinement::LeastSquares {
                pose = least_squares(
                    field,
                    &lvl.points,
                    pose,
                    lvl.radius,
                    cfg.min_contrast,
                    polarity,
                );
            }

            let (score, support) = score_pose(
                field,
                &lvl.points,
                pose.x,
                pose.y,
                pose.angle,
                pose.scale,
                polarity,
            );
            if score < cfg.min_score {
                continue;
            }

            let position = Point2f {
                x: up * pose.x + shift,
                y: up * pose.y + shift,
            };
            out.push(ShapeMatch {
                pose: pose_from(position, pose.angle, pose.scale, model.origin()),
                position,
                score,
                support,
                level: last,
            });
        }

        let (w0, h0) = (fields[0].width(), fields[0].height());
        suppress(
            mask,
            (w0, h0),
            &model.level(0).expect("level 0 exists").points,
            out,
            cfg.max_overlap,
            cfg.max_matches,
            nms_scratch,
        )
    }
}

/// `Translation(position) ∘ sR ∘ Translation(−origin)`, as a similarity.
fn pose_from(position: Point2f, angle: f32, scale: f32, origin: Point2f) -> Similarity2f {
    let (sn, cs) = wrap_angle(angle).sin_cos();
    let t = Vec2f {
        x: position.x - scale * (cs * origin.x - sn * origin.y),
        y: position.y - scale * (sn * origin.x + cs * origin.y),
    };
    similarity_from_parts(t, wrap_angle(angle), scale)
}

/// A level-0 step scaled to level `l`, or the model's own derived step.
///
/// The model radius halves per level, so one pixel of tip motion corresponds to
/// twice the angle: an explicitly configured level-0 step doubles per level.
#[inline]
fn step_at(configured: f32, derived: f32, level: usize) -> f32 {
    if configured > 0.0 {
        configured * (1usize << level) as f32
    } else {
        derived
    }
}

/// Narrow `want` to what `allowed` permits, keeping the result ordered.
fn intersect(want: (f32, f32), allowed: (f32, f32)) -> (f32, f32) {
    let lo = want.0.max(allowed.0);
    let hi = want.1.min(allowed.1);
    if hi < lo { (lo, lo) } else { (lo, hi) }
}

#[allow(clippy::too_many_arguments)]
fn sweep_angles(
    field: &DirectionField,
    points: &[ModelPoint],
    span: Span,
    ang: &Grid,
    scale: f32,
    polarity: Polarity,
    bound: Bound,
    threshold: f32,
    rot: &mut Vec<RotPoint>,
    maps: &mut MapBuffers,
    out: &mut Vec<Candidate>,
) {
    let n = ang.len();
    // The angle dimension of the local-maximum test needs the map before and
    // after the one being tested. For a full circle the ring is closed by
    // evaluating the last angle up front and keeping a copy of the first.
    if ang.cyclic && n > 2 {
        fill_map(
            field,
            points,
            ang.angle_at(n - 1),
            scale,
            span,
            polarity,
            bound,
            rot,
            &mut maps.prev,
        );
    } else {
        maps.prev.fill(f32::NEG_INFINITY);
    }
    fill_map(
        field,
        points,
        ang.angle_at(0),
        scale,
        span,
        polarity,
        bound,
        rot,
        &mut maps.cur,
    );
    if ang.cyclic && n > 2 {
        maps.first.copy_from_slice(&maps.cur);
    }

    for ai in 0..n {
        if ai + 1 < n {
            fill_map(
                field,
                points,
                ang.angle_at(ai + 1),
                scale,
                span,
                polarity,
                bound,
                rot,
                &mut maps.next,
            );
        } else if ang.cyclic && n > 2 {
            maps.next.copy_from_slice(&maps.first);
        } else {
            maps.next.fill(f32::NEG_INFINITY);
        }

        collect_local_maxima(
            &maps.prev,
            &maps.cur,
            &maps.next,
            span,
            ang.angle_at(ai),
            scale,
            threshold,
            out,
        );

        core::mem::swap(&mut maps.prev, &mut maps.cur);
        core::mem::swap(&mut maps.cur, &mut maps.next);
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_map(
    field: &DirectionField,
    points: &[ModelPoint],
    angle: f32,
    scale: f32,
    span: Span,
    polarity: Polarity,
    bound: Bound,
    rot: &mut Vec<RotPoint>,
    buf: &mut [f32],
) {
    rotate_into(points, angle, scale, rot);
    let mw = span.width();
    for iy in 0..span.height() {
        let row = &mut buf[iy * mw..(iy + 1) * mw];
        score_span(
            field,
            rot,
            span.y0 + iy as i32,
            span.x0,
            polarity,
            bound,
            row,
        );
    }
}

/// Track one candidate from level `l+1` into level `l`.
///
/// The position window is 5×5 because two independent errors stack: the halving
/// itself is ambiguous by ±1, and the coarse localisation is good to about ±1.
/// Angle and scale get 5 steps each for the same reason — the model radius
/// doubles when descending a level, so the coarse winner lands within ±1 of the
/// fine step, and 5 leaves a margin for noise.
#[allow(clippy::too_many_arguments)]
fn refine_candidate(
    field: &DirectionField,
    points: &[ModelPoint],
    span: Span,
    cand: Candidate,
    angle_step: f32,
    scale_step: f32,
    angles: (f32, f32),
    scales: (f32, f32),
    polarity: Polarity,
    bound: Bound,
    rot: &mut Vec<RotPoint>,
) -> Option<Candidate> {
    let (bx, by) = (2 * cand.x, 2 * cand.y);
    let mut best: Option<Candidate> = None;

    for si in -2i32..=2 {
        let scale = cand.scale * (1.0 + scale_step).powi(si);
        if scale < scales.0 * 0.99 || scale > scales.1 * 1.01 {
            continue;
        }
        for ai in -2i32..=2 {
            let angle = cand.angle + ai as f32 * angle_step;
            if angle < angles.0 - angle_step || angle > angles.1 + angle_step {
                continue;
            }
            rotate_into(points, angle, scale, rot);
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let (qx, qy) = (bx + dx, by + dy);
                    if !span.contains(qx, qy) {
                        continue;
                    }
                    let Some(score) = score_at(field, rot, qx, qy, polarity, bound) else {
                        continue;
                    };
                    if best.is_none_or(|b| score > b.score) {
                        best = Some(Candidate {
                            x: qx,
                            y: qy,
                            angle,
                            scale,
                            score,
                        });
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{intersect, pose_from, step_at};
    use vm_primitives::Point2f;

    #[test]
    fn pose_maps_the_model_origin_onto_the_reported_position() {
        let origin = Point2f { x: 30.0, y: -12.0 };
        let position = Point2f { x: 640.0, y: 512.0 };
        let pose = pose_from(position, 0.7, 1.3, origin);
        let q = pose * nalgebra::Point2::new(origin.x, origin.y);
        assert!((q.x - position.x).abs() < 1e-3, "{q}");
        assert!((q.y - position.y).abs() < 1e-3, "{q}");
    }

    #[test]
    fn pose_carries_the_requested_angle_and_scale() {
        let pose = pose_from(Point2f { x: 5.0, y: 5.0 }, -0.4, 0.75, Point2f::default());
        assert!((pose.isometry.rotation.angle() + 0.4).abs() < 1e-5);
        assert!((pose.scaling() - 0.75).abs() < 1e-5);
    }

    #[test]
    fn step_scaling_follows_the_halving_radius() {
        assert!((step_at(0.01, 9.9, 0) - 0.01).abs() < 1e-9);
        assert!((step_at(0.01, 9.9, 3) - 0.08).abs() < 1e-9);
        // Zero means "use the model's own derived step".
        assert!((step_at(0.0, 9.9, 3) - 9.9).abs() < 1e-9);
    }

    #[test]
    fn range_intersection_never_widens_and_never_inverts() {
        assert_eq!(intersect((-1.0, 1.0), (-0.5, 0.5)), (-0.5, 0.5));
        assert_eq!(intersect((0.1, 0.2), (-1.0, 1.0)), (0.1, 0.2));
        // Disjoint ranges collapse to a single value rather than inverting.
        let r = intersect((2.0, 3.0), (-1.0, 1.0));
        assert!(r.0 <= r.1);
    }
}
