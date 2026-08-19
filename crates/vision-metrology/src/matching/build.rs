//! Shape model creation, from a reference image or from geometry alone.

use vm_primitives::{
    Edge2DDetector, Edgel, Error, ImageView, Pixel, Point2f, Polyline2f, Pyramid, Rect2f,
    SmoothKind, Vec2f, Vec2fExt,
};

use super::config::ShapeModelConfig;
use super::model::{ModelPoint, ShapeModel, ShapeModelLevel};

/// Hard ceiling on automatically chosen pyramid levels.
const MAX_AUTO_LEVELS: usize = 5;
/// Hard ceiling on explicitly requested pyramid levels.
const MAX_LEVELS: usize = 8;
/// A level with a smaller radius than this cannot be searched meaningfully:
/// one angle step would sweep the whole model past itself.
const MIN_LEVEL_RADIUS: f32 = 6.0;
/// Level-0 pixels of context added around the ROI so that coarse downsamples
/// see real image content rather than a hard cut.
const ROI_MARGIN: usize = 8;
/// Points closer than this to the extracted sub-image border are boundary
/// artefacts of the crop, not object edges.
const BORDER_DROP: f32 = 2.0;
/// Averaged directions shorter than this inside one coarse cell are
/// contradictory — the cell straddles two opposite edges of a thin feature,
/// exactly where a box-downsampled scene also loses the edge.
const MIN_MEAN_DIR: f32 = 0.3;

/// Which way the normals of a geometry-only contour model point.
///
/// The choice is *geometric*, not photometric: [`Outward`](Self::Outward) means
/// away from the region the contour encloses. Whether that coincides with the
/// dark-to-bright direction depends on whether the part is darker or brighter
/// than its background — which a contour does not record. See
/// [`ShapeModel::from_polylines`] for why this usually means you want
/// [`Polarity::IgnoreLocal`](super::Polarity::IgnoreLocal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContourOrientation {
    /// Normals point away from the enclosed region.
    #[default]
    Outward,
    /// Normals point into the enclosed region.
    Inward,
}

/// A point as extracted, before it is made relative to the reference point.
#[derive(Debug, Clone, Copy)]
struct RawPoint {
    /// Position in this level's coordinate frame.
    p: Point2f,
    /// Unit gradient direction.
    t: Vec2f,
    /// Gradient magnitude, used only to pick a survivor during decimation.
    strength: f32,
}

/// Level-`l` coordinate of a level-0 coordinate.
///
/// The 2×2 box downsample maps level-`l` pixel `c` to the level-0 interval
/// `[2^l·c, 2^l·(c+1))`, whose centre is `2^l·c + (2^l − 1)/2`. Inverting that
/// gives the shift below. Offsets between two points at the same level scale by
/// a clean `2^l` because the shift cancels.
#[inline]
fn to_level(v: f32, level: usize) -> f32 {
    let s = (1usize << level) as f32;
    (v - 0.5 * (s - 1.0)) / s
}

/// Reusable builder for [`ShapeModel`]s.
///
/// Owns the reference-image pyramid and the edge detector, so building many
/// models in a row allocates only the models themselves.
///
/// # Example
/// ```no_run
/// use vision_metrology::{Image, Rect2f};
/// use vision_metrology::matching::{ShapeModelBuilder, ShapeModelConfig};
///
/// # fn main() -> Result<(), vision_metrology::Error> {
/// let img: Image<u8> = Image::new_fill(640, 480, 0);
/// let roi = Rect2f { x: 220.0, y: 160.0, width: 200.0, height: 160.0 };
///
/// let mut builder = ShapeModelBuilder::new();
/// let model = builder.build(&img.as_view(), roi, &ShapeModelConfig::default())?;
/// println!("{} levels, {} points at level 0", model.num_levels(), model.point_count(0));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct ShapeModelBuilder {
    pyr: Pyramid,
    det: Edge2DDetector,
}

impl ShapeModelBuilder {
    /// Create a builder with empty scratch buffers.
    pub fn new() -> Self {
        Self {
            pyr: Pyramid::new(),
            det: Edge2DDetector::new(),
        }
    }

    /// Build a model from a reference image of any [`Pixel`] type and a
    /// rectangular ROI.
    ///
    /// The ROI's own pyramid is what the model points are detected on — see
    /// system-design invariant 3: model and scene must suffer identical
    /// box-downsample aliasing.
    ///
    /// # Errors
    /// - [`Error::InvalidConfig`] for a non-positive ROI, a reversed or
    ///   non-positive `scale_range`, or a reversed `angle_range`.
    /// - [`Error::OutOfBounds`] when the ROI does not lie inside the image.
    /// - [`Error::InsufficientData`] when too few edge points survive.
    pub fn build<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        roi: Rect2f,
        cfg: &ShapeModelConfig,
    ) -> Result<ShapeModel, Error> {
        let crop = validate(img.width(), img.height(), roi, cfg)?;
        let sub = img.subview(crop.x0, crop.y0, crop.w, crop.h)?;
        self.pyr.build(&sub, crop.levels);
        self.finish(crop, roi, cfg)
    }

    fn finish(
        &mut self,
        crop: Crop,
        roi: Rect2f,
        cfg: &ShapeModelConfig,
    ) -> Result<ShapeModel, Error> {
        let built = self.pyr.num_levels();
        let mut raw: Vec<Vec<RawPoint>> = Vec::with_capacity(built);

        for level in 0..built {
            let img = self.pyr.level(level).expect("level < num_levels");
            let (lw, lh) = (img.width(), img.height());
            let edgels = self.det.detect(&img.as_view(), &cfg.edge);

            // Sub-image level-l coordinates translate to full-image level-l
            // coordinates by a pure shift, because the crop origin was aligned
            // to a multiple of 2^(levels-1).
            let shift = (
                crop.x0 as f32 / (1usize << level) as f32,
                crop.y0 as f32 / (1usize << level) as f32,
            );
            let roi_l = Rect2f {
                x: to_level(roi.x, level),
                y: to_level(roi.y, level),
                width: roi.width / (1usize << level) as f32,
                height: roi.height / (1usize << level) as f32,
            };

            let mut pts = Vec::with_capacity(edgels.len());
            for e in &edgels {
                if e.strength < cfg.min_contrast {
                    continue;
                }
                // Reject the crop's own border, where the Scharr kernel
                // clamps and manufactures an edge that is not on the object.
                if e.p.x < BORDER_DROP
                    || e.p.y < BORDER_DROP
                    || e.p.x > lw as f32 - 1.0 - BORDER_DROP
                    || e.p.y > lh as f32 - 1.0 - BORDER_DROP
                {
                    continue;
                }
                let p = Point2f::new(e.p.x + shift.0, e.p.y + shift.1);
                if !roi_l.contains(p) {
                    continue;
                }
                pts.push(RawPoint {
                    p,
                    t: e.n,
                    strength: e.strength,
                });
            }
            raw.push(pts);
        }

        let origin = match cfg.origin {
            Some(o) => o,
            None => centroid(raw.first().map_or(&[][..], Vec::as_slice)).ok_or(
                Error::InsufficientData {
                    need: cfg.min_points_per_level.max(1),
                    got: 0,
                },
            )?,
        };

        assemble(raw, origin, cfg)
    }
}

impl ShapeModel {
    /// Build a model from edgels already extracted from a reference image.
    ///
    /// [`Edgel::n`] already carries the photometric dark-to-bright direction, so
    /// this path is compatible with [`Polarity::Match`](super::Polarity::Match).
    ///
    /// # Errors
    /// [`Error::InsufficientData`] when fewer than `min_points_per_level` edgels
    /// survive, [`Error::InvalidConfig`] for an invalid angle or scale range.
    pub fn from_edgels(edgels: &[Edgel], cfg: &ShapeModelConfig) -> Result<Self, Error> {
        let pts: Vec<RawPoint> = edgels
            .iter()
            .filter(|e| e.strength >= cfg.min_contrast)
            .map(|e| RawPoint {
                p: e.p,
                t: e.n,
                strength: e.strength,
            })
            .collect();
        from_raw(pts, cfg)
    }

    /// Build a model from explicit positions and unit directions.
    ///
    /// The intended source is CAD geometry or a synthetic fixture. Directions
    /// need not be normalised; zero-length directions are dropped.
    ///
    /// # Errors
    /// [`Error::SizeMismatch`] when the two slices differ in length,
    /// [`Error::InsufficientData`] when too few points survive.
    pub fn from_directed_points(
        points: &[Point2f],
        dirs: &[Vec2f],
        cfg: &ShapeModelConfig,
    ) -> Result<Self, Error> {
        if points.len() != dirs.len() {
            return Err(Error::SizeMismatch {
                expected: points.len(),
                actual: dirs.len(),
            });
        }
        let pts: Vec<RawPoint> = points
            .iter()
            .zip(dirs)
            .filter_map(|(&p, &d)| {
                let t = d.normalized_or_zero();
                (t.norm() > 0.5).then_some(RawPoint {
                    p,
                    t,
                    strength: 1.0,
                })
            })
            .collect();
        from_raw(pts, cfg)
    }

    /// Build a model from contour polylines.
    ///
    /// Tangents come from the same symmetric finite difference
    /// `contour::GraphEdge::compute_geometry` uses, and the
    /// normal is the tangent rotated by 90°, with the sign fixed by the signed
    /// area of the polyline.
    ///
    /// # Polarity
    /// A contour records *geometry*, not photometry: nothing in it says whether
    /// the part is darker or brighter than its background. The sign chosen here
    /// is therefore a guess, and building with
    /// [`Polarity::Match`](super::Polarity::Match) on a guess that happens to be
    /// wrong yields a silently near-zero score. **Use
    /// [`Polarity::IgnoreLocal`](super::Polarity::IgnoreLocal) or
    /// [`IgnoreGlobal`](super::Polarity::IgnoreGlobal) for contour and CAD
    /// models**, under which the sign is irrelevant by construction. When a
    /// reference image exists, [`from_edgels`](Self::from_edgels) is the correct
    /// path.
    ///
    /// # Errors
    /// [`Error::InsufficientData`] when too few points survive.
    pub fn from_polylines(
        polylines: &[Polyline2f],
        orientation: ContourOrientation,
        cfg: &ShapeModelConfig,
    ) -> Result<Self, Error> {
        let mut pts = Vec::new();
        for poly in polylines {
            let p = &poly.points;
            if p.len() < 3 {
                continue;
            }
            // Shoelace signed area. Positive area means `t.perp()` points into
            // the enclosed region; an open polyline gives ~0 and the sign is
            // arbitrary, which is exactly why contour models want a
            // polarity-insensitive score.
            let mut area = 0.0f32;
            for i in 0..p.len() {
                let a = p[i];
                let b = p[(i + 1) % p.len()];
                area += a.x * b.y - b.x * a.y;
            }
            let sign = match (orientation, area > 0.0) {
                (ContourOrientation::Outward, true) | (ContourOrientation::Inward, false) => -1.0,
                _ => 1.0,
            };

            for i in 0..p.len() {
                let prev = p[(i + p.len() - 1) % p.len()];
                let next = p[(i + 1) % p.len()];
                let t = (next - prev).normalized_or_zero();
                if t.norm() < 0.5 {
                    continue;
                }
                pts.push(RawPoint {
                    p: p[i],
                    t: t.perp() * sign,
                    strength: 1.0,
                });
            }
        }
        from_raw(pts, cfg)
    }
}

// ---------------------------------------------------------------------------
// Shared assembly
// ---------------------------------------------------------------------------

/// The aligned crop the reference pyramid is built on.
#[derive(Debug, Clone, Copy)]
struct Crop {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    levels: usize,
}

fn validate(
    img_w: usize,
    img_h: usize,
    roi: Rect2f,
    cfg: &ShapeModelConfig,
) -> Result<Crop, Error> {
    check_ranges(cfg)?;
    if !(roi.width > 0.0 && roi.height > 0.0) {
        return Err(Error::InvalidConfig("roi must have positive extent"));
    }
    if roi.x < 0.0
        || roi.y < 0.0
        || roi.x + roi.width > img_w as f32
        || roi.y + roi.height > img_h as f32
    {
        return Err(Error::OutOfBounds);
    }

    let levels = if cfg.num_levels == 0 {
        MAX_AUTO_LEVELS
    } else {
        cfg.num_levels.min(MAX_LEVELS)
    };

    // The crop origin must be a multiple of 2^(levels-1) so that the model's
    // pyramid aggregates the same pixel blocks the scene's pyramid does.
    // Otherwise the two are downsampled out of phase and their coarse gradient
    // directions disagree for reasons that have nothing to do with the object.
    let align = 1usize << (levels - 1);
    let x0 = (roi.x as usize).saturating_sub(ROI_MARGIN) / align * align;
    let y0 = (roi.y as usize).saturating_sub(ROI_MARGIN) / align * align;
    let x1 = ((roi.x + roi.width).ceil() as usize + ROI_MARGIN).min(img_w);
    let y1 = ((roi.y + roi.height).ceil() as usize + ROI_MARGIN).min(img_h);

    Ok(Crop {
        x0,
        y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
        levels,
    })
}

fn check_ranges(cfg: &ShapeModelConfig) -> Result<(), Error> {
    if !(cfg.scale_range.0 > 0.0 && cfg.scale_range.1 >= cfg.scale_range.0) {
        return Err(Error::InvalidConfig(
            "scale_range must be positive and ordered",
        ));
    }
    if cfg.angle_range.1 < cfg.angle_range.0 {
        return Err(Error::InvalidConfig("angle_range must be ordered"));
    }
    Ok(())
}

fn centroid(pts: &[RawPoint]) -> Option<Point2f> {
    if pts.is_empty() {
        return None;
    }
    let n = pts.len() as f32;
    let (sx, sy) = pts
        .iter()
        .fold((0.0f32, 0.0f32), |(ax, ay), p| (ax + p.p.x, ay + p.p.y));
    Some(Point2f::new(sx / n, sy / n))
}

/// Build a model from a single level-0 point set by decimating it per level.
///
/// Used by the geometry-only constructors, where there is no image to
/// downsample. Directions inside one coarse cell are averaged and renormalised;
/// a cell whose mean direction is short holds contradictory edges and is
/// dropped, mirroring what a box downsample does to the scene.
fn from_raw(pts: Vec<RawPoint>, cfg: &ShapeModelConfig) -> Result<ShapeModel, Error> {
    check_ranges(cfg)?;
    let origin = match cfg.origin {
        Some(o) => o,
        None => centroid(&pts).ok_or(Error::InsufficientData {
            need: cfg.min_points_per_level.max(1),
            got: 0,
        })?,
    };

    let levels = if cfg.num_levels == 0 {
        MAX_AUTO_LEVELS
    } else {
        cfg.num_levels.min(MAX_LEVELS)
    };

    let mut per_level = Vec::with_capacity(levels);
    for level in 0..levels {
        let at_level: Vec<RawPoint> = pts
            .iter()
            .map(|r| RawPoint {
                p: Point2f::new(to_level(r.p.x, level), to_level(r.p.y, level)),
                t: r.t,
                strength: r.strength,
            })
            .collect();
        per_level.push(if level == 0 {
            at_level
        } else {
            merge_into_cells(&at_level, 1.0)
        });
    }

    assemble(per_level, origin, cfg)
}

/// Turn per-level raw points into a finished model.
fn assemble(
    raw: Vec<Vec<RawPoint>>,
    origin: Point2f,
    cfg: &ShapeModelConfig,
) -> Result<ShapeModel, Error> {
    check_ranges(cfg)?;
    let need = cfg.min_points_per_level.max(1);
    let mut levels: Vec<ShapeModelLevel> = Vec::with_capacity(raw.len());

    for (level, pts) in raw.into_iter().enumerate() {
        let ref_l = Point2f::new(to_level(origin.x, level), to_level(origin.y, level));
        let kept = if cfg.max_points == 0 || pts.len() <= cfg.max_points {
            pts
        } else {
            decimate(&pts, cfg.max_points)
        };

        if kept.len() < need || (level > 0 && kept.is_empty()) {
            break;
        }

        let mut points: Vec<ModelPoint> = kept
            .iter()
            .map(|r| ModelPoint {
                d: r.p - ref_l,
                t: r.t,
            })
            .collect();

        let radius = points.iter().fold(0.0f32, |m, p| m.max(p.d.norm()));
        if level > 0 && radius < MIN_LEVEL_RADIUS {
            break;
        }

        stratify(&mut points);
        let bbox = bounding_box(&points);
        levels.push(ShapeModelLevel {
            angle_step: (1.0 / radius.max(1.0)).clamp(0.001, core::f32::consts::FRAC_PI_8),
            scale_step: (1.0 / radius.max(1.0)).clamp(0.005, 0.25),
            radius,
            bbox,
            points,
        });
    }

    if levels.is_empty() {
        return Err(Error::InsufficientData { need, got: 0 });
    }

    let smooth = if cfg.edge.pre_smooth {
        cfg.edge.smooth_kind
    } else {
        SmoothKind::None
    };
    Ok(ShapeModel::from_parts(
        levels,
        origin,
        cfg.angle_range,
        cfg.scale_range,
        cfg.polarity,
        smooth,
    ))
}

fn bounding_box(points: &[ModelPoint]) -> Rect2f {
    let (mut x0, mut y0) = (f32::INFINITY, f32::INFINITY);
    let (mut x1, mut y1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in points {
        x0 = x0.min(p.d.x);
        y0 = y0.min(p.d.y);
        x1 = x1.max(p.d.x);
        y1 = y1.max(p.d.y);
    }
    Rect2f {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    }
}

/// Keep the strongest point per `k × k` cell, with `k` chosen so the survivors
/// number roughly `target`.
///
/// Spatial uniformity is not a preference here. Any decimation that favours
/// strong points concentrates the model on the highest-contrast part of the
/// contour, which inflates the score of a partially occluded instance and
/// destroys the `score ≈ 1 − occluded_fraction` reading that `min_score`
/// depends on.
fn decimate(pts: &[RawPoint], target: usize) -> Vec<RawPoint> {
    if pts.len() <= target || target == 0 {
        return pts.to_vec();
    }
    let k = ((pts.len() as f32 / target as f32).sqrt().ceil()).max(1.0);
    let mut out = merge_into_cells(pts, k);
    // One pass of grid decimation can overshoot when the points are unevenly
    // spread; widen the cell until the budget is met.
    let mut cell = k;
    while out.len() > target && cell < 4096.0 {
        cell *= 1.5;
        out = merge_into_cells(pts, cell);
    }
    out
}

/// Collapse points onto a grid of `cell`-sized squares.
///
/// The survivor of a cell keeps the strongest point's position but the mean of
/// the cell's directions, so a cell that straddles two opposite edges yields a
/// short mean and is dropped instead of picking one side arbitrarily.
fn merge_into_cells(pts: &[RawPoint], cell: f32) -> Vec<RawPoint> {
    if pts.is_empty() {
        return Vec::new();
    }
    let (mut x0, mut y0) = (f32::INFINITY, f32::INFINITY);
    let (mut x1, mut y1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in pts {
        x0 = x0.min(p.p.x);
        y0 = y0.min(p.p.y);
        x1 = x1.max(p.p.x);
        y1 = y1.max(p.p.y);
    }
    let nx = (((x1 - x0) / cell).floor() as usize).saturating_add(1);
    let ny = (((y1 - y0) / cell).floor() as usize).saturating_add(1);
    if nx.saturating_mul(ny) > 1 << 24 {
        return pts.to_vec();
    }

    let mut best: Vec<i32> = vec![-1; nx * ny];
    let mut acc: Vec<Vec2f> = vec![Vec2f::default(); nx * ny];
    let mut count: Vec<u32> = vec![0; nx * ny];
    let mut order: Vec<usize> = Vec::new();

    for (i, p) in pts.iter().enumerate() {
        let cx = (((p.p.x - x0) / cell).floor() as usize).min(nx - 1);
        let cy = (((p.p.y - y0) / cell).floor() as usize).min(ny - 1);
        let c = cy * nx + cx;
        acc[c] += p.t;
        count[c] += 1;
        let b = best[c];
        if b < 0 {
            best[c] = i as i32;
            order.push(c);
        } else if p.strength > pts[b as usize].strength {
            best[c] = i as i32;
        }
    }

    // Morton order over the surviving cells gives a spatially coherent list,
    // which is what the stratifying permutation later samples from.
    order.sort_unstable_by_key(|&c| morton(c % nx, c / nx));

    let mut out = Vec::with_capacity(order.len());
    for c in order {
        // `acc[c]` sums unit vectors, so `|acc| / count` is how much the cell's
        // directions agree: 1 when identical, 0 when exactly opposed.
        let agreement = acc[c].norm() / count[c] as f32;
        if agreement < MIN_MEAN_DIR {
            continue;
        }
        let t = acc[c].normalized_or_zero();
        if t.norm() < 0.5 {
            continue;
        }
        let i = best[c] as usize;
        out.push(RawPoint {
            p: pts[i].p,
            t,
            strength: pts[i].strength,
        });
    }
    out
}

fn morton(x: usize, y: usize) -> u64 {
    #[inline]
    fn spread(v: usize) -> u64 {
        let mut n = (v as u64) & 0xffff_ffff;
        n = (n | (n << 16)) & 0x0000_ffff_0000_ffff;
        n = (n | (n << 8)) & 0x00ff_00ff_00ff_00ff;
        n = (n | (n << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
        n = (n | (n << 2)) & 0x3333_3333_3333_3333;
        n = (n | (n << 1)) & 0x5555_5555_5555_5555;
        n
    }
    spread(x) | (spread(y) << 1)
}

/// Reorder a spatially coherent point list so that every prefix is spread over
/// the whole contour.
///
/// Greedy early termination evaluates a prefix and gives up if its partial sum
/// falls below a bound. If the prefix were one connected arc, an occlusion
/// covering that arc would abort a pose that is in fact correct. Sampling the
/// coherent list with a golden-ratio stride makes any prefix a low-discrepancy
/// sample of the whole shape, so no single localised occlusion can dominate it.
fn stratify(points: &mut [ModelPoint]) {
    let n = points.len();
    if n < 3 {
        return;
    }
    let mut step = ((n as f32) * 0.618_034).round().max(1.0) as usize;
    while gcd(step, n) != 1 {
        step += 1;
        if step >= n {
            step = 1;
            break;
        }
    }
    let src = points.to_vec();
    for (j, dst) in points.iter_mut().enumerate() {
        *dst = src[(j * step) % n];
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}
