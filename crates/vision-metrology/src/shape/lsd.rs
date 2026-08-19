//! Line Segment Detector (LSD).
//!
//! Implements the gradient-coherence region-growing approach from:
//! > von Gioi et al., "LSD: A Fast Line Segment Detector with a False Detection
//! > Control", IEEE TPAMI, 2010.
//!
//! # Algorithm summary
//! 1. Compute Scharr gradients `(gx, gy)` and angle image `θ = atan2(gy, gx)`.
//! 2. Bucket-sort all pixels by decreasing gradient magnitude.
//! 3. Seed on the highest-gradient unused pixel; grow an 8-connected region by
//!    keeping neighbours whose angle differs from the region's main angle by
//!    less than `ang_th`.
//! 4. Fit a least-squares line through the region (weighted by magnitude).
//! 5. Validate with the NFA criterion: accept when `log10(NFA) < log_eps`.
//! 6. Mark region pixels as used.
//!
//! # Coordinate convention
//! Pixel centers are at integer coordinates. Endpoints are reported in this
//! frame: `p1.x ∈ [0, W-1]`, `p1.y ∈ [0, H-1]`.
//!
//! # Border mode
//! Gradient computation uses `Clamp` border replication (consistent with the
//! rest of the crate).

use core::f32::consts::PI;

use vm_primitives::pyr::level_to_base;
use vm_primitives::{ImageView, Pixel, Point2f, PreSmooth, Pyramid, PyramidConfig, Vec2f};

use super::nfa::log10_nfa;

// ---------------------------------------------------------------------------
// Public output type
// ---------------------------------------------------------------------------

/// A detected line segment with subpixel endpoints.
///
/// Endpoints `p1`, `p2` are in image pixel coordinates (centers at integers).
/// `width` is the estimated support-region width in pixels (perpendicular
/// extent of the gradient-coherent region).
/// `nfa` is `log10(NFA)`: negative = statistically significant, positive = noise.
/// `angle` is the segment orientation in `(-π/2, π/2]` radians.
#[derive(Debug, Clone, PartialEq)]
pub struct LineSegment2f {
    /// First endpoint.
    pub p1: Point2f,
    /// Second endpoint (order: p1 is the pixel with smaller projection on the
    /// fitted line direction).
    pub p2: Point2f,
    /// Unit normal to the segment (perpendicular to `p2 - p1`).
    pub normal: Vec2f,
    /// Support-region width in pixels (perpendicular bounding-box extent).
    pub width: f32,
    /// `log10(NFA)`. Negative values indicate a statistically significant line.
    pub nfa: f32,
    /// Euclidean length from `p1` to `p2` in pixels.
    pub length: f32,
    /// Segment orientation in radians, in `(-π/2, π/2]`.
    pub angle: f32,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`LsdDetector`].
///
/// All fields have documented defaults; use `LsdConfig::default()` as the
/// starting point and override individual fields.
#[derive(Debug, Clone, PartialEq)]
pub struct LsdConfig {
    /// Pyramid levels to descend before detecting. `0` runs at full resolution.
    ///
    /// Each level halves both dimensions. Detecting on a coarser level is
    /// cheaper and suppresses high-frequency noise; endpoints are always
    /// reported in full-resolution coordinates via
    /// [`pyr::level_to_base`](vm_primitives::pyr::level_to_base).
    ///
    /// Default `1`, matching the reference implementation's intent of
    /// detecting on a reduced image.
    pub downscale_levels: u32,

    /// Anti-alias pre-filter applied before each decimation step.
    ///
    /// Only meaningful when `downscale_levels > 0`. A plain box mean has no
    /// stop-band, so unfiltered decimation can turn fine texture into spurious
    /// coherent gradients. Default [`PreSmooth::Binomial121`].
    pub pre_smooth: PreSmooth,

    /// Gradient-angle tolerance in **degrees**.
    ///
    /// Neighbours are added to a growing region when their angle differs from
    /// the region's main angle by at most `ang_th`. Default: `22.5°` (π/8).
    pub ang_th: f32,

    /// NFA acceptance threshold in log₁₀ domain.
    ///
    /// A segment is accepted when `log10(NFA) < log_eps`. Default `0.0` means
    /// at most one false alarm expected per image on average.
    pub log_eps: f32,

    /// Minimum density of aligned pixels in a region (ratio of aligned to total).
    ///
    /// Regions below this density are rejected before NFA evaluation.
    /// Default: `0.7`.
    pub density_th: f32,

    /// Number of gradient-magnitude bins for pseudo-sorting pixels.
    ///
    /// Higher values give finer pseudo-ordering at the cost of slightly more
    /// memory for the bucket arrays. Default: `1024`.
    pub n_bins: usize,

    /// Minimum accepted segment length in pixels (in input-image coordinates).
    ///
    /// Segments shorter than this are discarded. Default: `3.0`.
    pub min_length: f32,
}

impl Default for LsdConfig {
    fn default() -> Self {
        Self {
            downscale_levels: 1,
            pre_smooth: PreSmooth::Binomial121,
            ang_th: 22.5,
            log_eps: 0.0,
            density_th: 0.7,
            n_bins: 1024,
            min_length: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal gradient helpers
// ---------------------------------------------------------------------------

/// Compute Scharr gradients and gradient magnitude for a packed f32 image.
///
/// All arrays have length `w * h`. Clamp border mode is used.
fn compute_gradients(
    src: &[f32],
    w: usize,
    h: usize,
    gx: &mut [f32],
    gy: &mut [f32],
    mag: &mut [f32],
) {
    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(h - 1);
        for x in 0..w {
            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(w - 1);

            let p00 = src[ym1 * w + xm1];
            let p01 = src[ym1 * w + x];
            let p02 = src[ym1 * w + xp1];
            let p10 = src[y * w + xm1];
            let p12 = src[y * w + xp1];
            let p20 = src[yp1 * w + xm1];
            let p21 = src[yp1 * w + x];
            let p22 = src[yp1 * w + xp1];

            let gxx = (3.0 * p02 + 10.0 * p12 + 3.0 * p22) - (3.0 * p00 + 10.0 * p10 + 3.0 * p20);
            let gyy = (3.0 * p20 + 10.0 * p21 + 3.0 * p22) - (3.0 * p00 + 10.0 * p01 + 3.0 * p02);

            let idx = y * w + x;
            gx[idx] = gxx;
            gy[idx] = gyy;
            mag[idx] = (gxx * gxx + gyy * gyy).sqrt();
        }
    }
}

/// Compute level angle from gradient, normalised to `(-π/2, π/2]`.
///
/// We use the level-line angle (perpendicular to the gradient), which is the
/// orientation of the isophote. This is the convention used in LSD.
#[inline]
fn level_angle(gx: f32, gy: f32) -> f32 {
    // Level line direction is perpendicular to gradient: rotate 90°.
    // atan2(-gx, gy) gives the level-line angle; wrap to (-π/2, π/2].
    let a = f32::atan2(-gx, gy);
    // Map to (-π/2, π/2]
    if a <= -PI / 2.0 {
        a + PI
    } else if a > PI / 2.0 {
        a - PI
    } else {
        a
    }
}

/// Angular difference wrapped to `[0, π/2]` (undirected line angle distance).
#[inline]
fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = (a - b).abs();
    // Wrap to [0, π]: line angles are equivalent mod π
    if d > PI / 2.0 {
        d = (PI - d).abs();
    }
    d
}

// ---------------------------------------------------------------------------
// Region growing & line fitting
// ---------------------------------------------------------------------------

/// Weighted centroid and 2×2 inertia tensor for region fitting.
struct RegionStats {
    cx: f64,
    cy: f64,
    ixx: f64,
    iyy: f64,
    ixy: f64,
    total_weight: f64,
}

impl RegionStats {
    fn from_region(region: &[usize], mag: &[f32], w: usize) -> Self {
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mut total = 0.0f64;

        for &idx in region {
            let wt = mag[idx] as f64;
            let x = (idx % w) as f64;
            let y = (idx / w) as f64;
            cx += wt * x;
            cy += wt * y;
            total += wt;
        }

        if total <= 0.0 {
            return Self {
                cx: 0.0,
                cy: 0.0,
                ixx: 0.0,
                iyy: 0.0,
                ixy: 0.0,
                total_weight: 0.0,
            };
        }

        cx /= total;
        cy /= total;

        let mut ixx = 0.0f64;
        let mut iyy = 0.0f64;
        let mut ixy = 0.0f64;
        for &idx in region {
            let wt = mag[idx] as f64;
            let dx = (idx % w) as f64 - cx;
            let dy = (idx / w) as f64 - cy;
            ixx += wt * dx * dx;
            iyy += wt * dy * dy;
            ixy += wt * dx * dy;
        }

        Self {
            cx,
            cy,
            ixx,
            iyy,
            ixy,
            total_weight: total,
        }
    }
}

/// Fit a line through the region. Returns (centroid, unit_direction, main_angle).
///
/// The direction is the principal eigenvector of the 2×2 weighted inertia tensor,
/// i.e., the direction of maximum spread (the line direction).
///
/// We use the standard half-angle formula for the 2×2 symmetric eigenproblem:
/// `θ = 0.5 * atan2(2 * ixy, ixx - iyy)`. This avoids the degenerate case in
/// the explicit eigenvector formula when `ixy ≈ 0` and `ixx ≈ iyy`.
fn fit_line_to_region(stats: &RegionStats) -> Option<(Point2f, Vec2f, f32)> {
    if stats.total_weight <= 0.0 {
        return None;
    }
    // Principal axis angle: θ = 0.5 * atan2(2*ixy, ixx - iyy)
    // For ixy=0, ixx > iyy: atan2(0, positive) = 0 → θ=0 → dir=(1,0) (horizontal). Correct.
    // For ixy=0, iyy > ixx: atan2(0, negative) = π → θ=π/2 → dir=(0,1) (vertical). Correct.
    let theta = 0.5 * f64::atan2(2.0 * stats.ixy, stats.ixx - stats.iyy);
    let (cos_t, sin_t) = (theta.cos(), theta.sin());
    let dir = Vec2f::new(cos_t as f32, sin_t as f32);
    let center = Point2f::new(stats.cx as f32, stats.cy as f32);
    // Normalise line angle to (-π/2, π/2]
    let angle = theta as f32;
    let angle = if angle <= -PI / 2.0 {
        angle + PI
    } else if angle > PI / 2.0 {
        angle - PI
    } else {
        angle
    };
    Some((center, dir, angle))
}

/// Project all region pixels onto the line direction, find min/max projections
/// and max perpendicular distance (width), also count aligned pixels.
fn region_endpoints_and_width(
    region: &[usize],
    ang: &[f32],
    center: Point2f,
    dir: Vec2f,
    ang_th_rad: f32,
    region_angle: f32,
    w: usize,
) -> (Point2f, Point2f, f32, usize) {
    let perp = Vec2f::new(-dir.y, dir.x);
    let mut proj_min = f32::MAX;
    let mut proj_max = f32::MIN;
    let mut perp_max = 0.0f32;
    let mut aligned_count = 0usize;

    for &idx in region {
        let px = (idx % w) as f32;
        let py = (idx / w) as f32;
        let dx = px - center.x;
        let dy = py - center.y;
        let proj = dx * dir.x + dy * dir.y;
        let perp_dist = (dx * perp.x + dy * perp.y).abs();

        if proj < proj_min {
            proj_min = proj;
        }
        if proj > proj_max {
            proj_max = proj;
        }
        if perp_dist > perp_max {
            perp_max = perp_dist;
        }

        if angle_diff(ang[idx], region_angle) <= ang_th_rad {
            aligned_count += 1;
        }
    }

    let p1 = Point2f::new(center.x + proj_min * dir.x, center.y + proj_min * dir.y);
    let p2 = Point2f::new(center.x + proj_max * dir.x, center.y + proj_max * dir.y);
    // Width = 2× max perpendicular extent (one-sided → full width)
    let width = (2.0 * perp_max + 1.0).max(1.0);

    (p1, p2, width, aligned_count)
}

// ---------------------------------------------------------------------------
// Detector struct
// ---------------------------------------------------------------------------

/// Reusable scratch buffers for LSD.
///
/// Create once with [`LsdDetector::new`] and call [`LsdDetector::detect`] /
/// [`LsdDetector::detect`] on each frame.
pub struct LsdDetector {
    /// Source pyramid; only levels `0..=downscale_levels` are built.
    pyr: Pyramid,
    /// Packed f32 pixel buffer for the level being detected on.
    buf: Vec<f32>,
    /// Gradient x-component.
    gx: Vec<f32>,
    /// Gradient y-component.
    gy: Vec<f32>,
    /// Gradient magnitude.
    mag: Vec<f32>,
    /// Level-line angle image (per pixel, in `(-π/2, π/2]`).
    ang: Vec<f32>,
    /// Bucket arrays for pseudo-sorted pixel indices.
    buckets: Vec<Vec<usize>>,
    /// Per-pixel used flag (0 = unused, 1 = in active region, 2 = used).
    used: Vec<u8>,
    /// Scratch buffer for the current growing region (pixel linear indices).
    region: Vec<usize>,
    /// Region grow queue (BFS frontier).
    queue: Vec<usize>,
}

impl LsdDetector {
    /// Create a new detector with empty scratch buffers.
    pub fn new() -> Self {
        Self {
            pyr: Pyramid::new(),
            buf: Vec::new(),
            gx: Vec::new(),
            gy: Vec::new(),
            mag: Vec::new(),
            ang: Vec::new(),
            buckets: Vec::new(),
            used: Vec::new(),
            region: Vec::new(),
            queue: Vec::new(),
        }
    }

    /// Detect line segments in a grayscale image of any [`Pixel`] type.
    ///
    /// Endpoints are returned in **input-image** coordinates regardless of
    /// `cfg.downscale_levels`.
    pub fn detect<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        cfg: &LsdConfig,
    ) -> Vec<LineSegment2f> {
        if img.width() == 0 || img.height() == 0 {
            return Vec::new();
        }

        let level = cfg.downscale_levels as usize;
        self.pyr.build_with(
            img,
            level + 1,
            &PyramidConfig {
                pre_smooth: cfg.pre_smooth,
            },
        );

        // The pyramid stops early on small images; detecting on a level that
        // does not exist would silently change the reported scale.
        let Some(src) = self.pyr.level(level) else {
            return Vec::new();
        };
        let (w, h) = (src.width(), src.height());
        self.buf.clear();
        self.buf.extend_from_slice(src.data());

        self.run(w, h, cfg.downscale_levels, cfg)
    }

    // ------------------------------------------------------------------
    // Internal: main detection loop
    // ------------------------------------------------------------------

    fn run(&mut self, w: usize, h: usize, level: u32, cfg: &LsdConfig) -> Vec<LineSegment2f> {
        let n = w * h;
        if n == 0 {
            return Vec::new();
        }

        // --- Allocate / resize scratch ---
        self.gx.resize(n, 0.0);
        self.gy.resize(n, 0.0);
        self.mag.resize(n, 0.0);
        self.ang.resize(n, 0.0);
        self.used.resize(n, 0);
        self.used.fill(0);

        // --- Compute gradients and angle image ---
        compute_gradients(&self.buf, w, h, &mut self.gx, &mut self.gy, &mut self.mag);

        let mag_min_threshold = 2.0f32; // ignore near-zero gradient pixels
        for i in 0..n {
            self.ang[i] = if self.mag[i] > mag_min_threshold {
                level_angle(self.gx[i], self.gy[i])
            } else {
                f32::NAN // sentinel: below threshold, do not use as seed
            };
        }

        // --- Bucket sort by magnitude ---
        let max_mag = self.mag.iter().cloned().fold(0.0f32, f32::max);
        let n_bins = cfg.n_bins.max(1);
        self.buckets.resize_with(n_bins, Vec::new);
        for b in &mut self.buckets {
            b.clear();
        }
        if max_mag > 0.0 {
            for i in 0..n {
                if !self.ang[i].is_nan() {
                    let bin = ((self.mag[i] / max_mag) * (n_bins - 1) as f32).round() as usize;
                    let bin = bin.min(n_bins - 1);
                    // Reverse: we want highest magnitude first
                    self.buckets[n_bins - 1 - bin].push(i);
                }
            }
        }

        // NFA pre-computation
        // N_T = W * H * (W + H) as per LSD paper (number of possible rectangles)
        let n_tests = (w as f64) * (h as f64) * ((w + h) as f64) * 5.0;
        let ang_th_rad = cfg.ang_th.to_radians();
        // Level-line angles are in (-π/2, π/2] (a half-circle of range π).
        // A random level-line angle is uniform on this interval.
        // The region acceptance window is ±ang_th centred on the region's angle,
        // covering a total arc of 2*ang_th. Probability of a random pixel falling
        // inside:  p = 2 * ang_th / π.
        let p_align = 2.0 * ang_th_rad as f64 / core::f64::consts::PI;
        let step = (1u32 << level) as f32;
        let min_len_in_buf = (cfg.min_length / step).max(2.0);

        let mut segments = Vec::new();

        // --- Main seed loop ---
        'seed: for bin in 0..n_bins {
            for si in 0..self.buckets[bin].len() {
                let seed = self.buckets[bin][si];
                if self.used[seed] != 0 {
                    continue;
                }
                if self.ang[seed].is_nan() {
                    continue;
                }

                // ---- Region growing ----
                self.region.clear();
                self.queue.clear();

                self.used[seed] = 1;
                self.region.push(seed);
                self.queue.push(seed);

                // Initial region angle = seed angle
                let mut region_angle = self.ang[seed];

                while let Some(current) = self.queue.pop() {
                    let cx = current % w;
                    let cy = current / w;
                    let x0 = cx.saturating_sub(1);
                    let x1 = (cx + 1).min(w - 1);
                    let y0 = cy.saturating_sub(1);
                    let y1 = (cy + 1).min(h - 1);

                    for ny in y0..=y1 {
                        for nx in x0..=x1 {
                            let nidx = ny * w + nx;
                            if self.used[nidx] != 0 {
                                continue;
                            }
                            if self.ang[nidx].is_nan() {
                                continue;
                            }
                            if angle_diff(self.ang[nidx], region_angle) <= ang_th_rad {
                                self.used[nidx] = 1;
                                self.region.push(nidx);
                                self.queue.push(nidx);

                                // Update region angle: weighted by magnitude toward mean
                                // Use a running weighted mean approximation.
                                // Full recompute is expensive; we use the current seed angle
                                // as a stable anchor. (A proper LSD refines this iteratively.)
                                let wt = self.mag[nidx];
                                let seed_wt = self.mag[seed];
                                let region_len = self.region.len() as f32;
                                let total_wt = region_len * seed_wt + wt;
                                if total_wt > 0.0 {
                                    // Blend new angle in
                                    let da = self.ang[nidx] - region_angle;
                                    let da_wrapped = if da > PI / 2.0 {
                                        da - PI
                                    } else if da <= -PI / 2.0 {
                                        da + PI
                                    } else {
                                        da
                                    };
                                    region_angle += da_wrapped * wt / total_wt;
                                    region_angle = if region_angle <= -PI / 2.0 {
                                        region_angle + PI
                                    } else if region_angle > PI / 2.0 {
                                        region_angle - PI
                                    } else {
                                        region_angle
                                    };
                                }
                            }
                        }
                    }
                }

                // ---- Minimum size check ----
                let reg_n = self.region.len();
                if reg_n < 2 {
                    for &idx in &self.region {
                        self.used[idx] = 2;
                    }
                    continue 'seed;
                }

                // ---- Fit line ----
                let stats = RegionStats::from_region(&self.region, &self.mag, w);
                let Some((center, dir, line_angle)) = fit_line_to_region(&stats) else {
                    for &idx in &self.region {
                        self.used[idx] = 2;
                    }
                    continue 'seed;
                };

                // ---- Endpoints, width, aligned count ----
                let (p1_buf, p2_buf, width_buf, k) = region_endpoints_and_width(
                    &self.region,
                    &self.ang,
                    center,
                    dir,
                    ang_th_rad,
                    line_angle,
                    w,
                );

                // ---- Length check (in buffer coords) ----
                let dx = p2_buf.x - p1_buf.x;
                let dy = p2_buf.y - p1_buf.y;
                let length_buf = (dx * dx + dy * dy).sqrt();
                if length_buf < min_len_in_buf {
                    for &idx in &self.region {
                        self.used[idx] = 2;
                    }
                    continue 'seed;
                }

                // ---- Density check ----
                // Count pixels within ang_th of the line angle (aligned pixels).
                let density = k as f32 / reg_n as f32;
                if density < cfg.density_th {
                    for &idx in &self.region {
                        self.used[idx] = 2;
                    }
                    continue 'seed;
                }

                // ---- NFA check ----
                let log_nfa = log10_nfa(n_tests, reg_n as u64, k as u64, p_align);
                if log_nfa as f32 >= cfg.log_eps {
                    for &idx in &self.region {
                        self.used[idx] = 2;
                    }
                    continue 'seed;
                }

                // ---- Map back to input-image coordinates ----
                let p1 = Point2f::new(
                    level_to_base(p1_buf.x, level),
                    level_to_base(p1_buf.y, level),
                );
                let p2 = Point2f::new(
                    level_to_base(p2_buf.x, level),
                    level_to_base(p2_buf.y, level),
                );
                let length = length_buf * step;
                let width = width_buf * step;

                // Unit normal (perpendicular to line direction)
                let normal = Vec2f::new(-dir.y, dir.x);

                segments.push(LineSegment2f {
                    p1,
                    p2,
                    normal,
                    width,
                    nfa: log_nfa as f32,
                    length,
                    angle: line_angle,
                });

                // Mark region as used
                for &idx in &self.region {
                    self.used[idx] = 2;
                }
            }
        }

        segments
    }
}

impl Default for LsdDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{LsdConfig, LsdDetector};
    use vm_primitives::PreSmooth;

    /// Endpoints must land on the true edge at every downscale level, on both
    /// even and odd image widths.
    ///
    /// Before LSD used the pyramid it had its own `div_ceil` + border-clamp
    /// downsample and mapped positions back with a bare `p * (W / w)`, missing
    /// the `(2^l - 1)/2` term. At the default config that put every endpoint
    /// 0.50 px low on an even width and 0.95 px low on an odd one — silently,
    /// because no test looked at absolute position.
    #[test]
    fn endpoints_are_unbiased_at_every_downscale_level() {
        // Vertical step: x < 60 dark, x >= 60 bright. Under the pixel-centre
        // convention the edge sits at 59.5.
        const TRUE_X: f32 = 59.5;

        for w in [128usize, 129] {
            for levels in [0u32, 1, 2] {
                for pre in [PreSmooth::None, PreSmooth::Binomial121] {
                    let h = 128usize;
                    let data: Vec<u8> = (0..w * h)
                        .map(|i| if (i % w) >= 60 { 200u8 } else { 20 })
                        .collect();
                    let img = Image::from_vec(w, h, data).expect("valid image");

                    let mut det = LsdDetector::new();
                    let segs = det.detect(
                        &img.as_view(),
                        &LsdConfig {
                            downscale_levels: levels,
                            pre_smooth: pre,
                            ..LsdConfig::default()
                        },
                    );
                    assert!(
                        !segs.is_empty(),
                        "w={w} levels={levels} pre={pre:?}: no segments"
                    );

                    let xs: Vec<f32> = segs.iter().flat_map(|s| [s.p1.x, s.p2.x]).collect();
                    let mean = xs.iter().sum::<f32>() / xs.len() as f32;
                    assert!(
                        (mean - TRUE_X).abs() < 0.05,
                        "w={w} levels={levels} pre={pre:?}: mean endpoint x {mean:.4}, \
                         expected {TRUE_X} (bias {:+.4})",
                        mean - TRUE_X
                    );
                }
            }
        }
    }

    use super::*;
    use vm_primitives::Image;

    /// Build a synthetic binary step image with a horizontal edge at `edge_y`.
    /// Upper half is black (0), lower half is white (255).
    fn horizontal_step_image(w: usize, h: usize, edge_y: usize) -> Image<u8> {
        let data: Vec<u8> = (0..h)
            .flat_map(|y| {
                let v: u8 = if y >= edge_y { 255 } else { 0 };
                vec![v; w]
            })
            .collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    /// Build a synthetic binary step image with a vertical edge at `edge_x`.
    fn vertical_step_image(w: usize, h: usize, edge_x: usize) -> Image<u8> {
        let mut data = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = if x >= edge_x { 255 } else { 0 };
            }
        }
        Image::from_vec(w, h, data).expect("valid image")
    }

    /// Build a spatially uncorrelated noise image using a per-pixel hash.
    ///
    /// Each pixel value is derived from a bijective integer hash of its linear
    /// index so that adjacent pixels are statistically independent. This avoids
    /// the sequential correlation that arises from a single LCG stream.
    fn noise_image(w: usize, h: usize) -> Image<u8> {
        // Bijective 64-bit hash (finaliser from splitmix64).
        // Hashing the linear index gives spatially uncorrelated values.
        fn hash(mut x: u64) -> u8 {
            x ^= x >> 30;
            x = x.wrapping_mul(0xbf58476d1ce4e5b9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            (x >> 56) as u8
        }
        let data: Vec<u8> = (0..w * h).map(|i| hash(i as u64 ^ 0xdeadbeef)).collect();
        Image::from_vec(w, h, data).expect("valid image")
    }

    #[test]
    fn detects_horizontal_edge() {
        // A 128×64 image with a horizontal step at y=32 should yield at least one
        // near-horizontal segment whose midpoint y is close to 32.
        let w = 128usize;
        let h = 64usize;
        let edge_y = 32usize;
        let img = horizontal_step_image(w, h, edge_y);

        let mut det = LsdDetector::new();
        let cfg = LsdConfig {
            downscale_levels: 0, // full resolution so we stay in exact coordinates
            ..LsdConfig::default()
        };
        let segs = det.detect(&img.as_view(), &cfg);

        // At least one segment should be found
        assert!(
            !segs.is_empty(),
            "no segments detected on horizontal step edge"
        );

        // Find segment most aligned to horizontal (small |angle|) and check its y position
        let best = segs
            .iter()
            .filter(|s| s.angle.abs() < 0.3) // near-horizontal
            .min_by(|a, b| {
                let ya = (a.p1.y + a.p2.y) * 0.5;
                let yb = (b.p1.y + b.p2.y) * 0.5;
                let da = (ya - edge_y as f32).abs();
                let db = (yb - edge_y as f32).abs();
                da.partial_cmp(&db).expect("finite")
            });

        assert!(best.is_some(), "no near-horizontal segment found");
        let seg = best.unwrap();
        let mid_y = (seg.p1.y + seg.p2.y) * 0.5;
        assert!(
            (mid_y - edge_y as f32).abs() < 2.0,
            "horizontal edge y={mid_y:.2}, expected near {edge_y}"
        );

        // The segment should be long enough to span most of the image width
        assert!(
            seg.length > w as f32 * 0.5,
            "segment too short: {}",
            seg.length
        );
    }

    #[test]
    fn detects_vertical_edge() {
        // A 64×128 image with a vertical step at x=32 should yield at least one
        // near-vertical segment whose midpoint x is close to 32.
        let w = 64usize;
        let h = 128usize;
        let edge_x = 32usize;
        let img = vertical_step_image(w, h, edge_x);

        let mut det = LsdDetector::new();
        let cfg = LsdConfig {
            downscale_levels: 0,
            ..LsdConfig::default()
        };
        let segs = det.detect(&img.as_view(), &cfg);

        assert!(!segs.is_empty(), "no segments on vertical step edge");

        let best = segs
            .iter()
            .filter(|s| s.angle.abs() > PI / 2.0 - 0.3) // near-vertical (|angle| ≈ π/2)
            .min_by(|a, b| {
                let xa = (a.p1.x + a.p2.x) * 0.5;
                let xb = (b.p1.x + b.p2.x) * 0.5;
                let da = (xa - edge_x as f32).abs();
                let db = (xb - edge_x as f32).abs();
                da.partial_cmp(&db).expect("finite")
            });

        assert!(best.is_some(), "no near-vertical segment found");
        let seg = best.unwrap();
        let mid_x = (seg.p1.x + seg.p2.x) * 0.5;
        assert!(
            (mid_x - edge_x as f32).abs() < 2.0,
            "vertical edge x={mid_x:.2}, expected near {edge_x}"
        );
    }

    #[test]
    fn min_length_filter_rejects_short_segments() {
        // On a small 16×8 image the segments should be rejected by a large min_length.
        let img = horizontal_step_image(16, 8, 4);
        let mut det = LsdDetector::new();
        let cfg = LsdConfig {
            downscale_levels: 0,
            min_length: 200.0, // far larger than image dimensions
            ..LsdConfig::default()
        };
        let segs = det.detect(&img.as_view(), &cfg);
        assert!(segs.is_empty(), "expected no segments, got {}", segs.len());
    }

    #[test]
    fn noise_image_produces_few_or_no_segments() {
        // A noise-only image should have no statistically significant line segments
        // under the default NFA threshold.
        let img = noise_image(64, 64);
        let mut det = LsdDetector::new();
        let cfg = LsdConfig {
            downscale_levels: 0,
            log_eps: -1.0, // stricter threshold: NFA < 0.1
            ..LsdConfig::default()
        };
        let segs = det.detect(&img.as_view(), &cfg);
        // Under strict NFA control, noise should not produce strong false detections.
        // We allow up to 2 to account for coincidental patterns in the deterministic hash.
        assert!(
            segs.len() <= 2,
            "too many false detections on noise image: {}",
            segs.len()
        );
    }

    #[test]
    fn nfa_values_are_negative_for_real_edges() {
        // All accepted segments on a clean step image must have nfa < log_eps = 0.0
        let img = horizontal_step_image(64, 64, 32);
        let mut det = LsdDetector::new();
        let cfg = LsdConfig {
            downscale_levels: 0,
            ..LsdConfig::default()
        };
        let segs = det.detect(&img.as_view(), &cfg);
        for s in &segs {
            assert!(
                s.nfa < cfg.log_eps,
                "segment with non-negative NFA survived: nfa={}",
                s.nfa
            );
        }
    }

    #[test]
    fn detect_f32_consistent_with_detect_u8() {
        // detect_f32 on the same image (scaled to 0..1) should yield the same number
        // of segments as detect on the u8 version (gradients scale proportionally).
        let w = 64usize;
        let h = 64usize;
        let img_u8 = horizontal_step_image(w, h, 32);
        let img_f32_data: Vec<f32> = img_u8.data().iter().map(|&v| v as f32).collect();
        let img_f32 = Image::from_vec(w, h, img_f32_data).expect("valid f32 image");

        let mut det = LsdDetector::new();
        let cfg = LsdConfig {
            downscale_levels: 0,
            ..LsdConfig::default()
        };

        let segs_u8 = det.detect(&img_u8.as_view(), &cfg);
        let segs_f32 = det.detect(&img_f32.as_view(), &cfg);

        // Both should find at least one segment; counts should match (same data, same gradients)
        assert!(!segs_u8.is_empty());
        assert_eq!(
            segs_u8.len(),
            segs_f32.len(),
            "u8 and f32 paths produced different segment counts"
        );
    }

    #[test]
    fn empty_image_returns_empty() {
        let img: Image<u8> = Image::from_vec(0, 0, vec![]).expect("valid empty");
        let mut det = LsdDetector::new();
        let segs = det.detect(&img.as_view(), &LsdConfig::default());
        assert!(segs.is_empty());
    }
} // mod tests
