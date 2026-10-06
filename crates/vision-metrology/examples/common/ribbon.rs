//! A synthetic bead: an anti-aliased ribbon along a curve, with continuous ground truth.
//!
//! **Model.** A centreline [`Curve`], parametrised by arc length `s ∈ [0, L]`, carries a
//! ribbon of width `w(s)` ([`Width`]). Every point `p` has a nearest centreline location `s`
//! and a signed distance `d` from it along the normal there. The noise-free intensity is
//!
//! ```text
//! I(p) = bg(p) + (fg − bg)·g(s)·[Φ((w(s)/2 − d)/σ) − Φ((−w(s)/2 − d)/σ)] + extras
//! ```
//!
//! a box across the normal blurred by a 1-D Gaussian of `σ` px, on the background
//! `bg(p) = bg + gradient·(p − origin)`. `g(s)` is 0 inside a gap and 1 elsewhere. `fg > bg`
//! is a light bead, `fg < bg` a dark one. A pixel's value is the mean of `I` over the unit
//! square around its centre (integer coordinates are pixel centres), by 4 × 4 tensor
//! Gauss–Legendre quadrature; `tests/accuracy/bead.rs` checks it against the exact pixel
//! mean of a straight ribbon. Only pixels within `w_max/2 + 6σ + 2` px of the curve (squares
//! around curve samples) are integrated; the others are background.
//!
//! **Conventions.** x right, y down. `t(s)` is the unit tangent towards increasing `s`, and
//! `n(s) = (−t_y, t_x)`: on screen, +n points to the right of travel. `d > 0` is on the +n
//! side.
//!
//! **Curves.** A line; an arc, whose positive sweep turns clockwise on screen; an S-bend of
//! two tangent arcs turning opposite ways; a sine along a direction. The line and the arcs
//! are analytic. The sine's arc length is integrated (8-point Gauss–Legendre over a table of
//! 32 steps per period), and its nearest point is found by Newton's method from a scan.
//!
//! **Extras**, all optional:
//! - highlight stripes inside the bead: blurred boxes at a normal offset, absent in the gaps
//!   like the bead;
//! - distractor stripes parallel to the centreline, present along its whole length;
//! - straight distractor steps at any angle, over the whole image;
//! - a linear background gradient;
//! - curves that leave the image;
//! - seeded Gaussian noise ([`Raster::noisy`]), then [`Raster::to_u8`], [`Raster::to_u16`]
//!   (DN × 256) or [`Raster::to_f32`] (DN, unquantised).
//!
//! **Truth.** [`Ribbon::nearest`] maps a point to `(s, d)`. [`Ribbon::center`],
//! [`Ribbon::tangent`], [`Ribbon::normal`] and [`Ribbon::width`] give the geometry at `s`,
//! and [`Ribbon::length`] is `L`. [`Curve`] has the same in `f64`.
//!
//! **Limits.**
//! - The blur is 1-D along the normal, so the truth is exact by construction: the edges are
//!   at `d = ±w/2` and the centre at `d = 0`. An isotropic 2-D PSF on a curve of radius `R`
//!   would move the edges by about `σ²/(2R)`; keep `R` large where accuracy is asserted.
//! - The ends are square, without caps: a point whose nearest location is an end (`s = 0`
//!   or `L`) is outside the ribbon. The ends and the gap boundaries are hard cuts across the
//!   ribbon, not blurred, so pixels there are only approximate and carry no truth.
//! - A point equidistant from two parts of the curve (inside a tight bend) goes to the
//!   smaller `s`. Keep the radius of curvature above `w/2 + 6σ`, and above a parallel
//!   stripe's outer offset, so every rendered point has one nearest location.

// This module is `#[path]`-included by tests, benches and examples; each uses a subset of
// it, so per-root dead-code analysis is meaningless.
#![allow(dead_code)]

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI, TAU};
use std::ops::{Add, Mul, Range, Sub};

use vision_metrology::{Image, Point2f, Vec2f};

/// The pixel quadrature: 4-point Gauss–Legendre on `[−1, 1]`, as `(node, weight)`.
const PIXEL_RULE: [(f64, f64); 4] = [
    (-0.8611363115940526, 0.34785484513745374),
    (-0.3399810435848563, 0.6521451548625461),
    (0.3399810435848563, 0.6521451548625461),
    (0.8611363115940526, 0.34785484513745374),
];

/// The sine's arc-length quadrature: 8-point Gauss–Legendre on `[−1, 1]`, the nodes `±x`.
const ARC_RULE: [(f64, f64); 4] = [
    (0.1834346424956498, 0.362683783378362),
    (0.525532409916329, 0.3137066458778874),
    (0.7966664774136268, 0.22238103445337445),
    (0.9602898564975363, 0.10122853629037618),
];

/// Newton iterations stop at a step this small, in px of abscissa, or after 50.
const NEWTON_TOL: f64 = 1e-12;

// ── vectors ──────────────────────────────────────────────────────────────

/// A point or a vector in `f64` pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct P2 {
    pub x: f64,
    pub y: f64,
}

impl P2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// The unit vector `angle` radians from +x towards +y (clockwise on screen).
    pub fn polar(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c, s)
    }

    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y
    }

    /// Rotated by +90°, `(−y, x)`: the normal of a tangent.
    pub fn perp(self) -> Self {
        Self::new(-self.y, self.x)
    }

    pub fn norm(self) -> f64 {
        self.x.hypot(self.y)
    }

    /// Rounded to a library point.
    pub fn point(self) -> Point2f {
        Point2f::new(self.x as f32, self.y as f32)
    }

    /// Rounded to a library vector.
    pub fn vec(self) -> Vec2f {
        Vec2f::new(self.x as f32, self.y as f32)
    }
}

impl From<Point2f> for P2 {
    fn from(p: Point2f) -> Self {
        Self::new(f64::from(p.x), f64::from(p.y))
    }
}

impl Add for P2 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y)
    }
}

impl Sub for P2 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y)
    }
}

impl Mul<f64> for P2 {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        Self::new(self.x * k, self.y * k)
    }
}

// ── curves ───────────────────────────────────────────────────────────────

/// A circular arc: `center + radius·(cos θ, sin θ)` for θ from `start_angle` through
/// `start_angle + sweep`, in radians. A positive sweep turns clockwise on screen and +n
/// points to the centre; a negative one turns anticlockwise and +n points away.
#[derive(Clone, Copy, Debug)]
pub struct Arc {
    pub center: P2,
    pub radius: f64,
    pub start_angle: f64,
    pub sweep: f64,
}

impl Arc {
    pub fn new(center: P2, radius: f64, start_angle: f64, sweep: f64) -> Self {
        assert!(radius > 0.0, "an arc needs a positive radius");
        assert!(
            sweep != 0.0 && sweep.abs() < TAU,
            "an arc's sweep is in (0, 2π)"
        );
        Self {
            center,
            radius,
            start_angle,
            sweep,
        }
    }

    pub fn length(&self) -> f64 {
        self.radius * self.sweep.abs()
    }

    fn angle(&self, s: f64) -> f64 {
        self.start_angle + self.sweep.signum() * s / self.radius
    }

    /// The point at arc length `s`.
    pub fn at(&self, s: f64) -> P2 {
        self.center + P2::polar(self.angle(s)) * self.radius
    }

    /// The unit tangent at arc length `s`.
    pub fn tangent(&self, s: f64) -> P2 {
        P2::polar(self.angle(s)).perp() * self.sweep.signum()
    }

    /// The arc length of the nearest location: the radial projection inside the sweep,
    /// otherwise the nearer end (the start on a tie).
    fn nearest_s(&self, p: P2) -> f64 {
        let v = p - self.center;
        let u = (self.sweep.signum() * (v.y.atan2(v.x) - self.start_angle)).rem_euclid(TAU);
        if u <= self.sweep.abs() {
            return self.radius * u;
        }
        let (a, b) = (p - self.at(0.0), p - self.at(self.length()));
        if a.dot(a) <= b.dot(b) {
            0.0
        } else {
            self.length()
        }
    }
}

/// A sine along a direction: `origin + u·dir + amplitude·sin(2πu/period)·dir.perp()` for the
/// abscissa `u ∈ [0, span]`. `dir` is a unit vector and `span` the extent along it; the arc
/// length is longer.
#[derive(Clone, Debug)]
pub struct Sine {
    pub origin: P2,
    pub dir: P2,
    pub span: f64,
    pub amplitude: f64,
    pub period: f64,
    /// The arc length at `u = i·step`.
    table: Vec<f64>,
    step: f64,
}

impl Sine {
    fn new(origin: P2, dir: P2, span: f64, amplitude: f64, period: f64) -> Self {
        assert!(
            span > 0.0 && period > 0.0,
            "a sine's span and period are positive"
        );
        let steps = (32.0 * span / period).ceil() as usize;
        let mut sine = Self {
            origin,
            dir: dir * (1.0 / dir.norm()),
            span,
            amplitude,
            period,
            table: vec![0.0; steps + 1],
            step: span / steps as f64,
        };
        for i in 0..steps {
            let (a, b) = (i as f64 * sine.step, (i + 1) as f64 * sine.step);
            sine.table[i + 1] = sine.table[i] + sine.integral(a, b);
        }
        sine
    }

    fn k(&self) -> f64 {
        TAU / self.period
    }

    /// The point at abscissa `u`.
    pub fn at(&self, u: f64) -> P2 {
        let k = self.k();
        self.origin + self.dir * u + self.dir.perp() * (self.amplitude * (k * u).sin())
    }

    /// `d at / du`.
    fn velocity(&self, u: f64) -> P2 {
        let k = self.k();
        self.dir + self.dir.perp() * (self.amplitude * k * (k * u).cos())
    }

    /// The point and unit tangent at abscissa `u`.
    fn frame(&self, u: f64) -> (P2, P2) {
        let v = self.velocity(u);
        (self.at(u), v * (1.0 / v.norm()))
    }

    /// `d² at / du²`.
    fn acceleration(&self, u: f64) -> P2 {
        let k = self.k();
        self.dir.perp() * (-self.amplitude * k * k * (k * u).sin())
    }

    /// `∫ |velocity| du` over `[a, b]`, exact to rounding over a table step.
    fn integral(&self, a: f64, b: f64) -> f64 {
        let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
        let speed = |u: f64| self.velocity(u).norm();
        let sum: f64 = ARC_RULE
            .iter()
            .map(|&(x, w)| w * (speed(mid - half * x) + speed(mid + half * x)))
            .sum();
        sum * half
    }

    fn length(&self) -> f64 {
        self.table[self.table.len() - 1]
    }

    /// The arc length at abscissa `u`; exactly `0` and `L` at the ends.
    pub fn s_at(&self, u: f64) -> f64 {
        if u >= self.span {
            return self.length();
        }
        let i = ((u / self.step) as usize).min(self.table.len() - 2);
        (self.table[i] + self.integral(i as f64 * self.step, u)).clamp(0.0, self.length())
    }

    /// The abscissa at arc length `s`, by Newton's method from the table.
    pub fn u_at(&self, s: f64) -> f64 {
        let s = s.clamp(0.0, self.length());
        let last = self.table.len() - 1;
        let i = self.table.partition_point(|&t| t <= s).clamp(1, last) - 1;
        let frac = (s - self.table[i]) / (self.table[i + 1] - self.table[i]);
        let mut u = (i as f64 + frac) * self.step;
        for _ in 0..50 {
            let du = (self.s_at(u) - s) / self.velocity(u).norm();
            u = (u - du).clamp(0.0, self.span);
            if du.abs() <= NEWTON_TOL {
                break;
            }
        }
        u
    }

    /// The abscissa of the nearest location, clamped to `[0, span]`: Newton's method on
    /// `(at(u) − p)·velocity(u) = 0` from `seed`, or from the best sample of a scan.
    fn nearest_u(&self, p: P2, seed: Option<f64>) -> f64 {
        let mut u = seed.unwrap_or_else(|| self.scan(p));
        for _ in 0..50 {
            let (r, v) = (self.at(u) - p, self.velocity(u));
            // Past the centre of curvature the second derivative is not positive; a
            // gradient step stands in for Newton's there.
            let second = v.dot(v) + r.dot(self.acceleration(u));
            let slope = if second > 0.0 { second } else { v.dot(v) };
            let next = (u - r.dot(v) / slope).clamp(0.0, self.span);
            if (next - u).abs() <= NEWTON_TOL {
                return next;
            }
            u = next;
        }
        u
    }

    /// The best of samples one table step apart over every abscissa that can be nearer to
    /// `p` than its projection onto `dir` is: `|u − u0| ≤ |at(u) − p| ≤ |at(u0) − p|`. The
    /// first best wins, so a tie goes to the smaller `u`.
    fn scan(&self, p: P2) -> f64 {
        let u0 = (p - self.origin).dot(self.dir).clamp(0.0, self.span);
        let reach = (p - self.at(u0)).norm();
        let (lo, hi) = ((u0 - reach).max(0.0), (u0 + reach).min(self.span));
        let n = ((hi - lo) / self.step).ceil().max(1.0) as usize;
        let mut best = (f64::INFINITY, lo);
        for j in 0..=n {
            let u = lo + (hi - lo) * j as f64 / n as f64;
            let r = self.at(u) - p;
            if r.dot(r) < best.0 {
                best = (r.dot(r), u);
            }
        }
        best.1
    }
}

/// A centreline, parametrised by arc length `s ∈ [0, length()]`.
#[derive(Clone, Debug)]
pub enum Curve {
    /// The segment from `a` to `b`.
    Line {
        a: P2,
        b: P2,
    },
    Arc(Arc),
    /// Two arcs, tangent where they meet, turning opposite ways.
    SBend([Arc; 2]),
    Sine(Sine),
}

/// Where a point projects onto a curve.
#[derive(Clone, Copy, Debug)]
struct Foot {
    s: f64,
    d: f64,
    /// Whether the nearest location is interior; one at an end is outside the ribbon.
    inside: bool,
    /// The sine's abscissa: a Newton seed for nearby points.
    seed: Option<f64>,
}

impl Curve {
    pub fn line(a: P2, b: P2) -> Self {
        assert!(a != b, "a line needs two distinct points");
        Self::Line { a, b }
    }

    pub fn arc(center: P2, radius: f64, start_angle: f64, sweep: f64) -> Self {
        Self::Arc(Arc::new(center, radius, start_angle, sweep))
    }

    /// From `start`, heading `heading` radians from +x: an arc of `radius` turning by
    /// `sweep`, then one of the same radius turning back by `−sweep`. The end heading equals
    /// the start's.
    pub fn s_bend(start: P2, heading: f64, radius: f64, sweep: f64) -> Self {
        let turn = sweep.signum();
        let a0 = heading - turn * FRAC_PI_2;
        let first = Arc::new(start - P2::polar(a0) * radius, radius, a0, sweep);
        let a1 = a0 + sweep;
        let c2 = first.center + P2::polar(a1) * (2.0 * radius);
        Self::SBend([first, Arc::new(c2, radius, a1 + turn * PI, -sweep)])
    }

    /// See [`Sine`]; `dir` need not be unit.
    pub fn sine(origin: P2, dir: P2, span: f64, amplitude: f64, period: f64) -> Self {
        Self::Sine(Sine::new(origin, dir, span, amplitude, period))
    }

    pub fn length(&self) -> f64 {
        match self {
            Self::Line { a, b } => (*b - *a).norm(),
            Self::Arc(arc) => arc.length(),
            Self::SBend([first, second]) => first.length() + second.length(),
            Self::Sine(sine) => sine.length(),
        }
    }

    /// The point at arc length `s`, clamped to the curve.
    pub fn center(&self, s: f64) -> P2 {
        self.frame(s).0
    }

    /// The unit tangent at arc length `s`, clamped to the curve.
    pub fn tangent(&self, s: f64) -> P2 {
        self.frame(s).1
    }

    /// The unit normal `tangent(s).perp()`.
    pub fn normal(&self, s: f64) -> P2 {
        self.tangent(s).perp()
    }

    /// The nearest location's arc length, clamped to `[0, L]`, and `p`'s signed distance
    /// along the normal there.
    pub fn nearest(&self, p: P2) -> (f64, f64) {
        let foot = self.foot(p, None);
        (foot.s, foot.d)
    }

    fn frame(&self, s: f64) -> (P2, P2) {
        let s = s.clamp(0.0, self.length());
        match self {
            Self::Line { a, b } => {
                let t = (*b - *a) * (1.0 / (*b - *a).norm());
                (*a + t * s, t)
            }
            Self::Arc(arc) => (arc.at(s), arc.tangent(s)),
            Self::SBend([first, second]) => {
                if s <= first.length() {
                    (first.at(s), first.tangent(s))
                } else {
                    let s = s - first.length();
                    (second.at(s), second.tangent(s))
                }
            }
            Self::Sine(sine) => sine.frame(sine.u_at(s)),
        }
    }

    /// The projection of `p`; `seed` is a nearby point's [`Foot::seed`], for the sine.
    fn foot(&self, p: P2, seed: Option<f64>) -> Foot {
        let len = self.length();
        let (s, seed) = match self {
            Self::Line { a, b } => (((p - *a).dot(*b - *a) / len).clamp(0.0, len), None),
            Self::Arc(arc) => (arc.nearest_s(p), None),
            Self::SBend([first, second]) => {
                let (s1, s2) = (first.nearest_s(p), second.nearest_s(p));
                let (r1, r2) = (p - first.at(s1), p - second.at(s2));
                let s = if r1.dot(r1) <= r2.dot(r2) {
                    s1
                } else {
                    first.length() + s2
                };
                (s, None)
            }
            Self::Sine(sine) => {
                let u = sine.nearest_u(p, seed);
                (sine.s_at(u), Some(u))
            }
        };
        // The sine's frame from its abscissa, not from `s` through a second solve.
        let (at, t) = match (self, seed) {
            (Self::Sine(sine), Some(u)) => sine.frame(u),
            _ => self.frame(s),
        };
        Foot {
            s,
            d: (p - at).dot(t.perp()),
            inside: s > 0.0 && s < len,
            seed,
        }
    }
}

// ── the ribbon and its truth ─────────────────────────────────────────────

/// The width along the ribbon, px.
#[derive(Clone, Copy, Debug)]
pub enum Width {
    Const(f64),
    /// `w0` at `s = 0` to `w1` at `s = L`, linear in `s`.
    Linear {
        w0: f64,
        w1: f64,
    },
}

/// The truth: a centreline, its width profile, and the gaps where the bead is absent.
#[derive(Clone, Debug)]
pub struct Ribbon {
    pub curve: Curve,
    pub profile: Width,
    /// Arc-length intervals `[s0, s1]` without bead.
    pub gaps: Vec<(f64, f64)>,
}

impl Ribbon {
    pub fn new(curve: Curve, profile: Width) -> Self {
        Self {
            curve,
            profile,
            gaps: Vec::new(),
        }
    }

    /// No bead over `s ∈ [s0, s1]`.
    pub fn gap(mut self, s0: f64, s1: f64) -> Self {
        self.gaps.push((s0, s1));
        self
    }

    /// The arc length `L` of the centreline.
    pub fn length(&self) -> f64 {
        self.curve.length()
    }

    /// The centreline point at `s`.
    pub fn center(&self, s: f64) -> Point2f {
        self.curve.center(s).point()
    }

    /// The unit tangent at `s`.
    pub fn tangent(&self, s: f64) -> Vec2f {
        self.curve.tangent(s).vec()
    }

    /// The unit normal at `s`, `(−t_y, t_x)`.
    pub fn normal(&self, s: f64) -> Vec2f {
        self.curve.normal(s).vec()
    }

    /// The width at `s`, px.
    pub fn width(&self, s: f64) -> f64 {
        match self.profile {
            Width::Const(w) => w,
            Width::Linear { w0, w1 } => w0 + (w1 - w0) * (s / self.length()).clamp(0.0, 1.0),
        }
    }

    /// Whether the bead is there at `s`: strictly between the ends and outside every gap.
    pub fn present(&self, s: f64) -> bool {
        s > 0.0 && s < self.length() && !self.gaps.iter().any(|&(a, b)| s >= a && s <= b)
    }

    /// The nearest centreline location's arc length (`0` or `L` beyond an end) and `p`'s
    /// signed distance along the normal there.
    pub fn nearest(&self, p: Point2f) -> (f64, f64) {
        self.curve.nearest(p.into())
    }
}

// ── the scene ────────────────────────────────────────────────────────────

/// A blurred box across the normal: `width` px wide, centred `offset` px along +n from the
/// centreline, adding `contrast` DN (negative darkens).
#[derive(Clone, Copy, Debug)]
pub struct Stripe {
    pub offset: f64,
    pub width: f64,
    pub contrast: f64,
}

/// A straight blurred step over the whole image, `contrast·Φ(((p − point)·normal)/σ)`: the
/// +normal side is `contrast` DN brighter (negative darkens). `normal` is a unit vector.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub point: P2,
    pub normal: P2,
    pub contrast: f64,
}

/// One ribbon on a background, with optional extras; levels are in DN.
#[derive(Clone, Debug)]
pub struct Scene {
    pub width: usize,
    pub height: usize,
    pub ribbon: Ribbon,
    /// The background level at `gradient_origin`.
    pub bg: f64,
    /// The bead level: above `bg` for a light bead, below for a dark one.
    pub fg: f64,
    /// The blur across the normal, px.
    pub sigma: f64,
    /// The background's change per px.
    pub gradient: P2,
    pub gradient_origin: P2,
    pub highlights: Vec<Stripe>,
    pub parallels: Vec<Stripe>,
    pub steps: Vec<Step>,
}

impl Scene {
    /// A `width × height` image of `ribbon` at level `fg` on a flat `bg`, blurred by `sigma`
    /// px across the normal.
    pub fn new(width: usize, height: usize, ribbon: Ribbon, bg: f64, fg: f64, sigma: f64) -> Self {
        assert!(sigma > 0.0, "the fixture needs a blur");
        Self {
            width,
            height,
            ribbon,
            bg,
            fg,
            sigma,
            gradient: P2::new(0.0, 0.0),
            gradient_origin: P2::new(0.0, 0.0),
            highlights: Vec::new(),
            parallels: Vec::new(),
            steps: Vec::new(),
        }
    }

    /// A linear background: `bg` at `origin`, changing by `per_px` DN per px.
    pub fn gradient(mut self, origin: P2, per_px: P2) -> Self {
        self.gradient_origin = origin;
        self.gradient = per_px;
        self
    }

    /// A stripe inside the bead, absent in its gaps like the bead.
    pub fn highlight(mut self, stripe: Stripe) -> Self {
        self.highlights.push(stripe);
        self
    }

    /// A distractor stripe parallel to the centreline, along its whole length.
    pub fn parallel(mut self, stripe: Stripe) -> Self {
        self.parallels.push(stripe);
        self
    }

    /// A distractor step over the whole image.
    pub fn step(mut self, step: Step) -> Self {
        self.steps.push(step);
        self
    }

    /// The background level at `p`; linear, so also its pixel mean.
    pub fn background(&self, p: P2) -> f64 {
        self.bg + self.gradient.dot(p - self.gradient_origin)
    }

    /// The noise-free pixel means, in DN.
    pub fn render(&self) -> Raster {
        let (w, h) = (self.width, self.height);
        let pixel = |i: usize| P2::new((i % w) as f64, (i / w) as f64);
        let background = |p| self.background(p) + self.steps_at(p);
        let mut data: Vec<f64> = (0..w * h).map(|i| background(pixel(i))).collect();
        for (i, near) in self.band().into_iter().enumerate() {
            if near {
                data[i] += self.ribbon_pixel(pixel(i));
            }
        }
        Raster {
            width: w,
            height: h,
            data,
        }
    }

    /// How far from the centreline the ribbon and its stripes reach, blur included.
    fn reach(&self) -> f64 {
        let stripes = self.highlights.iter().chain(&self.parallels);
        // The width is linear in `s`, so widest at an end.
        let widest = self
            .ribbon
            .width(0.0)
            .max(self.ribbon.width(self.ribbon.length()));
        let half = stripes.fold(0.5 * widest, |m, st| {
            m.max(st.offset.abs() + 0.5 * st.width)
        });
        half + 6.0 * self.sigma + 2.0
    }

    /// The pixels within `reach()` of the curve: squares around samples at most 1 px apart.
    fn band(&self) -> Vec<bool> {
        let (w, h) = (self.width, self.height);
        let reach = self.reach();
        let mut near = vec![false; w * h];
        let len = self.ribbon.length();
        let n = len.ceil().max(1.0) as usize;
        for i in 0..=n {
            let c = self.ribbon.curve.center(len * i as f64 / n as f64);
            let (Some(xs), Some(ys)) = (around(c.x, reach, w), around(c.y, reach, h)) else {
                continue;
            };
            for y in ys {
                near[y * w + xs.start..y * w + xs.end].fill(true);
            }
        }
        near
    }

    /// The steps' pixel mean at `p`: 0 or the contrast away from a step, the quadrature
    /// near one.
    fn steps_at(&self, p: P2) -> f64 {
        let reach = 6.0 * self.sigma + 1.0;
        let sigma = self.sigma;
        let mut sum = 0.0;
        for st in &self.steps {
            let dist = (p - st.point).dot(st.normal);
            sum += if dist > reach {
                st.contrast
            } else if dist < -reach {
                0.0
            } else {
                st.contrast * pixel_mean(p, |q| phi((q - st.point).dot(st.normal) / sigma))
            };
        }
        sum
    }

    /// The ribbon's pixel mean at `p`, every node seeded from the pixel centre's projection.
    fn ribbon_pixel(&self, p: P2) -> f64 {
        let curve = &self.ribbon.curve;
        let seed = curve.foot(p, None).seed;
        pixel_mean(p, |q| self.ribbon_at(curve.foot(q, seed)))
    }

    /// The ribbon, its highlights and the parallel stripes at a point projecting to `foot`.
    fn ribbon_at(&self, foot: Foot) -> f64 {
        if !foot.inside {
            return 0.0;
        }
        let sigma = self.sigma;
        let stripe = |st: &Stripe| st.contrast * boxed(foot.d - st.offset, st.width, sigma);
        let mut v: f64 = self.parallels.iter().map(stripe).sum();
        if self.ribbon.present(foot.s) {
            v += (self.fg - self.bg) * boxed(foot.d, self.ribbon.width(foot.s), sigma);
            v += self.highlights.iter().map(stripe).sum::<f64>();
        }
        v
    }
}

/// The pixel indices within `reach` of `c` on an axis of `n` pixels.
fn around(c: f64, reach: f64, n: usize) -> Option<Range<usize>> {
    let lo = (c - reach).ceil().max(0.0);
    let hi = (c + reach).floor().min(n as f64 - 1.0);
    (lo <= hi).then(|| lo as usize..hi as usize + 1)
}

/// The mean of `f` over the unit pixel centred on `p`, by tensor Gauss–Legendre.
fn pixel_mean(p: P2, mut f: impl FnMut(P2) -> f64) -> f64 {
    let mut sum = 0.0;
    for &(xi, wi) in &PIXEL_RULE {
        for &(yj, wj) in &PIXEL_RULE {
            sum += wi * wj * f(p + P2::new(0.5 * xi, 0.5 * yj));
        }
    }
    0.25 * sum
}

/// A box of width `w` centred on 0, blurred by `sigma`, at `d`.
fn boxed(d: f64, w: f64, sigma: f64) -> f64 {
    phi((0.5 * w - d) / sigma) - phi((-0.5 * w - d) / sigma)
}

/// The standard normal CDF, through Numerical Recipes' `erfcc` (fractional error below
/// 1.2e-7). The tail is computed for `|z|`, so `Φ(−z) = 1 − Φ(z)` to rounding and a box is
/// symmetric about its centre.
fn phi(z: f64) -> f64 {
    let x = z.abs() * FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * x);
    let poly = [
        -0.822_152_23,
        1.488_515_87,
        -1.135_203_98,
        0.278_868_07,
        -0.186_288_06,
        0.096_784_18,
        0.374_091_96,
        1.000_023_68,
        -1.265_512_23,
    ]
    .iter()
    .fold(0.170_872_77, |acc, &c| acc * t + c);
    let tail = 0.5 * t * (poly - x * x).exp();
    if z >= 0.0 { 1.0 - tail } else { tail }
}

// ── rasters and noise ────────────────────────────────────────────────────

/// Pixel values in DN, row-major, before quantisation.
#[derive(Clone, Debug, PartialEq)]
pub struct Raster {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f64>,
}

impl Raster {
    pub fn at(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }

    /// With Gaussian noise of `sigma_dn` DN added, seeded by `seed`.
    pub fn noisy(&self, sigma_dn: f64, seed: u64) -> Self {
        let mut rng = Gauss(seed);
        let data = self
            .data
            .iter()
            .map(|&v| v + sigma_dn * rng.normal())
            .collect();
        Self { data, ..*self }
    }

    /// Rounded and clipped to 8 bits.
    pub fn to_u8(&self) -> Image<u8> {
        self.image(|v| v.round().clamp(0.0, 255.0) as u8)
    }

    /// Scaled by 256, rounded and clipped to 16 bits.
    pub fn to_u16(&self) -> Image<u16> {
        self.image(|v| (256.0 * v).round().clamp(0.0, 65535.0) as u16)
    }

    /// In DN, unquantised.
    pub fn to_f32(&self) -> Image<f32> {
        self.image(|v| v as f32)
    }

    fn image<T>(&self, f: impl Fn(f64) -> T) -> Image<T> {
        let data = self.data.iter().map(|&v| f(v)).collect();
        Image::from_vec(self.width, self.height, data).expect("the raster's own size")
    }
}

// Seeded Gaussian noise, splitmix64 and Box–Muller: deterministic, no RNG crate
// (invariant 12).
struct Gauss(u64);

impl Gauss {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }
}
