# Measuring a located part

Matching answers *where the part is*. `measure` answers *what its dimensions
are*: it turns a found pose into a set of subpixel edge positions and a fit
you can gate a tolerance on.

```text
ShapeMatcher::find  ->  ShapeMatch::pose  ->  MetrologyModel::apply  ->  Fit + residuals
       where                 fixture              measure + fit            the measurement
```

![Caliper anatomy: a rect caliper box over a synthetic edge, with the extracted 1-D profile plotted alongside it](assets/caliper-anatomy.png)

## What a caliper is

A [`Caliper`] places a geometry on the image, averages intensity *across* it
into a 1-D profile, and runs the existing subpixel `Edge1DDetector` *along*
that profile. The averaging is where the precision comes from: `n`
interpolated samples per profile entry drop noise by `1/√n` while leaving an
edge perpendicular to the scan exactly as sharp as it was.

That last part is also the constraint: widen a caliper (`half_width`) only
while the edge stays parallel to the averaging direction. On a curved edge, a
wide caliper starts averaging across the transition rather than along it,
which is exactly the problem the next section is about.

```rust
use vision_metrology::measure::{Caliper, MeasureConfig, MeasureRect};
use vision_metrology::{Image, Point2f};

// A vertical step: dark left of x = 30, bright from x = 30 on.
let mut data = vec![20u8; 64 * 64];
for y in 0..64 {
    for x in 30..64 {
        data[y * 64 + x] = 200;
    }
}
let img = Image::from_vec(64, 64, data).unwrap();

let mut cal = Caliper::rect(
    MeasureRect {
        center: Point2f::new(32.0, 32.0),
        angle: 0.0,
        half_len: 20.0,
        half_width: 10.0,
    },
    MeasureConfig::default(),
);

let edges = cal.measure(&img.as_view()).expect("an edge");
println!("edge at {:.3}", edges[0].p.x); // 29.5 — the pixel-centre convention
```

## Rect, arc, and radial — and why radial exists

There are four placements, and the choice is about the geometry of the edge
being crossed, not a style preference:

| Placement | Scans | Averages | Use for |
|---|---|---|---|
| [`MeasureRect`] | along its own axis | across, on a **straight chord** | a straight edge, at any angle |
| [`MeasureArc`] | along a circular arc | radially | a feature that *crosses* a circular path (a gear tooth, a slot, the tab on a can end) |
| [`MeasureRadial`] | radially | along the arc, at constant radius | the circular edge itself |
| [`MeasureStrip`] | from `start` to `end` | across, on a straight line | a scan with exact endpoints and sample counts ([below](#strips)) |

`MeasureRect` and `MeasureArc` sound interchangeable with `MeasureRadial` for
measuring a circle, and picking wrong is a bias you won't see unless you go
looking for it.

**The chord-bias story.** A rect caliper averages along a straight *chord*.
Place one radially across a circle of radius 40 with `half_width = 5`, and the
samples 5 px to either side of the centre line don't sit at radius 40 — they
sit at radius `√(40² + 5²) ≈ 40.31`, on the far side of the true edge. The
averaged profile is contaminated by pixels that belong to the wrong side of
the transition, and the detected edge reads **low**. Measured on an
anti-aliased disc (so the true edge sits at exactly the nominal radius, not
wherever the pixel grid happens to fall): a 32-caliper rect-based fit measured
**39.88 px** against a true radius of 40.00 — a −0.12 px bias that *grows*
with `half_width` and shrinks with radius, which is the worst kind of bias
because it moves depending on how you tuned an unrelated parameter.

`MeasureRadial` scans **radially** and averages **along the arc** instead: at
whatever radius a sample sits, its cross-offset is applied as an angle
(`s / radius`), not a straight perpendicular. Every averaged sample lands at
the *same* radius as the caliper centre, so the profile is never contaminated
across the edge. The same setup measures **39.990 px** — a twelvefold
reduction in bias, and it no longer grows with `half_width`.
`MetrologyShape::Circle` uses `MeasureRadial` for exactly this reason; reach
for `MeasureArc` only when the feature you are measuring *crosses* the circle
rather than *is* the circle.

## Strips

A [`MeasureStrip`] is a straight scan from `start` to `end`, averaged across.
It covers the same ground as a rect, but it is specified sample for sample,
which is what reproducing another implementation's numbers, or a scan line
taken from a drawing, needs:

- **Exact endpoints.** `samples` points include both ends, `length / (samples − 1)`
  apart, and the last one sits exactly on `end`.
- **Explicit counts.** `across` lines are spread evenly over `±half_width`:
  `half_width = 7.0, across = 15` is one line per pixel over a 15 px strip.
  Left as `None`, a count follows the rect rule: one sample per `profile.step`
  along, about one line per pixel across.
- **`t` from the start.** An edge's `t` is its distance from `start`, in pixels,
  not a signed offset from a centre. Reversing the strip moves an edge from `t`
  to `length − t`.
- **`sigma` stays in pixels.** It is converted to samples with the strip's true
  spacing, so the same `sigma` smooths the same distance at any `samples`.

```rust
use std::num::NonZeroUsize;
use vision_metrology::measure::{Caliper, MeasureConfig, MeasureStrip, OffImage, ProfileConfig};
use vision_metrology::{Image, Point2f};

// A bright bar on columns 16..48.
let data: Vec<f32> = (0..9 * 64)
    .map(|i| if (16..48).contains(&(i % 64)) { 1.0 } else { 0.0 })
    .collect();
let img = Image::from_vec(64, 9, data).unwrap();

let strip = MeasureStrip {
    start: Point2f::new(0.0, 4.0),
    end: Point2f::new(63.0, 4.0),
    half_width: 0.0,
    samples: NonZeroUsize::new(64),
    across: NonZeroUsize::new(1),
};
let cfg = MeasureConfig {
    threshold: 0.01,
    profile: ProfileConfig { off_image: OffImage::Reject, ..ProfileConfig::default() },
    ..MeasureConfig::default()
};
let mut cal = Caliper::strip(strip, cfg);
let edges = cal.measure(&img.as_view()).expect("two edges");
println!("{:.2} {:.2}", edges[0].t, edges[1].t); // 15.50 47.50
```

A reference usually also wants strict bounds. With `profile.off_image =
OffImage::Reject`, a placement with any sample outside `[0, w − 1] × [0, h − 1]`
is rejected with `RejectReason::OffImage` before edges are searched. The
default, `OffImage::Fill`, samples the outside with `profile.border` and
measures anyway. Both apply to every placement, not only strips.

## Locating the edge

`MeasureConfig::locate` decides what "the edge" means on a profile. There are
three methods:

| Method | The edge is where the smoothed profile… | Edges | Use for |
|---|---|---|---|
| `Locate::GradientPeak { refine }` (default) | changes fastest | every one over `threshold` | most measurements, several edges per caliper, pairs |
| `Locate::MidpointCrossing { endpoint_samples, min_contrast }` | crosses the mean of its two end levels | one, the crossing nearest the middle | a single edge between two flat, clean levels |
| `Locate::HalfContrast { flank_near_px, flank_far_px, tol_px, max_iter, min_contrast }` | crosses the mean of the levels just either side of it | each gradient edge, refined | asymmetric edges, shaded backgrounds |

On a clean, symmetric edge the three agree. They part when the edge is not
symmetric: a shadow on one side, a bevel, a long tail. The gradient peak
follows the steepest point of the transition; the two level methods follow its
50 % point, which on such an edge is somewhere else. The 50 % point is what a
drawing usually means by a width or a datum. Both level methods read the
profile after the Gaussian smoothing of `profile.sigma`.

**`MidpointCrossing`** takes its two levels from the ends of the profile: the
medians of the first and last `endpoint_samples` samples. It is the simplest
definition and the most fragile. The scan has to start and end on flat material
on either side of exactly one edge, and anything that moves the ends moves the
answer: shading, a second edge, a scratch near the end. Its checks run in a
fixed order, and each failure names itself:

1. the end levels differ by less than `min_contrast`: `LowContrast`;
2. their order (rising when the far end is brighter) is not the polarity asked
   for, by `polarity` or by the first entry of a `StrongestInOrder` sequence:
   `WrongPolarity`;
3. the profile never crosses the mean level with that polarity: `NoCrossing`.

`threshold` plays no part, and the edge's `amplitude` is the contrast between
the two levels.

**`HalfContrast`** starts from the gradient edges (a three-point parabola,
then `threshold`, `polarity` and `select` as usual) and moves each to the
crossing of its *local* level. The levels are the medians `flank_near_px` to
`flank_far_px` before and after the current position, the crossing nearest it
within `flank_near_px` is the next position, and the flanks are re-centred
until the position moves by `tol_px` or less. Because the flanks sit around the
crossing they define, the answer does not depend on where the gradient put the
seed. Shading along the scan moves both flanks alike, so it does not move the
edge; a second edge further than `flank_far_px` away does not either. The
method needs room: an edge closer than `flank_far_px` to the end of the profile
has no flank there and reports `NoCrossing`, as does one whose crossing ends
up more than `flank_near_px` from its gradient peak. Two gradient edges that
converge on one crossing (a staircase read as two steps) report
`IncompleteSequence`.

```rust
use std::num::NonZeroUsize;
use vision_metrology::measure::{Caliper, Locate, MeasureConfig, MeasureStrip};
use vision_metrology::{Image, Point2f};

// A step at column 16 on a [0, 1] image.
let data: Vec<f32> = (0..9 * 64).map(|i| if i % 64 >= 16 { 1.0 } else { 0.0 }).collect();
let img = Image::from_vec(64, 9, data).unwrap();
let strip = MeasureStrip {
    start: Point2f::new(0.0, 4.0),
    end: Point2f::new(63.0, 4.0),
    half_width: 0.0,
    samples: NonZeroUsize::new(64),
    across: NonZeroUsize::new(1),
};
let cfg = MeasureConfig {
    threshold: 0.01,
    locate: Locate::HalfContrast {
        flank_near_px: 3.0,
        flank_far_px: 8.0,
        tol_px: 0.01,
        max_iter: NonZeroUsize::new(5).unwrap(),
        min_contrast: 0.05,
    },
    ..MeasureConfig::default()
};
let mut cal = Caliper::strip(strip, cfg);
let t = cal.measure(&img.as_view()).expect("one edge")[0].t;
let level = cal.levels()[0];
println!("{t:.2} between {:.2} and {:.2}", level.before, level.after); // 15.50 between 0.00 and 1.00
```

`Caliper::levels` reports, for the last call, the levels each level-located
edge sits between, the level crossed and how many iterations it took. Its `x`
is in profile samples.

## `MeasureConfig`

```rust
pub struct MeasureConfig {
    pub threshold: f32,
    pub polarity: PolaritySelect,
    pub select: EdgeSelect,
    pub locate: Locate,
    pub max_obliquity_deg: f32,
    pub profile: ProfileConfig, // sigma, derivative, step, border, off_image
}
```

- **`threshold`** — the minimum `|DoG response|` to report an edge, on the
  input pixel scale like every other threshold in this workspace (re-tune for
  `u16`/`f32`).
- **`polarity`** (`PolaritySelect::{Any, Rising, Falling}`) — which
  transitions count. A caliper that should only ever see a dark-to-bright
  edge and instead reports a bright-to-dark one is a useful signal that
  something is wrong with the part, not just noise to filter.
- **`select`** (`EdgeSelect::{All, First, Last, Strongest, StrongestInOrder}`) —
  which of the surviving edges to keep when more than one crosses the threshold.
  `Strongest` is the sane default once a model's geometry is already
  approximately right (`MetrologyObject::new` picks it) — a caliper on a
  nominal edge should report *that* edge, not every edge it happens to cross.
  `StrongestInOrder(EdgeSequence { first, second })` reads one or two edges in
  scan order — a bar is `Rising` then `Falling`. For each entry it takes the
  strongest edge of that polarity strictly after the previous choice (equal
  strength: the earlier one). "After" is along the profile, not along `t`,
  which runs backwards on an arc with a negative extent.
- **`locate`** — how an edge position is found on the profile
  ([above](#locating-the-edge)). `Locate::GradientPeak { refine }` takes a local
  extremum of the derivative and refines it with `SubpixRefine::Parabolic3` (the
  default), `Gaussian3` (a parabola through the logarithms, exact for a
  Gaussian-shaped peak), `Centroid` or `None`.
- **`profile.sigma`** — the Gaussian σ of the smoothing, in pixels. Roughly the
  edge blur to expect: too small and noise produces spurious edges, too large and
  neighbouring edges merge.
- **`profile.derivative`** — `Derivative::DerivativeOfGaussian` (the default)
  convolves with the analytic derivative of a Gaussian of radius `⌈3σ⌉`;
  `Derivative::SmoothThenCentral { radius_px }` smooths with a Gaussian of the given
  half-width and takes central differences, the textbook operator.
- **`profile.step`** — profile sampling step along the scan axis, in pixels.
  `1.0` is one entry per pixel; oversampling (`0.5`) buys resolution on a
  sharp edge at proportional cost. `sigma` stays in pixels, so the same
  `sigma` smooths the same distance at any step.
- **`profile.border`** — sampling behaviour when the caliper overhangs the image.
- **`profile.off_image`** — `OffImage::Fill` (the default) measures a caliper
  that overhangs the image, sampling the outside with `profile.border`;
  `OffImage::Reject` rejects it with `RejectReason::OffImage` before looking for
  edges.
- **`max_obliquity_deg`** — the obliquity gate. A caliper that crosses an edge
  at a glancing angle reports a position along its own scan axis rather than
  the edge's true normal, and the two differ by `1/cos θ`; at a corner there
  is no meaningful crossing at all. Comparing the local image gradient
  against the scan direction and rejecting beyond this angle is what keeps a
  bad caliper *out* of a fit rather than merely down-weighted. `180.0`
  disables the check.

## `RejectReason`

`Caliper::measure` returns `Result<&[MeasureEdge], RejectReason>` — a caliper
that finds nothing is a *result*, not an error, and which gate rejected it is
the difference between "the part is missing" and "the search window is too
short":

| Reason | Meaning |
|---|---|
| `ProfileTooShort` | fewer than 3 profile samples — the placement is degenerate (near-zero `half_len`) |
| `NoEdge` | no response reached `threshold` anywhere in the window |
| `WrongPolarity` | edges were found, but none had the polarity `MeasureConfig::polarity` (or the first entry of an `EdgeSequence`) asked for |
| `TooOblique` | the best edge crossed at more than `max_obliquity_deg` from the scan direction |
| `OffImage` | the caliper reached outside the image: always with `OffImage::Reject`, and with `Fill` when the partly filled profile held no edge (in place of `NoEdge`, `LowContrast` or `NoCrossing`) |
| `IncompleteSequence` | `StrongestInOrder` found its first edge but no edge of the next polarity after it, or `HalfContrast` moved edges out of order |
| `LowContrast` | the levels a `MidpointCrossing` or `HalfContrast` edge sits between differ by less than `min_contrast` |
| `NoCrossing` | a level method found no crossing: none of the midpoint level with the expected polarity, or none near a half-contrast edge |

There is deliberately no variant of `measure` that discards this and returns
an empty slice instead — `Ok(&[])` is unrepresentable, because an extraction
that found nothing always has a reason. On a production line the distinction
between `NoEdge` and `TooOblique` is the difference between "the part is
missing" and "the recipe is mis-taught", and only a typed reason tells you
which without staring at the image.

## The metrology model: `find → pose → apply → fit`

A single caliper measures one edge. [`MetrologyModel`] is what scales that up
to a part: it holds nominal primitives (lines, circles, arcs) in the part's
**own frame** — the frame the part was taught in — distributes calipers along
each, and fits the measured points robustly with the `fit` module.

```rust
use vision_metrology::measure::{
    MetrologyFit, MetrologyModel, MetrologyObject, MetrologyShape,
};
use vision_metrology::{Image, Point2f, Similarity2f};

// A bright disc of radius 30 centred at (64, 64), anti-aliased so the true
// edge sits at exactly r = 30.
let (w, h) = (128usize, 128usize);
let data: Vec<u8> = (0..w * h)
    .map(|i| {
        let (x, y) = ((i % w) as f32 - 64.0, (i / w) as f32 - 64.0);
        let cover = (30.5 - (x * x + y * y).sqrt()).clamp(0.0, 1.0);
        (20.0 + 180.0 * cover).round() as u8
    })
    .collect();
let img = Image::from_vec(w, h, data).unwrap();

let mut model = MetrologyModel::new();
model.add(MetrologyObject::new(MetrologyShape::Circle {
    center: Point2f::new(64.0, 64.0),
    radius: 30.0,
    arc: None,
}));

// Identity fixture: the model is already in image coordinates.
let results = model.apply(&img.as_view(), &Similarity2f::identity());
let MetrologyFit::Circle(fit) = &results[0].as_ref().expect("measured").fit else {
    panic!("a circle")
};
println!("r = {:.3}, rms = {:.3}, n_used = {}", fit.model.radius, fit.rms, fit.n_used);
```

`fixture` is normally [`ShapeMatch::pose`] from a shape-matching search — that
is the whole point of the pattern: **teach the model once, in the part's own
frame, then let the pose carry it to wherever the part actually landed.**
`MetrologyModel::apply` returns one `Result<MetrologyResult, Error>` **per
object, in `objects()` order** — index `i` is always object `i`, so a failed
measurement is visible as `Err` at its own slot rather than silently
shortening the output. `MetrologyResult::hits` carries the caliper edges the
fit actually used, which is the fastest way to see *why* a fit came out the
way it did (usually one caliper that latched onto the wrong nearby edge).

A rotating, scaling fixture moves the calipers with the part: `caliper_len`
and `caliper_width` scale with `fixture.scaling()`, and a `MetrologyShape::Circle`'s
arc start angle rotates with `fixture`'s rotation. `MetrologyObject::fit`
carries a `FitConfig`, and a robust loss there is what keeps a single
bad caliper — a scratch, a print defect, a highlight — from moving the fitted
geometry: it still shows up in `max_dev`, just not in the answer.

## Worked example: `inspect_canend`

`examples/inspect_canend.rs` runs the whole chain on real can-end frames and
is the reference for how the pieces fit together:

1. **Teach**, once, on the first frame: build a `ShapeModel` of the tab from
   an ROI, find it to get a `taught_pose`, and separately fit the rim
   directly (`fit_circle` with RANSAC, no fixture yet — there's nothing to
   apply one to). Express the rim's centre and radius **in the tab's own
   frame** by undoing `taught_pose`: `rim_center_model = taught_pose.inverse() * rim.center`.
2. **Build the model** once: a `MetrologyModel` with one `MetrologyObject::new(MetrologyShape::Circle { center: rim_center_model, radius: rim_radius_model, .. })`,
   tuned with `Tukey` loss since print, dents and the tab itself put other
   edges near the rim.
3. **Inspect**, per frame: `matcher.find(...)` locates the tab and gives a
   fresh pose; `metrology.apply(&img, &m.pose)` measures the rim *in that
   frame's tab pose*, wherever the tab actually turned up; `fit.max_dev`
   against a tolerance is the pass/fail.

That last step is what makes the reported repeatability mean something: every
frame re-derives the rim from wherever the tab was found, so the spread of
the measured radius across frames is the fixture *and* the measurement
combined, not a restatement of where the part happened to sit under a fixed
camera. Measured on set1 (Tukey(2 px), 96 calipers, tolerance 2 px on
`max_dev`): 100/100 frames measured across two lighting conditions, mean rim
radius 365.2–365.7 px, σ ≈ 0.3 px. The full table is in
[performance and accuracy](performance.md#real-data-the-can-end-chain).

```text
cargo run --release -p vision-metrology --example inspect_canend -- \
  --scene-dir /path/to/canend/set1/normal/dome \
  --roi 420,350,420,320 --rim-radius 367 --tolerance 1.5
```

The can-end dataset is not distributed with this repository.

Units are **pixels** throughout `measure`. To report millimetres, convert the
fitted primitive through a camera calibration with the `metric` module
(`metric::pixel_to_plane`).

## See also

- [Shape-based object detection](shape-matching.md) — how `ShapeMatch::pose`,
  the fixture this module applies, is found in the first place.
- [Performance and accuracy](performance.md) — the caliper's measured bias and
  noise envelope, and the can-end reference numbers.

[`Caliper`]: ../crates/vision-metrology/src/measure/caliper.rs
[`MeasureRect`]: ../crates/vision-metrology/src/measure/placement.rs
[`MeasureArc`]: ../crates/vision-metrology/src/measure/placement.rs
[`MeasureRadial`]: ../crates/vision-metrology/src/measure/placement.rs
[`MeasureStrip`]: ../crates/vision-metrology/src/measure/placement.rs
[`MetrologyModel`]: ../crates/vision-metrology/src/measure/model.rs
[`ShapeMatch::pose`]: ../crates/vision-metrology/src/matching/matcher.rs
