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

There are three placements, and the choice is about the geometry of the edge
being crossed, not a style preference:

| Placement | Scans | Averages | Use for |
|---|---|---|---|
| [`MeasureRect`] | along its own axis | across, on a **straight chord** | a straight edge, at any angle |
| [`MeasureArc`] | along a circular arc | radially | a feature that *crosses* a circular path (a gear tooth, a slot, the tab on a can end) |
| [`MeasureRadial`] | radially | along the arc, at constant radius | the circular edge itself |

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

## `MeasureConfig`

```rust
pub struct MeasureConfig {
    pub sigma: f32,
    pub threshold: f32,
    pub polarity: PolaritySelect,
    pub select: EdgeSelect,
    pub step: f32,
    pub max_obliquity_deg: f32,
    pub border: BorderMode<f32>,
}
```

- **`sigma`** — the Gaussian σ of the 1-D derivative-of-Gaussian kernel, in
  pixels. Roughly the edge blur to expect: too small and noise produces
  spurious edges, too large and neighbouring edges merge.
- **`threshold`** — the minimum `|DoG response|` to report an edge, on the
  input pixel scale like every other threshold in this workspace (re-tune for
  `u16`/`f32`).
- **`polarity`** (`PolaritySelect::{Any, Rising, Falling}`) — which
  transitions count. A caliper that should only ever see a dark-to-bright
  edge and instead reports a bright-to-dark one is a useful signal that
  something is wrong with the part, not just noise to filter.
- **`select`** (`EdgeSelect::{All, First, Last, Strongest}`) — which of the
  surviving edges to keep when more than one crosses the threshold.
  `Strongest` is the sane default once a model's geometry is already
  approximately right (`MetrologyObject::new` picks it) — a caliper on a
  nominal edge should report *that* edge, not every edge it happens to cross.
- **`step`** — profile sampling step along the scan axis. `1.0` is one entry
  per pixel; oversampling (`0.5`) buys resolution on a sharp edge at
  proportional cost, and `sigma` is in the same units, so halving `step`
  means doubling `sigma` for equivalent smoothing.
- **`max_obliquity_deg`** — the obliquity gate. A caliper that crosses an edge
  at a glancing angle reports a position along its own scan axis rather than
  the edge's true normal, and the two differ by `1/cos θ`; at a corner there
  is no meaningful crossing at all. Comparing the local image gradient
  against the scan direction and rejecting beyond this angle is what keeps a
  bad caliper *out* of a fit rather than merely down-weighted. `180.0`
  disables the check. Idea adapted from the caliper in `rtvt-pano`.
- **`border`** — sampling behaviour when the caliper overhangs the image.

## `RejectReason`

`Caliper::measure` returns `Result<&[MeasureEdge], RejectReason>` — a caliper
that finds nothing is a *result*, not an error, and which gate rejected it is
the difference between "the part is missing" and "the search window is too
short":

| Reason | Meaning |
|---|---|
| `ProfileTooShort` | fewer than 3 profile samples — the placement is degenerate (near-zero `half_len`) |
| `NoEdge` | no response reached `threshold` anywhere in the window |
| `WrongPolarity` | edges were found, but none had the polarity `MeasureConfig::polarity` asked for |
| `TooOblique` | the best edge crossed at more than `max_obliquity_deg` from the scan direction |
| `OffImage` | the caliper reached outside the image, so the profile is partly border fill |

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
radius 365.2–365.7 px, σ ≈ 0.3 px — see `system-design.md` for the full table.

```text
cargo run --release -p vision-metrology --example inspect_canend -- \
  --scene-dir ~/privatedata/canend/set1/normal/dome \
  --roi 420,350,420,320 --rim-radius 367 --tolerance 1.5
```

Units are **pixels** throughout `measure`. Millimetres arrive with the
`metric` module (roadmap B5), which converts a fitted primitive through a
calibration.

## See also

- [Shape-based object detection](shape-matching.md) — how `ShapeMatch::pose`,
  the fixture this module applies, is found in the first place.
- `docs/system-design.md` — the chord-bias measurement and the design
  decisions behind `RejectReason` and the `Result`-returning `measure`/`apply`.

[`Caliper`]: ../crates/vision-metrology/src/measure/caliper.rs
[`MeasureRect`]: ../crates/vision-metrology/src/measure/caliper.rs
[`MeasureArc`]: ../crates/vision-metrology/src/measure/caliper.rs
[`MeasureRadial`]: ../crates/vision-metrology/src/measure/caliper.rs
[`MetrologyModel`]: ../crates/vision-metrology/src/measure/model.rs
[`ShapeMatch::pose`]: ../crates/vision-metrology/src/matching/matcher.rs
