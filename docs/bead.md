# Tracking a bead

An adhesive or sealant bead, a seam, or a stripe seen at an angle usually has a path that
is already roughly known: from a reference part, the previous frame, a fixture pose or a
robot program. [`BeadTracker`] takes that path as a prior polyline and returns:
- the refined centreline;
- the bead's position and width at every station along it;
- the statistics that say how far to trust both.

Its cost grows with the number and length of the search profiles, not with the image area.

```text
prior ─► stations ─► strips along the normals ─► edge pairs ─► robust, regularised solve
            ▲                                                         │
            └──────────── move along the normals, resample ◄──────────┘   (tracking passes)

refined stations ─► stricter strips ─► one hit or reason per station + statistics   (final stage)
```

![Bead tracking: an S-shaped bead with a gap and a brighter step beside one edge, tracked from a prior 4 px off with a bump in it](assets/bead-tracking.png)

The prior is grey, the final stage's strips faint blue, the refined centreline green and
the final edges yellow. The rejected stations are red at the gap, where the strips find no
edge, and orange at the start, where the step lies within `clearance` of the bead's edge.
`examples/bead_track.rs` draws the same overlay for a scene of its own.

## When to use it

- **The bead follows a curve, and you have a prior for it.** Each station measures along
  its own normal, so the tool follows any open curve.
  - A prior that is translated, rotated or smoothly bent by up to the tracking reach
    (15 px by default) converges onto the bead in a few passes. Smoothly means
    wavelengths well above `2π·bending_px`, about 25 px by default. The measured basins
    are in [performance and accuracy](performance.md#bead-tracking-convergence).
  - Local error in the prior converges much more slowly, because each pass suppresses
    short corrections ([The regulariser](#the-regulariser)). Examples are a kink, a bump
    shorter than that, or the chords of a coarse polygon on a bend.
- **Use a [`Caliper`](measure.md) or a `MetrologyModel` instead** for the edges of a
  located part: lines and circles in the part's own frame.
- **Use `laser` instead** for a stripe that stays roughly parallel to an image axis: it
  scans rows or columns and needs no prior.
- **It does not cover:** closed or branching curves, or finding a bead with no prior at
  all.

```rust
use vision_metrology::measure::{BeadConfig, BeadTracker};
use vision_metrology::{Image, Point2f};

// A vertical light bead 40 px wide, centred on x = 100, anti-aliased.
let (w, h) = (200usize, 160usize);
let data: Vec<u8> = (0..w * h)
    .map(|i| {
        let x = (i % w) as f32;
        let cover = (x + 0.5 - 80.0).clamp(0.0, 1.0) - (x + 0.5 - 120.0).clamp(0.0, 1.0);
        (40.0 + 160.0 * cover).round() as u8
    })
    .collect();
let img = Image::from_vec(w, h, data).unwrap();

// The prior: 6 px off the bead's centreline.
let prior = [Point2f::new(94.0, 20.0), Point2f::new(94.0, 140.0)];
let mut tracker = BeadTracker::new(BeadConfig::default()).expect("a valid config");
let bead = tracker.track(&img.as_view(), &prior).expect("a usable prior");

for s in &bead.samples {
    match s.hit {
        Ok(hit) => println!("{:?}: width {:.3}", s.point, hit.pair.width),
        Err(reason) => println!("{:?}: {}", s.point, reason.as_str()),
    }
}
let stats = bead.summary.stats.expect("the bead was found");
println!("support {}, width {:.3} ± {:.3}", bead.summary.support, stats.width_mean, stats.width_std);
```

`BeadTracker` owns both stages' calipers and every buffer. Reuse it from frame to frame:
after the first call at a given size, `track` allocates only its result.

## Conventions

- **Stations.** The prior is resampled to `N` stations, uniform in arc length:
  `round(L / spacing) + 1` of them, at least 2, for a prior of length `L`. `N` is fixed
  for the call, so station `i` is the same station in every pass and in the result. The
  last station is the prior's end exactly.
- **Tangent.** A station's tangent `t` points towards the curve's end. It is the chord over
  `±tangent_window_px` of arc length. Near the ends the window shrinks, so the chord stays
  centred on its station, and the two end stations take the chord to their neighbour. A
  centred chord is exact on a circle and smooths a coarse prior's corners.
- **Normal.** `n = t.perp() = (−t_y, t_x)`. With y down, that is to the right of travel as
  drawn on screen. Reversing the prior reverses both `t` and `n`.
- **Offsets** are signed distances along `+n`, in pixels.
- **Edge order.** Each strip scans from `−n` to `+n`. A pair's `first` edge is on the `−n`
  side and its `second` on the `+n` side.
- **Polarity.** A `Light` bead is brighter than its background: along the scan, a rising
  edge, then a falling one. A `Dark` bead is the reverse. The same bead gives the same
  pair whichever way round the prior runs.

## Two stages

Every station is measured twice, by two separate calipers:

| Stage | Config | Default | Purpose |
|---|---|---|---|
| Tracking | `BeadConfig::track` | reach ±15 px, ±2 px along the curve, step 1 px, no obliquity gate | Evidence for where the bead is. Wide and permissive. |
| Final | `BeadConfig::measure` | reach ±3 px, ±1 px along the curve, step 0.5 px, obliquity ≤ 30° | The reported positions and widths. Narrow and strict. |

The tracking passes move the curve. The final stage places new strips on the refined
stations and only measures: there is no regularisation and no feedback into the curve.
Keeping the stages apart is what lets the search be generous while the reported
dimensions stay strict:
- a pair that only the wide tracking search accepts cannot reach the result;
- each stage keeps its own caliper, whose profile length is fixed for the stage.

A strip's endpoints are stored in `f32`, so the spacing of its samples, and with it σ in
samples, differs between stations by a few ulps. Each caliper therefore refills its
smoothing kernel in place at most stations. That costs no allocation and little time
([speed](performance.md#speed)), and occasionally the kernel radius changes by one
sample. The effect on an edge position stays around 1e-4 px.

Each strip is a [`MeasureStrip`](measure.md#strips) along the station's normal, long
enough for a pair at the stage's reach:

```text
half-length = max_offset + max_width/2 + clearance + 3σ + step
```

It is rounded up so that its length is a whole number of profile steps.

## Choosing the pair: the gates

Each strip's caliper returns every edge of either polarity (`EdgeSelect::All`,
`PolaritySelect::Any`). The tracker sorts them along the scan and keeps the pairs that pass
every gate, in this order:

1. **The caliper.** When the caliper finds no edge, its own [`RejectReason`](measure.md#rejectreason)
   is the station's: `BeadReject::Caliper(..)`. An edge located outside the image, in
   border fill, is not evidence, so the tracker drops it. A strip with no edge left
   reports `Caliper(OffImage)`.
2. **`NoPair`.** No edge of the bead's leading polarity is followed by one of the trailing
   polarity. Edges between the two are allowed, so a highlight inside the bead does not
   break its pair.
3. **`Width`.** The pair is narrower than `min_width` or wider than `max_width` (inclusive).
4. **`Offset`.** The pair's midpoint is outside the stage's window, `±max_offset` from the
   station. On a bend, the window's concave side stops at `0.9 / |κ|`, short of the centre
   of curvature, so no observation asks the curve to cross it. The window gates
   evidence; the step scale ([below](#the-regulariser)) is what keeps a step from
   folding the curve.
5. **`Clearance`.** This gate runs only when `clearance` is set. Another edge lies within
   `clearance` px outside either edge of the pair. Use it when a distractor beside the
   bead would otherwise sit close enough to bias an edge. A station there is rejected
   rather than measured.

When no pair survives, the reason is the gate that removed the last one. The surviving
pairs are scored by their weaker edge, discounted by their offset:

```text
s = min(a₁, a₂) / (1 + (offset / max_offset)²)
```

The best score wins. A tie goes to the smaller `|offset|`, then to the earlier edges.

6. **`Ambiguous`.** This gate runs only when `min_margin` is set. The winner's margin over
   the best other pair, `1 − s₂/s₁`, is below `min_margin`.

`BeadHit::confidence` is that margin times the balance of the two amplitudes, `min/max`. It
is for diagnostics; no gate reads it.

## The regulariser

A pass observes an offset `d̂ᵢ` at each station that found the bead. It then solves for the
corrections `dᵢ` that minimise

```text
Σ wᵢ ρ(dᵢ − d̂ᵢ)  +  λ0 Σ dᵢ²  +  (ℓ1/h)² Σ (Δd)ᵢ²  +  (ℓ2/h)⁴ Σ (Δ²d)ᵢ²
```

and moves each station by `dᵢ` along its normal. In plain terms:
- **Stations without a pair have weight 0.** The penalties bridge them, so a gap is filled
  smoothly from both sides.
- **`loss` (`ρ`) is robust.** The default is Huber with a 1 px constant, so one wrong pair
  pulls the curve far less than its offset; Tukey removes it. A pass first solves by
  least squares, then reweights up to `irls_iters` times; Tukey anneals from the largest
  residual down to its constant over those solves. A robust loss also down-weights a
  large local error of the prior's own, which is one reason that error decays slowly.
- **`damping` (`λ0`) is trust in the prior.** At 0, the default, nothing holds the curve
  back from the data.
- **`tension_px` (`ℓ1`) and `bending_px` (`ℓ2`) penalise the correction's slope and
  curvature, not the bead's.** The tracker has no preferred shape: a translated or rotated
  prior is corrected in one pass at nearly full strength.
- **The bending length sets the shortest correction a pass makes.** A correction that
  varies along the curve with angular frequency `ω` passes at
  `1 / (1 + λ0 + ℓ1²ω² + ℓ2⁴ω⁴)`. A correction whose wavelength is shorter than about
  `2π·ℓ2` is suppressed; a longer one passes. The default `ℓ2` of 4 px puts that boundary
  at about 25 px. A bead cannot bend on a scale much shorter than its own width, so a
  short correction is more often noise or a wrong pair than the bead.
- **The penalties act on each pass's correction, not on the curve.** The curve keeps
  converging towards the measured centres over passes. The regulariser sets how robustly
  and how fast. A short-scale error in the prior shrinks by the suppressed fraction each
  pass, so it takes many passes, while station noise is held back in every pass, so the
  pass count bounds how much of it reaches the centreline.
- **The penalties are lengths, so the result does not depend on `spacing`.** The energy is
  the discretisation of a continuous one. Halving the spacing doubles the stations, not
  the stiffness.

Then the pass takes the step. A step scale below 1 shortens it if the full step would
fold the curve:
- no station may move more than 0.9 of the way to its centre of curvature;
- no segment between adjacent stations may keep less than a tenth of its length along its
  old direction.

The moved stations are resampled to the same `N`, and the loop runs again, up to `passes`
times. It stops early, `Converged`, once a pass's solved correction is below `tol` at every
station and was applied in full. `Converged` says the loop stopped moving the curve, not
that the curve sits on the bead. A coarse prior can stop moving while its short-scale
error remains, and the final stage's `center_rms` and `center_max_dev` are the evidence
of fit.

A pass that finds the bead at fewer than `min_support` of its stations, or at none, does not
move the curve. The loop then stops with `TooFewValid`, and the final stage still runs on
the curve as it stands.

The tuning defaults:
- **`tension_px` 2.** It keeps a long gap or a bare end from tilting the curve, at a cost
  of a fraction of a percent on a rotated prior.
- **`bending_px` 4.** It balances local error in the prior against noise in the curve
  ([the bending length](performance.md#the-bending-length)).
  - For a coarse or locally wrong prior, use a shorter bending length, 2 or 3 px. A
    shorter length corrects short-scale error in fewer passes but passes more noise into
    the curve. More passes help less.
  - For a smooth prior on a noisy bead, a longer one, up to 8 px, gives a quieter curve.
  - Either way, read `center_max_dev` to see whether the curve sits on the bead.
- **`tangent_window_px` 10.** Shorter is noisier on a coarse prior. Longer tilts the
  normals where the curvature changes, and a tilted normal makes the width read long, by
  `1/cos` of the tilt.

## Reading the result

[`TrackedBead`] holds the following:
- **`centerline`**: the refined stations, and `spacing` between them.
- **`samples`**: one per station, from the final stage. Each has the station's `point` and
  `normal`, and either a `BeadHit` (the pair, its `offset` from the station and its
  `confidence`) or the `BeadReject` that removed the last pair.
- **`summary`**:
  - `support`: the fraction of stations with a hit;
  - `longest_gap`: the longest run of stations without one, in pixels of arc length;
  - `stats`: `None` without a hit. Otherwise `n_used`, `center_rms` and `center_max_dev`
    (how far the measured midpoints sit from the refined curve), and the width's mean,
    standard deviation, minimum and maximum;
  - `rejects`: the stations rejected, counted by reason.
- **`track`**: one `BeadPass` per tracking pass, and `stop`, why the loop ended. A pass
  has its support, gap and rejections. Its `solve` is `None` when it found too few pairs
  to solve. Otherwise `solve` holds the correction it applied, the residual of the
  evidence against the solved correction, its step scale and its reweighted solves.

Gate an inspection on these, not on the curve alone:

```rust
use vision_metrology::measure::{BeadStop, TrackedBead};

fn bead_ok(bead: &TrackedBead) -> bool {
    let Some(stats) = bead.summary.stats else {
        return false; // nothing was found
    };
    bead.track.stop != BeadStop::TooFewValid
        && bead.summary.support >= 0.95
        && bead.summary.longest_gap <= 20.0      // px: the longest allowed interruption
        && stats.center_rms <= 0.25              // px: the curve sits on the measurements...
        && stats.center_max_dev <= 0.5           // ...everywhere, not just on average
        && stats.width_min >= 45.0
        && stats.width_max <= 55.0
}
```

- **A large `center_rms` or `center_max_dev`** means the refined curve is not where the
  final stage measured the bead. There are three causes:
  - the tracking ran out of passes (`track.stop`, and the last pass's
    `solve.correction_max`);
  - the prior carried a local error that the bending length suppresses (shorten
    `bending_px` or add passes);
  - the bead has structure shorter than `2π·bending_px`.

  A station whose curve is off by more than the final stage's reach is rejected with
  `Offset` instead.
- **A rejection reason that dominates `rejects`** usually names the misconfiguration:
  - `Width`: the width range;
  - `Offset`: the reach;
  - `Caliper(NoEdge)`: the threshold, or a missing bead;
  - `Clearance`: a distractor.

## Seeing why

`track` keeps only what the next call reuses. When a station is rejected, or the curve
is not where you expect it, `measure::diagnostics::explain_bead` tracks once more and keeps
every station's evidence:

```rust
use vision_metrology::measure::BeadTracker;
use vision_metrology::measure::diagnostics::explain_bead;
use vision_metrology::{Error, ImageView, Point2f};

fn why(
    tracker: &mut BeadTracker,
    img: &ImageView<'_, u8>,
    prior: &[Point2f],
) -> Result<(), Error> {
    let trace = explain_bead(tracker, img, prior)?;
    for (i, st) in trace.measure.iter().enumerate() {
        let Err(reason) = st.hit else { continue };
        println!(
            "station {i}: {} (window {:?} px, {} edges on the strip)",
            reason.as_str(),
            st.window,
            st.caliper.edges.len()
        );
    }
    Ok(())
}
```

| `BeadTrace` field | What it holds |
|---|---|
| `result` | what `track` returns for the same tracker, image and prior, to the bit |
| `passes` | one `BeadPassTrace` per tracking pass: its `stations`, and per station the `weights` the solve gave the observations and the `corrections` the pass applied |
| `measure` | the final stage's stations, parallel to `result.samples` |

Each `BeadStationTrace` holds:
- the station's `point`, `tangent` and `normal`;
- the `window`, the offsets `(lo, hi)` a pair's midpoint had to fall in;
- the `strip` its caliper measured, from `−n` to `+n`, with the station at its middle;
- the caliper's whole `CaliperTrace` ([Seeing why](measure.md#seeing-why)). An edge's `t`
  is its distance from the strip's start, so its offset from the station is `t` minus
  half the strip's length;
- the `hit`: the pair, or the gate that rejected every pair.

A tracking pass's stations are where the pass measured, before it moved the curve. A
pass's support, rejections and solve are the same entry of `result.track.passes`.

Reading a rejected station:
- **`Caliper(..)`.** The caliper's own reason: read its trace as for any caliper.
  `Caliper(OffImage)` with edges in the trace means every edge lay in border fill.
- **`NoPair`, `Width`.** `caliper.edges` holds every edge the gates paired, with its
  polarity and position. Look for the bead's two edges and their distance.
- **`Offset`.** A pair of a valid width is there, but its midpoint is outside `window`:
  the curve is further from the bead than the stage's reach. In a tracking pass that is
  the prior's error. In the final stage, the tracking did not bring the curve there.
- **`Clearance`, `Ambiguous`.** Another edge lies within `clearance` of the pair, or a
  second pair scores nearly as well; `caliper.edges` shows which.
- **A hit with a weight near 0 in a pass** was an outlier to the robust loss: its pair
  disagrees with its neighbours'.

`explain_bead` runs the tracker once, through the same code as `track`, with every strip
measured through `diagnostics::explain`. Its result is `track`'s to the bit, and the
tracker's later results are unchanged. It allocates a caliper trace per station and
pass, so it belongs in a tool that shows why a bead measured what it did, not in the
inspection loop. With the `serde` feature, a `BeadTrace` serializes.

In Python, `tracker.explain(image, prior)` takes `track`'s arguments. Each pass is arrays,
one row per station; the final stage is a list of `BeadStationTrace`:

```python
trace = tracker.explain(image, prior)
trace.result               # the TrackedBead that track() returns
p = trace.passes[0]
p.points, p.normals, p.windows        # (N, 2) float32
p.observed, p.weights, p.corrections  # (N,) float32; observed is NaN where rejected
p.reject[i], p.calipers[i].edges      # why station i was rejected, and what it saw
st = trace.measure[i]                 # the final stage's station i
st.reject, st.window, st.caliper.candidates
```

A rejected station is part of the trace and never raises.

## From frame to frame

The refined centreline is a polyline in image coordinates, so the previous result is the
next frame's prior as it stands:

```rust
use vision_metrology::measure::{BeadTracker, TrackedBead};
use vision_metrology::{Error, ImageView};

fn next(
    tracker: &mut BeadTracker,
    frame: &ImageView<'_, u8>,
    last: &TrackedBead,
) -> Result<TrackedBead, Error> {
    tracker.track(frame, &last.centerline)
}
```

When the part moves, move the prior with it. A `ShapeMatch::pose`, or any `Similarity2f`
or `Affine2f`, maps every point:

```rust
use vision_metrology::{Affine2f, Point2f, Similarity2f};

fn moved(prior: &[Point2f], pose: &Similarity2f) -> Vec<Point2f> {
    prior.iter().map(|&p| pose * p).collect()
}

fn warped(prior: &[Point2f], affine: &Affine2f) -> Vec<Point2f> {
    prior.iter().map(|&p| affine * p).collect()
}
```

A pure shift measured between frames with `corr::displacement` is a translation of the
prior:

```rust
use vision_metrology::corr::{DisplacementConfig, displacement};
use vision_metrology::{Error, ImageView, Point2f, Rect2f};

fn shifted(
    prev: &ImageView<'_, u8>,
    curr: &ImageView<'_, u8>,
    prior: &[Point2f],
) -> Result<Vec<Point2f>, Error> {
    let cfg = DisplacementConfig {
        // A textured window that moves with the bead.
        window: Rect2f { x: 300.0, y: 200.0, width: 96.0, height: 96.0 },
        ..DisplacementConfig::default()
    };
    let d = displacement(prev, curr, &cfg)?;
    Ok(prior.iter().map(|&p| p + d.shift).collect())
}
```

A prior moved this way is off by a translation or a smooth deformation. When that is
within the tracking reach, the tracker converges in a few passes. Error in the prior's
own shape converges more slowly ([When to use it](#when-to-use-it)). Either way,
`summary` says whether the curve sits on the bead.

## Python

```python
import numpy as np
import vision_metrology as vm

tracker = vm.BeadTracker(vm.BeadConfig(min_width=30.0, max_width=80.0, polarity="light"))
prior = np.array([[94.0, 20.0], [94.0, 140.0]], dtype=np.float32)   # (N, 2)
bead = tracker.track(image, prior)          # uint8, uint16 or float32

bead.centerline        # (N, 2) float32: the next frame's prior
bead.width, bead.offset, bead.center   # per station; NaN where rejected
bead.reject            # per station: None, or "no_edge", "width", "offset", ...
bead.support, bead.longest_gap, bead.center_rms, bead.width_mean
bead.stop              # "converged", "pass_limit" or "too_few_valid"
bead.rejects           # {"width": 3, ...}, in a fixed order
```

An invalid config, or a prior with fewer than two points, a non-finite point or no length,
raises `ValueError`. A missing bead does not raise: every station is rejected.

## Limitations

- **Open curves only.** It does not handle closed or branching curves, or more than one
  bead per call.
- **The tracker moves the curve sideways only.** Stations past the bead's end are
  rejected, and the curve's ends do not move along it. Where no station supports it, at a
  gap or a bare end, the curve is interpolated or extrapolated by the penalties, and
  `longest_gap` says how far.
- **Strips are straight.** On a bend of radius `R`, averaging `±half_width` along a
  straight strip pulls the measured centre towards the concave side by about
  `half_width² / (6R)`. Keep `half_width` small on tight bends.
- **Short-scale error in the prior decays over passes, not in one.** This includes a kink,
  a bump narrower than `2π·bending_px`, or a coarse polygon's chords on a bend. The
  penalties act on each pass's correction. With the defaults such an error can remain
  after the last pass, and even under `Converged`, because each correction is too small
  to apply. `center_rms` and `center_max_dev` show what remains. A shorter
  `bending_px`, more passes, a previous result or a densely sampled path all help.
- **Thresholds are on the input pixel scale.** Scale `threshold` with the image, e.g. by
  256 for a `u16` image that holds 8-bit data shifted up.
- **`Locate::HalfContrast` is fragile near distractors.** It refines every edge on the
  strip, and one that fails rejects the whole strip. Its flanks must not reach the
  opposite edge (`flank_far_px < min_width`), and `Locate::MidpointCrossing` is not
  allowed: it finds a single edge.

## See also

- [Measuring a located part](measure.md): the caliper and strip the tracker is built on,
  edge location, and `RejectReason`.
- [Performance and accuracy](performance.md): the tracker's speed, its accuracy on
  synthetic beads, its convergence basins, and the bending length's trade-off.

[`BeadTracker`]: ../crates/vision-metrology/src/measure/bead/mod.rs
[`TrackedBead`]: ../crates/vision-metrology/src/measure/bead/result.rs
