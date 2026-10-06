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

## When to use it

- **The bead follows a curve, and you have a prior for it.** Each station measures along
  its own normal, so the tool follows any open curve, and a prior that is off by up to the
  tracking reach (15 px by default) converges onto the bead.
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
- each caliper keeps its own profile length and smoothing kernel across stations.

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
   of curvature, so a correction can never fold the curve.
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
  pulls the curve far less than its offset; Tukey removes it. They are applied by
  reweighted solves (`irls_iters`).
- **`damping` (`λ0`) is trust in the prior.** At 0, the default, nothing holds the curve
  back from the data.
- **`tension_px` (`ℓ1`) and `bending_px` (`ℓ2`) penalise the correction's slope and
  curvature, not the bead's.** The tracker has no preferred shape: a translated or rotated
  prior is corrected in one pass at nearly full strength.
- **The bending length sets the shortest correction a pass makes.** A correction that
  varies along the curve with angular frequency `ω` passes at `1 / (1 + ℓ1²ω² + ℓ2⁴ω⁴)`. A
  correction whose wavelength is shorter than about `2π·ℓ2` is suppressed; a longer one
  passes. The default `ℓ2` of 8 px puts that boundary at about 50 px. A bead cannot bend on
  a scale much shorter than its own width, so a shorter correction is noise or a wrong
  pair.
- **The penalties are lengths, so the result does not depend on `spacing`.** The energy is
  the discretisation of a continuous one. Halving the spacing doubles the stations, not
  the stiffness.

Then the pass takes the step: a step scale below 1 shortens it if the full step would fold
the curve. The moved stations are resampled to the same `N`, and the loop runs again,
up to `passes` times. It stops early, `Converged`, once a pass moves no station by more
than `tol`.

A pass that finds the bead at fewer than `min_support` of its stations, or at none, does not
move the curve. The loop then stops with `TooFewValid`, and the final stage still runs on
the curve as it stands.

The tuning defaults:
- **`tension_px` 2.** It keeps a long gap or a bare end from tilting the curve, at a cost
  of a fraction of a percent on a rotated prior.
- **`bending_px` 8.** It gave the most accurate curve on noisy beads with a smooth prior.
  Shorter values follow a coarse polygon prior's chords more closely but pass more noise.
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
- **`track`**: one `BeadPass` per tracking pass (its support and gap, the correction it
  applied, the residual of the evidence against the solved correction, its step scale and
  rejections), and `stop`, why the loop ended.

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
        && stats.center_rms <= 0.25              // px: the curve sits on the measurements
        && stats.width_min >= 45.0
        && stats.width_max <= 55.0
}
```

- **A large `center_rms`** means the refined curve is not where the final stage measured
  the bead. Either the tracking did not converge (check `track.stop` and the last pass's
  `correction_max`), or the bead has structure shorter than `2π·bending_px`.
- **A rejection reason that dominates `rejects`** usually names the misconfiguration:
  - `Width`: the width range;
  - `Offset`: the reach;
  - `Caliper(NoEdge)`: the threshold, or a missing bead;
  - `Clearance`: a distractor.

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

The prior only has to be within the tracking reach. The tracker does the rest, and
`summary` says whether it managed.

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
- **High-frequency error in the prior decays over passes, not in one.** The penalties act
  on each pass's correction. A coarse polygon prior on a tight bend leaves its chords'
  short-scale error in the curve, and `center_rms` shows what remains. A previous result
  or a densely sampled path makes a better prior.
- **Thresholds are on the input pixel scale.** Scale `threshold` with the image, e.g. by
  256 for a `u16` image that holds 8-bit data shifted up.
- **`Locate::HalfContrast` is fragile near distractors.** It refines every edge on the
  strip, and one that fails rejects the whole strip. Its flanks must not reach the
  opposite edge (`flank_far_px < min_width`), and `Locate::MidpointCrossing` is not
  allowed: it finds a single edge.

## See also

- [Measuring a located part](measure.md): the caliper and strip the tracker is built on,
  edge location, and `RejectReason`.

[`BeadTracker`]: ../crates/vision-metrology/src/measure/bead/mod.rs
[`TrackedBead`]: ../crates/vision-metrology/src/measure/bead/result.rs
