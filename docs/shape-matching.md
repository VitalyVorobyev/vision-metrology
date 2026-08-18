# Shape-based object detection

Locate a modelled contour in an image under translation, rotation and uniform
scale — robustly against occlusion, clutter and changes in illumination.

```rust
use vision_metrology::{
    Rect2f, ShapeMatcher, ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig,
};

let roi = Rect2f { x: 420.0, y: 350.0, width: 420.0, height: 320.0 };
let model = ShapeModelBuilder::new()
    .build_u8(&reference.as_view(), roi, &ShapeModelConfig::default())?;

let cfg = ShapeSearchConfig { min_score: 0.6, ..Default::default() };
for m in ShapeMatcher::new().find_u8(&scene.as_view(), &model, &cfg) {
    println!("score {:.3} at {:?}, {:.1} deg", m.score, m.position, m.angle().to_degrees());
}
```

Run it on your own images:

```bash
cargo run --release -p vision-metrology --example shape_matching -- \
    --model-image ref.bmp --roi 420,350,420,320 --scene scene.bmp --out overlay.png
```

## What the score means

For a model point with unit gradient direction `tᵢ` placed at scene position
`pᵢ`, the term is `cᵢ = (R·tᵢ) · ĝ(pᵢ)`, where `ĝ` is the scene's unit gradient
direction and is **zero** wherever the gradient is weaker than
`ShapeSearchConfig::min_contrast`. The score is the mean over all `n` model
points.

Dividing by `n` — the full point count, not the number of points that
contributed — is what makes the score readable:

| Score | Meaning |
|-------|---------|
| ~1.0 | every model edge found a scene edge pointing the same way |
| ~0.7 | roughly 30 % of the contour is missing, occluded, or misoriented |
| ~0.0 | no agreement at all |

Because only *directions* are compared, the measure is unaffected by any
monotonic change in illumination: halving the contrast leaves the score alone
until edges fall below `min_contrast`.

`ShapeMatch::support` counts the points that found any gradient at all. When
`support / point_count` is high but `score` is low, the scene has edges where
the model expects them but pointing the wrong way — the signature of a wrong
pose on a busy scene, or of the wrong `Polarity`.

## Choosing a polarity

| Mode | Score | Accepts |
|------|-------|---------|
| `Match` (default) | `(1/n) Σ cᵢ` | only the polarity the model was built with |
| `IgnoreGlobal` | `abs((1/n) Σ cᵢ)` | the object **or** its full contrast inversion |
| `IgnoreLocal` | `(1/n) Σ abs(cᵢ)` | each edge inverted independently |

Use `IgnoreGlobal` when the same part is imaged under inverted contrast —
bright-field versus dark-field illumination is the standard case. Use
`IgnoreLocal` sparingly: it scores a contour highly even when every second edge
is inverted, which no real object does, so it is the most prone to false
positives on cluttered scenes.

**Models built from geometry need a polarity-insensitive mode.** A contour or a
CAD outline records geometry, not photometry: nothing in a polyline says whether
the part is darker or brighter than its background, so `ShapeModel::from_polylines`
has to guess the normal's sign. Guessing wrong under `Polarity::Match` yields a
score near zero — a failure that looks like a bug. Build contour models with
`IgnoreLocal` or `IgnoreGlobal`; when a reference image exists, use
`ShapeModel::from_edgels` and keep `Match`.

## Getting a clean model

The single most effective knob is `ShapeModelConfig::min_contrast`, the gradient
floor a reference-image edge must clear to enter the model.

On a low-relief part — a stamped metal surface, say — the edge detector's
automatic threshold also admits faint shading gradients. Those do not repeat
frame to frame, so they contribute nothing but noise and *dilute* the score,
because the sum is divided by the full point count. Raising `min_contrast` drops
them. Measured on 1280×1024 can-end frames, with everything else at its default:

| `min_contrast` | frames found | median score |
|---|---|---|
| 0 (default) | 50 / 50 | 0.785 |
| 200 | 50 / 50 | 0.863 |
| 400 | 50 / 50 | **0.998** |

The units are Scharr response on the input pixel scale: a clean black/white step
in `u8` gives a gradient magnitude of about 2000. Re-tune for `u16` and `f32`
input, whose pixel scales differ by orders of magnitude — and note that
`ShapeSearchConfig::min_contrast`, which gates the *scene*, needs the same
treatment.

Two further model knobs:

- **`max_points`** (default 512 per level) caps the model size. Decimation is
  always spatially uniform; keeping the strongest `n` points instead would
  concentrate the model on the highest-contrast part of the contour and break
  the `score ≈ 1 − occluded_fraction` reading.
- **`origin`** sets the reference point. The default — the level-0 centroid —
  minimises the model radius, hence the number of angle steps the search needs.

## Speed

`greediness` controls early termination. At `0.0` the abort bound is provably
safe: it never rejects a pose that would have scored at least `min_score`, which
makes it the reference for tests. The default `0.9` is much faster and *can*
miss a match whose first-evaluated points are the occluded ones — which is why
model points are stored in a spatially stratified order, so that any prefix
samples the whole contour rather than one arc.

### Where the time actually goes

On a 1280×1024 scene with an 800-point model and a full 360° search:

| | Time |
|---|---|
| whole `find_u8` call | 7.8 ms |
| ...of which: building the direction-field pyramid | ~5.3 ms |
| ...of which: the pose search itself | ~2.5 ms |
| same call at `greediness = 0.0` | 11.2 ms |
| model creation | 0.49 ms |

**Scene preprocessing dominates, not the search.** Micro-optimising the scoring
loop is the wrong move here; restricting the level-0 and level-1 fields to the
bounding boxes of surviving candidates is the one that would pay.

`max_candidates` (default 128) bounds how many coarse-level candidates descend
the pyramid. On a textured scene the cap can bite; `ShapeMatcher::truncated()`
reports when it did, which distinguishes "not present" from "gave up".

## Persisting a model

With the `serde` feature, a model built offline ships to the machine as a
versioned JSON document:

```rust
let json = model.to_json()?;              // {"format_version":1,"model":{...}}
let model = ShapeModel::from_json(&json)?; // refuses unknown versions
```

Python mirrors this as `model.save(path)` / `ShapeModel.load(path)`. The
`format_version` gate means an old runtime refuses a newer document instead of
silently mis-reading it.

## Pose and coordinates

`ShapeMatch::pose` is a `Similarity2f` that maps **reference-image coordinates
to scene coordinates**, with the model origin already baked in:

```text
pose = Translation(position) ∘ sR ∘ Translation(−model.origin())
```

So a fiducial or a measurement-ROI corner measured on the reference image maps
into the scene with `pose * p`, no offset arithmetic. `position` is separately
available as the place the model's own reference point landed.

Positions are level-0 pixel centres, as everywhere in this workspace.

## Angle ranges are not wrapped

`angle_range` is used verbatim. A range straddling ±π is written unwrapped —
170° to 190° is `(2.967, 3.316)`, not `(2.967, −2.967)`. Only the angle of a
finished `ShapeMatch` is wrapped into `(−π, π]`.

## Refinement

| Mode | Cost | Accuracy |
|------|------|----------|
| `None` | free | ±0.5 px, ±½ angle step |
| `Interpolate` (default) | ~8 score evaluations | 0.1–0.3 px |
| `LeastSquares` | one pass per iteration over the model | 0.02–0.05 px |

`LeastSquares` is correspondence-free: it samples the gradient magnitude at
`p − n`, `p` and `p + n` for each model point and fits a parabola to get the
signed distance to the true edge. There is no nearest-neighbour search and no
per-iteration allocation.

Rank deficiency is expected, not exceptional. A circular part has radial
normals, so `cross(vᵢ, nᵢ) = 0` everywhere and its rotation is genuinely
unobservable — a circle has no orientation. The solver falls back from 4 to 3 to
2 degrees of freedom rather than inverting a singular matrix, and returns the
unrefined pose if even that fails.

## Real-data results

`examples/shape_matching.rs` has a `--scene-dir` mode that runs a model over a
whole directory and reports the spread. Measured on 1280×1024 beverage can ends
(model built from the first frame of each folder, full 360° search, single
core), with the per-folder `min_contrast` from the table above:

| Folder | Found | Median score | Angle span | Median time |
|---|---|---|---|---|
| dome illumination | 50 / 50 | 0.998 | 352° | 15 ms |
| bright field | 50 / 50 | 0.995 | 353° | 13 ms |
| dark field | 50 / 50 | 0.961 | 352° | 15 ms |
| second product, dome | 48 / 48 | 0.998 | 343° | 63 ms |
| conveyor carrier, bright field | 19 / 19 | 0.998 | 231° | 18 ms |

A model built under dome illumination and searched against the *dark-field*
images of the same parts — fully inverted contrast — scores 0.908 with
`Polarity::IgnoreGlobal` and is not found at all with `Polarity::Match`.

## Reference

Steger, *Similarity Measures for Occlusion, Clutter, and Illumination Invariant
Object Recognition*, DAGM 2001.
