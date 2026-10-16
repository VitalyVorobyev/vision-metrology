# Synthetic sequence benchmark

No public dataset has many registered observations of one bead-like curve. This
generator is the public stand-in. It renders image sequences of a ribbon laid on a moving,
textured substrate and seen by one to three cameras, and it writes their exact truth. The
output is the neutral `sequence.json` manifest of
[`tools/registration_eval`](../registration_eval/README.md#the-manifest), with a few extra
fields, plus a `truth.json`. Fusion, rig poses and bead ends report on it.

Dependencies: numpy, scipy and Pillow. The self-check also uses the `vision_metrology`
bindings' `ProfileSampler` when they are installed.

```bash
python tools/sequence_bench/generate.py --list
python tools/sequence_bench/generate.py --preset fusion-multiview --out-dir /tmp/bench/fusion-multiview
python tools/sequence_bench/generate.py --all --out-dir /tmp/bench
python tools/sequence_bench/generate.py --preset fusion-regerr --reg-sigma 1.0 --reg-drift 0.1 --out-dir /tmp/bench/reg-1
python tools/sequence_bench/generate.py --preset fusion-single --invert --out-dir /tmp/bench/fusion-single-dark
python tools/sequence_bench/validate.py
python -m pytest tools/sequence_bench
```

Outputs go outside the repository, or under its git-ignored `data/`. `--seed`, `--frames`,
`--invert` (the image becomes `255 − I`, the geometry is unchanged), `--reg-sigma` and
`--reg-drift` override a preset. Each run prints, per view, the share of laid stations
with both walls observable and with one.

## Determinism

A scenario is a pure function of its configuration (invariant 12). Every random draw comes
from `numpy.random.default_rng([seed, stream, time, view])`, one stream each for the
texture, the image noise, the blank's noise, the occlusions and the registration error.
JSON floats are rounded to six decimals. Two runs write byte-identical files, and a test
checks it.

## Frames and conventions

- **Ribbon coordinates** hold the bead: its centreline, walls and caps.
- **Substrate coordinates** are fixed to the moving part. The ribbon is fixed to them,
  except in the slip variant, where `slips[t]` maps ribbon to substrate.
- **Plane coordinates** are fixed to the rig. `poses[t] = (tx, ty, θ)` maps substrate to
  plane by `q = R(θ)p + t`.
- **Image coordinates** are pixels, with integer coordinates at pixel centres. A view's
  `plane_to_image` is a 3×3 matrix (affine, or projective), and image points are
  `H·[q, 1]` dehomogenized.

Every axis has x right and y down. Lengths are in substrate units (`su`). A view's scale is
in px/su, and the presets use scales near 1. The centreline is a chain of lines and
circular arcs, continued by tangent half-lines past both ends, parametrised by arc length
`s`. At `s`, `t` is the unit tangent and `n = (−t_y, t_x)`, so +n points to the right of
travel on screen. A point's ribbon coordinates `(s, d)` are its nearest centreline location
and its signed offset along `n`. **The left wall** is at `d = −left(s)` and **the right
wall** at `d = +right(s)`.

## The scene

- **Path and motion.** A tool fixed at plane point `tool` lays material along the
  centreline at `speed` su per frame. `poses[t]` puts the tool at `curve(s_tool[t])`
  with the tangent along plane +x, so straight runs translate and arcs also rotate. The
  laid bead trails along plane −x.
- **The ribbon.** The material fraction is
  `m = E((d + left(s))/soft_l)·E((right(s) − d)/soft_r)` times its caps and gap, where `E` is
  a Gaussian CDF (`edge: erf`) or a logistic (`edge: logistic`). Each wall's half width is
  `half + slope·(s − s₀) + amp·sin(2π(s − s₀)/period + phase)`, with `s₀` the first
  station, and has its own softness, so the walls are asymmetric. The intensity is
  `substrate·(1 − m) + (level + interior + highlight)·m`: the material covers the
  substrate. `level` above the substrate's mean makes a bright bead; `--invert` gives the
  other polarity.
- **Growth.** With `growth`, material exists up to `laid[t] = s_tool[t] − lag`, so the end
  cap moves with the tool. Without it, the whole ribbon exists from the first frame.
- **Caps** at `ribbon.start` and at the laid end: `half_disc` (centre at the chord's
  midpoint, radius half the chord), `flat`, or `skewed` by `skew_deg`. Caps are exact where
  the centreline is straight, as in every cap preset.
- **Gap.** `ribbon.gap = [s0, s1]`: no material there, with soft square ends.
- **Interior stripes**, bead-fixed: boxes in `d` added to the bead's level.
- **Substrate.** A DoG-band texture on a 1 su lattice, evaluated by cubic spline. A
  `parallel` stripe follows the centreline at an offset band. `distractors` are soft
  rectangles at a station and offset, opaque, of their own level. Both are substrate
  features and also appear in the blank.
- **Views.** `centre`, `angle_deg`, `scale`, `aspect`, `shear` and `projective` define
  `plane_to_image`. `window` is the range of frames a view records, the camera handover.
  `shake` moves one view's image by `px` per frame over a range of frames: that camera
  moves although the rig is assumed fixed.
- **Camera-fixed fields.** `shading`, a gradient and a vignette, multiplies the image.
  `highlight`, a Gaussian blob at a fixed image position, adds to the material only.
- **Occlusions.** `occlusions` draws at most one ellipse per frame and view, at `rate`.
  `occluders` are explicit rectangles or ellipses. `outside` marks image regions as not
  part of the image: they read 0 and are invisible.
- **Missing observations.** `drops` removes frames from chosen views.
- **Registration error.** `registration_error` perturbs the poses handed to consumers,
  never the truth. Translation noise has σ `sigma_px` (view-0 pixels), with a drift of
  `drift_px` per frame along plane x. Rotation noise has σ `sigma_px` over half the
  view-0 diagonal. Frame 0 stays exact.
- **Rendering.** A pixel is the mean of the scene over its unit square, by `quadrature` ×
  `quadrature` Gauss–Legendre nodes (4 by default). Then come seeded Gaussian noise of σ
  `noise` DN, the optional inversion and rounding to `uint8`.

The configuration keys and defaults are in `presets.py` (`DEFAULTS`).

## Output

```text
sequence.json   the manifest
truth.json      the full truth
frames/v<view>/<time>.png      8-bit images
masks/v<view>/<time>.png       registration validity
visibility/v<view>/<time>.png  geometry visibility
blanks/v<view>/<time>.png      the same frame without the ribbon or occluders (when `blank`)
```

The **visibility** mask is 0 on every pixel that an occluder or an `outside` region
touches. The **registration** mask is 0 there, dilated by 1 px, and on the material
(`m > 0.01`) dilated by 3 px, since a growing end, a highlight or a sliding bead does not
move with the substrate.

### The manifest

It follows the [registration evaluator's format](../registration_eval/README.md#the-manifest).
`tools/registration_eval` reads it unchanged. The additions:

- **Frames** are listed view by view, each view's frames in time order, and `group` is the
  view. Every view is therefore one group, and the only handovers are between views. Each
  entry also has `view`, `time` and, when blanks are rendered, `blank`. A dropped frame is
  `{image: null, group: null, view, time}`. Frames outside a view's window are not listed.
- **`rig`**: the consumer's inputs. `units`, `times`, `tool` (plane point), `poses` (the
  given substrate→plane poses, with any injected error), and per view `id`, `shape`,
  `plane_to_image` (the nominal calibration, without shake) and `window`.
- **`truth.motion`**: per frame, the image similarity `[tx, ty, angle, scale]` from frame 0
  through the substrate, `null` when the views are not similarities.
- **`truth.boundaries`**: per frame, image polylines of the walls over runs of observable
  stations, every `boundary_every`-th station.
- **`truth.poses`**: the exact poses. **`truth.file`** names `truth.json`.
- **`truth.caps`**: per frame, the caps near the view, with `which`, `kind`, `s`, the image
  `chord` (the walls' end points), the image `rim` polyline from the left wall's end to the
  right's, and `observable` per rim point.
- **`annotations`**: every `annotate_every`-th station where both walls are observable,
  per frame, as `{frame, station, s, point, normal, left, right}`. `point` is the image of
  the centreline, `normal` the unit image direction of the station normal, and `left` and
  `right` the exact image walls on that line.
- **`seed`**: the annotated stations of the first frame that has two, from truth, so the
  registration evaluator's `--propagate` runs on it.

### `truth.json`

| Field | Meaning |
|---|---|
| `preset`, `issue`, `config` | the scenario and its full configuration |
| `times`, `frames` | the number of times, and `[time, view]` per manifest frame |
| `tool`, `s_tool`, `laid` | the tool's plane point, its arc length per time, the laid end per time |
| `poses`, `poses_given` | exact and given substrate→plane poses per time |
| `slips`, `moves_on_substrate` | ribbon→substrate poses per time, and whether they move |
| `views`, `image_offsets` | nominal `shape` and `plane_to_image` per view; the camera motion `[dx, dy]` per time and view, in px |
| `dropped` | `[time, view]` pairs |
| `ribbon` | `polarity`, `start`, `end`, `gap`, and per cap its `kind`, `s` and `rim` in ribbon coordinates (the end cap at the last time) |
| `stations` | per station: `s`, `point`, `normal`, `left`, `right` (half widths), `left_point`, `right_point`, `in_gap`, in ribbon coordinates |
| `visibility` | per time and view, one character per station, or `null` when the view has no image then |
| `occlusions` | `{time, view, kind, geometry, level}` per occluder, in image pixels |

Visibility codes: `n` not laid (yet); `g` in the gap; otherwise a digit whose bit 1 says
the left wall is observable and bit 2 the right: `0` neither, `1` left only, `2` right
only, `3` both. A wall is observable when the pixel that holds its image point lies in the
image and is visible. A station's image walls are
`apply(H_actual · se2(poses[t]) · se2(slips[t]), left_point)`, where `H_actual` is
`plane_to_image` preceded by the translation `image_offsets[t][view]`.
`render.Scenario.ribbon_to_image(t, v)` builds it.

## Presets

| Preset | For | Case |
|---|---|---|
| `tiny` | CI | every nuisance at once on two 64×40 views over six frames, one of them projective |
| `fusion-single` | [#73] | the single-observation case: stations 0–59 with half width 12 + 0.04k, logistic walls, the internal reversal, the soft exterior bump, every sample of stations 25–28 occluded, and the samples beyond offset 10 on the left side outside the image from station 45 on. Bright; `--invert` gives the dark one |
| `fusion-multiview` | [#73] | three 160×96 views with a 30 su bead (asymmetric walls) and a camera-fixed highlight, a dark distractor also in the blank, a third view ahead of the tool that never sees the bead, a three-frame gap, occlusions, shading, and a handover to the second view |
| `fusion-parallel` | [#73] | `fusion-multiview` with a substrate groove parallel to the right wall |
| `fusion-moved` | [#73] | `fusion-multiview` whose bead slides 0.15 su per frame on its substrate: fusion must report a mismatch |
| `fusion-regerr` | [#73] | `fusion-multiview` with registration error in the given poses; sweep it with `--reg-sigma` and `--reg-drift` |
| `fusion-junction` | [#73] | a dark bead crossing a blank-present structure of its own level, where both walls vanish |
| `rig-bridged` | [#74] | three views 120° apart around the tool (one sheared), a slow rotation along a 300 su arc; every view drops time 20 |
| `rig-gross` | [#74] | the same rig; the third camera moves 2.5, −1.5 px per frame over times 15–24 |
| `rig-single-view` | [#74] | the same rig; two views drop time 20, so its steps have one view |
| `cap-half-disc` | [#78] | a 20 su bright stripe closed by a half disc of radius 10 |
| `cap-flat` | [#78] | the stripe cut square |
| `cap-skewed` | [#78] | the stripe cut at 30° to its normal |
| `cap-clipped` | [#78] | a half disc whose rim leaves the image |
| `cap-distractor` | [#78] | a half disc with a blank-present dark edge 6 su past its rim |
| `validate-straight` | the generator | the straight ribbon of the self-check |

## Self-check

`validate.py` renders `validate-straight`: a straight ribbon with asymmetric walls on a
flat substrate, at 7° and scale 1 in one view and at −4° and scale 1.3 in the other. It
locates every wall with two measurements that share no code with the renderer:
first-difference centroids per image column (numpy), and `ProfileSampler` derivative
extrema along the station normals. It reports the largest and the RMS error along the
image normal, on the unquantized pixel means and on the 8-bit images.
`test_generator_self_check` bounds them.

## Reporting on it

An operator's numbers go to `docs/performance.md`, one row per preset, with the preset, the
seed (86 unless stated), the commit, and the operator's settings, frozen before the run.
Errors are signed, along the station normal, in px of the view the measurement came from.
Report the bias and σ, and the count they rest on.

- **Fusion.** Per wall, the error of the fused walls at stations with a code of 1–3 in at
  least one frame, beside the single-frame tracker's or extraction's error on the same
  stations. Also report:
  - walls reported at stations that were never observable, or in the gap (should be 0);
  - the width error;
  - whether `fusion-moved` was rejected;
  - the error and the rejected share against `--reg-sigma` and `--reg-drift`;
  - whether the parallel line was flagged.
- **Rig poses.** Per step, the translation error in su and the rotation error in degrees
  against `truth.poses`, the provenance per step against the case (bridged at time 20,
  single-view at time 20, the third view's weight over times 15–24), and the path-length
  ratio.
- **Caps.** The rim's radial error along rays from the chord's midpoint, against
  `truth.caps` (`scene.ray_hit` intersects a ray with a rim polyline): bias and σ over the
  rays, and the share of rays reported unmeasured.

[#73]: https://github.com/VitalyVorobyev/vision-metrology/issues/73
[#74]: https://github.com/VitalyVorobyev/vision-metrology/issues/74
[#78]: https://github.com/VitalyVorobyev/vision-metrology/issues/78
