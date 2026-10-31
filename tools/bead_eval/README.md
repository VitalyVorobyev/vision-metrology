# Bead-tracker evaluation on real data

Offline scripts that run `BeadTracker`, through the `vision_metrology` Python bindings, on
real curvilinear structures: the cracks of the DamSegment dataset, and the vessels of HRF.
They answer five questions:
- does the tracker lock onto a real crack and follow it from a perturbed prior?
- do its own quality signals tell a call that locked from one that did not?
- without a prior, can a ridge detector find candidate centrelines good enough to seed
  it ([acquisition](#acquisition-without-a-prior))?
- does choosing both walls together along the whole curve lock more often and wrongly
  less often, and does its max-marginal ambiguity flag the wrong locks
  ([coupled extraction](#coupled-two-boundary-extraction))?
- how far are the walls from where a person puts them, against the annotator's own
  repeatability ([hand-annotated cross-sections](#hand-annotated-cross-sections))?

This is **development tooling**, not a library demo, and it is not run in CI. For
runnable library examples see [`examples/python/`](../../examples/python).

**What these numbers are, and what they are not.** The reference geometry comes from
hand-drawn masks: polygons rasterised to pixels, wider than the dark line they enclose
and offset from it by up to a pixel or two. It is pixel-level truth. The evaluation
therefore measures robustness: locking, following, the basin of convergence, the
tracker's own quality signals and its speed. It does not measure subpixel accuracy,
which is validated on synthetic fixtures with exact ground truth
([performance and accuracy](../../docs/performance.md)). The measured numbers are in
[`docs/performance.md`](../../docs/performance.md#real-data-bead-tracking-on-damsegment-cracks).

## Setup

```bash
python -m venv tools/bead_eval/.venv && . tools/bead_eval/.venv/bin/activate
pip install -r tools/bead_eval/requirements.txt
pip install -r tools/bead_eval/requirements-baselines.txt   # optional, for acquire_eval.py
# the bindings, built from this checkout into the active venv:
(cd crates/vm-python && maturin develop --release)   # or: pip install ./crates/vm-python
```

Run the scripts with `python -I` (isolated mode): the dataset is downloaded data, and
isolated mode keeps the current directory and user site-packages off `sys.path`. The
scripts add their own directory explicitly.

## The DamSegment data

DamSegment, by Vahidreza Gharehbaghi, Caroline R. Bennett, Rémy Lequesne, Hang Zhao and
Jian Li, Mendeley Data, V1, 2025, [doi:10.17632/z5z6gtt5t4.1](https://doi.org/10.17632/z5z6gtt5t4.1),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is not
distributed with this repository. Its source, checksum and dev/test split are in the
[dataset registry](../datasets/README.md); to fetch, verify and extract it into the
git-ignored `data/damsegment/`:

```bash
python3 tools/datasets/fetch.py damsegment
```

`--split dev` or `--split test` on `paths.py` restricts every later step to one side of
the registered split, by image. The default, `all`, is what the published numbers use.

The archive holds `Damage Segmentaion/{Easy,Medium,Hard}/`, with
the dataset's own spelling, each with 500 `Images/*.jpg` (640×640 RGB) and
`Labels/{Mask,Pascal VOC,Yolo}`. The masks are RGB: cracks are pure red `(255, 0, 0)`,
spalling pure blue `(0, 0, 255)`, the rest black. In the Pascal VOC polygons, cracks are
`category_id` 0 (thin and elongated, about 19k polygons) and spalling is 1 (blobby, about
480). Rasterised over 180 of the images, 99.9% of the category-0 polygons' pixels are
red in the masks. Only cracks are used.

## Running it

```bash
D=data/damsegment/extracted            # or the 'Damage Segmentaion' folder itself
python -I tools/bead_eval/paths.py       --data-dir $D   # about 40 s
python -I tools/bead_eval/priors.py      --data-dir $D
python -I tools/bead_eval/run_tracker.py --data-dir $D   # about 2 min
python -I tools/bead_eval/baselines.py   --data-dir $D --workers 6   # about 10 min
python -I tools/bead_eval/report.py      --data-dir $D   # about 8 min
```

Each script takes `--data-dir` and `--out-dir`; outputs default to `<data-dir>/output/`.
The steps hand over through files there, so any step can be re-run alone:

| Output | Written by | Holds |
|---|---|---|
| `paths/<difficulty>/<image>.json`, `paths/index.json` | `paths.py` | reference centrelines and widths, per image |
| `priors.json` | `priors.py` | each path's perturbations, as seeded parameters |
| `runs/tracker/…/*.npz`, `meta.json` | `run_tracker.py` | every call's refined curve, widths, reasons, summary and time |
| `runs/active_contour/…` | `baselines.py` | the same, for the baseline |
| `report.md`, `report.json`, `overlays/*.png` | `report.py` | the metrics, and a few overlays for a look |
| `signals_report.md`, `signals_report.json` | `signals.py` | whether the tracker's own signals separate locked calls, and how close they come to `Converged` |
| `acquire_report.md`, `acquire_report.json`, `acquire_overlays/*.png` | `acquire_eval.py` | acquisition without a prior, and tracking from what it found |
| `coupled_report.md`, `coupled_report.json`, `coupled_report_rows.json` | `coupled_eval.py` | the coupled extraction against the tracker, and every call's row |

Never commit anything from the output directory.

### The steps

- **`paths.py`: reference centrelines.** Per image, the crack pixels of the mask are
  skeletonised (`skimage.morphology.skeletonize`). Spurs shorter than 12 px are pruned,
  and the skeleton is split at junctions and endpoints into non-branching paths. Each path
  is cut back 6 px from a junction and smoothed along its length (Gaussian, σ 3 px), then
  resampled at 1 px. Paths of at least 80 px are kept. Each gets its local width from
  the mask's distance transform, `2 · EDT − 1`, sampled on the path and smoothed.
- **`priors.py`: perturbed priors.** Each path gets the unperturbed prior and a sweep of:
  - translation perpendicular to its chord;
  - rotation about its midpoint, sized by how far its ends move;
  - a sine of wavelength `L/2` along its normals;
  - a Gaussian bump (σ 10 px of arc);
  - Douglas–Peucker simplification to 10 and to 6 vertices;
  - a quarter of its length truncated off one end.

  The signs, phases, bump centres and truncated ends are seeded per path, so the priors
  are reproducible. `priors.json` stores their parameters, and `priors.build_prior`
  rebuilds them.
- **`run_tracker.py`: the tracker.** One `BeadTracker` per path, reused over its priors.
  The image is BT.601 luma, `0.299 R + 0.587 G + 0.114 B`, as float32 on the 8-bit scale.
  The time is the `track()` call alone, including the bindings' conversions. The settings:

  | Setting | Value | Why |
  |---|---|---|
  | `polarity` | `"dark"` | a crack is darker than the concrete |
  | `min_width`, `max_width` | `max(2, w/4)`, `2w`, for a mask width `w` | the masks are about twice as wide as the dark line |
  | `track.max_offset` | 8 px | the sweep runs to 12 px, so its top straddles the reach |
  | `spacing` | 2 px | cracks are a few pixels wide and wiggle on that scale |
  | `threshold` | 3 grey levels | faint cracks |
  | `measure.max_obliquity_deg` | 180° (off) | a rough crack wall's gradient direction is unreliable |
  | everything else | library defaults | including `sigma` 1 px and 3 passes |

  `--threshold 5 --obliquity 30 --name tracker_defaults` reruns with the library's own
  values, and `report.py --compare tracker_defaults` adds them to the report.
- **`baselines.py`: `skimage.segmentation.active_contour`.** An open snake from the same
  priors, resampled at the same spacing. It runs on the same luma, scaled to [0, 1] and
  smoothed with a Gaussian of σ 3 px, which sets its capture range. It uses
  `w_line = −1` (drawn to dark), `w_edge = 0`, and `boundary_condition="free"`, so neither
  end is held where the prior put it. Everything else is scikit-image's defaults. It is
  slow, so it runs on a seeded subset of 40 paths per difficulty, and the report compares
  the two methods on exactly those calls.
- **`report.py`: the metrics,** per difficulty and overall, from the unperturbed prior and
  from every prior within the reach, and the basin per perturbation:
  - **centre distance**: each refined station's distance to the reference centreline.
    Stations past either end of the reference are left out, and the statistics are
    pooled over stations;
  - **on the crack**: within half the local mask width of the centreline, plus 1 px;
  - **tangent error**: against the reference, each tangent the chord over ±4 px;
  - **support**: the tracker's own fraction of measured stations;
  - **width**: the final stage's, against the mask width. That is a sanity check only:
    the masks are wider than the visible crack;
  - **locked**: at least 90% of the stations on the crack;
  - **converged**: at least 90% within 1 px of the curve the same method returns from
    the unperturbed prior. It needs no annotation, and it is the basin;
  - **false lock**: not locked, but with support of at least 0.5, so the tracker's own
    summary would not flag it;
  - **runtime**: the median per call.
- **`signals.py`: the tracker's own signals.** On a seeded subset of paths (100 per
  difficulty), every prior within the reach is tracked with `run_tracker.py`'s settings,
  and again with `min_margin` 0.1, 0.2 and 0.3. For each signal (`support`, `center_rms`,
  `longest_gap`, the median `confidence` of the final stage's hits, and the support with a
  `min_margin`), it reports how well it separates the calls that locked from those that
  did not: the AUC, and the share of unlocked calls that pass a threshold set to keep 90%
  of the locked ones. It also reads every call's passes: the last solved pass's
  correction and residual, and how often candidate stopping tests would hold at each
  pass, with the median correction and residual at each pass. It writes
  `signals_report.md` and `signals_report.json`.

  ```bash
  python -I tools/bead_eval/signals.py --data-dir $D   # about 1 min; after priors.py
  python -I tools/bead_eval/signals.py --data-dir $D --passes 10 --name signals_10pass
  ```

## Acquisition without a prior

`acquire_eval.py` asks whether candidate centrelines found from the image alone, with no
prior, are good enough to seed the tracker. It is an evaluation, not a library feature.
The algorithmic reference is C. Steger, "An Unbiased Detector of Curvilinear
Structures", *IEEE PAMI* 20(2), 1998.

```bash
pip install -r tools/bead_eval/requirements-baselines.txt   # optional: ridge-detector
python -I tools/bead_eval/acquire_eval.py --data-dir $D --workers 6   # about 30 min; after paths.py
python -I tools/bead_eval/acquire_eval.py --out-dir /tmp/acq --datasets synthetic   # no data needed
```

It writes `acquire_report.md`, `acquire_report.json`, and under `acquire_overlays/` three
synthetic scenes with each detector's paths. Times are measured per image inside each
worker; for timing, run with `--workers 1`.

**The sources** (`acquire.py`), each with a sweep of threshold settings and one default
whose paths are tracked:
- **scikit-image's `sato` and `meijering` ridge filters**, at scales `w / (2√3)` and
  `w / 2` for each expected width `w`. The response is thresholded with hysteresis: the
  high threshold is the larger of Otsu's and `median + k·s`, where `s` is the median
  absolute deviation scaled to a standard deviation, and the low one is halfway from the
  median. Each `k` in 2, 3, 5 and 8 is a setting; 3 is the default. The mask then goes
  through `paths.py`'s skeleton graph, so the paths are non-branching. Their width is the
  mask's `2 · EDT − 1`, which says more about the threshold than about the line.
- **`ridge-detector`**, optional, from `requirements-baselines.txt`: a multi-scale
  detector after Steger, with sub-pixel points and widths. It is MIT-licensed but
  describes itself as an adaptation of the GPL ImageJ Ridge Detection plugin, so it is
  used as a black box only, through its public calls. Nothing of it is copied or adapted
  here, and nothing derived from it may go into the library. Two of its behaviours are
  worked around from outside:
  - a dark line is passed as a light line on the inverted image, where its contrast
    thresholds mean what they say;
  - it rounds its contrast thresholds down to a whole number of a unit that grows with
    the line width (about 19 grey levels at 8 px), so for a wide line they round to 0.
    It therefore runs on the image mean-pooled by `ceil(w_max / 8)`, and its points and
    widths are scaled back.

  Its settings are the low and high contrasts 10/20, 20/40 and 40/80 grey levels; 20/40
  is the default. When it is not installed, the script says so and skips it.

**The data.**
- **Synthetic ribbons** (`ribbons.py`), 1280×1024, rendered in numpy with exact truth: the
  model of the library's own test fixture, a light ribbon of contrast 60 DN on 80 DN,
  blurred across by σ 1 px. A line, an arc (R 300 px) and a sine (amplitude 40 px, period
  400 px), each 3, 8 and 30 px wide, under four conditions: clean (noise 2 DN), noisy
  (8 DN), a distractor (a 40 DN step 10 px beyond the ribbon's edge, parallel to its
  chord), and a 30 px gap. Two noise seeds each, 72 images. The sources expect the true
  width.
- **DamSegment cracks**, 640×640, against `paths.py`'s reference paths. The sources
  expect dark lines 3, 5 and 8 px wide, the range of the visible cracks.

**The metrics,** per source and setting:
- **recall**: the share of reference paths with at least 80% of their length within
  `max(2 px, w/2)` of an acquired path. **One-path recall** asks it of a single acquired
  path, which is what a prior needs;
- **precision**: the share of acquired length near a reference. On DamSegment that is
  within 1 px of the crack mask, so an unannotated dark line, such as a formwork seam,
  counts against it;
- **centre error**: each acquired point near a reference, its distance to it;
- **width**: the source's estimate there, against the truth or the mask width;
- **time** per image, of the whole acquisition and of the filter alone. The skeleton
  graph is pure Python, so the filter time is the part a compiled implementation would
  compete with.

**Acquire, then track.** Each reference path that one acquired path covers (one-path
recall) is tracked three times with the same settings: from that acquired path, cut to
its longest run within the tolerance; from the reference itself; and from the reference
translated 4 px across its chord, with `priors.py`'s seeded sign. The synthetic ribbons
use the library's defaults with a width range of `0.5w` to `1.5w`, and DamSegment uses
`run_tracker.py`'s settings. The metrics are `report.py`'s, with *converged* measured
against the run from the reference itself. Cutting the acquired path uses the reference,
so this measures how good a found path is as a prior, not how to pick the right one.

## Coupled two-boundary extraction

`coupled.py` is a prototype of a global choice of both walls along a curve, from one
image. The tracker picks each station's edge pair on its own; here the pairs along the
whole curve are one shortest path, and the path's max-marginals say what a different
choice at each station would cost. `coupled_eval.py` compares it with the tracker and with
the library's operator, `BeadTracker` with coupled pairing (the `rust` method). The
measured numbers are in
[`docs/performance.md`](#results).

```bash
python -I tools/bead_eval/coupled_eval.py --data-dir $D --workers 6   # about 1 min; after priors.py
python -I tools/bead_eval/coupled_eval.py --data-dir $D --split test --methods tracker dense rust \
    --rust-set passes=1 --workers 8   # the published library numbers
python -I tools/bead_eval/coupled_eval.py --out-dir /tmp/c --datasets synthetic   # no data needed
python -m pytest tools/bead_eval/test_coupled.py   # needs pytest
```

`--split dev|test` keeps the paths of the registry split's images
(`tools/datasets/splits/damsegment.json`) before the seeded subset is drawn; the default,
`all`, is the set the prototype was measured on. `--rust-set key=value` overrides a
`BeadCoupling` field, or `track_step`, `measure_step` (keeping the threshold per px),
`passes` and `clearance`. The `rust` method's ambiguity and evidence are its first
pass's `path_margin` and `path_evidence`. Its settings, one tracking pass and otherwise
the defaults, were chosen on the development split; the paired `rust − dense` rows
compare it with the prototype.

**Stations and profiles.** The prior is resampled at `spacing` px, with chord tangents over
±10 px that shrink at the ends. Each station runs one strip caliper along its normal with
the tracking stage's settings, every edge of either polarity, and a profile step of
0.25 px. The caliper's signed derivative response is the evidence, and its sub-pixel edges
are the candidate walls and the final wall positions: each wall the dense variant chooses
moves to the nearest caliper edge of its polarity within 1 px. There is no second edge
detector. The caliper's response is a difference per profile sample, so its threshold and
amplitudes are rescaled to per px.

**Costs.** A dark bead's left wall falls and its right wall rises; a light bead's the
reverse. With the derivative `g` signed so the expected polarity is positive and softly
saturated, `e(g) = (g/g0) / (1 + |g|/g0)`, so one strong edge cannot dominate, a station
costs per px of curve:
- present, with walls `l < r`, centre `c` and width `w`:
  `−e(g(l)) − e(g(r)) + λ_w·huber(ln(w/w0)/s_w) + λ_0·(c/reach)²`;
- absent: `−2·e(absent_level)`.

Between stations `h` px apart the walls pay `κ·(Δl² + Δr²)/h + κ_w·Δw²/h`, with steps bounded
by `max_wall_slope·h` and `max_width_slope·h`, and a change between present and absent pays
`switch_cost`. The data terms are per px and the smoothness goes as `1/h`, so the solution
does not depend on the station spacing; a test checks that.

**The two variants.**
- **Sparse**: a Viterbi over the caliper's candidate pairs per station (polarity order,
  width range, centre within the reach). A gap of up to 20 px is a skip edge whose
  smoothness and step bounds scale with its length; a longer one passes through absent
  states and pays `reacquire_cost` to resume.
- **Dense**: a dynamic programme over every (centre, width) bin, centres 0.25 px and
  widths 0.5 px apart, so both walls fall on profile samples. Since
  `Δl² + Δr² = 2Δc² + Δw²/2`, the transition is separable in centre and width, and the
  minimum over it is two banded passes. The absent layer keeps a geometry, so a gap is
  bridged smoothly.

Both are solved exactly, forward and backward; tests check both against brute force on
small problems, max-marginals included.

**Ambiguity.** At a present station, the margin is the cost of the best path that puts a
wall more than `tol` px from the chosen one there, minus the best path's. The call's
ambiguity is the 0.25 quantile of its stations' margins at `tol` 5 px: low means another
line was almost as good. `evidence`, the best path's mean gain per px over absence, is a
second whole-call signal. The margins come from the first extraction, whose window is the
prior's: re-stationing on the chosen line would move a competitor out of the window.

**Settings.** Reach, spacing, width range, threshold and profile sigma are
`run_tracker.py`'s. The costs were chosen on a seeded tuning subset of 40 paths per
difficulty, disjoint from the evaluation's paths, and on the synthetic scenes:
`g0` 10 DN/px, `absent_level` 4, `κ` 0.2, `κ_w` 0.5, slopes 1 and 0.5, `λ_0` 0.1,
`switch_cost` 2, and no width prior (`λ_w` 0), which lowered the lock rate on cracks,
whose width varies. `--subset tune` reruns the tuning subset, and `--set key=value`
overrides any cost.

**The data.**
- **DamSegment**: `signals.py`'s calls, every prior within the reach on 100 seeded paths
  per difficulty, with `report.py`'s lock. A path is *lockable* when the call from the
  reference itself locks for every method.
- **Synthetic** (`synthetic.py`): beads with walls known exactly, on a line, an arc and a
  sine, with two noise seeds: clean; asymmetric walls of varying width; a stripe 2 px
  outside one wall; an identical parallel bead 14 px away; a gap; an occluding disc; and
  a highlight inside the bead. Priors are the truth moved 0, ±3, ±6 and +7 px along its
  normals (+7 is midway to the parallel bead) and a 3 px sine.

**The metrics,** with 90% bootstrap intervals over paths, since a path that fails tends to
fail from every prior:
- lock rate and false locks (not locked, support of at least 0.5);
- the AUC of each signal for telling locked calls from the others;
- **wrong at equal coverage**: each method's signal accepts the same share of all calls as
  locked as the tracker does at support 0.5, and the share accepted but not locked is
  read off. Paired differences from the tracker are bootstrapped too;
- on the synthetic scenes, at every visible station, the wall error: more than 1 px is a
  wrong wall, and bias and σ are over the rest;
- the median time per call: the prototype is Python around the bindings' caliper.

## Hand-annotated cross-sections

A public benchmark of both walls, annotated by hand on CC BY 4.0 images: HRF vessels and
DamSegment cracks. The annotations are our own work, committed under CC BY 4.0 in
[`tools/datasets/annotations/`](../datasets/annotations), with the stations frozen before
anyone annotates. No numbers are published until the annotations exist.

**Status: awaiting annotation.** The stations are frozen; nobody has annotated them yet.

| | HRF | DamSegment |
|---|---:|---:|
| stations, dev / test | 36 / 84 | 36 / 84 |
| images with stations, dev / test | 11 / 30 | 36 / 84 |
| width range (mask width) | 3–20 px | 3–14 px |
| profile reach, `half_length_px` | ±20 px | ±12 px |
| frame | the green channel, 8-bit | BT.601 luma, float32 |

**Stations** (`crosssections.py`). Per image of the registry split, the dataset's mask goes
through `paths.py`'s skeleton graph into non-branching centreline segments. Candidates
lie every 10 px along each segment. A candidate is kept when:
- it is at least `max(10, 2w)` px from its segment's ends, and `max(15, 2w + 6)` px from
  every skeleton junction;
- the chord over `±max(8, 1.5w)` px is at least 0.95 of the arc, so the segment is
  locally straight;
- the mask width `w` is in the range;
- across the normal, at the station and `w/2` either side, the mask is one run with
  nothing else in reach;
- the profile lies inside the frame and its field of view;
- the contrast is at least 2 grey levels. The contrast is the weaker side's background
  median minus the core mean.

Per split, about 30% of the stations go to dev and the rest to test, by image, as the
registry split assigns them. Each split gets an equal quota per cell of the width ×
contrast terciles, pooled over both splits. Candidates are taken in seeded SHA-256 rank
order, at most `ceil(1.5 · count / images)` per image, and at least 40 px apart on one
image. A frame of the manifest is one segment, and the frames of one image share its
file. The cell of each station is stored with it.

```bash
python3 tools/datasets/fetch.py hrf                      # 73 MB; damsegment is 110 MB
python -I tools/bead_eval/crosssections.py hrf           # about 1 min; damsegment the same
python -I tools/bead_eval/crosssections.py hrf --check   # the frozen stations reproduce
```

It writes `data/<id>/crosssections/`: `sequence.json`, `curves.json`, `frames/`,
`visibility/` and `masks/`. It never overwrites the committed stations: a rerun compares
them and fails on any difference.

**Annotating.** One session per dataset and annotator, on the
[annotation page](../registration_eval/README.md#cross-section-annotation-annotatepy):

```bash
python tools/registration_eval/annotate.py --manifest data/hrf/crosssections/sequence.json \
  --curves data/hrf/crosssections/curves.json \
  --annotations tools/datasets/annotations/hrf/annotations.json --annotator <initials>
```

Do the same with `damsegment` in place of `hrf`. The page opens at
`http://127.0.0.1:8766/`. It takes its profile reach from the file, and it resumes where
it stopped. Follow the [protocol](../registration_eval/README.md#annotation-protocol):
- place each wall at the profile's inflection;
- mark a wall not visible rather than guess it;
- pass 1 visits all 120 stations, and pass 2 visits them again, blind, in another order;
- take a break between the passes.

At about 30 s per section per pass, one dataset takes about 2 hours: 120 sections × 2
passes. Both datasets take about 4 hours, which splits naturally into four sittings. A
second annotator adds their own two passes to the same file. Commit the file when a pass
ends.

**Evaluation** (`crosssection_eval.py`). For each annotated frame of a split, the tracker
runs along the segment's curve in two modes:
- `tracker`: per-station pairing;
- `coupled`: coupled pairing, with one tracking pass.

Both modes use `run_tracker.py`'s reach, spacing, sigma and obliquity. The width range is
`max(2, 0.25w)` to `2w` for DamSegment and `max(2, 0.5w)` to `2w` for HRF, where `w` is
the segment's median mask width. The edge threshold is 3 grey levels on DamSegment and 0.5
on HRF, whose vessel edges change by a few grey levels per pixel. For the same reason, HRF's
coupled mode uses `absent_level` 1 and `saturation` 3. These HRF values are provisional:
they were chosen on dev against synthetic annotations, and they are frozen on dev against
the real annotations before test runs.

An annotated station's walls are where its normal crosses the tracker's two wall
polylines, between consecutive measured stations. A station in a gap has no wall.
`annotations_eval.py` then scores the walls: wall bias and σ, width error, coverage, and
the annotator's two-pass σ beside them. The report adds 90% bootstrap intervals over
images.

```bash
python -I tools/bead_eval/crosssection_eval.py --manifest data/hrf/crosssections/sequence.json \
  --annotations tools/datasets/annotations/hrf/annotations.json --split dev --out /tmp/hrf-dev.json
# once the settings are frozen on dev:
python -I tools/bead_eval/crosssection_eval.py ... --split test --held-out --out /tmp/hrf-test.json
```

**Testing without annotations.** `crosssection_synth.py` writes **synthetic** annotations,
outside the repository only. They are the mask's own walls plus 0.25 px of noise per pass,
under the annotator `synthetic-mask`, and the protocol is marked `"synthetic": true`. The
evaluator labels its report as synthetic. `tools/check-invariants.py` refuses a committed
annotation file that holds any. `test_crosssections.py` runs the whole chain on a rendered
bead whose walls are known exactly.

```bash
python -I tools/bead_eval/crosssection_synth.py --manifest data/hrf/crosssections/sequence.json \
  --annotations tools/datasets/annotations/hrf/annotations.json --out /tmp/hrf-synthetic.json
python -I tools/bead_eval/crosssection_eval.py --manifest data/hrf/crosssections/sequence.json \
  --annotations /tmp/hrf-synthetic.json --split dev
python -m pytest tools/bead_eval/test_crosssections.py
```

## GAN_Synth_Adhesive: local, qualitative use only

[GAN_Synth_Adhesive](https://github.com/RicardoSPeres/GAN_Synth_Adhesive) publishes real
1024×1024 images of structural adhesive beads with defects (the `adhesive1024.zip`
release). The repository has **no licence file**, so all rights are reserved by its
authors. You may download the release yourself and look at the tracker on it locally.
Nothing derived from it is committed or published here: no images, no overlays, no
numbers. For any other use, contact the authors. Cite their paper:

> R. S. Peres, M. Azevedo, S. O. Araújo, M. Guedes, F. Miranda and J. Barata,
> "Generative Adversarial Networks for Data Augmentation in Structural Adhesive
> Inspection", *Applied Sciences* 11(7), 3086, 2021,
> [doi:10.3390/app11073086](https://doi.org/10.3390/app11073086).

The images come without centrelines, so you supply the priors. Write a JSON file that maps
each image, relative to `--data-dir`, to its beads:

```json
{
  "good/0001.png": [
    {"points": [[120, 900], [400, 610], [880, 540]], "polarity": "dark",
     "min_width": 20.0, "max_width": 70.0}
  ]
}
```

Points are pixel coordinates, with pixel centres at integers. Then run:

```bash
python -I tools/bead_eval/adhesive.py --data-dir /path/to/adhesive1024 --priors priors.json
```

It tracks every prior with the library's default settings (`--reach`, `--spacing`,
`--threshold`, `--sigma` and `--obliquity` override them). It writes an overlay per bead
under `<out-dir>/adhesive/`: the prior, the refined centreline, the final edges and the
rejected stations. It also writes each bead's summary to `<out-dir>/adhesive.json`. The
script works on any folder of images, so you can try it on your own captures the same
way.

## Results

### Finding a bead without a prior

The tracker needs a prior within its reach. For a first frame with no reference part, CAD
or robot path or taught frame, something else has to find the bead. These runs measure
how well a ridge detector does that, and whether what it finds is a good prior. They are
an offline evaluation with Python packages, not a library feature.

**The detectors,** each given the bead's expected width:
- scikit-image's `sato` and `meijering` ridge filters, at scales of `w/(2√3)` and `w/2`.
  The response is thresholded with hysteresis, relative to its median and spread, then
  skeletonised and split at junctions into non-branching paths. Their width is the
  thresholded mask's;
- [`ridge-detector`](https://pypi.org/project/ridge-detector/), a Python implementation
  after C. Steger's detector ("An Unbiased Detector of Curvilinear Structures", *IEEE
  PAMI* 20(2), 1998), with sub-pixel points and widths. On a bead wider than 8 px it runs
  on a mean-pooled image, because its contrast thresholds round to 0 at large scales.

**Synthetic beads,** 1280×1024 with exact truth: a light ribbon of 60 DN contrast, blurred
by σ 1 px, along a line, an arc (R 300 px) or a sine; 3, 8 or 30 px wide. Each is
rendered clean (2 DN of noise), noisy (8 DN), beside a 40 DN step edge 10 px beyond the
bead, or broken by a 30 px gap, with two noise seeds: 72 images. *Found* means at least
80% of the bead within `max(2 px, w/2)` of one path; *precision* is the share of found
length that lies on the bead.

| Detector | Found, without a gap | Precision: clean / noisy / beside the step | Centre, median / p95 | Width read at 3 / 8 / 30 px | Time per image at 3 / 8 / 30 px |
|---|---:|---:|---:|---:|---:|
| `sato` | 100% | 100% / 100% / 62% | 0.12 / 1.99 px | 2.6 / 7.1 / 29.6 px | 0.39 / 0.21 / 0.57 s |
| `meijering` | 100% | 100% / 100% / 39% | 0.12 / 1.40 px | 2.6 / 7.1 / 29.3 px | 0.41 / 0.23 / 0.59 s |
| `ridge-detector` | 100% | 100% / 7% / 100% | 0.03 / 0.54 px | 4.1 / 8.4 / 31.4 px | 1.3 / 1.3 / 0.09 s |

Each found path, cut to where it lies on the bead, is then the prior of a `track` call
with the default config and a width range of `0.5w` to `1.5w`. *Locked* means 90% of the
stations within `w/2 + 1 px` of the true centreline, and *converged* 90% within 1 px of the
curve tracked from the true centreline:

| Prior | Calls | Refined centreline, median / p95 from the truth | Locked | Converged |
|---|---:|---:|---:|---:|
| found by `sato` | 60 | 0.033 / 0.147 px | 100% | 100% |
| found by `meijering` | 54 | 0.037 / 0.139 px | 100% | 100% |
| found by `ridge-detector` | 54 | 0.022 / 0.148 px | 100% | 100% |
| the true centreline | 60 | 0.018 / 0.111 px | 100% | – |
| the true centreline, moved 4 px | 60 | 0.018 / 0.108 px | 100% | 100% |

**Real cracks.** The same detectors on the DamSegment images and reference paths of the
section above (Gharehbaghi et al., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)),
expecting dark lines 3, 5 and 8 px wide. *Precision* is the share of found length within
1 px of the crack mask, so an unannotated dark line, such as a formwork seam, counts
against it. Each found piece is tracked with the config of the section above:

| Detector, setting | Cracks touched | Found in one piece | Precision | Paths per image | Time per image | Tracked from a found piece: locked / converged |
|---|---:|---:|---:|---:|---:|---:|
| `sato`, threshold 3 | 38% | 14% | 14% | 113 | 0.29 s | 99% / 91% |
| `sato`, threshold 8 | 20% | 8% | 35% | 26 | 0.22 s | – |
| `meijering`, threshold 3 | 42% | 15% | 20% | 85 | 0.28 s | 100% / 89% |
| `ridge-detector`, contrast 20/40 | 81% | 24% | 5% | 508 | 1.8 s | 99% / 83% |

*Touched* asks for 80% of a crack's reference path within `max(2 px, w/2)` of any found
path, *found in one piece* of a single path. The threshold is in robust deviations of the
filter's response above its median. From the reference path itself, the same cracks lock
in 96–97% of the calls, and from a 4 px translation of it, 95% lock and 82–88% converge.

How to read them:
- **On a clean part every detector finds the bead, and what it finds is as good a prior as
  the truth.** The tracker ends on the curve it reaches from the true centreline.
  - A skeleton path, a tenth of a pixel off the centre at the median and up to 2 px at
    the 95th percentile, is enough: the tracker measures the centre and the width
    itself. Steger's sub-pixel centre adds nothing the tracker needs.
  - From a skeleton path, the refined curve keeps a little of its pixel-scale wiggle
    after three passes: 0.033 px against 0.018 px from the truth
    ([limitations](../../docs/bead.md#limitations)).
- **Each detector also reports what is not the bead.**
  - `meijering` at every width, and `sato` mostly at 30 px, follow the step edge.
  - `ridge-detector` passes noise ridges at its default contrast on the noisy 3 px bead,
    about 1100 paths per image; its strictest setting removes them.
  - A gap breaks the bead into two paths, except under `sato` at 30 px, whose 15 px scale
    bridges it.
- **On cracks, the choice is the problem.** `ridge-detector` touches 81% of the cracks,
  but among about 500 paths per image, and only a quarter of the cracks come out in one
  piece. The filters find fewer cracks and fewer other lines. No setting gives both.
- **A found piece seeds the tracker as well as the annotation does.** From a found piece,
  99–100% of the calls lock, and 83–91% end on the curve tracked from the reference path,
  against 82–88% from a 4 px translation of it. The piece is cut to the crack with the
  reference's help, so these rows measure how good a found path is as a prior, not how
  to choose it.
- **The detectors filter the whole image at the bead's scale.** In scikit-image, the
  filter alone takes 0.15–0.53 s per 1280×1024 image and 0.18–0.20 s per 640×640 crack
  image, on one core, against 0.27 ms for a `track` call on a crack. Times are medians,
  measured one image at a time.

The dataset is not distributed with this repository.
[`tools/bead_eval/README.md`](#acquisition-without-a-prior)
reproduces these numbers.

### Choosing both walls along the curve

The tracker chooses each station's edge pair on its own. A prototype chooses the pairs of
all stations together, as one shortest path over the profiles with smoothness on both
walls and an explicit "absent" state, and reads an ambiguity for the whole call from the
path's max-marginals: what the best answer with a wall more than 5 px elsewhere costs.
It comes in two variants: a Viterbi over the caliper's candidate pairs (*sparse*), and a
dynamic programme over every centre and width bin of the profile (*dense*). Both refine
the walls with the caliper's own sub-pixel edges. It is Python around the bindings'
caliper, an offline evaluation and not a library feature.

On the DamSegment calls of the [tracker's signals](../../docs/performance.md#real-data-bead-tracking-on-damsegment-cracks):
5400 calls from the priors within the reach on 300 paths, with the same reach, width range
and threshold. Intervals are 90% bootstrap intervals over paths. *Wrong at equal coverage*
lets each method's support accept as many locked calls as the tracker's support of at
least 0.5 does, 84.5% of all calls, and counts the calls it accepts that did not lock:

| Method | Locked | Not locked, support ≥ 0.5 | Wrong at equal coverage | AUC: support / ambiguity / path evidence | Time per call |
|---|---:|---:|---:|---:|---:|
| tracker | 85.3% [82.6, 88.0] | 13.0% [10.6, 15.4] | 13.0% [10.5, 15.4] | 0.71 / – / – | 0.4 ms |
| sparse | 86.3% [84.3, 88.2] | 7.1% [5.7, 8.8] | 12.4% [10.0, 14.3] | 0.70 / 0.54 / 0.70 | 13 ms |
| dense | 91.0% [88.8, 93.3] | 8.8% [6.6, 11.0] | 7.1% [5.0, 8.8] | 0.58 / 0.50 / 0.72 | 19 ms |

- **The dense choice locks more and wrongly less.** Paired with the tracker on the same
  calls, it locks 5.7 percentage points more [4.1, 7.3] and, at equal coverage, accepts
  6.0 points fewer wrong locks [4.1, 7.8]. On the 257 paths whose reference itself locks
  for every method, it locks 98.0% of the calls against 94.1%, and the wrong locks at equal
  coverage fall from 5.5% to 1.4%.
- **The sparse choice does not.** At equal coverage it accepts about as many wrong locks
  as the tracker, 0.7 points fewer [−1.5, 3.8]. It has fewer false locks only because its
  support is lower on every call: 64% on average, against the tracker's 90% and the dense
  variant's 99%.
- **The max-marginal ambiguity does not predict wrong locks on cracks.** Its AUC is 0.50
  (dense) and 0.54 (sparse), and 0.59–0.63 on the lockable paths, under support's. The
  wrong locks are not close calls. In 86% of the dense variant's, the path it took has at
  least as much evidence as the best path along the annotation (54% on the lockable
  paths), so another structure is genuinely stronger. They are partial: a median of 82% of
  the stations stays on the crack, and 63% of those that leave it lie in the outer fifth
  at either end, where 40% would if they were spread evenly. The best path's mean evidence
  is a better whole-call signal, with an AUC of 0.72.
- **It does flag a competing parallel line.** On synthetic beads with an identical bead
  14 px away, the six calls whose prior lies midway between the two score an ambiguity
  below 0.85, and every other synthetic call scores above 4. Half of the six follow the
  wrong bead, and they are the only wrong locks of either variant, so the ambiguity's AUC
  on the synthetic set is 1.00, against 0.43 for the dense variant's support. The
  tracker's support also separates them there, because the tracker loses stations
  between the two beads.
- **Accuracy is the caliper's.** On the synthetic walls (clean, asymmetric and varying in
  width, with a gap or an occluder), both variants find every visible wall within 1 px,
  with a bias within 0.005 px and a σ of 0.046 px against the tracker's 0.051 px. Where
  a nearby edge merges with a wall in the derivative, every method reads the same biased
  wall, and a choice between edges cannot undo it: a stripe 2 px outside the wall pulls it
  0.43 px inwards, and a highlight inside the bead 0.38 px.
- **Time.** The dense programme has 65 centres by a median of 28 widths per station on
  these paths. The Python prototype takes 19 ms per call, median, of which 3 ms is
  sampling the profiles; the tracker takes 0.4 ms in Rust. Measured with 8 worker
  processes.

The settings were chosen on a tuning subset of 120 other paths and on the synthetic
scenes. [`tools/bead_eval/README.md`](#coupled-two-boundary-extraction)
reproduces the numbers.

### The library's operator

`BeadPairing::Coupled` is the dense variant in the library. `coupled_eval.py` runs it as
the `rust` method: `run_tracker.py`'s config with coupled pairing at its defaults and one
tracking pass. The tracking stage's profile step stays at 1 px, so the path's grid is
1 px there and 0.5 px in the final stage, against the prototype's 0.25 px. The settings
were chosen on the registry split's development images (`tools/datasets/splits/damsegment.json`,
100 seeded paths per difficulty): there, a second and third tracking pass locked 1.9
points less often, and a 0.25 px profile step changed the lock rate by half a point or
less. The test split was then run once.

| Set | Method | Locked | Wrong at equal coverage | AUC: support / ambiguity / path evidence | Time per call |
|---|---|---:|---:|---:|---:|
| test split, 5400 calls on 300 paths | tracker | 85.6% [83.4, 87.8] | 13.2% [11.0, 15.2] | 0.71 / – / – | 0.4 ms |
| | dense (Python) | 91.4% [89.4, 93.2] | 6.5% [4.7, 8.2] | 0.60 / 0.57 / 0.75 | 25 ms |
| | coupled (Rust) | 91.0% [89.1, 93.0] | 5.8% [4.2, 7.5] | 0.71 / 0.59 / 0.77 | 0.6 ms |
| #76's set, 5400 calls on 300 paths | tracker | 85.3% [82.6, 88.0] | 13.0% [10.5, 15.4] | 0.71 / – / – | 0.4 ms |
| | dense (Python) | 91.0% [88.8, 93.3] | 7.1% [5.0, 8.8] | 0.58 / 0.50 / 0.72 | 23 ms |
| | coupled (Rust) | 91.1% [89.2, 93.2] | 5.8% [4.1, 7.5] | 0.67 / 0.50 / 0.72 | 0.6 ms |

- **The Rust operator reproduces the prototype.** On the test split, paired on the same
  calls, it locks 0.4 points less than the dense prototype [−1.3, +0.5] and accepts 0.7
  points fewer wrong locks at equal coverage [−1.8, +0.4]; against the tracker it locks
  5.4 points more [3.6, 7.2] and accepts 7.4 fewer wrong locks [5.3, 9.6]. On #76's set it
  locks 0.1 points more than the prototype [−0.7, +1.0].
- **Its support is a better signal than the prototype's.** It has fewer stations with a
  wall that no caliper edge supports, so support separates locked calls about as well as
  the tracker's does, and its false locks (support ≥ 0.5, not locked) are 6.0% against
  the prototype's 8.2%.
- **The margin still does not predict wrong locks on cracks**, as for the prototype (AUC
  0.50–0.59). The path's mean evidence is the best whole-call signal.
- **Synthetic walls.** On `synthetic.py`'s scenes it matches the prototype's lock and
  coverage in every condition, the ambiguity's AUC included (1.00 on the parallel bead).
  Its walls snap to the final stage's 0.5 px profile, so their σ is the tracker's,
  0.052 px, against the prototype's 0.046 px on a 0.25 px profile.
- **Development sequence A** (private, aggregates only). On the 445 frames with saved
  ribbon geometry, from that geometry and from it moved ±4 px along its normals (1335
  calls, the same width range and a reach of ±8 px), the tracker stays on the reference
  in 93.6% of calls with per-station pairing and in 100% with coupled pairing (one pass).
  Both walls are measured at a median 79% of stations per call with per-station pairing
  and 74% with coupled pairing, which measures one wall at a further 25% of stations, most
  of them where the other wall leaves the image.

`tools/bead_eval/README.md` reproduces the public numbers.
