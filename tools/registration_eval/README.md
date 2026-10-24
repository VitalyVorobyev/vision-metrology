# Sequence registration evaluation

Development tooling for `corr::TranslationEstimator` and the experimental
`corr::SimilarityEstimator` (`vision_metrology.experimental` in Python). It reads one neutral manifest per image sequence
(`sequence.json`, below), registers adjacent frames, composes caller-owned sequence
poses with reciprocal anchors, and writes reports, overlays and a standalone viewer. It also
checks edge locators for sub-pixel locking without ground truth, and measures wall accuracy
against hand-annotated cross-sections.
The tool knows nothing about where a sequence came from. A dataset with its own on-disk
layout gets a small adapter that *writes* a manifest; private adapters live in the
git-ignored `tools/private/`.

Dependencies: the installed `vision_metrology` bindings (`maturin develop --release` in
`crates/vm-python`), numpy, Pillow, scipy, and pytest for the checks. Registration and
local measurements run in Rust. Python reads the manifest, applies the caller-side
preprocessing, and writes the reports.

```bash
python tools/registration_eval/synthetic.py --out-dir /tmp/registration-synthetic/seq
python tools/registration_eval/run.py --manifest /tmp/registration-synthetic/seq/sequence.json \
  --out-dir /tmp/translation-out --propagate
python tools/registration_eval/similarity_run.py \
  --manifest /tmp/registration-synthetic/seq/sequence.json --out-dir /tmp/similarity-out \
  --propagation-report /tmp/translation-out/report.json
python tools/registration_eval/phase_histogram.py \
  --manifest /tmp/registration-synthetic/seq/sequence.json --out-dir /tmp/phase-out
python -m pytest tools/registration_eval
```

Both evaluators take several `--manifest` paths. Outputs must lie outside the repository
and outside every manifest's directory.

## The manifest

`sequence.json` is a JSON object. Paths are relative to the manifest's directory and may
not leave it. Coordinates are pixel centres (`(0, 0)` is the centre of the top-left
pixel), in the frame's own image.

| Field | Required | Meaning |
|---|---|---|
| `version` | no | `1`, the default |
| `name` | no | the sequence's name in reports; defaults to the manifest directory's name |
| `convention` | no | `"pixel_centres"`, the only one supported |
| `split` | no | `"development"` or `"held_out"`; held-out sequences refuse `--propagate` |
| `provenance` | no | free-form: where frames, masks and groups came from |
| `frames` | yes | the frames, in order; a frame's index is its position here |
| `seed` | no | seed geometry for propagation |
| `truth` | no | known geometry, for synthetic or independently measured sequences |
| `comparison` | no | another method's estimates, reported against but never truth |
| `annotations` | no | hand-annotated cross-sections: a list, or the path of an annotations file |

Each entry of `frames` is `{image, mask?, visibility?, group?}`:
- `image`: a single-channel image Pillow can read (8- or 16-bit), or a 2-D `.npy` array.
  Every frame has the same shape. Several frames may name the same file, which is read
  once.
- `mask`: registration validity, nonzero where the pixel is substrate evidence. Without
  it the whole frame is valid. A mask should exclude everything that does not move with
  the substrate: stationary foreground, unlit regions, and the tracked object itself.
- `visibility`: nonzero where seed geometry may be observed. Without it the whole frame is
  visible. It is separate from `mask`, because the tracked object is visible but is not
  registration evidence.
- `group`: a camera or a segment, any JSON scalar, `0` by default. Adjacent frames in
  different groups are a handover: they are never registered, and a handover ends a
  composed segment. `null` puts the frame in no group; such a frame may have no `image`.

`seed` is `{frame, polyline, provenance, walls?, sections?}`. `polyline` is the seed
curve in frame `frame`, as `[[x, y], ...]`. `walls` are boundary polylines for overlays.
Each section is `{s, normal, left, right, center?, width?}`: an arc position, a unit
normal, and the two boundary points along it, or `null` where the boundary is not
supported. Propagation transports the seed with accumulated translation; it never edits
the seed.

`truth` is `{motion?, boundaries?}`. `motion` holds one pose per frame (or `null`),
`[tx, ty, angle, scale]`, mapping frame 0 to that frame by `q = s R p + t`. Reports then
include each accepted pair's maximum corner error against it. `boundaries` holds each
frame's boundary polylines, for boundary evaluation.

`comparison` is `{provenance, shifts}`: one previous-to-current `[dx, dy]` (or `null`)
per adjacent pair, from another estimator. Reports state the distance to it.

`annotations` is a list of annotated cross-sections, or the relative path of the
annotations file that `annotate.py` writes (a named file that does not exist yet reads as
empty). Each annotation is `{frame, station, normal, left, right, left_visible,
right_visible, annotator, pass, timestamp}`:
- `station` is `[x, y]` on the frame's curve, and `normal` the unit normal there.
- `left` and `right` are signed offsets in px along `normal` from `station`, with
  `left < right`. Each is `null` when that boundary is not visible, and its `*_visible`
  flag is then false.
- `pass` is 1 or 2, the blind repeat. `timestamp` is ISO 8601 UTC.

An inline entry with no `annotator` is an exact cross-section from a generator, such as
`tools/sequence_bench`'s `{frame, station, s, point, normal, left, right}` with wall points.
It is carried through unchanged and never scored as an annotation.

The annotations file is `{version, protocol, sections, annotations}`. `sections` lists every
station with its arc position and `source`: `uniform`, `span`, or `user` (added on the
page), or `stratified`, chosen by another tool such as `tools/bead_eval/crosssections.py`.
`protocol` records the seed, the count, the curve source and the profile settings. A
resumed file keeps its `half_length_px` unless `--half-length` is given.

## Preprocessing

`--preprocess dog`, the default, applies a masked normalized difference of Gaussians
(σ 1 and 4 px) to each frame before registration. Excluded pixels contribute no filter
mass, so a mask also keeps them out of the filtered neighbours. `--preprocess none`
registers the original pixels. The estimators themselves take original pixel units and
choose no filter.

## Translation and propagation: `run.py`

`report.json` holds every pair: support, regional consistency, photometric residuals,
forward/backward closure, and the distances to `truth` and `comparison` when present.
`--stride N` samples adjacent pairs at that interval; it does not register N-frame jumps.
Estimator latency includes the Python conversion. Preparation latency includes I/O, masks
and filtering. Native-only timings are in the Criterion `corr` bench.

`--propagate` transports one immutable seed into future and past frames with cumulative
substrate translation: the curve, the walls and the sections all move together.
`BeadTracker` measures the visible part of the transported curve. Independent wall
calipers search ±4 px along each section's normal on the original image, and need
exactly one edge of the expected light-object polarity; several candidates are reported
as ambiguous. Those measurements never update registration or the seed, and rejected
sections stay explicit. Propagation stops at registration failure, a handover, loss of
the visible curve, the sequence end or `--max-frames`. PNG overlays show the transported
walls in amber and the measured wall points in green.

## Similarity, reciprocal sequences and review: `similarity_run.py`

It compares translation with the experimental local similarity on identical prepared
inputs. A sequence step needs an accepted reverse registration and at most 0.25 px of
corner closure (`--max-closure`). An independent seed anchor is attempted every eight
supported steps (`--anchor-every`), under the same reciprocal policy. A handover, a
registration loss or a failed reciprocity check starts a new segment. Failed anchors stay
unavailable and never update a pose. Reports keep both the uncorrected composition and
the anchored trajectory.

`index.html` is a standalone viewer: sequence and group selectors, frame scrubbing,
playback, blending, excluded pixels, mapped grids, outcomes, closure and trajectories. A
`--propagation-report` from `run.py --propagate` adds its transported walls (amber) and
measured wall points (green); that overlay never updates the similarity pose. All pixels
are embedded; there are no external scripts or image requests. Open it directly, or serve
only its output directory on loopback:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory /tmp/similarity-out
```

Source pixels, geometry and detailed reports of a private dataset stay in local,
untracked outputs. Only aggregate numbers belong in the repository's documentation,
under opaque sequence labels; they are in `docs/performance.md`.

## Sub-pixel locking: `phase_histogram.py`

A ground-truth-free check that located edges do not cluster at particular sub-pixel
phases. For every frame it runs the bindings' `detect_edges` (default `EdgeConfig`) and
keeps the edgels whose pixel lies inside the frame's mask, eroded by `--erode` px (default
5), and whose normal is within `--max-angle` degrees (default 25) of an image axis. Each
edgel contributes the coordinate along that axis where its tangent crosses its own pixel
row or column. `--mask` chooses `visibility` (the default), `validity` (the registration
mask), `both` or `none`. `--refine dog` re-locates the same edgels with a continuous,
tapered derivative of Gaussian, a reference that does not lock; `--refine sampler` does the
same with the bindings' `ProfileSampler`, dropping an edgel whose samples reach an excluded
pixel. `--sigma` sets the kernel's σ (default 1 px). The re-location searches ±1.1 px
around each edgel, a reach sized for σ near 1 px.

`phase_histogram.json` and `summary.txt` hold, per sequence and pooled, the 10-bin
histogram of fractional parts normalized to the uniform rate and the locking index, its
max/min bin ratio (`null` when a bin is empty). They are aggregates only. A flat histogram
is necessary, not sufficient, for an unbiased locator.

```bash
python tools/registration_eval/phase_histogram.py \
  --manifest /tmp/registration-synthetic/seq/sequence.json --out-dir /tmp/phase-out --refine dog
```

## Cross-section annotation: `annotate.py`

A local page for annotating two-edge cross-sections. It proposes stations on a curve per
frame: the manifest's seed polyline by default, or `--curves`, a file of
`{"curves": [{"frame": i, "polyline": [[x, y], ...]}]}` such as a tracked curve. `--count`
stations (default 60) are spread uniformly over the curves' joint arc length, one at
random in each equal stratum, from `--seed`. Each `--span FRAME,S0,S1,COUNT` adds stations
on a hard span, and clicking near the curve in the page's overview adds one more. A
station's normal is the curve's chord over ±2 px of arc, turned by +90°.

Each task shows the frame crop around the station and the profile along the normal,
sampled every 0.05 px over ±`--half-length` px (default 12). The profile is a cubic spline
through the original pixels; `--profile sampler` reads the bindings' `ProfileSampler`
instead (σ `--sigma`). Samples outside the frame or outside `visibility` are hatched.
Click the profile to place the active boundary. `←`/`→` nudge it by 0.05 px (`Shift`:
0.5 px), `Tab` switches sides, `V` marks a boundary not visible, the wheel or `+`/`−`/`0`
zoom, and `Enter` submits.

Pass 1 visits every section in one shuffled order and pass 2 in another. No task shows an
earlier answer. Every answer is written at once, so a session resumes where it stopped.
Progress is per `--annotator`. The annotations file defaults to the one the manifest
names, or `annotations.json` beside the manifest, so that annotations of private data stay
with the data. The server listens on loopback only.

```bash
python tools/registration_eval/annotate.py --manifest /tmp/registration-synthetic/seq/sequence.json \
  --annotator me --count 20
```

## Annotation evaluation: `annotations_eval.py`

It scores an operator's walls against the annotations. A results file is
`{"operator": name, "sections": [{frame, station, normal?, left, right}]}`. Each wall is
`null` when the operator found none, an `[x, y]` point, or an offset along the row's normal.
A row matches the annotated section of the same frame whose station lies within
`--match-px` (default 0.5). `--stations-out` writes the annotated stations as an unfilled
results file for an operator to complete.

The reference at a section is the mean of all its visible annotations. A side that some
annotations mark visible and others not is *disputed*, and it is left out. The report
gives:
- per wall and pooled: the bias, σ, RMS and largest error;
- the width error: bias, σ and RMS;
- coverage: visible walls found, hidden walls reported (spurious) and disputed walls;
- the annotator's spread, from pass 2 against pass 1 of the same section: the inter-pass
  bias, a single annotation's σ (`std(difference)/√2`) for a wall and for the width, and
  visibility agreement;
- the wall σ with the reference's own noise removed.

`--annotator` and `--source uniform|span|user|stratified` restrict the annotations. The report holds
aggregates only. `--per-section` adds per-section errors, which carry geometry and stay
local. A held-out sequence needs `--held-out`.

```bash
python tools/registration_eval/annotations_eval.py --manifest seq/sequence.json \
  --results /tmp/operator.json --out /tmp/annotation-report.json
```

## Annotation protocol

**What a boundary is.** Place a boundary where the intensity changes fastest across it:
the inflection of the profile, at the extremum of the grey derivative trace. Mark a
boundary not visible when that change is not visible: occlusion, glare, or no contrast.
Do not guess one.

**Choosing sections.**
- Use the uniform stations for the headline numbers. They sample the curve as the
  operator meets it, and the fixed seed makes the set reproducible.
- Add the hard spans on top: junctions, tight curvature, low contrast, glare,
  occlusion boundaries, and frames where the tracker struggled. Give them as `--span`
  before pass 1 starts, or add them on the overview, and report them separately with
  `--source`.
- Choose stations from the curve and the frame alone, before any operator result is
  looked at.

**How many.** Annotate 50–100 sections per sequence, each twice, about 100–200 answers.
A bias is known to about σ/√n. Spread the sections over the sequence's frames; many
stations on one frame are correlated.

**The blind repeat.** Finish pass 1 before pass 2, and take a break between them. Add
stations before pass 1 ends. A station added later still gets both passes, but its pass 2
may come soon after its pass 1. Several annotators share one file, each with their own
`--annotator`.

**Development and held-out.** The split is per sequence: the manifest's `split`. Choose
the held-out sequences before annotating, and annotate them with the same protocol. Tune
operators on development sequences only. Run a held-out sequence once, with frozen
settings; `annotations_eval.py` refuses one without `--held-out`. When a single sequence
must be split, split it into contiguous frame ranges with one manifest each. Never
interleave stations, because neighbouring stations and frames share their errors.

**Reporting.** Report each operator's wall bias and σ, its width error and its coverage
beside the annotator's σ and the number of sections. A wall σ near the reference's own
noise (the annotator's σ/√2 for two passes) cannot be resolved by these annotations.
Report the uniform and the hard-span sections separately. Annotations of private data stay
with the data, and only aggregates under opaque labels go into `docs/performance.md`.

## Regional-check diagnosis: `--region-check off`, `diagnose.py`, `false_reject.py`

Both evaluators take `--region-check off`: the regional statistics are still computed and
reported, but never reject (diagnosis only; propagation refuses it). `diagnose.py` reads
such reports and prints aggregates: regional disagreement percentiles, its rank
correlation with forward/backward closure, closure p95 on either side of a px limit, and,
for similarity, each anchor's correction against the summed regional disagreement of the
steps it corrects.

`false_reject.py` measures how often the regional check rejects a motion its model
represents exactly. Each trial renders a fresh band-limited texture at 320×97 (a sum of
cosines whose frequencies follow the masked DoG band's power, integrated over the pixel
area), moves it by a known translation (`--motion translation`) or a small similarity
(`similarity`), adds Gaussian read noise, quantizes to 8 bits and excludes a horizontal
band from both masks. `rotation` (0.01 rad by default) and `affine` (±0.5% anisotropic
scale) are the negative controls that must be rejected. `--no-check` keeps the
statistics without their decision, to choose a threshold from the distribution.

```bash
python tools/registration_eval/false_reject.py --motion similarity --contrast 4 --noise 4 \
  --trials 400 --out /tmp/similarity-false-rejects.json
```

`hpatches.py` takes the same questions to real texture: HPatches (registered in
`tools/datasets/registry.json`, fetched into the git-ignored `data/`, split by sequence
in `tools/datasets/splits/hpatches.json`). On 320×97 crops of image 1 it compares
`--pairs self` (image 1 resampled by a known translation or small similarity: real
texture, exact model), `rectified` (image k resampled through the ground-truth
homography, then moved by the known map: real illumination or viewpoint change, with the
homography's own error in the truth) and `natural` (the same crop of image k).
`--residual rotation|affine` adds the negative controls. Tune on `--split dev`; run
`test` once with the settings frozen. Both tools take `--max-chi2` and `--bias-floor` to
try other `RegionCheck` values.

## The synthetic sequence: `synthetic.py`

A textured substrate under known similarity motion, with a light two-boundary bead that
rides on it and a stationary block. Masks exclude the bead and the block; visibility
excludes the block. The sequence splits into two groups, so its middle pair is a
handover. `truth` holds every pose and both boundaries, and `seed` the bead at frame 0.
The tests use it to check the evaluators end to end, with no private data.
