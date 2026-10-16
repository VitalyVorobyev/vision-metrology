# Sequence registration evaluation

Development tooling for `corr::TranslationEstimator` and the experimental
`corr::SimilarityEstimator` (`vision_metrology.experimental` in Python). It reads one neutral manifest per image sequence
(`sequence.json`, below), registers adjacent frames, composes caller-owned sequence
poses with reciprocal anchors, and writes reports, overlays and a standalone viewer. It also
checks edge locators for sub-pixel locking without ground truth.
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
| `annotations` | no | hand-annotated cross-sections, carried through unchanged |

Each entry of `frames` is `{image, mask?, visibility?, group?}`:
- `image`: a single-channel image Pillow can read (8- or 16-bit), or a 2-D `.npy` array.
  Every frame has the same shape.
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

`annotations` is a list of `{frame, point, normal, left, right}` cross-sections.

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

## The synthetic sequence: `synthetic.py`

A textured substrate under known similarity motion, with a light two-boundary bead that
rides on it and a stationary block. Masks exclude the bead and the block; visibility
excludes the block. The sequence splits into two groups, so its middle pair is a
handover. `truth` holds every pose and both boundaries, and `seed` the bead at frame 0.
The tests use it to check the evaluators end to end, with no private data.
