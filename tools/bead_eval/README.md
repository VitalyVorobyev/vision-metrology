# Bead-tracker evaluation on real data

Offline scripts that run `BeadTracker`, through the `vision_metrology` Python bindings, on
real curvilinear structures: the cracks of the DamSegment dataset. They answer three
questions:
- does the tracker lock onto a real crack and follow it from a perturbed prior?
- do its own quality signals tell a call that locked from one that did not?
- without a prior, can a ridge detector find candidate centrelines good enough to seed
  it ([acquisition](#acquisition-without-a-prior))?

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
distributed with this repository. To fetch it:

```bash
mkdir -p data/damsegment && cd data/damsegment
curl -L -o "Damage Segmentaion.zip" \
  "https://data.mendeley.com/public-files/datasets/z5z6gtt5t4/files/bd928c77-1194-4ee3-ac1a-f6ba3036e193/file_downloaded"
shasum -a 256 "Damage Segmentaion.zip"
# d87849a70d7a2e280e49d3687d89a5850902804554a35d779c941ea71ffcff1a
mkdir extracted && unzip -q "Damage Segmentaion.zip" -d extracted
```

`data/` is git-ignored. The archive holds `Damage Segmentaion/{Easy,Medium,Hard}/`, with
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
