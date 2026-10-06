# Bead-tracker evaluation on real data

Offline scripts that run `BeadTracker`, through the `vision_metrology` Python bindings, on
real curvilinear structures: the cracks of the DamSegment dataset. They answer one
question: does the tracker lock onto a real crack and follow it from a perturbed prior?

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
