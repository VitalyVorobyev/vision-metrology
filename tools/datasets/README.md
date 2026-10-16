# Public datasets

The registry of every external dataset this repository evaluates on, or has considered.
Nothing from a dataset is committed here: the data goes into the git-ignored `data/`, and
only its source, licence, checksums and split of IDs are tracked.

| File | Holds |
|---|---|
| `registry.json` | one entry per dataset: source, licence as stated, citation, route, size, files with SHA-256, split, status |
| `fetch.py` | downloads a `verified` entry into `data/<id>/`, verifies its checksums and extracts it; prints the manual steps for any other |
| `make_split.py` | writes a dataset's split file from the fetched data, by a fixed seeded rule, and `--check` reproduces it |
| `splits/<id>.json` | the dev and test IDs, never any data |
| `annotations/<id>/` | our own annotations of a dataset, under CC BY 4.0, with a `LICENSE` that attributes the source |

```bash
python3 tools/datasets/fetch.py --list
python3 tools/datasets/fetch.py damsegment                  # 110 MB
python3 tools/datasets/fetch.py hpatches                    # 1.2 GB
python3 tools/datasets/fetch.py hpatches --check            # verify only
python3 tools/datasets/make_split.py hpatches --check       # the split reproduces
```

Both scripts use the standard library only and take `--data-root` for a `data/` folder
elsewhere, such as the main checkout's from a worktree.

## Statuses

- **`verified`**: the licence, as the owners state it at the source, allows our use and
  publishing aggregate results with attribution. Only these are fetched, and only these
  may be named in [`docs/performance.md`](../../docs/performance.md).
- **`licence-unclear`**: no licence is stated, or the terms limit use to non-commercial
  research or education, and whether this project's use qualifies needs the owners'
  written confirmation. Not used for published numbers until then.
- **`rejected`**: the terms forbid our use, or the data cannot be obtained. Kept so that
  nobody researches it again.

Each entry records the date-sensitive facts it rests on, such as a source page that no
longer answers, in its `route` or `notes`.

## Intake checklist

A new dataset is registered before anything is evaluated on it:

1. **The licence allows the use and publishing aggregate results.** Record it as stated
   at the source, with its URL: an SPDX id, or the exact terms quoted. If none is stated,
   the status is `licence-unclear`. Never infer a licence from a mirror or an aggregator.
2. **Nothing from the dataset is committed**: no images, crops, overlays, annotations or
   per-item results. Data goes under `data/<id>/`, outputs outside the repository. The one
   exception is [our own annotations](#our-own-annotations), which are not the dataset's.
3. **Files and checksums**: a direct URL, pinned to a revision where the host allows it,
   with its byte size and SHA-256, so `fetch.py` reproduces it. A dataset behind a
   sign-up gets its route in `route` and no `files`.
4. **The split is fixed and committed before any evaluation**: dev about 30%, test about
   70%, drawn by `make_split.py`'s rule within strata. Split by the unit that keeps
   correlated items together: a sequence, or an image when one image yields many items.
5. **Settings are frozen on dev before the test split is run.** Test runs once per frozen
   configuration. A change of settings after seeing test numbers is a new dev round.
6. **Results go to [`docs/performance.md`](../../docs/performance.md)**, with the dataset,
   its citation, the split, and bootstrap intervals where the numbers are statistics.
   `tools/check-invariants.py` fails when `performance.md` names a dataset that is not
   registered as `verified`.

## The splits

- **DamSegment**, by image, within each difficulty: 150 dev and 350 test images in each
  of Easy, Medium and Hard. One image yields several crack paths, so paths of one image
  never straddle the split. `tools/bead_eval/paths.py --split dev` (or `test`) builds the
  reference paths of one side; the default, `all`, reproduces the published numbers,
  which predate the split and pool all 1500 images.
- **HRF**, by image, within each subject group (healthy, diabetic retinopathy,
  glaucoma): 4 dev and 11 test images in each. One image yields many cross-sections.
- **HPatches**, by sequence, within illumination (`i_`) and viewpoint (`v_`): 17 and 18
  dev sequences, 40 and 41 test. Every frame pair is drawn within one sequence.

## Our own annotations

When no public dataset annotates what a benchmark needs, we annotate a `verified`
dataset ourselves. The annotations are our work, so they are committed, under
`annotations/<id>/`, with:
- a `LICENSE` that puts them under CC BY 4.0 and attributes the source dataset;
- a pointer from the registry entry, `annotations.path`;
- `protocol.dataset` set to the entry's id inside each file.

They hold geometry only, such as station positions and wall offsets, never pixels.
Synthetic stand-ins for testing are never committed. `tools/check-invariants.py` enforces
all of this.

- **`hrf/`, `damsegment/`**: cross-sections with two walls each, for the
  [two-boundary benchmark](../bead_eval/README.md#hand-annotated-cross-sections). The
  stations are frozen, 120 per dataset, split by image as above. Annotation is pending.
