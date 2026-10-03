# Roadmap

Open work, in priority order, with acceptance criteria. Status values: `planned`,
`in progress`. When an item lands it leaves this file: user-visible changes go to
[`CHANGELOG.md`](../../CHANGELOG.md), and decisions to an ADR. Known debt that is not
scheduled lives in [`backlog.md`](backlog.md).

## Where the library stands

The measurement chain runs end to end on real data: rectify → locate → fixture → calipers
→ robust fit → millimetres → pass/fail ([system design](system-design.md)). The work ahead:

1. the lab moves onto the shared `@vitavision/*` UI packages;
2. a textbook caliper baseline that CaliperBench can run;
3. the remaining gaps: `filter`, accuracy coverage, blob features, bindings.

---

## Track L: the lab on `@vitavision/*` packages, `in progress`

The lab is built on `@vitavision/ui`, `stage2d`, `charts` and `workbench`
([ADR-0015](adr/0015-the-lab.md)). What remains is moving its lab-specific canvas layers and
shell pieces upstream where a second app needs them, then using the released components.

### L3: upstream components in lab-ui, `planned`
These go into lab-ui as one ticket per PR. Each has two consumers (this lab and
CaliperBench, or another sibling app):

| Ticket | Package | Component |
|---|---|---|
| U1 | ui | numeric `NumberInput` value API; `Popover`/`DropdownMenu`/`Listbox`; `Kbd` |
| U2 | stage2d | imperative stage handle and `initialView`; `ImageLayer` with pixelated zoom and resolution tiers; tooltip'd `StageButton`; `StageLayersMenu` |
| U3 | stage2d | rectangle ROI editor layer |
| U4 | stage2d | tool model on `ImageStage` + selectable `PolylineSet` layer (hover, marquee, ids) |
| U5 | stage2d | overlay role tokens; additive `MeasureOverlay` `polyline` primitive and per-primitive `id`/`state` |
| U6 | charts | responsive frame, hover/pick/cursor, bands and markers, sequential colormaps |
| U7 | workbench | thumbnail strip / sequence navigator; `FileDrop` path adapter for Tauri |

Single-consumer proposals (datum handle, nav rail, stepper, status bar, triptych) are filed
as lab-ui issues and stay local here.

**Accept:** each component is released, with stories and tests, and passes lab-ui's
definition-of-done checks.

### L4: use the upstreamed components, `in progress`
- Delete the local ROI layer, contour selection, interaction hook, image layer and image
  grid in favour of U2–U7, as each is released.
- `MeasureOverlay` changes stay additive, because `OverlayPrimitiveOut` in the contract
  mirrors it.

**Accept:** contract parity passes and the Teach workflow test passes unchanged.

### L5: Find and Verify inventories, `planned`
- Find gets a match inventory that is hover-linked to the canvas, selectable, steppable
  and framable, as Teach's contour inventory is.
- Verify gets the same per caliper, built on U4/U5 and on M9's `explain_model`.

**Accept:** both views are driven end to end on a real capture.

---

## Track M: caliper baseline for CaliperBench, `planned`

[CaliperBench](https://github.com/VitalyVorobyev/caliperbench) scores edge localization,
paired edges (width and centre) and end caps on real and synthetic strips. Its contract is
a strip (`start`, `end`, `width`, `samples`, `across`) plus an ordered polarity list, and
the answer is edge distances from `start`. External methods run through a JSONL protocol.
This track gives vision-metrology:

- textbook methods that reproduce CaliperBench's own baselines (`gradient_parabolic`,
  `gradient_integer`, `midpoint_crossing`) to within 1e-4 px;
- a half-contrast method matching its reference edge definition;
- a Gaussian peak refinement;
- a runner that speaks the protocol.

No obliquity gate or other non-textbook logic is used by the runner. The design extends
[ADR-0008](adr/0008-calipers.md).

| Step | Content | Accept |
|---|---|---|
| M3 | `MeasureStrip{start, end, half_width, samples, across}`, `OffImage::Reject`, `t` measured from the start | strip tests ported from CaliperBench's baseline tests |
| M4 | `EdgeSelect::StrongestInOrder` + `RejectReason::IncompleteSequence` | tie and strict-order tests |
| M5 | `level1d.rs` (`LevelCrossing1D`) + `Locate::{MidpointCrossing, HalfContrast}` | analytic tests; checks run in CaliperBench's order |
| M6 | `diagnostics::explain` → `CaliperTrace` | trace edges equal `measure` output bit-for-bit |
| M7 | `examples/caliperbench_run.rs` + golden cross-check fixture | ≥ 99.9% identical rows against `caliperbench run` on synth-v1, with every mismatch a listed exact tie |
| M8 | Accuracy rows on pixel-integrated steps and bars (PSF σ 0–2.5, oblique strips, 8-bit noise) | envelopes pinned and published in `docs/performance.md` |
| M9 | `diagnostics::explain_model`: per-caliper traces plus the fit in one pass | lab backend and Tauri drop their second measurement pass |

Each step ships Python parity and updates `docs/measure.md`.

---

## B3: `filter`, `planned`

- Separable and recursive (Deriche / van Vliet) Gaussian, sliding-window box mean, an
  O(1)-per-radius histogram median, and grayscale erode/dilate/open/close/top-hat
  (van Herk–Gil-Werman). `edge/conv1d.rs` folds in here.
- It feeds the pyramid pre-smooth and illumination correction. The rank-filter scope stops
  at median ([ADR-0016](adr/0016-scope-what-we-do-not-build.md)).

**Accept:** each filter matches a naive reference bit-for-bit on seeded fixtures; median
cost is measured constant in radius.

## C1: accuracy coverage, `in progress`

The suite (`tests/accuracy.rs`) covers edges, the rect caliper, circle fitting, shape
matching (translation, rotation, scale), rectified crops and displacement. Open rows:

| Operator | Sweep | Report |
|---|---|---|
| `fit_ellipse` | point count, arc extent, noise, outlier fraction | axis bias, centre σ |
| `LaserExtractor` | stripe width, saturation, tilt | centre bias, σ |
| `Caliper` arc / radial | radius, caliper width, blur, noise | radial bias, σ |

**Accept:** each row has an envelope pinned at about 1.5× the measured value and appears in
`docs/performance.md`.

## C2: blob features, `planned`

`ComponentStats` has label, pixel count, bounding box and centroid. Add second-order moments
(orientation, elongation), convex hull, minimum-area rectangle, circularity and
rectangularity.

**Accept:** analytic shapes (ellipse, rotated rectangle) recover their parameters within a
pinned tolerance; Python parity.

## C3: bindings and CI, `planned`

- Generate the vm-python config conversions instead of mirroring them by hand
  (`src/config/*.rs`, about 1,600 lines).
- Python bindings for `laser`, `segment::watershed` / region growing and
  `build_graph_from_edgels`.
- `cargo publish --dry-run` in CI.
- A weekly miri job over every `unsafe` block.

---

## Later

- **First crates.io and PyPI release** of `vm-primitives`, `vision-metrology` and the wheel.
  Gate: `cargo publish --dry-run` for both crates, a README and docs review, and CHANGELOG
  cut.
- **Direct `calibration-rs` dependency** once its solver chain moves to nalgebra 0.35. It
  replaces the `metric` mirror types ([ADR-0003](adr/0003-metric-calibration-bridge.md)).
- **A shared raster substrate.** Several sibling crates carry their own `ImageView`.
  Publishing `core::raster` as a common crate would end that, and is worth doing after the
  first release settles the API ([ADR-0004](adr/0004-core-raster-and-geometry.md)).
