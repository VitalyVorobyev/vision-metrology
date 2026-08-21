# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

#### Desktop lab: the image and its overlay moved apart, and the contours were inert

The Teach screen looked finished and behaved like a demo. Six things, one root cause between
the first three.

- **Image and overlay drifted apart on window resize.** `CanvasStage` handed
  `ZoomPanCanvas` a full-size frame with **no aspect ratio**, then drew the photograph with
  `object-contain` (letterboxing to the *image's* aspect) while every overlay SVG used
  `preserveAspectRatio="none"` (stretching to the *frame's*). The two agreed at exactly one
  window shape and nowhere else, and the same mismatch went through `contentUnder`, so ROI
  drags and contour hit-tests were off by the letterbox as well.
- **Contours, ROI and datum did not move with the image at all.** They were mounted as
  *siblings* of the canvas rather than children, because `ZoomPanCanvas` called
  `setPointerCapture` on every `pointerdown` and left no way for an interactive layer to live
  inside it. So they stayed pinned at fit scale while the picture zoomed and panned away
  underneath them.
- **The datum read as decoration.** Its handles were `strokeWidthFor(zoom) * 4` inside that
  stretched viewBox — about **three screen pixels** — and drawn in the same cyan as the
  hundred and sixty-six contours around them.

All three are gone by construction: the lab now builds on `@vitavision/lab-ui@0.2.0`'s
`ImageStage`, whose stage element is laid out at exactly the image's pixel size and carries
the whole transform. A layer is a child `<svg viewBox="0 0 W H">` and is registered with the
photograph at every viewport size; panning is the default reading of a press and a layer opts
out by claiming it. Handles are sized through `stage.imageLength`, so they are a constant
number of *screen* pixels at any zoom, and the datum has its own colour.

- **No zoom controls.** A wheel and a 10-pixel "Fit" chip were the whole vocabulary, and
  zoom bottomed out *at* fit. There is now a toolbar over the canvas — zoom out / percentage
  menu / zoom in / Fit / 1:1 — plus `+` `-` `0` `1`, and zooming out below fit. Double-click
  toggles fit against **the view you were just at** rather than against 1:1.
- **166 contours and 7365 points with nothing to do about them.** `teach_preview` has always
  returned each contour's arc `length` and `mean_strength` — the two facts that separate the
  part from the bench it is sitting on — and neither reached the screen. There is now an
  inventory: sortable, filterable by keep state, hover-linked to the canvas in both
  directions, with select / keep / drop / invert, `↑`/`↓` to step through, `Delete` to drop,
  `Enter` to keep only the selection, `F` to frame it, and edge points drawn at 3× and above.
  Selection and keep state are separate: clicking a contour *was* dropping it, so there was
  no way to look at one without changing the model, and no way to act on several.
- **A curated selection could be built against a different extraction.** Contour ids are
  positions in the extraction — `models_create` re-runs `contours_in_roi` and trusts
  determinism — so `keep_contours` only means anything for the exact `(image, roi,
  min_contrast)` it came from, and nothing in the request records which. Moving the box after
  curating built a model out of whichever contours landed on those indices, silently. The
  panel now tracks the preview's own inputs, marks it stale and blocks the build until it is
  re-extracted; while nothing has been curated yet it re-extracts on its own instead.
- **Building wiped the evidence.** The contours were cleared on success, so "is the model
  what I picked?" — the question a build actually raises — had nothing on screen to answer
  it. They stay, and the model's points are a layer over them.

#### Desktop lab: overlays were half a pixel off the pixels they marked

Every result the canvas draws — a detected edge, a contour vertex, a fitted circle's centre —
is in the library's own convention, where `i` means the **centre** of pixel `i`
(`AGENTS.md`). CSS is the other one: an `<img>` at its natural size puts pixel `i` across
`[i, i + 1)`, so SVG's `i` is that pixel's leading edge. Drawing a measured point at its raw
coordinate therefore put it on the *boundary* of the pixel it was measured in, and reading a
pointer back named the pixel up and to the left of the one under the cursor.

Half a pixel, and it scales with the zoom: 0.4 screen pixels at fit, four at 8× — invisible
exactly where overlays get glanced at, and plainly wrong exactly where someone zooms in to
check whether one lands on the edge it claims to mark. `@vitavision/lab-ui@0.3.0` carries the
offset in `toImage`/`toScreen` and in `imageViewBox`, which every layer here now uses.

#### Desktop lab: the theme toggle's choice was not what the pre-paint script read

`index.html`'s no-flash script read `"vitavision-theme"` while `ThemeToggle` wrote
`"metrology-lab-theme"` and `main.tsx` called `initTheme()` with neither. A dark-mode user
got a light flash on every start — which in a desktop window looks like a slow load — and a
reload could come back in the palette they had just changed away from. One constant now
(`shell/theme.ts`), used by all three.

#### Desktop lab: a black window on every start, and no way to see why

- `lab/frontend`'s new routed shell put a `ThemeToggle` in its header, and
  `@vitavision/lab-ui` renders that inside a Radix tooltip — which **throws** when no
  `TooltipProvider` is mounted above it. React 19 unmounts the whole root on an uncaught
  render error, and `index.html`'s `color-scheme: dark` then paints the empty document
  black, so `bun run tauri dev` opened a window with nothing in it and no message
  anywhere. `main.tsx` now mounts `TooltipProvider` at the root, next to the router
  context `PageHeader`'s `<Link>` already required. Reproducible in the browser build too;
  it was never Tauri-specific.
- **The shell now says when it breaks.** `shell/CrashScreen.tsx` adds an error boundary
  around the tree plus `error`/`unhandledrejection` handlers installed before the first
  render, painting the message and stack into `#root`. It imports no design system, uses
  no Tailwind class and no token — inline styles only, because a crash screen that needs
  the stylesheet is another black window on the day the stylesheet is what failed. It
  paints only when the root is otherwise empty, so a live UI keeps reporting its own
  errors.
- **Startup no longer dies over one unreadable file.** `AppState::rehydrate` skipped
  nothing: a sidecar this build cannot parse, or a `ShapeModel` whose format it no longer
  accepts, propagated out of Tauri's `setup` hook through `.expect(...)` — a panic *after*
  the window exists, i.e. the same black rectangle. Each registry now drops the entry it
  cannot read (with a line on stderr) and keeps the rest, and `setup` returns its
  remaining errors instead of panicking on them.

#### `birdseye_mosaic` example: the mosaic plane is measured, not assumed

- The real-data bird's-eye example placed its plane **perpendicular to camera0's optical
  axis** at the two cameras' axis-convergence distance, because
  `25_09_17_Table_Calibration/calibration.json` records intrinsics, extrinsics and hand-eye
  but **no target pose**. The distance was about right; the orientation was never measured.
  The target is in fact tilted **37.81°** away from that axis, so each camera picked up its
  own projective error and the published composite showed the two cameras' halves at
  different scales with the checker grid stepping across the tint boundary — not a mosaic.
  `docs/assets/birdseye-mosaic.png` (README gallery) is regenerated.
- The example now **estimates the plane from the two frames**: a coarse translation-invariant
  tilt sweep (score = median tracked `corr::displacement` ZNCC, which a wrong *distance*
  cannot fool because the tracker absorbs a bulk shift), then coarse-to-fine rounds of
  track → ray-to-ray correspondence → RANSAC + least squares for `v = n/d` in
  `x1 ~ (R + t·vᵀ) x0`. Knowing `R`/`t` from the calibration makes that solve **linear in
  three unknowns**, with none of a homography decomposition's ambiguity. Recovered plane:
  `n = (0.1584, 0.5923, 0.7900)` in camera0's frame, piercing camera0's axis at
  **276.35 mm**; last round 47/47 windows tracked and 47/47 inliers, tracked residual p50
  **0.12 grid px** at 0.0128 mm/px, fit reprojection p50 **0.11 px**. No library change —
  `metric::plane_grid_map` / `warp` are used unmodified, per the mosaic wave's "not a
  library module" decision. Runtime ~16 s release.
- The example's **seam metric is now ZNCC over the jointly-valid overlap, and it gates the
  run** (`bail!` below 0.75). The old `max − min` raw-intensity statistic saturates on a
  hard-edged coded checkerboard whether or not the mosaic is registered, which is precisely
  why the broken asset shipped with a paragraph explaining away a p95 of 251/255. Measured:
  **0.9927** on the estimated plane, **0.0656** on the plane the example used to assume.
- The example now needs the `corr` feature as well as `metric`.

### Changed

#### Desktop lab: an inspector you can work in, and a frame switcher on every screen

- **The right panel is the work, not instructions for it.** Three numbered steps whose bodies
  were read-only sentences are replaced by the region as four editable numbers, the
  extraction and whether it still describes them, the contour inventory, the datum as
  numbers beside its handles, and what the model came out as — per-level point counts
  included, because a top pyramid level with a handful of points is where a model quietly
  fails. The build button is pinned to the foot of the column so a hundred and sixty-six rows
  cannot scroll it away.
- **The column is resizable and dense.** It was a fixed `22rem` at page density; a panel
  body's `p-4` inside a section's `gap-3` inside the column's own padding is three margins
  deep before a control appears. It now drags between 18 and 40 rem, remembers its width, and
  renders under `@vitavision/lab-ui`'s `compact` density — which drops leading and padding
  and keeps hit targets.
- **Frames can be changed from anywhere.** The header's `8.bmp · model-2` was dead text, so
  changing frame meant a round trip through Library — impossible to discover on Find, whose
  whole task is running a model against a different frame. It is now a switcher with
  thumbnails, prev/next, and `[` / `]`.
- **Layers are toggleable** — region, kept and dropped contours separately, edge points,
  datum, model points — from the canvas toolbar, which is what makes "what is actually in the
  model" a thing you can see rather than infer.

### Added

#### v0.2 substrate reset (Track A, PR #24)

- **`Pixel`**, a sealed trait over `u8`/`u16`/`f32`. Every `_u8`/`_u16`/`_f32` triplet
  collapsed into one generic entry point: **40 entry points → 16**. `laser`'s private
  `ScanPixel` deleted.
- **`Pyramid`** generic over `Pixel`, `unsafe`-free (5 blocks removed, and 1.1% *faster*
  than the raw-pointer version it replaced); `downsample.rs` 489 → 189 lines. Optional
  `PreSmooth::Binomial121`. `level_to_base` / `base_to_level` are the single
  implementation of invariant 2.
- **`Point2f` / `Vec2f` are nalgebra aliases**; 7 conversion functions and ~250 lines of
  hand-written operators gone. `Vec2fExt::normalized_or_zero` guards the NaN trap this
  introduced.
- **Module hygiene**: no glob re-exports, `prelude` on both crates, every domain module a
  default-on feature, CI feature-matrix job. `Error` is `#[non_exhaustive]`.
- **Six defects fixed**, three of which were silently corrupting measurements:
  `multiscale` mapped level-*l* edgels without the `(2^l − 1)/2` term, biasing every fitted
  circle centre 0.07–0.10 px (module deleted); LSD carried a second, divergent downsample
  and mapped endpoints back without the half-pixel term (−0.50 px at the default config,
  −0.95 px on odd widths); `downsample2x2_mean_*_into` guarded a release-mode
  out-of-bounds write with a `debug_assert!`; `Pyramid::ensure` was public and could
  construct 0×0 levels; `[profile.release]` was untuned (`lto = "thin"`,
  `codegen-units = 1`, worth −2.6% / −1.9%); four stale claims in the persistent-context
  docs.
- Verified: all gates green including the feature matrix and `cargo doc -D warnings`;
  `shape_find_1280x1024_360deg` 3.42 → 3.36 ms and clutter 6.61 → 6.42 ms; canend set1
  150/150 across three folders with ZNCC p50 0.912–0.961, identical to four decimals.

#### `fit` — robust primitive fitting (Track B1, PR #24)

- `fit_line` (TLS + IRLS + RANSAC), `fit_circle` (Taubin → Gauss–Newton on the true
  residual), `fit_ellipse` (Fitzgibbon → geometric), each returning `Fit<M>` with `rms` /
  `max_dev` / `n_used` / `inliers`, and `RobustLoss::{None, Huber, Tukey}`. `Circle2f`,
  `Ellipse2f` and `Conic2f` moved down to `vm-primitives`; `Circle2f` is new. Short arcs
  (30°–180°) stay within 0.05 px. Bench (M4 Pro): `fit_circle_500pts` 2.6 µs, `+tukey`
  4.2 µs, `fit_line_500pts` 1.7 µs, `fit_ellipse_100pts` 1.6 µs,
  `fit_ellipse_ransac_1000pts` 430 µs.

#### `measure` — calipers and the metrology model (Track B2, PR #24)

- `Caliper` over rect / arc / **radial** placements, typed `RejectReason`, an obliquity
  gate and sub-pixel stepping (the last three adapted from the `rtvt-pano` caliper).
  `MetrologyModel` applies nominal geometry at a fixture pose (`ShapeMatch::pose`) and
  fits robustly. A rect caliper averages along a *chord*, which biases a curved edge
  inward — 39.88 px on a nominal-40 disc; `MeasureRadial` averages along the arc: 39.990 px.
- `examples/inspect_canend` runs find → fixture → metrology model → pass/fail on real
  frames: 100/100 measured across two lighting conditions, σ ≈ 0.3 px on the rim radius,
  every caliper surviving the robust fit.

#### v0.3 API reset (Track D, PR #25)

Six commits, each with the full gate set green before the next started. No algorithm
moved; everything about how the API *states* things did.

1. **Visibility sweep.** The flat domain re-export block left the crate root (invariant
   17), `prelude` extended to `fit` / `measure` / `segment`, and nine leaked internals
   pulled back: the laser stage functions (`coarse_center_{u8,u16,f32}`,
   `best_pair_with_prior` — a triplet that also broke invariant 19),
   `contour::MAX_KERNEL_PTS`, the `pyr::downsample2x2_mean*` free functions,
   `matching::create_shape_model`, `core::transform_point_iso`, the `edge` submodule paths
   (`edge::edge2d::Edgel` → `edge::Edgel`), and `match_point_scores` (now only
   `matching::diagnostics::match_point_scores`).
2. **Config split, sentinels, `Contrast`.** `ShapeSearchConfig` 14 flat fields → 8 + a
   nested `ShapeSearchTuning`; `LaserExtractConfig` likewise. Every `0`/`0.0` = "auto"
   became `Option<NonZeroUsize>` / `Option<f32>` / a named enum (`Hysteresis`,
   `CenterSmoothing`) — invariant 10 rewritten. `min_contrast` gained a unit:
   `Contrast::{Raw, FractionOfRange}`.
3. **Result honesty.** `Caliper::measure_checked` → `measure` returning
   `Result<&[MeasureEdge], RejectReason>`, with the lossy twin deleted;
   `MetrologyModel::apply` returns one `Result` per object, in object order; `hits()`
   folded into `MetrologyResult`.
4. **One batched model-format bump** (2 → 3): opaque `save`/`load` + `to_bytes`/
   `from_bytes`, `FORMAT_VERSION` internal, `ModelPoint`/`ShapeModelLevel` read-only
   through accessors, and backlog R3 closed — the pyramid `PreSmooth` is stored in the
   model and the search reads it from there, so invariant 3 cannot be violated.
5. **`core` split into `raster` (zero nalgebra) and `geom`.** Module split only, both
   private, no public path changed.
6. **`shape` → `lsd`** (module + feature), and `DirectionField`'s stateful tiling protocol
   became a `TiledField<'a>` session that owns `ensure_rect`.

Verified: all gates green, per-feature clippy by hand (which caught one stale
`#[cfg(feature = "serde")]` that `--all-features` could not); every self-asserting example
passes; `match_shape` held or improved on all six cases (360° 3.408 → 3.370 ms, scale
sweep 17.57 → 16.83 ms); canend set1 `inspect_canend` **100/100** measured (dome mean
365.237 px σ 0.282, dark 365.696 px σ 0.307) and shape matching 150/150 found across
dome / bright / dark with dome p50 **0.998**, unchanged to three decimals.

#### vm-python parity wave (Track D.1, PR #27)

- Config mirror (nested `ShapeSearchTuning`, `roi`/`angle_range`/`scale_range`, a tagged
  `Contrast` pyclass rather than a bare float); `measure`/`fit_line`/`contour`/`morph`
  bindings (`Caliper` raising `MeasureRejected`, `MetrologyModel` returning
  `MetrologyResult | MetrologyError` per object); dtype dispatch replacing every `_u8`
  name (`EdgeDetector`, `LsdDetector`, `ShapeModel`/`ShapeMatcher` all accept
  `uint8`/`uint16`/`float32`); `.pyi` + `py.typed` wired into the wheel via maturin's
  `python-source` mixed layout — which needed a bridging `__init__.py` (the compiled
  extension lands as a *submodule* of the same-named package, not merged into it; verified
  by installing the built wheel, not just compiling). `laser` and
  `segment::watershed`/region-growing got no binding this wave (recorded in
  `docs/backlog.md`).

#### `warp` — build once, apply per frame (Track B4, PR #31)

- `Map` with `affine` / `projective` / `polar` / `from_fn` constructors, `Interp::{Nearest,
  Bilinear}`, `apply` and `apply_with_mask` over caller-owned flat buffers. Every
  constructor is expressed over `from_fn`, so there is one coordinate-precompute path and
  one hand-rolled bilinear gather. `apply_with_mask` marks a destination pixel `255` iff
  every interpolation tap it read fell inside the source — derived from which of `apply`'s
  two paths ran, not a second pass. `Map` allocates only at construction.
- Verified: identity affine reproduces the source exactly (both interpolation modes); a
  +0.5 px affine translation reproduces a linear ramp to 1e-4; affine ∘ affine⁻¹ recovers
  the source to 1e-3 away from the clamped border; a projective embedding of an affine
  matrix matches the affine path bit-for-bit; `polar` followed by a `from_fn` inverse
  recovers a two-axis disc pattern to 0.79 intensity units; the mask marks exactly the
  out-of-source taps. Bench (M4 Pro, 640×480, bilinear, `Map` built outside the loop):
  `affine_apply_640x480_bilinear` ≈ 510 µs, `polar_apply_640x480_bilinear` ≈ 494 µs
  (~1.6–1.7 ns/destination-pixel, no per-apply allocation).

#### rectify — canonical-pose crops (Track B4.1, PR #32)

- `CropSpec { rect, px_per_unit, normalize_scale }` and
  `ShapeMatch::{model_frame_pose, model_frame_map}` in
  `crates/vision-metrology/src/matching/crop.rs`; `matching`'s Cargo feature now implies
  `warp`. `CropSpec::output_size()` depends only on the spec, never on the match.
  Recommended usage (`Map::apply_with_mask` + `BorderMode::Constant`, not the crate's
  usual `Clamp`) is documented at the API.
- C1 row `rectify_repeatability`: one taught L-shape re-rendered at 12 seeded subpixel
  poses (±0.5 px translation, ±1° rotation, ±1% scale), found and rectified each time.
  Measured (2026-08-20): **bias 0.88, σ 1.69** in 8-bit intensity units (under 1% of full
  range); envelope pinned at 1.3 / 2.5.
- `examples/align_crops` runs teach → find → rectify on real glue-rig frames
  (`data/42781`): 20/20 frames found, mean crop validity 0.97.

#### `metric` — the calibration bridge (Track B5, PR #33)

- `PinholeIntrinsics`, `BrownConrady5`, `CameraModel`, `Pose3` (= `Isometry3f`,
  camera-from-reference), `Plane3`, `PlaneGrid` on nalgebra 0.35. Alloc-free
  `distort_pixel`/`undistort_pixel` (fixed 20-iteration Newton, converged < 1e-6
  normalized), `pixel_to_ray`, `ray_plane_intersect`, `pixel_to_plane`,
  `homography_plane_to_image` + `plane_grid_map`, `undistort_map`. `metric::io` imports
  calibration-rs's `RigExtrinsicsExport` (meters → mm on import) and the
  `table_calibration` tool's `calibration.json` (already mm).
- Golden-parity tests against a hand-crafted `RigExtrinsicsExport` fixture (round-trips
  `distort_pixel`/`undistort_pixel` to < 1e-6 normalized over a grid, `pixel_to_plane` by
  hand on axis-aligned cases) and a trimmed real `table_calibration.json`.
- Python: `CameraModel`/`PinholeIntrinsics`/`BrownConrady5`/`Plane3`/`PlaneGrid`
  pyclasses, `Pose3` as a `(4, 4)` float64 array, vectorized `pixel_to_plane` over
  `(N, 2)` arrays (NaN row on a miss), `plane_grid_map`/`undistort_map` returning the
  existing `Map` pyclass, `load_rig_extrinsics`/`load_table_calibration`.
- Lab: `POST/GET /api/calibration` (format detected by shape), `POST /api/measure`
  augmented with `calibration_id`/`camera_index`/`plane`, mm fields on fitted circle
  centre/radius and hit caliper edges, a px/mm toggle in the Measure tab.
- **Rectify-first 3-D acceptance** (`tests/metric_rectify.rs`): a synthetic SDF L-shape
  rendered on the reference frame's `z = 0` plane through a tilted calibrated camera
  (pinhole + BC5, tilt 0/10/20/30/40° about the reference x-axis) via the forward model,
  then rectified back with `plane_grid_map` + `apply_with_mask`. A `ShapeModel` taught on
  the 0° crop is found in every tilt's crop, score 1.0000 throughout, position error
  0.0004 / 0.0037 / 0.0023 / 0.0120 / 0.0029 px — max **0.012 px**, an order of magnitude
  inside the 0.1 px target.

#### `corr` — cross-correlation matching and displacement (Track B6, PR #34)

- `corrmatch` moves from dev-dependency to a real, optional dependency (`corr` feature,
  default-on). `corr::CorrTemplate` / `find` / `find_topk` are a thin
  zero-copy-when-contiguous adapter over corrmatch's own pyramid, angle bank, beam search
  and quadratic subpixel refinement. `CorrMatch` is deliberately not `ShapeMatch` (raw
  correlation coefficient, no scale search). `u8`-only, and says so in the module docs.
- `corr::displacement`: a bounded, rotation-off ZNCC search restricted to a `search`-pixel
  margin around the window's previous position, optionally followed by translation-only
  inverse-compositional Lucas-Kanade (`Refine::LucasKanade`, a 2x2 normal-equations solve
  on the window's own Scharr gradient computed once, 3 iterations by default).
- C1 rows `displacement_quadratic` / `displacement_lk`: a 10x10 grid of exact fractional
  shifts (0.0–0.9 px, both axes) from a continuous aperiodic value-noise texture, two
  noise levels, worst cell. Quadratic-only bias 0.0239 px / sigma 0.0164 px;
  +Lucas-Kanade bias 0.0204 px / sigma 0.0141 px — about a 15% reduction in both.
  Envelopes pinned at ~1.5x measured.
- Real-data cross-check (`tests/glue_displacement.rs`, agreement not a gate) over 39
  consecutive `data/42781` frame pairs, middle strip: mean `|shift|` 1.024 px, max
  4.003 px (inside the ±10 px search bound), mean score 0.946, cumulative displacement
  (-39.84, 2.22) px.
- Bench (`benches/corr.rs`, M4 Pro): `find` on a VGA scene with a 64x64 template —
  rotation off 4.60 ms, rotation on 22.1 ms. `displacement` on a 320x97 window —
  quadratic-only 1.60 ms, +Lucas-Kanade 1.71 ms (+7%).
- Python: `CorrTemplate`, `find`/`find_topk`/`displacement`,
  `CorrTemplateConfig`/`CorrConfig`/`DisplacementConfig` + nested tuning mirrors, `Refine`
  as a tagged pyclass. `u8`-only end to end — a `uint16`/`float32` array is a `TypeError`
  at the PyO3 boundary. Lab: `POST /api/displacement` and a Motion tab.
- No API gaps found in corrmatch 0.2.5; nothing was reported back upstream.

#### mosaic — bird's-eye composite of N calibrated cameras (Track B7, PR #35)

- **Not a library module**: `metric::plane_grid_map` per camera plus
  `warp::Map::apply_with_mask` already have everything a mosaic needs; the only new logic
  is compositing, which lives in `crates/vision-metrology/tests/mosaic.rs` and (re-derived
  deliberately) in `examples/birdseye_mosaic.rs` and the lab's `routers/mosaic.py`.
- Compositing rule: nearest-camera-centre priority, ties by camera index, no blending by
  default; uncovered pixels are `source_id = 255`. Feathering is opt-in and display-only.
- Fixture test (`tests/mosaic.rs`, CI gate): three virtual pinhole+BC5 cameras 80 mm
  apart, standing off 500 mm, 7 antialiased fiducial discs at known mm positions.
  Measured (2026-08-20, M4 Pro): full coverage inside the check rectangle (0 uncovered
  pixels); worst-of-7 fiducial centroid position error **0.0067 px** (target 0.05 px);
  seam disparity p50 0.000, p95 0.000, max 7.000 (envelope pinned at 3.0 on p95).
- Real-data example (`examples/birdseye_mosaic.rs`, demo not a gate) on
  `~/vision/data/25_09_17_Table_Calibration/` (2 real cameras): the calibration's
  reference frame is `camera0`'s own frame, whose `z = 0` sits at camera0's optical
  centre — degenerate for `plane_grid_map` — so the example recovered the closest-approach
  distance between the two cameras' optical axes instead: **273.90 mm**, with the axes
  passing within **0.406 mm** of each other, and placed the mosaic plane perpendicular to
  camera0's optical axis there. *That orientation was a guess and it was wrong* — see the
  Fixed entry above; the numbers this bullet originally carried (coverage 59.0%/65.0%,
  seam disparity p50 47.00 / p95 251.00) described a composite that was not registered.
  `docs/assets/birdseye-mosaic.png` is the README gallery row.
- One new binding, `project_plane_points` (vectorized forward reprojection), rather than
  duplicating the distortion formula in Python.
- Lab: `POST /api/mosaic` + Bird's-eye tab — `calibration_id` + `[{camera_index,
  image_id}]` + an optional grid spec (auto-fit from the cameras' image-border footprints
  via `pixel_to_plane` when omitted); response carries `image_url`/`source_id_url`,
  per-camera coverage, union/overlap fractions and seam disparity p50/p95;
  `?feather=true` switches to the display-only feather.

#### `scale` — estimate-then-verify (Track B8, PR #37)

- `estimate_scale_moments`, `estimate_scale_logpolar`, `find_scale_invariant`, and
  `ShapeModel::resample_at`. Model format 3 → **4**, backward-loading: every model now
  stores its pre-decimation level-0 edge points (`TeachPoint`); a format-3 document still
  loads with `teach_point_count() == 0`, and `resample_at`/`estimate_scale_logpolar`
  refuse cleanly rather than resampling from already-decimated points. The resampled
  model's own `scale_range` is pinned to `(0.95, 1.05)`. `warp::Map::log_polar` is new
  this wave.
- **Decision 9g — offset-collapse dedup: three designs built and measured, all three
  rejected, nothing shipped.** See `docs/system-design.md` for the full account.
- C1 rows `estimate_verify_scale_bias` / `estimate_verify_position`: same 12-scale ×
  3-rotation grid as the scan rows but on a model taught with the *default* `scale_range`,
  recovered via `find_scale_invariant` — 100% found-rate at every scale, |bias| 0.0014
  (scale, fraction) / 0.0218 px (position), sigma 0.0003 / 0.0039 px, identical to the
  scan row cell for cell. `scale_estimate_vs_scan_cost` (timed, M4 Pro): wide scan ~1.6 s
  vs. estimate-then-verify 663–745 ms on the identical scene — **~2.2–2.4x** faster.
- canend parity, bit-for-bit against the recorded baseline: dome 365.237 px / σ 0.282 px,
  dark 365.696 px / σ 0.307 px, both set1 folders, `--tolerance 1.5`.
- Python: `ShapeModel.resample_at`/`.teach_point_count`, `Map.log_polar`,
  `estimate_scale_moments`/`estimate_scale_logpolar`,
  `find_scale_invariant_roi`/`find_scale_invariant_center`,
  `MomentScaleConfig`/`LogPolarScaleConfig`/`ScaleInvariantConfig`.

#### `measure::diagnostics::layout` — caliper placement, shared (Track B2.1, PR #36)

- `CaliperShape::{Rect, Radial}`, `CaliperPlacement { object_index, caliper_index, shape }`
  and `layout(&MetrologyModel, &Similarity2f) -> Vec<CaliperPlacement>`.
  `MetrologyModel::apply` and `layout` both call one private function
  (`model::caliper_placements`). `layout` needs no image. Python:
  `MetrologyModel.layout(x, y, angle, scale, origin)`. The lab's `vm_lab/geometry.py` is
  deleted; both the FastAPI router and the Tauri command call `layout` instead of
  re-deriving placement. Overlay output verified unchanged (all pre-existing backend smoke
  tests pass without modification).

#### Accuracy regression suite (Track C1, PR #29)

- `crates/vision-metrology/tests/accuracy.rs`: a data-driven table (`ROWS: &[Row]`, one
  row per operator, `fn() -> Measured` plus a pinned envelope) covering `Edge1DDetector`,
  `Edge2DDetector`, `Caliper` (rect), `fit_circle` and `ShapeMatcher` (translation +
  rotation), each fixture an antialiased Gaussian-CDF edge (or SDF+smoothstep L-shape)
  with exact subpixel ground truth. Envelopes measured once and pinned at ~1.5x, except
  `fit_circle`'s worst cell (30° arc + 10% outliers), a genuinely near-degenerate fit.
- The `Edge2DDetector` sweep found that `Hysteresis::Auto` is unusable at the
  heavy-blur/high-noise corner of the grid; the fixture characterises each blur level's
  clean peak once and holds a fixed `Hysteresis::Manual` threshold instead.
- Scale row (PR #31): `shape_matcher_scale_bias` / `shape_matcher_scale_position` sweep 12
  true scales geometrically spaced over 0.5–2.0× at 3 rotations each, model taught at
  scale 1.0 with `scale_range = (0.45, 2.1)`. Measured: found-rate **100% at every one of
  the 12 scales** (36/36), scale bias ≤ 0.0014 (0.14%), position bias ≤ 0.022 px.
  Envelopes pinned at ~1.5–2x (scale bias 0.003, scale σ 0.001, position bias 0.04 px,
  position σ 0.01 px); found-rate pinned as a per-scale regression guard
  (`BASELINE_FOUND_RATE`).

#### Tauri desktop shell for the lab (Track C4, PR #36)

- `lab/frontend/src-tauri`, a Tauri v2 crate (`vm-lab-desktop`) that calls
  `vision-metrology`/`vm-primitives` directly — commands and events, no HTTP, no PyO3 —
  behind the same `LabBackend` TypeScript interface the browser build's `httpBackend`
  implements. One frontend bundle, transport chosen at runtime by `getBackend()`.
- Standalone Cargo workspace: `src-tauri/Cargo.toml` carries its own empty `[workspace]`
  table, verified with `cargo metadata` from the repo root (still exactly
  `vm-primitives`/`vision-metrology`/`vm-python`).
- Contract fixtures, the anti-drift gate: `lab/contract/fixtures/` holds golden
  request/response JSON plus small deterministic synthetic PNGs (a disc for
  teach/find/measure/rectify; two value-noise textures shifted by an exact known
  `(4.0, 3.0)` px for displacement) for the six core operations, generated from the
  FastAPI backend (`lab/backend/scripts/export_contract_fixtures.py`) and replayed by two
  independent tests — `lab/backend/tests/test_contract_fixtures.py` and
  `lab/frontend/src-tauri/tests/contract_parity.rs`.
- Command surface: `images_upload`/`images_list`/`image_data`,
  `models_create`/`models_list`, `find`, `measure`, `rectify` + `rectify_crop`,
  `displacement`, `calibration_upload`/`calibration_list`. `find` emits `lab://progress`
  started/finished events. State is a small Rust port of `store.py`.
- Verified: `bunx tauri build` produced a full bundle (`.app` + `.dmg`, macOS arm64) on
  the first clean attempt; `bun run tauri dev` launched Vite on `:5174` plus a native
  window; frontend typecheck / vitest (48 pass, 8 new for `tauriBackend.ts`) / build all
  green; root workspace `fmt`/`clippy`/`cargo metadata` re-verified unaffected.

- Manually triggered `Benchmarks` workflow (`workflow_dispatch`): runs criterion
  for a selectable crate (optionally filtered by bench name), renders the
  results table into the job summary, and uploads the raw estimates as an
  artifact. Shared-runner numbers are indicative only; M4 Pro reference numbers
  stay in `docs/system-design.md`.
- `match_point_scores` diagnostics API: the individual score term of every
  level-0 model point at a recovered pose (mean reproduces the match score).
- `pose_audit` example: per-frame **independent ZNCC** of recovered poses via
  corrmatch (dev-dependency), three-panel diagnostic overlays (pose /
  checkerboard registration / per-point contributions), rim-relative
  repeatability via the RANSAC ellipse fitter, and an `xcheck` subcommand
  running corrmatch's own rotation search against the shape matcher
  (measured on canend: |Δpos| p95 0.74 px, |Δangle| p95 0.66°).
  Conventions of the bridge are pinned by `tests/corrmatch_bridge.rs`.
- Shared example overlay module; `shape_matching` and `pose_audit` render
  through the same code.
- Lazy tiled direction fields: below the top pyramid level the matcher builds
  gradient tiles only around surviving candidates, bit-identical to the full
  build (pinned by tests). Full-360° find on 1280×1024: 7.8 → 3.46 ms clean,
  10.4 → 6.57 ms cluttered; canend medians 5.6–25.5 ms at unchanged 256/256
  detection and bit-identical scores. New `shape_find_1280x1024_tracked_roi`
  bench: 1.49 ms with a previous-frame pose prior.
- Blocked span scoring (point-major over 32-position blocks, identical greedy
  abort semantics) and degenerate scale/angle collapse in candidate
  refinement (a fixed `scale_range` no longer sweeps five scales per level).
- `trace-cands` dev feature: per-level candidate counts and stage timings on
  stderr; `examples/stage_timing.rs` per-stage probe.
- `serde` feature: versioned `ShapeModel` persistence (`to_json` / `from_json`
  with an explicit `format_version` gate) and `Serialize`/`Deserialize` on the
  model and geometry types. Python: `ShapeModel.save(path)` / `ShapeModel.load(path)`.
- Tests: greedy-bound safety (T13 — greediness 0 is bit-identical to an
  exhaustive reference at every qualifying pose), fine-toothed-contour pyramid
  aliasing probe (R3), edge1d Centroid/threshold/typed-entry coverage,
  contour C4 and `min_component_size` behavior, multiscale config defaults,
  serialization round-trips (Rust + Python).
- Benches: `morph` (chamfer distance, Zhang-Suen thinning, open3x3), `edge1d`,
  `contour_smooth_polyline_5k_sigma2`, and a seeded cluttered scene for
  `match_shape` (`shape_find_1280x1024_360deg_clutter` — 10.4 ms vs 7.8 ms
  clean on M4 Pro, the honest baseline for the <5 ms work).
- Persistent-context documentation: `docs/system-design.md` (architecture,
  invariants, decision record), `docs/roadmap.md` (tracks and acceptance
  criteria), `docs/backlog.md` (known debt). `AGENTS.md`/`CLAUDE.md` now point
  to them, and the commit checklist requires keeping them current.

- **Shape-based object detection** (`vision_metrology::matching`). `ShapeModel`
  plus `ShapeMatcher` locate a modelled contour under translation, rotation and
  uniform scale, using the mean dot product of gradient directions (Steger,
  DAGM 2001). Invariant to any monotonic illumination change; the score reads as
  `1 - occluded_fraction` because low-contrast points contribute zero while the
  sum is still divided by the full point count.
  - `Polarity::{Match, IgnoreGlobal, IgnoreLocal}` controls which contrast
    reversals still count as the object.
  - `Refinement::{None, Interpolate, LeastSquares}`; the least-squares mode is
    correspondence-free and degrades to fewer degrees of freedom on symmetric
    parts rather than solving a singular system.
  - Models can also be built from edgels, from directed points, or from contour
    polylines (`ShapeModel::from_edgels` / `from_directed_points` /
    `from_polylines`).
- `vm_primitives::DirectionField`: a dense unit gradient direction field with a
  magnitude gate, in the layout an orientation-based scoring loop wants.
- `vm_primitives` transform helpers: `transform_point`, `transform_vec`,
  `transform_point_iso`, `similarity_from_parts`, `similarity_parts`,
  `Vec2f::perp`, `Vec2f::cross`, and `parabolic_peak_offset`.
- Python bindings for the new API: `ShapeModel`, `ShapeMatcher`, `ShapeMatch`,
  `ShapeModelConfig`, `ShapeSearchConfig`, and `find_shape_model`.
- `examples/shape_matching.rs` (synthetic self-asserting mode plus a real-data
  mode that writes overlay PNGs) and `benches/match_shape.rs`.

- `ContourBuildConfig::thin` (default `true`), which skeletonises the edgel
  occupancy mask before tracing.
- CI jobs for documentation warnings, the declared MSRV, the self-asserting
  examples, and the Python extension module.
- `deny.toml` and `.github/dependabot.yml`.
- `LICENSE-MIT` and `LICENSE-APACHE`, and a `CONTRIBUTING.md`.

### Removed

- **Breaking:** the chamfer-distance matcher — `EdgeModel`, `RigidEdgeMatcher`,
  `RigidMatchConfig`, `MatchConfig`, `MatchResult`, `RigidMatchResult`,
  `chamfer_score`, `normal_score`, `build_scene_chamfer`, `transform_points`
  and `icp_refine`, together with the `RigidMatcher` / `match_rigid_model`
  Python bindings. Its coarse metric ignored gradient orientation entirely, it
  had no pyramid and no early termination, and its angle range could not cross
  ±π. `morph::chamfer_distance_u8` is unaffected and remains public.

### Changed

- `laser/extractor.rs` (1717 lines) split into 8 focused modules; the
  u8/u16/f32 triplication collapsed into one generic implementation over a
  private `ScanPixel` trait. Public API unchanged; extract benches within
  noise of the previous numbers.
- `contour/build.rs` chain-tracing helpers take a borrowed `GridCtx`;
  all three `too_many_arguments` allows removed.
- Workspace MSRV raised from 1.89 to 1.91, ahead of the planned `corrmatch` and
  `box-image-pyramid` dev-dependencies (both declare 1.91). nalgebra 0.35 needs
  only 1.89; the crates were unpublished, so no compatibility promise changed.

- **Breaking:** `LaserExtractor::extract_line_u8`, `extract_line_u16` and
  `extract_line_f32` return `Result<LaserLine, Error>`. They previously
  asserted when `ColAccess::Transposed` was selected without a transposed view.
- Bumped nalgebra to 0.35, criterion to 0.8, and pyo3/numpy to 0.29. The
  workspace MSRV is now 1.89, set by nalgebra.
- `image` is a dev-dependency with only the `png` and `bmp` features, cutting
  the dev tree from 158 to 100 crates.
- The `vm-python` Rust lib target is named `vm_python` rather than
  `vision_metrology`, removing an output-filename collision. The Python module
  is still imported as `vision_metrology`.

### Fixed

- `watershed` was exponential in image size on flat regions and never completed
  at 1280x1024; it also mis-partitioned plateaus and produced 2 px boundaries
  biased toward the lowest-numbered seed.
- Contour graph construction shattered contours into fragments because the
  edgel mask was not thin: a ring produced 288 junctions and 432 edges instead
  of two closed loops.
- `ConicFitter::fit` panicked on non-finite input points instead of reporting a
  degenerate fit.

### Removed

- The `rayon` feature, which was declared but never used.
- The `crates/vm-gallery` crate and the duplicated figure-generation tooling.
