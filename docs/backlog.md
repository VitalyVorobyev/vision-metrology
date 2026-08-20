# Backlog

Known debt and deferred features, each with enough context that a future session (human or
agent) can pick it up cold. When an item is scheduled it moves into
[`roadmap.md`](roadmap.md); when it lands, delete it here and record the decision in
[`system-design.md`](system-design.md) if it changed one.

## Shape matching

- **`Contrast::FractionOfRange` needs a real-data calibration sweep before docs
  recommend it as the default advice.** `Contrast::Raw(400.0)` ≈
  `Contrast::FractionOfRange(0.098)` on the canend dataset (see `shape-matching.md`),
  but that equivalence rests on `SCHARR_FULL_STEP_GAIN = 16.0`, the unsmoothed Scharr
  operator's response to an *ideal, unblurred* unit step — real edges under
  `PreSmooth::Binomial121` or camera blur reach only a fraction of that gain, so the
  16× conversion is optimistic on real data. Sweep `FractionOfRange` against `Raw` on
  more than one dataset (ideally one with `PreSmooth::Binomial121` on) before the docs
  say more than "it exists and transfers across pixel types."
- **Anisotropic scale.** 5-DOF search with a different refinement Jacobian; deliberately
  excluded from v1. Design from scratch when a use case exists — do not bolt onto the
  4-DOF pose structs.
- **Quantized directions + SIMD score loop** (Track 2 lever 3, deferred with numbers).
  After lazy tiles + blocked span scoring, the cluttered fixture sits at 6.6 ms:
  top sweep 2.3 ms + candidate descent 4.2 ms, and the descent is dominated by
  well-scoring candidates that never abort — per-pose cost is the floor. i8 (nx, ny)
  with i16 dot products would cut that floor ~3–4× but changes score arithmetic
  (no longer bit-comparable to f32) — do it as its own PR with a documented
  tolerance policy. Clean-scene and real canend numbers are already 3.5–11 ms,
  inside the 30 ms full-pipeline budget, which is why this was deferred.
- **`PreparedScene` multi-model API** (was Track 2.5). With lazy tiled fields, the
  shareable per-scene work is only the pyramid (0.13 ms) + top field (0.02 ms) —
  the amortization argument mostly evaporated. Revisit only if a multi-model
  station measures the per-model overhead as material; the API carries real
  maintenance cost for a ~0.3 ms/model win. Tiles could additionally be shared
  across models with equal (smooth, min_contrast).
- **`rayon` parallel feature.** Top-level angle sweep is the natural fan-out. Single-thread
  performance comes first (Track 2); parallelism is a multiplier, not a fix. Keep results
  deterministic (stable reduction order) if added.
- **Timeout / anytime search** was deliberately rejected: non-deterministic results break
  the test contract. Revisit only with a deterministic budget (e.g. max poses evaluated).
- **Static scale bank for clutter without priors** (scale wave, PR #37). `scale::find_scale_invariant`
  needs an estimate first — a segmentable ROI (`estimate_scale_moments`) or an approximate
  center (`estimate_scale_logpolar`). Neither exists for a cluttered scene with **no** prior
  on where or how big the object is; the honest answer there remains a coarse discrete
  scan over `scale_range` at the top pyramid level (cheap there — `ShapeSearchTuning::last_level`
  already lets a caller stop before descending), or a small static bank of models pre-resampled
  at K fixed scales (`resample_at` makes building the bank itself trivial; the cost is K× the
  search, same shape as the plan's original "static bank" idea it deliberately avoided for the
  common case). Not attempted this wave — remains open for whichever real dataset needs it.
- **Offset-collapse score inflation at `scale < 1` — investigated, three fixes rejected by
  measurement (scale wave, PR #37 — decision 9g).** The full account — what the bug is, what each of the
  three designs cost, and why design 3 was measured *worse* than the bug it fixed — is in
  [`system-design.md`](system-design.md)'s "Scale-invariance" entry. What is left open, and the
  only thing a fourth attempt has to answer: **the search sweep's own candidate selection and
  the final reported score have to agree on whether a duplicate counts**, not just the reported
  number. That means either deduping consistently through the whole pipeline (design 2's +35%
  on `shape_find_1280x1024_scale_0p8_1p25`, unless a genuinely allocation-and-sort-free
  formulation is found) or changing how `min_score` interacts with a pose whose score is only
  known after the fact. Current behaviour is unfixed, documented on
  `matching::score::rotate_into`/`score_pose`, and pinned by
  `score::tests::offset_collapse_at_reduced_scale_is_a_known_unfixed_score_inflation`.

## Contour

- **Contour → primitive segmentation.** `contour` traces topology and reports per-edge
  tangent/curvature, but nothing turns a traced polyline into typed geometric primitives
  the way HALCON's `segment_contours_xld` does: split a `GraphEdge`'s polyline into runs
  at curvature breakpoints, classify each run as a line or an arc, and fit it with the
  `fit` module (`fit_line` / `fit_circle`). This is the piece that would let a caller go
  straight from "a contour was traced" to "here are its line segments and arcs, each
  with a fitted model and residuals" without hand-picking which polyline stretch to feed
  to `MetrologyObject`. Estimated ~300 lines over `contour` + `fit` — a curvature-breakpoint
  segmenter, per-run primitive classification (arc vs. line by curvature magnitude and
  variance), and the fit call. Not on any current track; add to `roadmap.md` Track B or C
  when it is scheduled.

## Filtering and segmentation — deliberately not investing further

- **Watershed.** `segment::watershed` already exists and is the segmentation module's
  region-splitting tool; marker-controlled or hierarchical variants are not planned.
  Revisit only if a concrete inspection case needs a segmentation `watershed` cannot
  produce with a reasonable marker set.
- **Rank filters beyond median.** B3 (`filter`, planned) scopes to the separable/recursive
  Gaussian, box mean, and an O(1)-per-radius histogram **median** — not the full
  rank-order family (min/max-of-k, percentile-k). Median covers the metrology use case
  (impulse-noise rejection on a caliper profile or an image pre-filter); the rest is
  unused generality until a concrete need appears.
- **FFT-based methods.** No planned use case in this workspace (no periodic-pattern
  removal, no frequency-domain correlation); pulling in an FFT dependency for one is not
  worth it pre-emptively. `corrmatch` (spatial-domain ZNCC) already covers the
  correlation need this crate has.

## Code health

- **File-size offenders.** Invariant 14 counts *code* lines (tests excluded), so measure
  that way, not by raw total. Over the cap today: `contour/build.rs` (802 code / 1175 total),
  `lsd/detect.rs` (757 / 983), `matching/build.rs` (737 / 737), `matching/matcher.rs`
  (653 / 692). Under it despite a large total: `edge/edge2d.rs` (578 / 859),
  `edge/gradient.rs` (527 / 783), `matching/score.rs` (423 / 683). Split opportunistically
  when a track touches them.
- **miri job** for the unsafe paths. `unsafe` is *not* confined to `laser/` — it lives in
  `vm-primitives/core/image.rs` (the `get_unchecked` family), `core/sample.rs`,
  `pyr/downsample.rs` (the contiguous-even kernels) and `edge/conv1d.rs`, with
  `laser/gather.rs` holding the fewest blocks. Scope a weekly job to
  `cargo miri test -p vm-primitives` plus `-p vision-metrology laser::`, next to the audit
  workflow, not on every PR.

## Measurement

- **Background-padding gate for calipers.** The `rtvt-pano` caliper additionally requires
  clean background for a few px beyond each edge and rejects rays that leave the image,
  which is what makes it robust to FOV truncation. Here that needs a mask or a region type
  to check against, so it waits on the `segment` rework. `RejectReason::OffImage` covers the
  leaving-the-image half already.
- **Two-pass centreline refinement.** Also from `rtvt-pano`: refine caliper centres from a
  rough polyline, then re-measure from the refined one so tangents stay continuous. That is
  a property of a tracked contour, not of a caliper — it belongs in a bead/stripe tool built
  on `measure`, whenever one exists.
- **`MeasureArc` obliquity** is checked against the arc *tangent*, which is right for
  features crossing the arc. A future "measure the arc's own edge" mode would want the
  radial direction instead.
- **Fuzzy / expected-position caliper scoring.** Today a caliper reports the strongest
  (or first/last/all) edge that clears `threshold` — there is no way to prefer an edge
  near where the nominal geometry predicts it over an equally strong but wrongly placed
  one. HALCON's `fuzzy_measure_pos` scores each candidate edge against an expected
  position/amplitude profile instead of a hard threshold, which is what keeps a caliper
  from latching onto a print mark or a highlight *of similar amplitude* to the real edge.
  Would need a scoring function on `MeasureEdge` positions relative to `MetrologyObject`'s
  nominal geometry, evaluated before `EdgeSelect` narrows the candidates.
- **Variation-model golden template (after B4).** A per-pixel tolerance band learned from
  a set of good parts (HALCON's `create_variation_model` family): teach on N reference
  frames, warp each to a common pose with `warp::Map` (B4), and store a per-pixel
  mean/σ. Inspection then flags pixels outside the learned band instead of comparing
  against nominal CAD geometry — useful for texture/print defects a caliper model can't
  describe. Blocked on B4 (`warp`), which supplies the pose-normalizing step; no design
  work started.

## Mosaic

- **Exposure/gain compensation across cameras.** The mosaic compositor (PR #35) picks
  one camera per pixel (nearest-camera-centre priority, no blending) precisely so a
  measurement always traces to one camera's own calibration — but it does nothing about
  two cameras disagreeing on *brightness* for the same physical patch (different exposure
  time, gain, or vignetting falloff). A per-camera
  gain/offset correction (estimated from the overlap region, applied before compositing)
  would shrink that gap and make the `feather` display mode less visually jarring at a
  seam. Not attempted this wave: the real-data example's own registration gate is ZNCC,
  which tolerates an offset/gain difference by construction, so nothing currently *needs*
  the correction. Would need its own accuracy
  fixture (known ground-truth exposure ratio) before being trusted the way `fit`/`measure`
  are.
- **The lab's `/api/mosaic` still composites on `z = 0` of the calibration's reference
  frame** (`lab/backend/src/vm_lab/routers/mosaic.py`, `_auto_fit_grid` / `_resolve_grid`).
  For an `import_table_calibration` rig that plane sits *at camera0's own optical centre* —
  a homography through it is singular, and there is no physical surface there — so the
  Bird's-eye tab is only meaningful for calibrations whose reference frame already sits on
  the target. `examples/birdseye_mosaic.rs` hit exactly this and now measures the plane from
  the images instead (tilt sweep → tracked correspondences → linear `n/d` solve, all in the
  example, no library module). The router was deliberately left alone: porting that
  estimator would mean either a new library/binding surface or a Python reimplementation,
  and the request schema has no way to *say* which plane a caller means. Minimum honest fix
  is to let `MosaicRequest` carry an explicit plane (or a reference-frame shift) and to
  return the overlap ZNCC the example now gates on, so a caller can at least see that a
  composite is unregistered.

## Testing

- **Laser extractor u16/f32 depth**: after the Track 1 split, the generic scan loop makes
  it cheap to run the full test matrix over all three pixel types — today u16/f32 have one
  cross-check test each.

## Python

- **`ShapeMatch.matrix()` convention docs** and a worked pixel→pose→pixel example in the
  Python README (users keep asking the equivalent HALCON question).
- **Wheel smoke on Windows** in python-wheels.yml (currently built but only imported on
  Linux).
- **`laser` has no Python binding.** Deliberately out of scope for the v0.3 Python-parity
  wave (config mirror, measure/fit/contour/morph, dtype dispatch, stubs) — every other
  default-on domain module got one. Add `LaserExtractor`/`LaserExtractConfig` bindings
  (nested `tuning`, same pattern as `ShapeSearchConfig`) when a caller needs it.
- **`segment::watershed` and edgel region growing have no Python binding.** `Segmenter`
  covers Otsu/adaptive threshold, CCL and component stats only.
- **`contour::build_graph_from_edgels`** (the raw-edgel constructor) isn't bound, only the
  detector-output convenience `build_contour_graph`. Add if a caller has edgels from
  somewhere other than `Edge2DDetector` (e.g. a laser stripe).
- **No per-caliper "explain" API for `MetrologyModel` — placement half closed, the
  re-measurement half remains.** `apply`'s result exposes the fit and the `hits` it used, but
  not which calipers were rejected or why, nor their raw profiles. The *placement* duplication
  this item originally flagged is fixed — both the FastAPI router and the Tauri command call
  `measure::diagnostics::layout`, the same code `apply` calls internally (see
  [`system-design.md`](system-design.md)'s `measure` entry). What's left: both call sites still
  invoke `Caliper::measure` a *second* time per caliper purely to recover the rejection reason
  and raw profile that `apply`'s own result does not surface — correct (both passes start from
  identical placements) but still two measurement passes over the same image data. A real
  `MetrologyModel::explain` (or an `apply` variant returning per-caliper
  `MeasureResult`/`MeasureRejected` alongside the fit) would remove that redundancy.
- **`MeasureConfigIn.polarity`'s wire strings may not reach the native binding correctly
  — found while building the Tauri command, not confirmed as a live bug.** The lab's
  Pydantic schema types this field `Literal["bright_to_dark", "dark_to_bright",
  "either"]`, and `routers/measure.py` passes it straight through to
  `vm.MeasureConfig(polarity=...)`. `vm-python/src/config/measure.rs`'s setter matches
  literal strings `"rising"`/`"falling"`, falling back to `Any` for anything else —
  which would make every one of the three documented wire values resolve to `Any`
  regardless of what was requested, since none of them is spelled `"rising"` or
  `"falling"`. Not exercised by any existing test (`lab/contract/fixtures/measure.json`'s
  request objects never set `measure.polarity`), so this was not chased down this wave;
  worth a real end-to-end check (upload a request with `polarity: "bright_to_dark"`,
  assert only falling edges are reported) before anyone relies on that field.

## Lab desktop shell (Tauri, W6)

- **Mosaic has no Tauri command** (the single record of this gap; `system-design.md` and
  `lab/README.md` point here). `routers/mosaic.py`'s compositor (~315 lines: grid
  auto-fit, nearest-camera-centre priority, `source_id` map, opt-in feather) was not
  ported to `lab/frontend/src-tauri`; `tauriBackend.ts`'s `mosaic`/`mosaicImageUrl`/
  `mosaicSourceIdUrl` throw a clear "not available in the desktop build" error. Porting
  it is mechanical (the compositing rule already exists twice — once in
  `crates/vision-metrology/tests/mosaic.rs`, once in `examples/birdseye_mosaic.rs` — so a
  third, Rust-native, non-test copy in a Tauri command is the established pattern, not a
  new design) but real work: grid auto-fit from camera footprints, the per-pixel
  nearest-camera-centre priority pass, PNG encoding of both the composite and the
  source_id palette map.
- **No packaged single-binary distribution verified — only a local, unsigned `.app`/
  `.dmg`.** `bunx tauri build` produced both without a fallback to `--no-bundle` (no
  code-signing identity was configured, and the build did not need one to succeed
  locally), but this was not installed from the `.dmg` or run past Gatekeeper — a real
  distribution would need a signing identity and (for macOS) notarization, neither
  attempted this wave.

## Waiting on upstream

- **Direct `vision-calibration` dependency** — blocked on tiny-solver/faer → nalgebra 0.35
  and a calibration-rs rebase (its MSRV 1.93 also above ours). Until then the `metric`
  module mirrors parameter types (Track 4).
- **corrmatch scale support** — corrmatch has rotation but no scale search; the external
  ZNCC score in pose_audit is documented as valid at scale ≈ 1 only. If a scaled use case
  appears, either add scale banks upstream or pre-scale the reference patch here.
- **corrmatch `u16`/`f32` support** (corr wave, PR #34) — `corr::CorrTemplate` /
  `find` / `displacement` are `u8`-only because corrmatch's published API (0.2.5) is.
  A `u16` industrial-camera path (12/16-bit sensors, same as the rest of this crate's
  `Pixel` dispatch) needs corrmatch itself to grow the dtype, not a quantizing cast
  here — filed as an upstream ask, not attempted in this wave. No other API gap was
  found against 0.2.5 for what B6 needed (template compile/rotation, bounded search,
  top-k, subpixel refine all covered the wrapper's requirements as published).
