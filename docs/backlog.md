# Backlog

Known debt and deferred features, each with enough context that a future session (human or
agent) can pick it up cold. When an item is scheduled it moves into
[`roadmap.md`](roadmap.md); when it lands, delete it here and record the decision in
[`system-design.md`](system-design.md) if it changed one.

## Shape matching

- **Arbitrary-region model ROI (mask) — priority raised (2026-08).** Models are built
  from a rectangle; on non-rectangular parts (the canend tab) the model picks up
  background edges near the corners. Mitigated today by the 2 px border drop +
  `min_contrast`, but that is a per-dataset tuning workaround, not a fix — the v0.3
  review flagged this as the highest-value item left in `matching` now that the config
  split and `Contrast` unit are landed. The real fix is `mask: Option<&ImageView<u8>>`
  on `ShapeModelConfig`. Needs the same mask handling on every pyramid level of the
  reference ROI.
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

- **Exposure/gain compensation across cameras.** The mosaic compositor (roadmap B7) picks
  one camera per pixel (nearest-camera-centre priority, no blending) precisely so a
  measurement always traces to one camera's own calibration — but it does nothing about
  two cameras disagreeing on *brightness* for the same physical patch (different exposure
  time, gain, or vignetting falloff). Real-data seam disparity
  (`examples/birdseye_mosaic.rs`) is large in absolute terms partly for this reason, on top
  of the hard-edged-target effect the example's own doc comment explains. A per-camera
  gain/offset correction (estimated from the overlap region, applied before compositing)
  would shrink that gap and make the `feather` display mode less visually jarring at a
  seam; not attempted this wave since the plan's acceptance bar for the real-data example
  was "produces nonzero overlap," not seam-intensity matching. Would need its own accuracy
  fixture (known ground-truth exposure ratio) before being trusted the way `fit`/`measure`
  are.

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
- **No per-caliper "explain" API for `MetrologyModel`.** `apply`'s result exposes the fit
  and the `hits` it used, but not which calipers were rejected or why, nor their raw
  profiles — useful for a UI (`lab/`'s Measure tab wants exactly this) or for debugging a
  bad fit. Today `lab/backend/src/vm_lab/routers/measure.py` reconstructs each caliper by
  hand, mirroring `MetrologyModel::measure_one`
  (`crates/vision-metrology/src/measure/model.rs`) — duplicated logic that silently goes
  stale if that function's placement math ever changes. A `MetrologyModel::explain`
  (or an `apply` variant returning per-caliper `MeasureResult`/`MeasureRejected`) would
  remove the duplication and give both Rust and Python callers a supported way to get it.

## Waiting on upstream

- **Direct `vision-calibration` dependency** — blocked on tiny-solver/faer → nalgebra 0.35
  and a calibration-rs rebase (its MSRV 1.93 also above ours). Until then the `metric`
  module mirrors parameter types (Track 4).
- **corrmatch scale support** — corrmatch has rotation but no scale search; the external
  ZNCC score in pose_audit is documented as valid at scale ≈ 1 only. If a scaled use case
  appears, either add scale banks upstream or pre-scale the reference patch here.
- **corrmatch `u16`/`f32` support** (corr wave, roadmap B6) — `corr::CorrTemplate` /
  `find` / `displacement` are `u8`-only because corrmatch's published API (0.2.5) is.
  A `u16` industrial-camera path (12/16-bit sensors, same as the rest of this crate's
  `Pixel` dispatch) needs corrmatch itself to grow the dtype, not a quantizing cast
  here — filed as an upstream ask, not attempted in this wave. No other API gap was
  found against 0.2.5 for what B6 needed (template compile/rotation, bounded search,
  top-k, subpixel refine all covered the wrapper's requirements as published).
