# Backlog

Known debt and deferred features, each with enough context that a future session (human or
agent) can pick it up cold. When an item is scheduled it moves into
[`roadmap.md`](roadmap.md); when it lands, delete it here and record the decision in
[`system-design.md`](system-design.md) if it changed one.

## Shape matching

- **Arbitrary-region model ROI (mask).** Models are built from a rectangle; on
  non-rectangular parts (the canend tab) the model picks up background edges near the
  corners. Mitigated today by the 2 px border drop + `min_contrast`; the real fix is
  `mask: Option<&ImageView<u8>>` on `ShapeModelConfig`. Needs the same mask handling on
  every pyramid level of the reference ROI.
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

## Code health

- **File-size offenders.** Invariant 14 counts *code* lines (tests excluded), so measure
  that way, not by raw total. Over the cap today: `contour/build.rs` (802 code / 1175 total),
  `shape/lsd.rs` (757 / 983), `matching/build.rs` (737 / 737), `matching/matcher.rs`
  (653 / 692). Under it despite a large total: `edge/edge2d.rs` (578 / 859),
  `edge/gradient.rs` (527 / 783), `matching/score.rs` (423 / 683). Split opportunistically
  when a track touches them.
- **`Edge2DDetector` should consume `DirectionField`** and delete its private
  `compute_scharr` — the Scharr kernel still exists twice (`edge2d.rs` and `gradient.rs`).
  LSD's third copy of the *downsample* is gone (it uses `pyr` now), but it still has its own
  Scharr. Pure refactor, no behavior change; verify with the existing edge2d tests + bench.
- **miri job** for the unsafe paths. `unsafe` is *not* confined to `laser/` — it lives in
  `vm-primitives/core/image.rs` (the `get_unchecked` family), `core/sample.rs`,
  `pyr/downsample.rs` (the contiguous-even kernels) and `edge/conv1d.rs`, with
  `laser/gather.rs` holding the fewest blocks. Scope a weekly job to
  `cargo miri test -p vm-primitives` plus `-p vision-metrology laser::`, next to the audit
  workflow, not on every PR.

## Testing

- **R3 — coarse-level aliasing on fine-toothed contours.** `PyramidF32` is a plain 2×2 box
  mean with no pre-smoothing; high-frequency contours (fine teeth, thin webs) can alias or
  vanish by level 3–4, so the true match dies at the coarse level. `coarse_score_factor`
  is a band-aid; the real fix is an optional binomial pre-smooth on the pyramid. The comb
  fixture test (`r3_fine_toothed_model_survives_the_pyramid`) pins the current behavior —
  an 8 px tooth pitch survives today because the auto level count stops where coarse
  points destabilize. **Half done:** `PreSmooth::Binomial121` now exists on `Pyramid` and is
  the LSD default. What remains is wiring it into `ShapeModel`: invariant 3 means the model
  and the scene must share the kernel, so the choice has to be stored in the model and the
  serialization format version bumped.
- **Laser extractor u16/f32 depth**: after the Track 1 split, the generic scan loop makes
  it cheap to run the full test matrix over all three pixel types — today u16/f32 have one
  cross-check test each.

## Python

- **`ShapeMatch.matrix()` convention docs** and a worked pixel→pose→pixel example in the
  Python README (users keep asking the equivalent HALCON question).
- **Wheel smoke on Windows** in python-wheels.yml (currently built but only imported on
  Linux).

## Waiting on upstream

- **Direct `vision-calibration` dependency** — blocked on tiny-solver/faer → nalgebra 0.35
  and a calibration-rs rebase (its MSRV 1.93 also above ours). Until then the `metric`
  module mirrors parameter types (Track 4).
- **corrmatch scale support** — corrmatch has rotation but no scale search; the external
  ZNCC score in pose_audit is documented as valid at scale ≈ 1 only. If a scaled use case
  appears, either add scale banks upstream or pre-scale the reference patch here.
