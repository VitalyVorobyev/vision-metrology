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
- **Quantized directions + SIMD score loop** (Track 2 lever 3). i8 (nx, ny), i8×i8→i32
  dot via `wide`; 4× less field memory traffic. Only if tiled fields + u8 Scharr leave
  the <5 ms target unmet — record the measured numbers here if skipped.
- **`rayon` parallel feature.** Top-level angle sweep is the natural fan-out. Single-thread
  performance comes first (Track 2); parallelism is a multiplier, not a fix. Keep results
  deterministic (stable reduction order) if added.
- **Timeout / anytime search** was deliberately rejected: non-deterministic results break
  the test contract. Revisit only with a deterministic budget (e.g. max poses evaluated).

## Code health

- **File-size offenders** (soft cap ~600 lines, see system-design invariant 14):
  `contour/build.rs` (1103), `shape/lsd.rs` (983), `vm-primitives/edge/edge2d.rs` (858),
  `vision-metrology/matching/build.rs` (737), `vm-python/config_py.rs` (588). Split
  opportunistically when a track touches them; `laser/extractor.rs` is handled by Track 1.
- **`Edge2DDetector` should consume `DirectionField`** and delete its private
  `compute_scharr` — the Scharr kernel exists twice (three times counting LSD). Pure
  refactor, no behavior change; verify with the existing edge2d tests + bench.
- **miri job** for the unsafe gather paths in `laser/` (all workspace unsafe lives there).
  Slow — scope it to `cargo miri test -p vision-metrology laser::` on a weekly schedule
  next to the audit workflow, not on every PR.

## Testing

- **R3 — coarse-level aliasing on fine-toothed contours.** `PyramidF32` is a plain 2×2 box
  mean with no pre-smoothing; high-frequency contours (fine teeth, thin webs) can alias or
  vanish by level 3–4, so the true match dies at the coarse level. `coarse_score_factor`
  is a band-aid; the real fix is an optional binomial pre-smooth on the pyramid. The comb
  fixture test (`r3_fine_toothed_model_survives_the_pyramid`) pins the current behavior —
  an 8 px tooth pitch survives today because the auto level count stops where coarse
  points destabilize. The pyramid pre-smooth remains unscheduled; a part with even finer
  teeth than the fixture may still need it.
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
