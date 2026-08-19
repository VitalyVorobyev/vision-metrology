# System design

The architecture of `vision-metrology` as it **is**, and the record of why it is that way.
This file, together with [`roadmap.md`](roadmap.md) (where we are going) and
[`backlog.md`](backlog.md) (known debt), is the persistent context for anyone — human or
agent — starting a work session on this repository. Read all three before changing code.

Keep this file truthful: when a decision here is superseded, rewrite the entry (and say
what replaced it), don't append contradictions.

## Layering

Three publishable crates, strict one-way dependencies:

```
vm-primitives  ──►  vision-metrology  ──►  vm-python
(low-level)         (domain algorithms)    (PyO3 bindings)
```

| Crate | Modules | Contents |
|---|---|---|
| `vm-primitives` | `core` | `Image<T>`/`ImageView<T>`, sampling (nearest/bilinear), `BorderMode`, geometry + nalgebra aliases (`Isometry2f`, `Similarity2f`, …), `Error` |
| | `pyr` | `PyramidF32`: 2×2 box-mean pyramid, f32 levels from u8/u16/f32 sources |
| | `edge` | 1D/2D subpixel DoG edges, edgels, edge pairs, `DirectionField` |
| | `morph` | binary morphology (parameterized SE), chamfer distance, Zhang-Suen thinning |
| `vision-metrology` | `contour` | contour graph with T/Y junctions, per-edge geometry, polyline smoothing |
| | `laser` | stripe extraction via opposite-polarity edge pairs (rows/cols, ROI + prior) |
| | `matching` | `ShapeModel`/`ShapeMatcher`: gradient-orientation shape-based detection |
| | `segment` | Otsu/adaptive thresholding, CCL, watershed, edgel region growing |
| | `shape` | LSD, Bookstein/Fitzgibbon conic fitting, RANSAC ellipse fitting |
| `vm-python` | — | numpy-in/numpy-out detectors; lib target named `vm_python` (see invariants) |

Both library crates re-export their full module APIs at the crate root.

## Invariants

These are load-bearing. Breaking one is a design change, not a refactor, and needs a
matching update here.

1. **Pixel centers.** Integer coordinate `i` means position `i as f32`. Everything —
   edgels, ROIs, poses, pyramid coordinate mapping — assumes it.
2. **Pyramid coordinate mapping.** Level-`l` coordinate of a level-0 point:
   `L_l(p) = (p − (2^l − 1)/2) / 2^l`; candidate propagation `q_l = 2·q_{l+1} + 0.5`.
3. **Model/scene same-aliasing rule.** A shape model's level-`l` points come from running
   the edge detector on level `l` of the *reference ROI's own pyramid*, never from
   decimating level-0 points — model and scene must suffer identical box-downsample
   aliasing or coarse scores are systematically depressed. Corollary: the model build and
   the scene search must use the **same downsample kernel**.
4. **Score semantics.** The shape-matching score divides by the **full** model point count
   `n`, never by the contributing count. That is what makes `score ≈ 1 − occluded_fraction`
   and gives `min_score` its meaning. Spatially uniform point decimation (grid cells) is
   mandatory for the same reason.
5. **Rust-native.** No OpenCV, no FFI in the library crates.
6. **Hot paths are allocation-free per scan/row.** Detectors/extractors own reusable
   scratch; the only per-call allocation allowed is the output container.
7. **`unsafe` policy.** Only small, justified blocks with `// SAFETY:` comments; guarding
   `assert!`s and the block they protect move together in any refactor.
   `unsafe_op_in_unsafe_fn` is denied workspace-wide.
8. **Error type.** `vm_primitives::Error` everywhere; `&'static str` payloads only.
9. **`'static` public outputs.** No lifetimes in public result types (PyO3 compatibility).
10. **Config-struct + reusable-detector API pattern** throughout; `0.0`/`0` mean "auto"
    where a config field supports it.
11. **Default border mode is `Clamp`** in core/edge unless configured otherwise.
12. **Determinism.** No RNG in library code; tests use synthetic fixtures (seeded if
    randomness is unavoidable); f32 sort ties broken explicitly (`(−score, x, y)`).
13. **MSRV 1.91**, edition 2024, nalgebra 0.35 as workspace dependency. Do not
    re-implement linear algebra.
14. **File-size policy.** Soft cap ~600 lines per source file (tests excluded from the
    count). Crossing it is a signal to split as part of the same change, not later.
    Current known offenders are tracked in `backlog.md`.
15. **vm-python parity.** A PR that adds public Rust API updates the bindings and a Python
    test in the same PR.
16. **Docs-as-memory.** A PR that changes scope, decisions, or invariants updates
    `system-design.md` / `roadmap.md` / `backlog.md` in the same PR.

## Decisions and why

### Chamfer matcher → gradient-orientation shape matching (PR #18, 2026-08)
The distance-only chamfer matcher was replaced wholesale (deleted, not deprecated: crate
was 0.1.0 and unpublished). Its metric ignored gradient orientation, it had no pyramid, no
greedy abort, and its angle range could not cross ±π. The replacement implements Steger's
DAGM 2001 similarity measure (the algorithm behind HALCON `create_shape_model` /
`find_shape_model`): score `S = (1/n) Σ (R·tᵢ)·ĝ(pᵢ)` with three polarity modes, greedy
early termination factored to one FMA + compare per 8-point chunk, coarse-to-fine pyramid
search with 3-D (x, y, α) local maxima, and correspondence-free least-squares pose
refinement (f64 normal equations, 4→3→2 DOF Cholesky fallback for symmetric parts).
`morph::chamfer_distance_u8` survives as an independent primitive.

### `DirectionField` lives in vm-primitives, not in matching
The matcher needs a dense gated unit-gradient field per pyramid level, stored across
levels. `GradientBuffers` borrows `&mut Edge2DDetector` and cannot be held per level; the
full edge pipeline (NMS + hysteresis + edgel build) is ~2× the necessary work. The field is
a pure image primitive with no matching semantics, hence vm-primitives.

### Model `min_contrast` is the knob that decides real-world performance
On low-relief parts (canend dataset), edge-detector auto-thresholds admit model points on
faint non-repeating surface shading. Because of invariant 4 those points *dilute* the
score rather than merely not helping. Raising `ShapeModelConfig::min_contrast` took
set1/dome from 0.785 → 0.998 median score and CP34 from 35/39 found to 39/39. Documented
with a tuning table in `shape-matching.md`.

### Cross-repo dependencies are crates.io releases only
The maintainer owns `corrmatch`, `box-image-pyramid`, and `calibration-rs`. When a feature
is missing there, it is implemented upstream, released, and bumped here — never a git or
path dependency in committed code. Keeps this workspace publishable and CI reproducible.

### MSRV 1.89 → 1.91 (2026-08)
nalgebra 0.35 needs only 1.89, but `corrmatch` and `box-image-pyramid` (planned
dev-dependencies for pose auditing and the u8 scene-path pyramid) declare 1.91. The crates
here were unpublished at the time, so no compatibility promise was broken. The MSRV CI job
builds `--all-targets`, which is why even dev-dependencies constrain the floor.

### vision-calibration: offline/runtime split
`calibration-rs` has the laser-plane pipeline we need (`LaserPlane`,
`LaserlinePlaneSolver`, `pixel_to_gripper_point`), but it is structurally pinned to
nalgebra **0.34** (tiny-solver/faer chain; nalgebra types cross its public API) and MSRV
1.93. Decision: vision-calibration remains the **offline** calibration system; a small
runtime `metric` module here mirrors only the parameter types (intrinsics, distortion,
plane) on nalgebra 0.35 and loads calibration-rs JSON exports, pinned by a golden-file
test. A direct dependency replaces the mirror when upstream reaches nalgebra 0.35.

### Shape-matching preprocessing dominates the search
Measured (M4 Pro, 1280×1024, full 360°): `find_u8` ≈ 7.8 ms = ~5.3 ms direction-field
pyramid + ~2.5 ms search. Below the top pyramid level the search reads only small windows
around candidates, so full-frame fine-level fields are mostly wasted work. The performance
plan (roadmap Track 2) is lazy tiled fields first, integer u8 Scharr second, quantized
directions + SIMD only if still needed — in that order, each gated on measurement.

### `multiscale` deleted, not fixed (2026-08)
`MultiScaleEdgeDetector` ran the 2-D detector on every pyramid level and merged the results
back to level 0. Three things were wrong with it and only the first was a bug:

1. It mapped a level-`l` edgel to level 0 as `p · 2^l`, omitting the `(2^l − 1)/2` term that
   invariant 2 requires. Level-2 positions were off by 1.5 px, level-3 by 3.5 px. Mixed with
   correct level-0 edgels this produced a *systematic* bias, not noise: `examples/measure_circles`
   measured every circle centre 0.07–0.10 px low in both axes, growing with radius as the
   coarse levels contributed more. With the module removed the same example measures 0.00 px.
   Its assertions (5 px on centre, 1.5 px on radius) were far too loose to notice.
2. Deduplication keyed on `idx * 2^l`, so a level-3 edgel claimed a single level-0 pixel
   rather than the 8×8 block it stood for, and `merge_duplicates` barely merged. The test
   only asserted `merged <= all`.
3. The reported `scale = base_sigma · 2^l` was fiction. A box-mean pyramid with a fixed-σ DoG
   at each level is not a Gaussian scale space, so the number had no operational meaning.

Fixing (1) alone would have left a module that is neither a scale space nor a sound merge,
with no consumer inside the workspace. The one sound idea — the level↔level-0 mapping — is
now `pyr::level_to_base` / `base_to_level`, the single implementation of invariant 2. If
genuine scale selection is ever needed, design it as a real scale space rather than reviving
this. `examples/measure_circles` and its Python twin now use `Edge2DDetector` directly, with
assertions tightened to 0.02 px on centre and 0.10 px on radius.

### Release profile is tuned, and benches inherit it (2026-08)
`[profile.release] lto = "thin", codegen-units = 1` in the root manifest, with
`[profile.bench] inherits = "release"` so measurements match what users ship. Measured on
`match_shape` before adopting: clean 360° 3.51 → 3.42 ms (−2.6%), clutter 6.74 → 6.61 ms
(−1.9%). Small, but free and permanent; all later numbers are against this profile. It costs
release build time, which is the trade we want on a library whose detection budget is ~5 ms.

## Performance numbers (M4 Pro, single thread, release)

Record per release. The target use case budgets ~30 ms for a full multi-stage
image analysis; detection is stage 1 and must leave room for the rest.

| Bench | post-#18 | post-tiling (Track 2) |
|---|---|---|
| `shape_model_create_1280x1024` | 0.49 ms | 0.49 ms |
| `shape_find_1280x1024_360deg` | 7.8 ms | **3.46 ms** |
| `shape_find_1280x1024_360deg_clutter` | 10.4 ms | **6.57 ms** |
| `shape_find_1280x1024_tracked_roi` | — | **1.49 ms** |
| `shape_find_1280x1024_360deg_greedy0` | 11.2 ms | 5.47 ms |
| `shape_find_1280x1024_scale_0p8_1p25` | 23.1 ms | 16.95 ms |
| `direction_field_1280x1024` (full frame) | 4.0 ms | 4.0 ms (lazily skipped in find) |
| `edge2d_detect_u8_1280x1024` | 5.6 ms | 5.6 ms |

Canend real data, full 360°, median per frame: set1 dome 15 → **5.6 ms**,
bright 16.9 → 9.2 ms, dark 15 → 11.5 ms, set2 dome 63 → **25.5 ms**, conveyor
10.6 ms, CP34 9.2 ms. Detection 256/256 preserved; per-frame scores
bit-identical to the pre-tiling code (verified against `main` on identical
flags).

Where the remaining time goes (cluttered fixture, per stage): top-level sweep
2.3 ms, candidate descent 4.2 ms, everything else <0.5 ms. The descent cost is
dominated by well-scoring candidates that legitimately never trigger the
greedy abort — reducing it further means quantized/SIMD scoring
(see backlog).
