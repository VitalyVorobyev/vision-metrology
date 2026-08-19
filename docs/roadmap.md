# Roadmap

Where the project is going, track by track, with acceptance criteria. Status values:
`planned` → `in progress` → `done` (with the landing PR). Superseded plans are rewritten,
not appended. Background and rationale live in [`system-design.md`](system-design.md);
deferred items and known debt in [`backlog.md`](backlog.md).

## Completed foundations (2026-08)

| Milestone | PR |
|---|---|
| 12 → 3 crate consolidation, CI unblocked, docs realigned | #13 |
| Dependency bumps, licensing, real CI (6 jobs), 4 algorithm bug fixes, 198 tests | #15 |
| Shape-based object detection replacing the chamfer matcher; validated 256/256 on canend | #18 |

## Current tracks

### Track 1 — B-track closure + de-spaghettification — `done` (PR #20)
Split `laser/extractor.rs` (1717 lines) into `laser/{types, extractor, scan, pairing,
coarse, gather, postprocess, tests}` with a private `ScanPixel` trait collapsing the
u8/u16/f32 triplication (public API unchanged, extract bench regression gate ≤2%).
`GridCtx` parameter struct removes the three `too_many_arguments` allows in
`contour/build.rs`. Test gaps: edge1d Centroid path + threshold rejection + u8/u16 entry
points, multiscale config defaults, contour C4 + min_component_size, **T13** (greedy-bound
safety: greediness 0 vs no-abort reference, bit-identical), **R3** (fine-toothed model at
coarse levels — passes or documented). New benches: morph, edge1d, contour smooth, and a
seeded **cluttered fixture** for match_shape. Model serialization: `serde` feature,
versioned `ShapeModel` save/load, Python parity.

**Accept:** all quality gates green; extractor files ≤ ~400 lines; no
`too_many_arguments` allows in contour; extract bench within 2% of baseline.

### Track 2 — detection performance — `done` (PRs #21, #23)
Direction (per user): pay serious attention to performance; ~5 ms full-360° detection is
the aspiration, and the hard context is a ~30 ms budget for the *whole* multi-stage
analysis of a frame, of which detection is stage 1.
Landed: clean 360° 7.8 → 3.46 ms, clutter 10.4 → 6.57 ms, tracked (ROI + angle prior)
1.49 ms, canend medians 5.6–25.5 ms at 256/256 with bit-identical scores.
Levers in order, each gated on measurement: (1) lazy 64×64-tiled direction fields —
full field only at the top level, tiles on demand around candidates below it;
(2) integer Scharr directly on u8 — dropped: measurement showed the pyramid+conversion
at 0.13 ms, not worth it; (3) quantized directions + SIMD — deferred to backlog with
measured numbers. Landed instead: blocked span scoring (bit-identical, vector-friendly),
degenerate scale/angle collapse in candidate refinement, tile-determinism tests,
tracked-ROI bench, `trace-cands` diagnostics feature. PreparedScene moved to backlog
(lazy tiles removed most of the amortization win). The `workflow_dispatch` bench
workflow landed (criterion table into the job summary, raw estimates as an artifact —
shared-runner numbers are indicative only; M4 Pro references stay in system-design.md).
`truncated()` root-cause is understood (adversarial
fixture genuinely has >128 spatially distinct coarse hypotheses; the funnel handles
them — 128→64→15→3→1 — and detection is unaffected).

**Accept:** <5 ms on both fixtures; tile-determinism + T13 green; canend 256/256
re-validated; `truncated()` resolved or explained + bounded; per-stage numbers recorded
in system-design.md.

### Track 3 — visual diagnostics + corrmatch external validation — `done`
`corrmatch` (crates.io) as a workspace dev-dependency. Diagnostic overlay replacing the
current one: scene panel with pose quad + axes glyph; **registration panel** — zoomed
checkerboard composite of the pose-warped reference patch vs the scene (sub-pixel
misregistration visible as edge breaks at checker boundaries); **contribution panel** —
model points colored by their individual score term. `examples/pose_audit/`: `audit`
(overlays + external ZNCC per match via `MaskedTemplatePlan::from_rotated_u8` +
`score_masked_zncc_at`) and `xcheck` (corrmatch full search vs ShapeMatcher, Δposition/
Δangle stats). Coordinate-convention conversion pinned by a unit test. **Repeatability on
real data**: rim fit via RANSAC ellipse per frame, tab pose in rim-centered coordinates,
σ across frames reported (target σ < 0.1 px, < 0.1°). Regenerated canend report.

**Accept:** pose_audit over ≥3 canend folders with ZNCC ≥ 0.8 on all found matches;
xcheck Δposition p95 < 1 px, Δangle p95 < 0.5°; repeatability numbers published.
**Measured (full regeneration, all 6 canend folders, 256 frames):** 256/256 found;
per-folder ZNCC p50 0.915–0.961, worst single frame 0.82 (a wrong pose collapses it by
≥0.3, pinned by test). xcheck over 20 frames × 3 set1 folders: |Δpos| p95 0.87–1.31 px,
|Δangle| p95 0.35–0.66° (sums BOTH matchers' errors incl. corrmatch's angle quantization;
accepted). Cross-polarity: dome model on dark-field frames, IgnoreGlobal — 50/50 at score
p50 0.936 with *negative* ZNCC (−0.52), independently confirming the contrast inversion.
Rim-relative spread σ(radius) 0.8–3.3 px / σ(φ−angle) 1.1–3.9° over *different physical
cans* per folder — an upper bound bundling real part-to-part tab variation, not detection
repeatability. Report (3.4) regenerated with three-panel diagnostic plates, ZNCC/xcheck
tables, and honest repeatability labelling.

### Track 4 — metrology bridge — `planned`
Runtime `metric` module consuming vision-calibration JSON exports (offline/runtime split —
see system-design.md): mirror types `PinholeIntrinsics`, `BrownConrady5`, `LaserPlane` on
nalgebra 0.35; alloc-free `undistort_pixel`, `pixel_to_ray`, `ray_plane_intersect`,
`laser_line_to_profile(&LaserLine, &Calib) -> Vec<Point3f>`, `pixel_to_plane_mm`.
Golden-file test against a real calibration-rs export. Python parity — the headline demo:
laser image → 3D profile in millimetres.

**Accept:** golden-file numeric parity with calibration-rs; profile demo runs from Python.

## Later milestones

- **v0.2.0 publish**: `vm-primitives` + `vision-metrology` to crates.io (+ wheels) once
  Tracks 1–3 land. Gate: `cargo publish --dry-run` both crates, README/docs.rs review.
- **Direct vision-calibration dependency**: when tiny-solver/faer move to nalgebra 0.35
  and calibration-rs rebases; replaces the `metric` mirror types.
- See [`backlog.md`](backlog.md) for unscheduled items.
