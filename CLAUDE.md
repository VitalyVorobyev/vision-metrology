# Claude Code — vision-metrology

Please read and follow **@AGENTS.md** (repo-wide conventions and invariants).

Persistent context lives in `docs/system-design.md` (architecture + decisions),
`docs/roadmap.md` (current tracks), and `docs/backlog.md` (known debt) — read them at the
start of a session and keep them updated when scope or decisions change.

## Quick repo map

Three published crates, two layers: `vm-primitives` (low-level building blocks) →
`vision-metrology` (domain algorithms) → `vm-python` (PyO3 bindings).

**Do not keep a module list here.** The canonical one — every module, in both library
crates, with what it contains — is the table in
[`docs/system-design.md`](docs/system-design.md#layering). Copies of it in this file,
`AGENTS.md`, `CONTRIBUTING.md` and the crate READMEs all drifted; now they all point there.

Names live at their module path. Both crates ship a `prelude`; crate-root re-exports are
explicit lists, never globs. Every `vision-metrology` module is a default-on feature.

## Key decisions

* `nalgebra 0.35` is a workspace dependency; use type aliases `Isometry2f / Similarity2f / Affine2f / Projective2f` from `vm_primitives` — do **not** re-implement linear algebra.
* Error type: `vm_primitives::Error` across all crates.
* All public output types must be `'static` / lifetime-free (PyO3 compatibility).
* Config-struct + reusable-detector API pattern throughout.

## What “good” looks like

* No per-scan allocations in extraction loops
* Tests are deterministic and explain the expected geometry
* Benches exist for the real hot functions

## When unsure

* Ask for the missing constraint (pixel format, expected ranges, thresholds, tolerances).
* Prefer a simple baseline API first; we can optimize once behavior is locked.
