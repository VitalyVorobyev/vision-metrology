# AGENTS.md

Guidance for coding agents in `vision-metrology`, a Rust library (with Python bindings and
a lab app) for high-precision industrial metrology.

## Read first

1. [`docs/dev/system-design.md`](docs/dev/system-design.md): layering, the numbered
   invariants, and the ADR index ([`docs/dev/adr/`](docs/dev/adr)).
2. [`docs/dev/roadmap.md`](docs/dev/roadmap.md): the open tracks and their acceptance
   criteria.
3. [`docs/dev/backlog.md`](docs/dev/backlog.md): known debt.
4. [`CONTRIBUTING.md`](CONTRIBUTING.md): the gates, tests, benchmarks, Python workflow and
   documentation rules.

Trust these over reconstructing state from git history.

## Rules

- **Gates:** run the [CONTRIBUTING gates](CONTRIBUTING.md#quality-gates) before every
  commit, and the affected benches when a hot path changed.
- **Invariants are design constraints:** [system design](docs/dev/system-design.md#invariants).
- **Parity in the same PR:** [invariant 15](docs/dev/system-design.md#invariants). A lab API
  change also updates the Tauri command and the contract fixtures
  ([`lab/contract/README.md`](lab/contract/README.md)).
- **Docs move with the code:** [invariant 16](docs/dev/system-design.md#invariants) and
  CONTRIBUTING's [documentation rules](CONTRIBUTING.md#documentation). Finished roadmap
  items move to `CHANGELOG.md` `[Unreleased]`.
- **Scoped commits; never revert unrelated changes.**
- **Shared UI goes upstream:** [ADR-0015](docs/dev/adr/0015-the-lab.md).

## Skills

Task-specific guidance is in `.claude/skills/`. `.agents/skills` links to it.
- `metrology-invariants`: coordinate and subpixel conventions.
- `tests-synthetic-fixtures`: fixtures with known ground truth.
- `hotpath-rust` and `criterion-bench`: performance work.
- `api-shaping`: public API changes.
- `laser-extract`: the `laser` module.
