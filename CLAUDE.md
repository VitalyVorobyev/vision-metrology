# Claude Code: vision-metrology

Follow **@AGENTS.md**. It points to the persistent context in `docs/dev/` and to the
gates in `CONTRIBUTING.md`.

- When a requirement is unclear (pixel format, expected ranges, thresholds, tolerances),
  ask for the missing constraint rather than guessing.
- Prefer a simple baseline API first; optimize once behaviour is locked by tests.
- Lab UI work that would help another app is done in `../lab-ui` (`@vitavision/*`), then
  consumed here.
