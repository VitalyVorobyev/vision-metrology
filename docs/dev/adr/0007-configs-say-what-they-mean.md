# ADR-0007: Configs say what they mean

- Status: Accepted
- Date: 2026-08-19

## Context

Invariant 10 sets the rule: plain config structs with `Default`, and no sentinel values.
Three problems recur under it:

- configs that mix **what is searched for** with **how hard the search works**;
- pairs of fields that encode one decision, so half-set combinations are representable and
  meaningless;
- thresholds whose meaning silently changes with the pixel type.

## Decision

- **Flat structs, no builders.** A config past about 8 fields keeps its "what" fields at the
  top level and moves the effort fields into a nested `tuning: XTuning` with its own
  `Default`. Examples: `ShapeSearchConfig`, `LaserExtractConfig`.
- **One decision, one type.** Coupled fields are one enum:
  - `Hysteresis::{Auto, Manual { low, high }}`: both thresholds derived, or both given;
  - `SmoothKind::{None, Binomial3}`: whether to pre-smooth and with what, in one value;
  - `CenterSmoothing::{None, Median { half_window }}`: the filter and its window are
    named, not hidden behind a boolean.
- **Thresholds carry a unit type.** `Contrast::Raw(f32)` is Scharr response on the input
  pixel scale (the default). `Contrast::FractionOfRange(f)` resolves to `f · 16 · (max − min)`
  of the image being processed, where 16 is Scharr's response to an ideal unit step, so it
  transfers between `u8` and `u16` unchanged. The model resolves it against the reference
  ROI and the search against the scene. A model built from bare edgels has no image, so it
  rejects `FractionOfRange` instead of inventing a range.

## Alternatives

- **Builders.** More API for the same expressiveness, and they don't cross into Python.
- **Sentinels such as `0 = auto`.** A magic value is indistinguishable from a legitimate one,
  does not survive a language boundary (`None` is what a Python caller writes), and the
  compiler cannot check it.

## Consequences

- Python mirrors each config as a class with keyword arguments; nested `tuning` structs
  become nested classes.
- `FractionOfRange`'s 16× gain assumes an ideal step and is optimistic on blurred edges.
  Calibrating it on real data is in the backlog.
