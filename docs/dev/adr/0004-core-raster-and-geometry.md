# ADR-0004: `core` is a nalgebra-free raster half plus a nalgebra geometry half

- Status: Accepted
- Date: 2026-08-19

## Context

Image buffers are the piece most often duplicated across vision crates. A crate that pins a
different nalgebra major version cannot share an image type with this one if the two live in
the same module. Separately, points and vectors need arithmetic, and the geometry types
(`Similarity2f`, …) already put nalgebra in the public API.

## Decision

- `core` has two private submodules behind one public path, `vm_primitives::core::…`.
  - **`raster`** (`Image<T>`, `ImageView<T>`, the sealed `Pixel` trait over u8/u16/f32,
    sampling, `BorderMode`, `Error`) **names no linear-algebra type**. `sample_bilinear_f32`
    takes bare `f32` coordinates.
  - **`geom`** holds the nalgebra aliases `Point2f`/`Vec2f`/`Similarity2f`/…, the `Vec2fExt`
    trait, transforms and the conic shapes.
- `Point2f` and `Vec2f` are nalgebra aliases rather than crate-local structs, so points cross
  into calibration-rs and corrmatch unconverted.

## Alternatives

- **Crate-local point and vector structs.** This is a parallel type system: seven conversion
  functions and about 250 lines of hand-written operators, in contradiction of ADR-0002.
- **One flat `core` module.** The nalgebra dependency becomes invisible, and extracting a
  shared raster crate later becomes a rewrite instead of a move.

## Consequences

- nalgebra's `normalize` divides unconditionally, so a zero vector becomes `NaN`, and a guard
  like `t.norm() < 0.5` is false for `NaN`. Every site that normalizes a possibly degenerate
  gradient or tangent calls `Vec2fExt::normalized_or_zero`, which a test pins.
- Vectors serialize as flat `[x, y]` arrays (nalgebra's serde form).
- `core::raster` can be extracted as a shared substrate later without touching geometry.
