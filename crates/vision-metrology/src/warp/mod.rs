//! Image warping: build a coordinate map once, apply it per frame.
//!
//! [`Map`] precomputes, for every destination pixel, the source coordinate it
//! should be sampled from. Building the map (`affine` / `projective` /
//! `polar` / `from_fn`) walks the destination grid once; [`Map::apply`] and
//! [`Map::apply_with_mask`] then read that buffer and gather — no
//! trigonometry, matrix multiply, or allocation happens per frame. This is
//! the "rectify" primitive the metrology chain sits on top of: fixture pose
//! (`matching`) says *where* the part is, `warp` moves its pixels into a
//! canonical frame calipers or a future variation-model can then measure
//! without repeating the geometry per call.
//!
//! ## `dst → src`: read this before calling anything here
//!
//! Every map — however built — answers "for this **destination** pixel,
//! where in the **source** do I look?" This is the gather direction: it
//! lets [`Map::apply`] visit every destination pixel exactly once with no
//! holes, which a forward (`src → dst`) scatter cannot guarantee. Concretely:
//!
//! ```text
//! dst[y][x] = src.sample( map(x, y) )
//! ```
//!
//! [`Map::affine`] and [`Map::projective`] therefore take the transform that
//! carries a **destination** coordinate to the matching **source**
//! coordinate — the inverse of the transform you would use to place the
//! source *into* the destination. If you have the forward transform (e.g. a
//! fixture pose from `matching`, which maps model space into the scene), the
//! map you build for warping the scene *into* model space is its inverse:
//! `nalgebra`'s `Isometry2`/`Similarity2`/`Affine2`/`Projective2` all provide
//! `.inverse()` for exactly this. `Map::polar` and `Map::from_fn` state their
//! own `dst → src` rule directly since there is no single matrix to invert.
//!
//! ## Pixel centers
//!
//! Per the workspace-wide convention (invariant 1), integer destination pixel
//! `(i, j)` means the coordinate `(i as f32, j as f32)` — its center, not its
//! corner. Every builder here evaluates its mapping at exactly that point.
//!
//! ## Validity is first-class
//!
//! [`Map::apply`] fills every destination pixel — out-of-source taps fall
//! back to `border`, same as any other sampling call in this workspace.
//! [`Map::apply_with_mask`] additionally reports, per destination pixel,
//! whether the sample is real image data or border fill: `255` iff **every**
//! interpolation tap it read (the one tap for [`Interp::Nearest`], all four
//! for [`Interp::Bilinear`]) fell inside the source image, `0` otherwise. A
//! caller that discards the mask silently mixes border-fabricated pixels
//! into a downstream measurement (a caliper, a variation-model pixel
//! average) with no way to tell them apart from real data — the mask is the
//! difference between "the part's edge is here" and "the warp's `Constant`
//! border happens to look like an edge."
//!
//! ## Prefiltering
//!
//! `apply`'s per-pixel gather is what the plain warp does — it is not a
//! resampling filter. Minifying by more than roughly 2× through a map built
//! here aliases exactly the way naive nearest/bilinear image scaling always
//! does. Downsample through [`crate::vm_primitives::pyr::Pyramid`] first
//! (or apply the map at a coarser destination size and read the matching
//! pyramid level as `src`) rather than asking one `Map` to both minify
//! heavily and resample; `filter` (roadmap B3, not yet landed) is not a
//! prerequisite for this module and is not needed for the common case of
//! rectifying a fixture pose at roughly unit scale.
//!
//! ## Quick start
//!
//! ```
//! use vision_metrology::warp::{Interp, Map};
//! use vision_metrology::{BorderMode, Image, Affine2f};
//!
//! let src = Image::from_vec(4, 4, (0u8..16).collect()).unwrap();
//! // Identity: dst(x, y) samples src(x, y).
//! let map = Map::affine(4, 4, &Affine2f::identity());
//! let mut dst = vec![0u8; 16];
//! map.apply(&src.as_view(), &mut dst, Interp::Nearest, BorderMode::Clamp).unwrap();
//! assert_eq!(dst, src.data());
//! ```

mod apply;
mod map;

pub use apply::Interp;
pub use map::Map;
