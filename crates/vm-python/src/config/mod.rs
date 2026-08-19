//! Python-visible configuration objects and conversion to native Rust configs.
//!
//! ## Design notes
//!
//! - **Sentinels.** Every "auto"/"unlimited" native `Option<T>` /
//!   `Option<NonZeroUsize>` maps to a Python `None`, one-to-one with
//!   invariant 10.
//! - **`Hysteresis`.** `Edge2DConfig::hysteresis: Hysteresis::{Auto, Manual}`
//!   is *not* exposed as its own enum type. `EdgeConfig.low_thresh` /
//!   `.high_thresh: Optional[float]` mirror it with the two-field sentinel
//!   pattern familiar from `argparse`-style optional pairs: both `None` is
//!   `Auto`; setting either selects `Manual`, and a threshold left at `None`
//!   defaults to `0.0`. This keeps `EdgeConfig()` a flat kwarg bag instead of
//!   introducing a second small type for a single either/or choice.
//! - **`Contrast`.** `matching::Contrast::{Raw, FractionOfRange}` *is* exposed
//!   as its own small tagged pyclass — `Contrast.raw(v)` /
//!   `Contrast.fraction_of_range(f)` — rather than collapsed onto a bare
//!   `float`. Unlike `Hysteresis`, a bare float would have to silently pick
//!   one variant, and picking the wrong one changes behaviour by up to 257x
//!   between `uint8` and `uint16` images (see [`Contrast`]'s own docs, and
//!   `matching::Contrast` in Rust). `min_contrast` parameters default to
//!   `Contrast.raw(...)` at the value the native `Default` uses.
//! - **Nested tuning.** `ShapeSearchConfig.tuning: ShapeSearchTuning` mirrors
//!   the Rust split one-to-one: the top-level fields say *what* is being
//!   searched for, `tuning`'s six fields say *how hard* the search works.

mod edge;
mod fit;
mod lsd;
mod matching;
mod measure;

pub use edge::EdgeConfig;
pub use fit::FitConfig;
pub use lsd::LsdConfig;
pub use matching::{Contrast, ShapeModelConfig, ShapeSearchConfig, ShapeSearchTuning};
pub use measure::MeasureConfig;
