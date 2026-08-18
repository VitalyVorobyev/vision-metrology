//! Shape-based object detection: find a modelled contour under translation,
//! rotation and uniform scale.
//!
//! The similarity measure is the mean dot product between the model's gradient
//! directions and the scene's, after Steger's *Similarity Measures for
//! Occlusion, Clutter, and Illumination Invariant Object Recognition* (DAGM
//! 2001). Because it compares **directions** and not intensities, it is
//! invariant to any monotonic change in illumination, and because low-contrast
//! points contribute exactly zero while the sum is still divided by the full
//! point count, the score reads directly as `1 − occluded_fraction`.
//!
//! # Workflow
//!
//! ```no_run
//! use vision_metrology::{
//!     Image, Rect2f, ShapeMatcher, ShapeModelBuilder, ShapeModelConfig, ShapeSearchConfig,
//! };
//!
//! # fn main() -> Result<(), vision_metrology::Error> {
//! // 1. Build a model from a reference image and a rectangular ROI.
//! let reference: Image<u8> = Image::new_fill(1280, 1024, 0);
//! let roi = Rect2f { x: 500.0, y: 380.0, width: 300.0, height: 270.0 };
//! let mut builder = ShapeModelBuilder::new();
//! let model = builder.build_u8(&reference.as_view(), roi, &ShapeModelConfig::default())?;
//!
//! // 2. Search a scene.
//! let scene: Image<u8> = Image::new_fill(1280, 1024, 0);
//! let mut matcher = ShapeMatcher::new();
//! let cfg = ShapeSearchConfig { min_score: 0.6, ..Default::default() };
//!
//! for m in matcher.find_u8(&scene.as_view(), &model, &cfg) {
//!     // `pose` maps reference-image coordinates straight into the scene.
//!     println!("score {:.2} at {:?}, {:.1} deg", m.score, m.position, m.angle().to_degrees());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Polarity
//!
//! [`Polarity`] decides which contrast reversals still count as the object.
//! A model built from an image carries the photometric dark-to-bright direction
//! and works with the default [`Polarity::Match`]. A model built from a contour
//! or from CAD does **not**: nothing in a polyline says whether the part is
//! darker or brighter than its background, so the normal's sign is a guess and
//! [`Polarity::Match`] on a wrong guess scores near zero for reasons that look
//! like a bug. Use [`Polarity::IgnoreLocal`] or [`Polarity::IgnoreGlobal`] for
//! geometry-derived models — see [`ShapeModel::from_polylines`].
//!
//! [`Polarity::IgnoreGlobal`] is also the mode for the same part imaged under
//! inverted contrast, such as bright-field versus dark-field illumination.
//!
//! # Angle ranges are not wrapped
//!
//! `angle_range` is used verbatim, so a range straddling ±π is expressed
//! unwrapped — 170° to 190° is `(2.967, 3.316)`, not `(2.967, -2.967)`. Only
//! the angle of a finished [`ShapeMatch`] is wrapped into `(-π, π]`.
//!
//! # Coordinate conventions
//!
//! Positions are level-0 pixel centres. [`ShapeMatch::pose`] already contains
//! the model origin, so a point measured in the reference image maps into the
//! scene with `pose * p` and no offset arithmetic.

mod build;
mod config;
mod matcher;
mod model;
mod nms;
mod refine;
mod score;
mod search;

pub use build::{
    ContourOrientation, ShapeModelBuilder, create_shape_model_f32, create_shape_model_u8,
    create_shape_model_u16,
};
pub use config::{Polarity, Refinement, ShapeModelConfig, ShapeSearchConfig};
pub use matcher::{ShapeMatch, ShapeMatcher};
#[cfg(feature = "serde")]
pub use model::SHAPE_MODEL_FORMAT_VERSION;
pub use model::{ModelPoint, ShapeModel, ShapeModelLevel};
