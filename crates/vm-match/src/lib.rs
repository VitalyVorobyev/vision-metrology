//! Rigid and similarity directed-edge object matching for the `vision-metrology`
//! workspace.
//!
//! This crate matches a set of model edgels (stored in an [`EdgeModel`]) against
//! scene edgels using a chamfer-distance coarse search followed by point-to-point
//! ICP refinement and normal-coherence scoring.
//!
//! ## Workflow
//! ```rust,no_run
//! use vm_match::{EdgeModel, RigidEdgeMatcher, RigidMatchConfig};
//! use vm_core::Rect2f;
//!
//! // 1. Build model from known edgels (e.g., extracted from a template image).
//! # let model_edgels = Vec::new();
//! let model = EdgeModel::from_edgels(model_edgels, 5);
//!
//! // 2. Configure the search.
//! let cfg = RigidMatchConfig {
//!     position_search: Rect2f { x: 0.0, y: 0.0, width: 640.0, height: 480.0 },
//!     min_score: 0.5,
//!     ..RigidMatchConfig::default()
//! };
//!
//! // 3. Run the matcher.
//! let mut matcher = RigidEdgeMatcher::new();
//! # let scene_edgels = Vec::new();
//! let result = matcher.match_model(&model, &scene_edgels, &cfg);
//! if let Some(r) = result {
//!     println!("Found! inlier fraction = {:.2}", r.score);
//! }
//! ```
//!
//! ## Normal convention (OQ-5)
//! `Edgel.n` points dark-to-bright (increasing intensity). When building a model
//! from a CAD boundary (no photometric context), the caller must assign outward
//! normals and is responsible for ensuring they align with the dark-to-bright
//! convention of the scene edgels. The normal-coherence scorer uses the dot product
//! of model vs. scene normals to reject false positives with inverted polarity.

mod icp;
mod matcher;
mod model;
mod rigid;
mod score;

pub use icp::icp_refine;
pub use matcher::{MatchResult, RigidEdgeMatcher, RigidMatchResult};
pub use model::EdgeModel;
pub use rigid::{MatchConfig, RigidMatchConfig};
pub use score::{build_scene_chamfer, chamfer_score, normal_score, transform_points};
