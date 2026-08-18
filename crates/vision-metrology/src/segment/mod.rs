//! Image segmentation for the `vision-metrology` workspace.
//!
//! This module provides four segmentation modes:
//!
//! 1. **Binary thresholding** — Otsu global optimal threshold and adaptive
//!    (local-mean) threshold.
//! 2. **Connected-component labeling (CCL)** — two-pass union-find with path
//!    compression; C4 and C8 connectivity.
//! 3. **Watershed** — marker-based priority-queue flood-fill on a gradient image.
//! 4. **Edgel region growing** — chamfer-distance driven region fill using a
//!    `ContourGraph` as the boundary source.
//!
//! All public functions follow the workspace conventions:
//! - Pixel centers at `i as f32`.
//! - Default border mode: `Clamp`.
//! - Hot paths are allocation-free (per-row or per-scan).
//! - Fallible functions return `Result<T, vm_primitives::Error>`.

mod ccl;
mod region;
mod threshold;
mod watershed;

pub use ccl::{CcLabel, ComponentStats, component_stats, label_connected_components_u8};
pub use region::{RegionGrowConfig, grow_regions};
pub use threshold::{AdaptiveThreshConfig, adaptive_threshold_u8, otsu_threshold_u8};
pub use watershed::watershed;
