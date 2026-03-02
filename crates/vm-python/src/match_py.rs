//! Python binding for `RigidEdgeMatcher`.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use vision_metrology::{EdgeModel, RigidEdgeMatcher};
use vm_primitives::edge::edge2d::{Edge2DDetector, Edgel as NativeEdgel};
use vm_primitives::{Point2f, Rect2f, Vec2f};

use crate::config_py::{EdgeConfig, RigidMatchConfig};
use crate::convert::image_from_numpy_u8;
use crate::types::MatchResult;

fn parse_model_edgels(model_edgels: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<NativeEdgel>> {
    let shape = model_edgels.shape();
    if shape.len() != 2 || shape[1] != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "model_edgels must be a (N, 5) float32 array [x, y, nx, ny, strength]",
        ));
    }

    let n = shape[0];
    let slice = model_edgels.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array not C-contiguous: {e}"))
    })?;

    Ok((0..n)
        .map(|i| NativeEdgel {
            p: Point2f {
                x: slice[i * 5],
                y: slice[i * 5 + 1],
            },
            n: Vec2f {
                x: slice[i * 5 + 2],
                y: slice[i * 5 + 3],
            },
            strength: slice[i * 5 + 4],
            idx: (slice[i * 5] as usize, slice[i * 5 + 1] as usize),
        })
        .collect())
}

/// Python-facing rigid edge matcher.
#[pyclass]
pub struct RigidMatcher {
    edge_det: Edge2DDetector,
    matcher: RigidEdgeMatcher,
    edge_cfg: EdgeConfig,
    match_cfg: RigidMatchConfig,
}

#[pymethods]
impl RigidMatcher {
    /// Create a matcher with optional explicit configs.
    #[new]
    #[pyo3(signature = (edge_config=None, match_config=None))]
    pub fn new(edge_config: Option<EdgeConfig>, match_config: Option<RigidMatchConfig>) -> Self {
        Self {
            edge_det: Edge2DDetector::new(),
            matcher: RigidEdgeMatcher::new(),
            edge_cfg: edge_config.unwrap_or_default(),
            match_cfg: match_config.unwrap_or_default(),
        }
    }

    /// Match `model_edgels` against `scene_img`.
    ///
    /// - `model_edgels`: `(N, 5)` float32 array with columns `[x, y, nx, ny, strength]`.
    /// - `scene_img`: `(H, W)` uint8 numpy array.
    ///
    /// Returns a `MatchResult` on success, or `None`.
    pub fn match_model<'py>(
        &mut self,
        py: Python<'py>,
        model_edgels: PyReadonlyArray2<'py, f32>,
        scene_img: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let raw_model_edgels = parse_model_edgels(model_edgels)?;
        let model = EdgeModel::from_edgels(raw_model_edgels, 5);

        let image = image_from_numpy_u8(py, &scene_img)?;
        let edge_cfg = self.edge_cfg.to_native()?;
        let scene_edgels = self.edge_det.detect_u8(&image.as_view(), &edge_cfg);

        let mut match_cfg = self.match_cfg.to_native()?;
        match_cfg.position_search = Rect2f {
            x: 0.0,
            y: 0.0,
            width: image.width() as f32,
            height: image.height() as f32,
        };

        let result = self.matcher.match_model(&model, &scene_edgels, &match_cfg);
        match result {
            Some(r) => {
                let (tx, ty) = r.translation();
                let obj = Py::new(
                    py,
                    MatchResult {
                        tx,
                        ty,
                        angle: r.angle(),
                        scale: r.scale(),
                        score: r.score,
                        inlier_count: r.inlier_count,
                    },
                )?;
                Ok(Some(obj.into()))
            }
            None => Ok(None),
        }
    }

    /// Return a copy of the edge-detector config.
    pub fn edge_config(&self) -> EdgeConfig {
        self.edge_cfg.clone()
    }

    /// Return a copy of the match config.
    pub fn match_config(&self) -> RigidMatchConfig {
        self.match_cfg.clone()
    }

    /// Replace edge-detector config.
    pub fn set_edge_config(&mut self, config: EdgeConfig) {
        self.edge_cfg = config;
    }

    /// Replace match config.
    pub fn set_match_config(&mut self, config: RigidMatchConfig) {
        self.match_cfg = config;
    }
}

pub(crate) fn match_rigid_model_impl<'py>(
    py: Python<'py>,
    model_edgels: PyReadonlyArray2<'py, f32>,
    scene_img: PyReadonlyArray2<'py, u8>,
    edge_config: EdgeConfig,
    match_config: RigidMatchConfig,
) -> PyResult<Option<Py<PyAny>>> {
    let mut matcher = RigidMatcher::new(Some(edge_config), Some(match_config));
    matcher.match_model(py, model_edgels, scene_img)
}
