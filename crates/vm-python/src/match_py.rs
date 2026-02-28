//! Python binding for `RigidEdgeMatcher`.
//!
//! Accepts model edgels as a `(N, 5)` float32 array `[x, y, nx, ny, strength]`
//! and a scene image as a `(H, W)` uint8 array. Runs the full edge-detect +
//! chamfer-match pipeline and returns a dict with `tx`, `ty`, `angle`, `score`.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use vm_core::{Point2f, Rect2f, Vec2f};
use vm_edge::edge2d::{Edge2DConfig, Edge2DDetector, Edgel};
use vm_match::{EdgeModel, RigidEdgeMatcher, RigidMatchConfig};

use crate::convert::image_from_numpy_u8;

/// Python-facing rigid edge matcher.
///
/// Accepts a set of model edgels and a scene image. Detects edges in the scene
/// image, then runs chamfer-based grid search + ICP refinement.
///
/// ## Example (Python)
/// ```python
/// matcher = vm_python.PyRigidMatcher()
/// model_edgels = np.array([[x, y, nx, ny, strength], ...], dtype=np.float32)
/// result = matcher.match_model(model_edgels, scene_img)
/// if result:
///     print(result['tx'], result['ty'], result['angle'], result['score'])
/// ```
#[pyclass]
pub struct PyRigidMatcher {
    edge_det: Edge2DDetector,
    edge_cfg: Edge2DConfig,
    matcher: RigidEdgeMatcher,
    match_cfg: RigidMatchConfig,
}

#[pymethods]
impl PyRigidMatcher {
    /// Create a new rigid matcher with default configuration.
    #[new]
    pub fn new() -> Self {
        Self {
            edge_det: Edge2DDetector::new(),
            edge_cfg: Edge2DConfig::default(),
            matcher: RigidEdgeMatcher::new(),
            match_cfg: RigidMatchConfig::default(),
        }
    }

    /// Match `model_edgels` against `scene_img`.
    ///
    /// - `model_edgels`: `(N, 5)` float32 array with columns `[x, y, nx, ny, strength]`.
    /// - `scene_img`: `(H, W)` uint8 numpy array.
    ///
    /// Returns a dict with keys `tx`, `ty`, `angle`, `score` on success, or `None`.
    pub fn match_model<'py>(
        &mut self,
        py: Python<'py>,
        model_edgels: PyReadonlyArray2<'py, f32>,
        scene_img: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        // Parse model edgels.
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

        let raw_model_edgels: Vec<Edgel> = (0..n)
            .map(|i| Edgel {
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
            .collect();

        let model = EdgeModel::from_edgels(raw_model_edgels, 5);

        // Detect scene edgels.
        let image = image_from_numpy_u8(py, &scene_img)?;
        let scene_edgels = self.edge_det.detect_u8(&image.as_view(), &self.edge_cfg);

        // Configure search to cover the full scene.
        let w = image.width() as f32;
        let h = image.height() as f32;
        self.match_cfg.position_search = Rect2f {
            x: 0.0,
            y: 0.0,
            width: w,
            height: h,
        };

        let result = self
            .matcher
            .match_model(&model, &scene_edgels, &self.match_cfg);

        match result {
            Some(r) => {
                let d = PyDict::new_bound(py);
                let t = r.transform.translation;
                let rot = r.transform.rotation.angle();
                d.set_item("tx", t.x)?;
                d.set_item("ty", t.y)?;
                d.set_item("angle", rot)?;
                d.set_item("score", r.score)?;
                Ok(Some(d))
            }
            None => Ok(None),
        }
    }
}
