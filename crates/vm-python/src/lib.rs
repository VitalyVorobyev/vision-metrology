//! PyO3 Python extension module for the `vision-metrology` workspace.
//!
//! ## Module: `vm_python`
//!
//! Exposes the following classes:
//!
//! | Python class | Rust backend | Description |
//! |---|---|---|
//! | [`PyEdgeDetector`] | `vm_edge::Edge2DDetector` | 2-D subpixel edge detection |
//! | [`PyLsdDetector`] | `vm_shape::LsdDetector` | Line Segment Detection |
//! | [`PyConicFitter`] | `vm_shape::ConicFitter` | Ellipse / conic fitting |
//! | [`PyRigidMatcher`] | `vm_match::RigidEdgeMatcher` | Rigid edge model matching |
//!
//! ## GIL strategy (OQ-4)
//! All image I/O uses `PyReadonlyArray2<u8>` (zero-copy view). The GIL must be
//! held during detection. This is the simplest correct approach; for
//! multi-threaded Python callers, copy image data to `np.array` before releasing
//! the GIL.
//!
//! ## Numpy array conventions
//! - Image arrays: `(H, W)` `uint8`, C-contiguous (row-major).
//! - Point arrays: `(N, 2)` `float32`.
//! - Edgel arrays: `(N, 5)` `float32` with columns `[x, y, nx, ny, strength]`.

mod convert;
mod detector;
mod match_py;
mod shape;

use pyo3::prelude::*;

use detector::PyEdgeDetector;
use match_py::PyRigidMatcher;
use shape::{PyConicFitter, PyLsdDetector};

/// `vm_python` — Python bindings for the vision-metrology workspace.
#[pymodule]
fn vm_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEdgeDetector>()?;
    m.add_class::<PyLsdDetector>()?;
    m.add_class::<PyConicFitter>()?;
    m.add_class::<PyRigidMatcher>()?;
    Ok(())
}
