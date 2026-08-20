//! Import the `table_calibration` tool's `calibration.json` format
//! (`intrinsic.cameraN.{matrix, distortion, frame_cols, frame_rows}` +
//! `extrinsic.cameraN.sensor2camera`).
//!
//! **Unit decision, by inspection, not by spec.** Unlike `RigExtrinsicsExport`
//! this format carries no documented unit for `sensor2camera`'s translation.
//! The real fixture (`25_09_17_Table_Calibration/calibration.json`, trimmed
//! into `tests/fixtures/table_calibration.json`) has `camera1`'s translation
//! at magnitude ~111 mm relative to `camera0` — a plausible two-camera rig
//! baseline in **millimetres**; read as meters it would claim an
//! 111-**meter** baseline, absurd for a benchtop rig. This importer treats
//! the translation as already being in millimetres and applies no
//! conversion. This is a magnitude judgment call about one real dataset, not
//! a documented contract — if a future export from this tool turns out to
//! use meters, this importer will be quietly wrong, which is why the
//! reasoning is spelled out here rather than asserted as fact.
//!
//! **Reference frame.** `camera0`'s `sensor2camera` is the identity in every
//! observed export, so this importer treats `camera0`'s own frame as the
//! reference frame and every camera's `sensor2camera` (however many
//! cameras) as directly the `Pose3` (camera-from-reference) this crate
//! wants — no extra composition needed.

use std::collections::BTreeMap;

use vm_primitives::Error;

use crate::metric::types::{BrownConrady5, CameraModel, PinholeIntrinsics, Pose3};

#[derive(serde::Deserialize)]
struct WireIntrinsicEntry {
    /// OpenCV order: `[k1, k2, p1, p2, k3]`.
    distortion: [f64; 5],
    /// Row-major 3x3: `[[fx, skew, cx], [0, fy, cy], [0, 0, 1]]`.
    matrix: [[f64; 3]; 3],
}

#[derive(serde::Deserialize)]
struct WireExtrinsicEntry {
    /// Row-major 4x4 homogeneous transform, rotation + translation.
    sensor2camera: [[f64; 4]; 4],
}

#[derive(serde::Deserialize)]
struct WireCalibration {
    intrinsic: BTreeMap<String, WireIntrinsicEntry>,
    extrinsic: BTreeMap<String, WireExtrinsicEntry>,
}

/// Parse `"camera7"` -> `7`, for sorting camera keys numerically rather than
/// lexicographically (`"camera10"` must sort after `"camera9"`).
fn camera_index(key: &str) -> Option<u32> {
    key.strip_prefix("camera")?.parse().ok()
}

fn convert_intrinsics(e: &WireIntrinsicEntry) -> CameraModel {
    let m = e.matrix;
    let [k1, k2, p1, p2, k3] = e.distortion;
    CameraModel {
        intrinsics: PinholeIntrinsics {
            fx: m[0][0] as f32,
            fy: m[1][1] as f32,
            cx: m[0][2] as f32,
            cy: m[1][2] as f32,
            skew: m[0][1] as f32,
        },
        distortion: BrownConrady5 {
            k1: k1 as f32,
            k2: k2 as f32,
            k3: k3 as f32,
            p1: p1 as f32,
            p2: p2 as f32,
        },
    }
}

fn convert_pose(e: &WireExtrinsicEntry) -> Pose3 {
    let m = e.sensor2camera;
    // Orthonormalize in f64 (`Rotation3::from_matrix` iterates to the
    // nearest orthonormal matrix rather than trusting the input exactly),
    // then cast the resulting quaternion's four components down to f32 —
    // `nalgebra::convert` only widens (`To: SupersetOf<From>`), so an
    // f64 -> f32 narrow has to be done component-wise.
    let rot64 = nalgebra::Matrix3::new(
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    );
    let q64 = nalgebra::UnitQuaternion::from_matrix(&rot64);
    let rotation = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        q64.scalar() as f32,
        q64.imag().x as f32,
        q64.imag().y as f32,
        q64.imag().z as f32,
    ));
    let translation = nalgebra::Translation3::new(m[0][3] as f32, m[1][3] as f32, m[2][3] as f32);
    Pose3::from_parts(translation, rotation)
}

/// Parse a `table_calibration` `calibration.json` document into
/// `(CameraModel, Pose3)` pairs, sorted by numeric camera index
/// (`camera0`, `camera1`, ...).
///
/// Only `intrinsic.cameraN` and `extrinsic.cameraN` are read; `handeye` and
/// any other top-level key are ignored.
///
/// # Errors
/// [`Error::InvalidConfig`] if `bytes` is not valid JSON matching the
/// expected shape, if `intrinsic` and `extrinsic` don't carry the same set
/// of camera keys, or if there are no cameras at all.
pub fn import_table_calibration(bytes: &[u8]) -> Result<Vec<(CameraModel, Pose3)>, Error> {
    let doc: WireCalibration = serde_json::from_slice(bytes)
        .map_err(|_| Error::InvalidConfig("not a valid table_calibration document"))?;

    if doc.intrinsic.is_empty() || doc.intrinsic.len() != doc.extrinsic.len() {
        return Err(Error::InvalidConfig(
            "table_calibration: intrinsic/extrinsic camera sets are empty or mismatched",
        ));
    }

    let mut indexed: Vec<(u32, &str)> = doc
        .intrinsic
        .keys()
        .map(|k| {
            camera_index(k)
                .map(|i| (i, k.as_str()))
                .ok_or(Error::InvalidConfig("camera key is not \"cameraN\""))
        })
        .collect::<Result<_, _>>()?;
    indexed.sort_by_key(|(i, _)| *i);

    indexed
        .into_iter()
        .map(|(_, key)| {
            let extrinsic = doc.extrinsic.get(key).ok_or(Error::InvalidConfig(
                "table_calibration: camera present in intrinsic but not extrinsic",
            ))?;
            let intrinsic = &doc.intrinsic[key];
            Ok((convert_intrinsics(intrinsic), convert_pose(extrinsic)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../../../tests/fixtures/table_calibration.json");

    #[test]
    fn imports_two_cameras_in_index_order() {
        let cams = import_table_calibration(FIXTURE).expect("valid fixture");
        assert_eq!(cams.len(), 2);

        let (cam0, pose0) = &cams[0];
        assert!((cam0.intrinsics.fx - 21_440.34).abs() < 1e-2);
        assert!((cam0.intrinsics.cx - 1023.5).abs() < 1e-6);
        assert!((cam0.distortion.k1 - 0.959_967_13).abs() < 1e-5);
        // camera0's sensor2camera is the identity.
        assert!(pose0.translation.vector.norm() < 1e-9);
        assert!(pose0.rotation.angle().abs() < 1e-9);

        let (cam1, pose1) = &cams[1];
        assert!((cam1.intrinsics.fx - 21_343.379).abs() < 1e-2);
        // Baseline magnitude ~111 mm, treated as already-mm (see module docs).
        assert!((pose1.translation.vector.x - 111.382_93).abs() < 1e-3);
        assert!(pose1.translation.vector.norm() > 50.0 && pose1.translation.vector.norm() < 200.0);
    }

    #[test]
    fn rejects_garbage() {
        assert!(import_table_calibration(b"{}").is_err());
    }
}
