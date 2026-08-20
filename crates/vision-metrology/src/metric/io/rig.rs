//! Import calibration-rs's `RigExtrinsicsExport` wire format
//! (`app/schemas-generated/diagnose_wire.json` in calibration-rs).

use vm_primitives::Error;

use crate::metric::types::{BrownConrady5, CameraModel, PinholeIntrinsics, Pose3};

/// The one export kind this importer accepts (`ExportKind::RigExtrinsics`).
const EXPECTED_KIND: &str = "rig_extrinsics";

#[derive(serde::Deserialize)]
struct WireIntrinsics {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    skew: f64,
}

#[derive(serde::Deserialize)]
struct WireDistortion {
    k1: f64,
    k2: f64,
    k3: f64,
    p1: f64,
    p2: f64,
    // `iters` (undistortion iteration count) is present on the wire but not
    // carried into `BrownConrady5` — see that type's docs: this crate's
    // `undistort_pixel` always runs its own fixed iteration count instead.
}

#[derive(serde::Deserialize)]
struct WireCamera {
    k: WireIntrinsics,
    dist: WireDistortion,
    // `proj` / `sensor` / `_phantom` are schema plumbing (always `null` for
    // the pinhole rigs this importer targets) and are not read.
}

/// Wire format of `nalgebra::Isometry3<f64>`: `{"rotation": [qx, qy, qz, qw],
/// "translation": [tx, ty, tz]}` (translation in **meters** — calibration-rs's
/// `Iso3Schema` proxy documents this explicitly, since `nalgebra` itself
/// carries no unit).
#[derive(serde::Deserialize)]
struct WireIso3 {
    rotation: [f64; 4],
    translation: [f64; 3],
}

#[derive(serde::Deserialize)]
struct WireRigExtrinsicsExport {
    kind: String,
    cameras: Vec<WireCamera>,
    cam_se3_rig: Vec<WireIso3>,
}

fn convert_camera(c: &WireCamera) -> CameraModel {
    CameraModel {
        intrinsics: PinholeIntrinsics {
            fx: c.k.fx as f32,
            fy: c.k.fy as f32,
            cx: c.k.cx as f32,
            cy: c.k.cy as f32,
            skew: c.k.skew as f32,
        },
        distortion: BrownConrady5 {
            k1: c.dist.k1 as f32,
            k2: c.dist.k2 as f32,
            k3: c.dist.k3 as f32,
            p1: c.dist.p1 as f32,
            p2: c.dist.p2 as f32,
        },
    }
}

/// Meters (the wire unit) to millimetres (this module's unit — see
/// `crate::metric`'s module docs).
const M_TO_MM: f64 = 1000.0;

fn convert_pose(iso: &WireIso3) -> Pose3 {
    let [qx, qy, qz, qw] = iso.rotation;
    // `nalgebra::Quaternion::new` takes (w, i, j, k) — the wire order is
    // (qx, qy, qz, qw), so the components are reordered here. Normalized
    // defensively rather than trusted, since the wire value is someone
    // else's floating-point output.
    let q = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        qw as f32, qx as f32, qy as f32, qz as f32,
    ));
    let [tx, ty, tz] = iso.translation;
    let t = nalgebra::Translation3::new(
        (tx * M_TO_MM) as f32,
        (ty * M_TO_MM) as f32,
        (tz * M_TO_MM) as f32,
    );
    Pose3::from_parts(t, q)
}

/// Parse a calibration-rs `RigExtrinsicsExport` document into
/// `(CameraModel, Pose3)` pairs, one per camera, in `cameras[]` index order.
///
/// `Pose3` is **camera-from-reference** (`cam_se3_rig`, `T_C_R`), converted
/// from the wire's meters to this crate's millimetres. Only `kind`,
/// `cameras` and `cam_se3_rig` are read — every other field of the export
/// (reprojection diagnostics, image manifest, per-view rig poses) is not
/// this module's concern and is silently ignored by `serde`'s default
/// "extra fields are fine" behavior.
///
/// # Errors
/// [`Error::InvalidConfig`] if `bytes` is not valid JSON, is not a
/// `RigExtrinsicsExport` document (wrong or missing `kind`), or
/// `cameras.len() != cam_se3_rig.len()`.
pub fn import_rig_extrinsics(bytes: &[u8]) -> Result<Vec<(CameraModel, Pose3)>, Error> {
    let doc: WireRigExtrinsicsExport = serde_json::from_slice(bytes)
        .map_err(|_| Error::InvalidConfig("not a valid RigExtrinsicsExport document"))?;
    if doc.kind != EXPECTED_KIND {
        return Err(Error::InvalidConfig(
            "JSON document's \"kind\" is not \"rig_extrinsics\"",
        ));
    }
    if doc.cameras.len() != doc.cam_se3_rig.len() {
        return Err(Error::InvalidConfig(
            "RigExtrinsicsExport: cameras[] and cam_se3_rig[] length mismatch",
        ));
    }
    Ok(doc
        .cameras
        .iter()
        .zip(doc.cam_se3_rig.iter())
        .map(|(c, p)| (convert_camera(c), convert_pose(p)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../../../tests/fixtures/rig_extrinsics.json");

    #[test]
    fn imports_two_cameras_with_meters_to_mm_conversion() {
        let cams = import_rig_extrinsics(FIXTURE).expect("valid fixture");
        assert_eq!(cams.len(), 2);

        let (cam0, pose0) = &cams[0];
        assert_eq!(cam0.intrinsics.fx, 800.0);
        assert_eq!(cam0.distortion.k1, -0.12);
        // translation [0, 0, 1.0] m -> [0, 0, 1000] mm.
        assert!((pose0.translation.vector.z - 1000.0).abs() < 1e-4);
        assert!(pose0.rotation.angle().abs() < 1e-6);

        let (_, pose1) = &cams[1];
        assert!((pose1.translation.vector.x - 100.0).abs() < 1e-3);
        // 15 degrees, in radians.
        assert!((pose1.rotation.angle() - 15.0f32.to_radians()).abs() < 1e-3);
    }

    #[test]
    fn rejects_wrong_kind() {
        let bad = br#"{"kind":"planar_intrinsics","cameras":[],"cam_se3_rig":[]}"#;
        assert!(import_rig_extrinsics(bad).is_err());
    }

    #[test]
    fn rejects_length_mismatch() {
        let bad = br#"{"kind":"rig_extrinsics","cameras":[],"cam_se3_rig":[
            {"rotation":[0,0,0,1],"translation":[0,0,0]}
        ]}"#;
        assert!(import_rig_extrinsics(bad).is_err());
    }

    #[test]
    fn rejects_garbage() {
        assert!(import_rig_extrinsics(b"not json").is_err());
    }
}
