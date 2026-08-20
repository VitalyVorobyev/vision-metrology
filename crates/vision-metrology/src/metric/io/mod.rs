//! Importers for calibration-rs's exported calibration formats.
//!
//! Wire structs live only in the two submodules and are never public — only
//! [`CameraModel`](super::CameraModel) / [`Pose3`](super::Pose3) escape this
//! module, per the offline/runtime split recorded in `docs/system-design.md`
//! ("vision-calibration: offline/runtime split"). Both importers return
//! `Vec<(CameraModel, Pose3)>` in camera-index order, `Pose3` always
//! **camera-from-reference** — see [`Pose3`](super::Pose3)'s own docs for
//! that direction convention, which every importer here follows regardless
//! of the source format's own field name.
//!
//! | Importer | Source | Reference frame | Units on disk |
//! |---|---|---|---|
//! | [`import_rig_extrinsics`] | calibration-rs `RigExtrinsicsExport` (`cam_se3_rig`, `T_C_R`) | the rig frame the export was solved in | translation in **meters** — converted to mm on import |
//! | [`import_table_calibration`] | the `table_calibration` tool's `calibration.json` (`extrinsic.cameraN.sensor2camera`) | `camera0`'s own frame (its `sensor2camera` is the identity in every observed export) | translation already in **millimetres** by magnitude (~100 mm two-camera baseline) — see [`import_table_calibration`]'s own docs for the reasoning, not asserted by the source format itself |

mod rig;
mod table;

pub use rig::import_rig_extrinsics;
pub use table::import_table_calibration;
