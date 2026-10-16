//! Stored shape models written before a change still load after it.
//!
//! `fixtures/shape_model_v5.json` is a format-5 document written by the
//! `ShapeModel::to_bytes` of the release that introduced format 5, from an
//! L-shaped polyline model (`IgnoreLocal`, `reference_angle` 0.25, 3 levels,
//! 88 teach points). It is frozen: never regenerate it. A change to the
//! in-memory model must keep reading it, and must keep writing it back byte for
//! byte, or bump the format version (ADR-0005). The format-3 and format-4
//! documents are derived from it by removing the fields those versions lacked.

use vision_metrology::matching::{Polarity, ShapeModel};

const V5: &[u8] = include_bytes!("fixtures/shape_model_v5.json");

#[test]
fn a_format_5_document_loads_and_writes_back_byte_for_byte() {
    let model = ShapeModel::from_bytes(V5).expect("a format-5 document loads");
    assert_eq!(model.num_levels(), 3);
    assert_eq!(model.teach_point_count(), 88);
    assert_eq!(model.polarity(), Polarity::IgnoreLocal);
    assert_eq!(model.reference_angle(), 0.25);
    assert!(model.point_count(0) > 0);

    let written = model.to_bytes().expect("serializes");
    assert!(
        written == V5,
        "the stored format changed: bump FORMAT_VERSION instead"
    );
}

/// Removes `fields` from the model object and stamps `version` on the envelope.
fn older(version: u32, fields: &[&str]) -> Vec<u8> {
    let mut doc: serde_json::Value = serde_json::from_slice(V5).expect("valid json");
    doc["format_version"] = serde_json::json!(version);
    let model = doc["model"].as_object_mut().expect("model is an object");
    for f in fields {
        assert!(model.remove(*f).is_some(), "{f} is in format 5");
    }
    serde_json::to_vec(&doc).expect("re-serializes")
}

#[test]
fn format_4_and_3_documents_still_load() {
    let v5 = ShapeModel::from_bytes(V5).expect("format 5 loads");

    // Format 4 predates `reference_angle`, which reads 0.0.
    let v4 = ShapeModel::from_bytes(&older(4, &["reference_angle"])).expect("format 4 loads");
    assert_eq!(v4.reference_angle(), 0.0);
    assert_eq!(v4.teach_point_count(), 88);

    // Format 3 also predates `teach_points`, so `resample_at` refuses.
    let v3 = ShapeModel::from_bytes(&older(3, &["reference_angle", "teach_points"]))
        .expect("format 3 loads");
    assert_eq!(v3.teach_point_count(), 0);
    assert!(v3.resample_at(1.2).is_err());

    for m in [&v4, &v3] {
        assert_eq!(m.num_levels(), v5.num_levels());
        for l in 0..v5.num_levels() {
            assert_eq!(m.point_count(l), v5.point_count(l));
            assert_eq!(m.model_geometry(l), v5.model_geometry(l));
        }
        assert_eq!(m.origin(), v5.origin());
        assert_eq!(m.polarity(), v5.polarity());
    }

    for bad in [2, 6] {
        assert!(
            ShapeModel::from_bytes(&older(bad, &[])).is_err(),
            "format {bad}"
        );
    }
}
