//! Replays `lab/contract/fixtures/*.json` through the Tauri command layer (native Rust,
//! `commands::*` plain functions over `&AppState` — no GUI, see `lib.rs`'s module docs)
//! and asserts numeric agreement with the golden captured from the FastAPI backend.
//!
//! This is the desktop-path half of the W6 anti-drift gate (plan decision 7); the
//! browser-path half is `lab/backend/tests/test_contract_fixtures.py`. See
//! `lab/contract/README.md` for what "agreement" means here: field-by-field numeric
//! comparison, not a byte-identical JSON shape — the two backends' response *types*
//! differ in places (this crate's own `types.rs`, not `lab_backend`'s Pydantic models),
//! but the fields both report (pose, score, measured radius/rms, mm values, displacement
//! dx/dy/score) must agree within float tolerance.

use std::path::PathBuf;

use serde_json::{Value, json};
use vm_lab_desktop::commands;
use vm_lab_desktop::state::AppState;
use vm_lab_desktop::types::{
    CropSpecIn, DisplacementRequest, FindRequest, MeasureObjectIn, MeasureRequest,
    MeasureShapeKind, ModelCreateRequest, PlaneIn, RectifyRequest,
};

const REL_TOL: f64 = 1e-2;
const ABS_TOL: f64 = 1e-2;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("contract")
        .join("fixtures")
}

fn golden(name: &str) -> Value {
    let path = fixtures_dir().join(format!("{name}.json"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("valid JSON")
}

/// Asserts `actual` agrees with `expected` at every key path present in `expected` —
/// `actual` may carry extra fields `expected` does not (this crate's response types are
/// not required to be a structural subset of the Python backend's), floats compare
/// within tolerance, everything else must match exactly.
fn assert_agrees(actual: &Value, expected: &Value, path: &str) {
    match expected {
        Value::Object(map) => {
            let Value::Object(actual_map) = actual else {
                panic!("{path}: expected an object, got {actual:?}");
            };
            for (k, v) in map {
                let a = actual_map
                    .get(k)
                    .unwrap_or_else(|| panic!("{path}.{k}: missing in actual response"));
                assert_agrees(a, v, &format!("{path}.{k}"));
            }
        }
        Value::Array(items) => {
            let Value::Array(actual_items) = actual else {
                panic!("{path}: expected an array, got {actual:?}");
            };
            assert_eq!(actual_items.len(), items.len(), "{path}: length mismatch");
            for (i, (a, e)) in actual_items.iter().zip(items).enumerate() {
                assert_agrees(a, e, &format!("{path}[{i}]"));
            }
        }
        Value::Number(n) => {
            let e = n.as_f64().expect("finite number");
            let a = actual
                .as_f64()
                .unwrap_or_else(|| panic!("{path}: expected a number, got {actual:?}"));
            let ok = (a - e).abs() <= ABS_TOL || (a - e).abs() <= REL_TOL * e.abs();
            assert!(
                ok,
                "{path}: {a} != {e} (tolerance {ABS_TOL} abs / {REL_TOL} rel)"
            );
        }
        other => {
            assert_eq!(actual, other, "{path}");
        }
    }
}

/// A fixture request's ids are the `$PLACEHOLDER` tokens (see `lab/contract/README.md`)
/// — this run's own freshly-assigned ids are substituted in before the command call.
struct Ids {
    image: String,
    frame_a: String,
    frame_b: String,
    model: String,
    calibration: String,
}

fn setup(state: &AppState) -> Ids {
    let disc = std::fs::read(fixtures_dir().join("disc.png")).expect("disc.png");
    let frame_a_bytes = std::fs::read(fixtures_dir().join("frame_a.png")).expect("frame_a.png");
    let frame_b_bytes = std::fs::read(fixtures_dir().join("frame_b.png")).expect("frame_b.png");
    let cal_bytes =
        std::fs::read(fixtures_dir().join("calibration.json")).expect("calibration.json");

    let image = commands::images::images_upload(state, "disc.png".into(), &disc)
        .expect("upload disc")
        .id;
    let frame_a = commands::images::images_upload(state, "frame_a.png".into(), &frame_a_bytes)
        .expect("upload frame_a")
        .id;
    let frame_b = commands::images::images_upload(state, "frame_b.png".into(), &frame_b_bytes)
        .expect("upload frame_b")
        .id;
    let calibration =
        commands::calibration::calibration_upload(state, "calibration.json".into(), cal_bytes)
            .expect("upload calibration")
            .id;

    // teach (same roi/min_contrast the fixtures were generated with).
    let model = commands::models::models_create(
        state,
        ModelCreateRequest {
            image_id: image.clone(),
            roi: [24.0, 24.0, 80.0, 80.0],
            min_contrast: 0.15,
            num_levels: None,
            // The parity fixtures predate curated teaching, and must keep
            // describing the plain rectangle-ROI build: `None` here is "keep
            // everything the ROI holds", the behaviour they were generated
            // against.
            keep_contours: None,
            origin: None,
            reference_angle: 0.0,
        },
    )
    .expect("teach")
    .id;

    Ids {
        image,
        frame_a,
        frame_b,
        model,
        calibration,
    }
}

#[test]
fn teach_matches_the_golden() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = AppState::new(dir.path().to_path_buf()).expect("state");
    let ids = setup(&state);

    let out = commands::models::models_list(&state)
        .into_iter()
        .find(|m| m.id == ids.model)
        .expect("model exists");
    let actual = json!({
        "roi": out.roi,
        "min_contrast": out.min_contrast,
        "num_levels_built": out.num_levels_built,
        "origin": out.origin,
        "point_counts": out.point_counts,
    });
    let golden = golden("teach");
    let expected = &golden["response"];
    let expected_subset = json!({
        "roi": expected["roi"],
        "min_contrast": expected["min_contrast"],
        "num_levels_built": expected["num_levels_built"],
        "origin": expected["origin"],
        "point_counts": expected["point_counts"],
    });
    assert_agrees(&actual, &expected_subset, "teach");
}

#[test]
fn find_matches_the_golden() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = AppState::new(dir.path().to_path_buf()).expect("state");
    let ids = setup(&state);

    let resp = commands::find::find(
        &state,
        FindRequest {
            image_id: ids.image.clone(),
            model_id: ids.model.clone(),
            min_score: 0.5,
            ..FindRequest::default()
        },
    )
    .expect("find");
    let actual = serde_json::to_value(&resp).expect("serialize");
    assert_agrees(&actual, &golden("find")["response"], "find");
}

#[test]
fn measure_matches_the_golden() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = AppState::new(dir.path().to_path_buf()).expect("state");
    let ids = setup(&state);

    let objects = vec![MeasureObjectIn {
        kind: MeasureShapeKind::Circle,
        label: Some("outer edge".to_string()),
        cx: Some(64.0),
        cy: Some(64.0),
        r: Some(24.0),
        arc: None,
        ax: None,
        ay: None,
        bx: None,
        by: None,
        n_calipers: 12,
        caliper_len: 8.0,
        caliper_width: 4.0,
        measure: None,
        fit: None,
    }];

    let resp = commands::measure::measure(
        &state,
        MeasureRequest {
            image_id: ids.image.clone(),
            model_id: ids.model.clone(),
            fixture: None,
            min_score: 0.5,
            objects: objects.clone(),
            calibration_id: None,
            camera_index: 0,
            plane: PlaneIn::default(),
        },
    )
    .expect("measure");
    let actual = serde_json::to_value(&resp).expect("serialize");
    let expected = &golden("measure")["response"];
    // The two backends' calipers arrays are the full detail from each — compare the
    // structural/numeric fields the contract cares about rather than every raw profile
    // sample, whose exact values depend on caliper-order-sensitive floating point sums.
    assert_agrees(&actual["fixture"], &expected["fixture"], "measure.fixture");
    assert_eq!(actual["fixture_source"], expected["fixture_source"]);
    let (a_obj, e_obj) = (&actual["objects"][0], &expected["objects"][0]);
    for field in [
        "kind",
        "circle_cx",
        "circle_cy",
        "circle_r",
        "rms",
        "max_dev",
        "n_used",
    ] {
        assert_agrees(
            &a_obj[field],
            &e_obj[field],
            &format!("measure.objects[0].{field}"),
        );
    }
    assert_eq!(
        a_obj["calipers"].as_array().unwrap().len(),
        e_obj["calipers"].as_array().unwrap().len(),
        "measure: caliper count"
    );

    // -- measure with calibration/mm --------------------------------------------------
    let resp_mm = commands::measure::measure(
        &state,
        MeasureRequest {
            image_id: ids.image,
            model_id: ids.model,
            fixture: None,
            min_score: 0.5,
            objects,
            calibration_id: Some(ids.calibration),
            camera_index: 1,
            plane: PlaneIn {
                nx: 0.0,
                ny: 0.0,
                nz: 1.0,
                d: -100.0,
            },
        },
    )
    .expect("measure mm");
    let actual_mm = serde_json::to_value(&resp_mm).expect("serialize");
    let expected_mm = &golden("measure_mm")["response"];
    let (a_obj, e_obj) = (&actual_mm["objects"][0], &expected_mm["objects"][0]);
    for field in ["circle_cx_mm", "circle_cy_mm", "circle_r_mm"] {
        assert_agrees(
            &a_obj[field],
            &e_obj[field],
            &format!("measure_mm.objects[0].{field}"),
        );
    }
}

#[test]
fn rectify_matches_the_golden() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = AppState::new(dir.path().to_path_buf()).expect("state");
    let ids = setup(&state);

    let resp = commands::rectify::rectify(
        &state,
        RectifyRequest {
            image_id: ids.image,
            model_id: ids.model,
            crop: CropSpecIn {
                rect: [24.0, 24.0, 80.0, 80.0],
                px_per_unit: 1.0,
                normalize_scale: true,
            },
            min_score: 0.5,
            max_matches: None,
            // The fixture is the plain rectify the contract describes: default
            // search everywhere else.
            angle_range: None,
            scale_range: None,
            refinement: None,
            min_contrast: None,
            tuning: None,
        },
    )
    .expect("rectify");

    let expected = golden("rectify");
    assert_eq!(
        resp.width,
        expected["response"]["width"].as_u64().unwrap() as usize
    );
    assert_eq!(
        resp.height,
        expected["response"]["height"].as_u64().unwrap() as usize
    );
    assert_eq!(resp.matches.len(), 1);
    let m = &resp.matches[0];
    let em = &expected["response"]["matches"][0];
    let actual = json!({"x": m.x, "y": m.y, "angle": m.angle, "scale": m.scale, "score": m.score, "validity": m.validity});
    let expected_fields = json!({"x": em["x"], "y": em["y"], "angle": em["angle"], "scale": em["scale"], "score": em["score"], "validity": em["validity"]});
    assert_agrees(&actual, &expected_fields, "rectify.matches[0]");
}

#[test]
fn displacement_matches_the_golden() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = AppState::new(dir.path().to_path_buf()).expect("state");
    let ids = setup(&state);

    let resp = commands::displacement::displacement(
        &state,
        DisplacementRequest {
            image_ids: vec![ids.frame_a, ids.frame_b],
            window: [20.0, 20.0, 70.0, 70.0],
            search_x: 10,
            search_y: 10,
            refine: "lucas_kanade".to_string(),
            lk_iters: 3,
            min_score: 0.5,
        },
    )
    .expect("displacement");
    let golden = golden("displacement");
    let expected = &golden["response"];
    assert_agrees(
        &json!({"cumulative_x": resp.cumulative_x, "cumulative_y": resp.cumulative_y}),
        &json!({"cumulative_x": expected["cumulative_x"], "cumulative_y": expected["cumulative_y"]}),
        "displacement.cumulative",
    );
    assert_eq!(
        resp.pairs.len(),
        expected["pairs"].as_array().unwrap().len()
    );
    let (p, ep) = (&resp.pairs[0], &expected["pairs"][0]);
    // `from_image_id`/`to_image_id` are this run's own ids, not comparable to the
    // golden's `$FRAME_A_ID`/`$FRAME_B_ID` placeholders — only the numeric fields agree.
    assert_agrees(
        &json!({"dx": p.dx, "dy": p.dy, "score": p.score}),
        &json!({"dx": ep["dx"], "dy": ep["dy"], "score": ep["score"]}),
        "displacement.pairs[0]",
    );
}
