//! The CaliperBench runner (`examples/caliperbench_run.rs`) against CaliperBench itself.
//!
//! `tests/fixtures/caliperbench_golden.json` holds small 8-bit images with CaliperBench
//! requests, and what CaliperBench's own baselines return for them: the prediction and the
//! lab trace. The cases cover a bar at subpixel phases .3 and .7, the same strip reversed,
//! 0.5 px spacing, an oblique strip 3 px wide, the one-line shift of a wide strip, the three
//! negatives (flat, ramp, an edge), a missing peak, an out-of-bounds strip and each outcome
//! of the midpoint method. None hangs on a comparison closer than 1e-6, so `f32` and
//! `float64` cannot decide one differently. Regenerate it with
//! `tools/gen_caliperbench_golden.py`.
//!
//! The runner must give the same status and reason, the same edges within 1e-4 px, and the
//! same intermediates: values within 1e-5, positions within 1e-4 px.

#[path = "../examples/common/caliperbench.rs"]
mod caliperbench;

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use caliperbench::{
    Method, Operator, Params, Request, RunOptions, Runner, gray_to_unit, pillow_luma, run,
};

const EDGE_TOL_PX: f64 = 1e-4;
const VALUE_TOL: f64 = 1e-5;

fn golden() -> Value {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/caliperbench_golden.json");
    serde_json::from_str(&fs::read_to_string(path).expect("golden fixture")).expect("valid JSON")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

fn number(v: &Value) -> f64 {
    v.as_f64().expect("a number")
}

fn assert_close(name: &str, what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: {what} has {} entries, CaliperBench {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{name}: {what}[{i}] = {g}, CaliperBench {w}"
        );
    }
}

/// Run one golden case and compare prediction and trace with CaliperBench's.
fn check_case(case: &Value) {
    let name = case["name"].as_str().expect("name");
    let (w, h) = (
        number(&case["width"]) as usize,
        number(&case["height"]) as usize,
    );
    let pixels: Vec<u8> = case["pixels"]
        .as_array()
        .expect("pixels")
        .iter()
        .map(|v| number(v) as u8)
        .collect();
    let img = gray_to_unit(w, h, &pixels);
    let method = Method::parse(case["method"].as_str().expect("method")).expect("a method");
    let params = Params::from_json(&case["params"].to_string()).expect("params");
    let request: Request = serde_json::from_value(case["request"].clone()).expect("request");

    let mut runner = Runner::new(method, Operator::Central, params);
    let (prediction, trace) = runner.predict(&img.as_view(), &request, true);
    let trace = trace.expect("a trace");

    // The prediction: status and reason exactly, edges to 1e-4 px.
    let expected = &case["expected"];
    match &prediction.outcome {
        Ok(edges) => {
            assert_eq!(
                expected["status"], "ok",
                "{name}: ok, CaliperBench {expected}"
            );
            let want = floats(&expected["edges_px"]);
            assert_close(name, "edges_px", edges, &want, EDGE_TOL_PX);
        }
        Err(reason) => {
            assert_eq!(expected["status"], "failed", "{name}: failed with {reason}");
            assert_eq!(expected["reason"], *reason, "{name}");
        }
    }

    // The trace: the same keys, the same intermediates.
    let want = case["trace"].as_object().expect("trace object");
    let mut keys: Vec<&str> = trace
        .keys()
        .map(String::as_str)
        .filter(|k| *k != "sample_id")
        .collect();
    let mut want_keys: Vec<&str> = want.keys().map(String::as_str).collect();
    keys.sort_unstable();
    want_keys.sort_unstable();
    assert_eq!(keys, want_keys, "{name}: trace keys");
    for key in ["profile", "smooth", "gradient", "levels"] {
        if let Some(v) = want.get(key) {
            assert_close(name, key, &floats(&trace[key]), &floats(v), VALUE_TOL);
        }
    }
    for key in ["step", "threshold"] {
        if let Some(v) = want.get(key) {
            let (g, w) = (number(&trace[key]), number(v));
            assert!(
                (g - w).abs() <= VALUE_TOL,
                "{name}: {key} = {g}, CaliperBench {w}"
            );
        }
    }
    if let Some(v) = want.get("candidates") {
        let (got, want) = (
            trace["candidates"].as_array().unwrap(),
            v.as_array().unwrap(),
        );
        assert_eq!(
            got.len(),
            want.len(),
            "{name}: candidates {got:?}, CaliperBench {want:?}"
        );
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g[2], w[2], "{name}: candidate polarity");
            let close = (number(&g[0]) - number(&w[0])).abs() <= VALUE_TOL
                && (number(&g[1]) - number(&w[1])).abs() <= EDGE_TOL_PX;
            assert!(close, "{name}: candidate {g}, CaliperBench {w}");
        }
    }
}

#[test]
fn the_runner_reproduces_caliperbench_on_the_golden_cases() {
    let golden = golden();
    let cases = golden["cases"].as_array().expect("cases");
    assert!(cases.len() >= 12, "{} cases", cases.len());
    for case in cases {
        check_case(case);
    }
}

#[test]
fn params_are_checked_like_caliperbench() {
    let p = Params::from_json(r#"{"sigma": 1.5, "radius": 4, "width_px": null}"#).expect("valid");
    assert_eq!((p.sigma, p.radius, p.width_px), (1.5, 4, None));
    assert_eq!(Params::from_json("{}").expect("empty"), Params::default());
    for bad in [
        r#"{"sigma": 0}"#,
        r#"{"radius": 0}"#,
        r#"{"radius": 2.5}"#,
        r#"{"endpoint_samples": 0}"#,
        r#"{"threshold": 0.1}"#,
        r#"{"samples": 3}"#,
        "[1, 2]",
        "not json",
    ] {
        assert!(Params::from_json(bad).is_err(), "{bad} should be refused");
    }
}

/// Values Pillow 12's `Image.convert("L")` gives for these RGB pixels.
#[test]
fn rgb_is_converted_with_pillows_luma() {
    let pixels = [
        ((200, 0, 0), 60),
        ((0, 200, 0), 117),
        ((0, 0, 200), 23),
        ((12, 240, 77), 153),
        ((255, 255, 255), 255),
        ((97, 13, 201), 60),
    ];
    for ((r, g, b), want) in pixels {
        assert_eq!(pillow_luma(r, g, b), want, "({r}, {g}, {b})");
    }
}

/// A scratch directory under the system temp dir, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let name = format!("vm-caliperbench-{tag}-{}-{nanos}", std::process::id());
        let dir = std::env::temp_dir().join(name);
        fs::create_dir_all(&dir).expect("scratch dir");
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A bright bar on columns 20..44 of a 16 × 64 image, as gray levels.
fn bar_levels() -> Vec<u8> {
    (0..16 * 64)
        .map(|i| {
            if (20..44).contains(&(i % 64)) {
                180
            } else {
                40
            }
        })
        .collect()
}

/// A request for a horizontal strip 3 px wide through the middle of a 16 × 64 image.
fn request_line(id: &str, image: &str, polarities: &str) -> String {
    let strip =
        r#"{"start_xy":[4.0,8.0],"end_xy":[60.0,8.0],"width_px":3.0,"samples":57,"across":3}"#;
    format!(
        r#"{{"schema_version":1,"sample_id":"{id}","image":"{image}","image_sha256":"{}","strip":{strip},"polarities":{polarities}}}"#,
        "a".repeat(64)
    )
}

fn read_rows(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .expect("output written")
        .lines()
        .map(|l| serde_json::from_str(l).expect("a JSON row"))
        .collect()
}

/// Write the test images: the bar as `L`; as RGB in pure green, and as the `L` image of
/// that RGB's Pillow luma; and as gray with alpha, a mode CaliperBench refuses.
fn write_images(root: &Path) {
    fs::create_dir_all(root.join("img")).unwrap();
    let levels = bar_levels();
    let gray = image::GrayImage::from_raw(64, 16, levels.clone()).unwrap();
    gray.save(root.join("img/gray.png")).unwrap();
    let green: Vec<u8> = levels
        .iter()
        .flat_map(|&v| {
            let g = (0..=255u8)
                .find(|&g| pillow_luma(0, g, 0) >= v)
                .unwrap_or(255);
            [0, g, 0]
        })
        .collect();
    let luma: Vec<u8> = green
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[r, g, b]| pillow_luma(r, g, b))
        .collect();
    let luma = image::GrayImage::from_raw(64, 16, luma).unwrap();
    luma.save(root.join("img/luma.png")).unwrap();
    let rgb = image::RgbImage::from_raw(64, 16, green).unwrap();
    rgb.save(root.join("img/rgb.png")).unwrap();
    let la: Vec<u8> = levels.iter().flat_map(|&v| [v, 255]).collect();
    let la = image::GrayAlphaImage::from_raw(64, 16, la).unwrap();
    la.save(root.join("img/la.png")).unwrap();
}

/// The whole runner, end to end: PNGs and a requests file in, prediction rows out.
#[test]
fn run_writes_one_protocol_row_per_request() {
    let dir = Scratch::new("run");
    let root = dir.0.join("data");
    write_images(&root);

    let requests = dir.0.join("requests.jsonl");
    let lines = [
        request_line("bar", "img/gray.png", r#"["rising","falling"]"#),
        request_line("rgb", "img/rgb.png", r#"["rising","falling"]"#),
        request_line("luma", "img/luma.png", r#"["rising","falling"]"#),
        request_line("negative", "img/gray.png", "[]"),
        request_line("missing_peak", "img/gray.png", r#"["falling","rising"]"#),
        request_line("la", "img/la.png", r#"["rising"]"#),
        request_line("absent", "img/absent.png", r#"["rising"]"#),
    ];
    fs::write(&requests, lines.join("\n") + "\n").unwrap();

    let opts = RunOptions {
        method: Method::GradientParabolic,
        operator: Operator::Central,
        params: Params::default(),
        requests,
        data_root: root,
        output: dir.0.join("out/predictions.jsonl"),
        trace: Some(dir.0.join("out/trace.jsonl")),
    };
    let summary = run(&opts).expect("the run completes");
    assert_eq!((summary.ok, summary.failed), (4, 3));

    let rows = read_rows(&opts.output);
    let ids: Vec<&str> = rows
        .iter()
        .map(|r| r["sample_id"].as_str().unwrap())
        .collect();
    let want_ids = [
        "bar",
        "rgb",
        "luma",
        "negative",
        "missing_peak",
        "la",
        "absent",
    ];
    assert_eq!(ids, want_ids, "rows come in request order");
    for row in &rows {
        let mut keys: Vec<&str> = row
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        let mut want = vec![
            "edges_px",
            "runtime_ms",
            "sample_id",
            "schema_version",
            "status",
        ];
        if row["status"] == "failed" {
            want.push("reason");
            want.sort_unstable();
            assert_eq!(row["edges_px"], serde_json::json!([]), "{row}");
        }
        assert_eq!(keys, want, "{row}");
        assert_eq!(row["schema_version"], 1);
        assert!(number(&row["runtime_ms"]) >= 0.0);
    }
    // The bar's edges sit at x = 19.5 and 43.5: 15.5 and 39.5 from the strip's start.
    let bar = floats(&rows[0]["edges_px"]);
    assert_close("bar", "edges_px", &bar, &[15.5, 39.5], 0.05);
    // RGB goes through Pillow's luma, so it measures exactly as its luma image does.
    assert_eq!(rows[1]["edges_px"], rows[2]["edges_px"]);
    // A negative task reports its strongest edge as a false detection.
    assert_eq!(rows[3]["status"], "ok");
    assert_eq!(floats(&rows[3]["edges_px"]).len(), 1);
    let reasons: Vec<&str> = rows[4..]
        .iter()
        .map(|r| r["reason"].as_str().unwrap())
        .collect();
    assert_eq!(
        reasons,
        ["missing_peak", "unsupported_image_mode", "image_unreadable"]
    );

    let traces = read_rows(opts.trace.as_ref().unwrap());
    assert_eq!(traces.len(), rows.len());
    assert_eq!(traces[0]["profile"].as_array().unwrap().len(), 57);
    assert!(
        traces[6].get("profile").is_none(),
        "an image that never loaded has no trace"
    );
}

#[test]
fn an_unreadable_requests_file_fails_the_run() {
    let dir = Scratch::new("bad");
    let base = RunOptions {
        method: Method::MidpointCrossing,
        operator: Operator::Central,
        params: Params::default(),
        requests: dir.0.join("missing.jsonl"),
        data_root: dir.0.clone(),
        output: dir.0.join("out.jsonl"),
        trace: None,
    };
    assert!(run(&base).is_err(), "a missing requests file");

    let line = |image| request_line("a", image, "[]");
    let bad = [
        line("x.png").replace("\"polarities\"", "\"extra\":1,\"polarities\""),
        [line("x.png"), line("y.png")].join("\n"),
        line("/etc/x.png"),
        line("../x.png"),
        line("x.png").replace("\"across\":3", "\"across\":1"),
        "{".to_string(),
    ];
    for (i, text) in bad.iter().enumerate() {
        let requests = dir.0.join(format!("bad{i}.jsonl"));
        fs::write(&requests, text).unwrap();
        let opts = RunOptions {
            requests,
            ..base.clone()
        };
        assert!(run(&opts).is_err(), "case {i} should be refused: {text}");
        assert!(
            !opts.output.exists(),
            "nothing is written for a refused file"
        );
    }
}
