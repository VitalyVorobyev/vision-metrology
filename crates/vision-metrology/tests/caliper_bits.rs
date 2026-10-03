//! Bit-identity pin for the caliper.
//!
//! Every case measures a deterministic synthetic image and hashes the exact `f32` bits of
//! every reported edge (position, `t`, amplitude, polarity) or the rejection reason. A
//! refactor that is meant to change nothing must keep every hash.
//!
//! To print the current hashes (for a change that is *meant* to move results), run
//! `CALIPER_PIN_PRINT=1 cargo test -p vision-metrology --test caliper_bits -- --nocapture`.

use vision_metrology::measure::{
    Caliper, EdgeSelect, MeasureArc, MeasureConfig, MeasureEdge, MeasureRadial, MeasureRect,
    MetrologyFit, MetrologyModel, MetrologyObject, MetrologyShape, PolaritySelect, RejectReason,
};
use vision_metrology::{BorderMode, EdgePolarity, Image, Point2f, Similarity2f};

/// FNV-1a over 32-bit words.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn word(&mut self, w: u32) {
        for b in w.to_le_bytes() {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn f(&mut self, v: f32) {
        self.word(v.to_bits());
    }
}

fn hash_edges(h: &mut Fnv, edges: &[MeasureEdge]) {
    h.word(edges.len() as u32);
    for e in edges {
        h.f(e.p.x);
        h.f(e.p.y);
        h.f(e.t);
        h.f(e.amplitude);
        h.word(match e.polarity {
            EdgePolarity::Rising => 1,
            EdgePolarity::Falling => 2,
        });
    }
}

fn hash_result(r: Result<&[MeasureEdge], RejectReason>) -> u64 {
    let mut h = Fnv::new();
    match r {
        Ok(edges) => hash_edges(&mut h, edges),
        Err(reason) => h.word(0xdead_0000 + reason as u32),
    }
    h.0
}

/// Deterministic xorshift noise in `[-amp, amp]`.
struct Noise(u64);

impl Noise {
    fn next(&mut self, amp: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (((self.0 >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
    }
}

/// Anti-aliased bright bar between `x0` and `x1` (fractional), rotated by `angle`
/// about the image centre, with seeded noise.
fn bar(w: usize, h: usize, x0: f32, x1: f32, angle: f32, noise: f32, seed: u64) -> Image<u8> {
    let (s, c) = angle.sin_cos();
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let mut rng = Noise(seed);
    let data = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 - cx, (i / w) as f32 - cy);
            let u = c * x + s * y + cx;
            let cover = ((u - x0 + 0.5).clamp(0.0, 1.0)) * ((x1 - u + 0.5).clamp(0.0, 1.0));
            (30.0 + 170.0 * cover + rng.next(noise))
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect();
    Image::from_vec(w, h, data).expect("valid image")
}

fn disc(w: usize, h: usize, c: Point2f, r: f32) -> Image<u8> {
    let data = (0..w * h)
        .map(|i| {
            let p = Point2f::new((i % w) as f32, (i / w) as f32);
            let cover = (r + 0.5 - (p - c).norm()).clamp(0.0, 1.0);
            (20.0 + 180.0 * cover).round() as u8
        })
        .collect();
    Image::from_vec(w, h, data).expect("valid image")
}

fn wedge(w: usize, h: usize, c: Point2f) -> Image<u8> {
    let data = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let phi = (y - c.y).atan2(x - c.x);
            if (0.0..30.0f32.to_radians()).contains(&phi) {
                20u8
            } else {
                200
            }
        })
        .collect();
    Image::from_vec(w, h, data).expect("valid image")
}

fn cases() -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let bars: Vec<(&str, Image<u8>)> = vec![
        ("bar_axis", bar(128, 96, 40.3, 70.8, 0.0, 0.0, 1)),
        ("bar_rot17_noise", bar(128, 96, 41.7, 66.2, 0.3, 4.0, 7)),
        ("bar_rot40_noise", bar(128, 128, 38.1, 80.4, 0.7, 2.0, 11)),
    ];
    let selects = [
        ("all", EdgeSelect::All),
        ("first", EdgeSelect::First),
        ("last", EdgeSelect::Last),
        ("strongest", EdgeSelect::Strongest),
    ];
    let polarities = [
        ("any", PolaritySelect::Any),
        ("rising", PolaritySelect::Rising),
        ("falling", PolaritySelect::Falling),
    ];
    for (name, img) in &bars {
        for angle in [0.0f32, 0.12, 0.5] {
            for (half_len, half_width) in [(40.0f32, 0.0f32), (40.0, 5.0), (33.3, 7.5)] {
                for step in [1.0f32, 0.5, 0.37] {
                    for sigma in [1.0f32, 1.7] {
                        for (sname, select) in selects {
                            for (pname, polarity) in polarities {
                                let cfg = MeasureConfig {
                                    sigma,
                                    step,
                                    select,
                                    polarity,
                                    threshold: 4.0,
                                    ..MeasureConfig::default()
                                };
                                let rect = MeasureRect {
                                    center: Point2f::new(63.4, 47.6),
                                    angle,
                                    half_len,
                                    half_width,
                                };
                                let mut cal = Caliper::rect(rect, cfg);
                                out.push((
                                    format!(
                                        "rect/{name}/a{angle}/l{half_len}/w{half_width}/s{step}/\
                                         g{sigma}/{sname}/{pname}"
                                    ),
                                    hash_result(cal.measure(&img.as_view())),
                                ));
                            }
                        }
                        // Pairs ignore select and polarity.
                        let mut cal = Caliper::rect(
                            MeasureRect {
                                center: Point2f::new(63.4, 47.6),
                                angle,
                                half_len,
                                half_width,
                            },
                            MeasureConfig {
                                sigma,
                                step,
                                threshold: 4.0,
                                ..MeasureConfig::default()
                            },
                        );
                        let pairs = cal.measure_pairs(&img.as_view());
                        let mut h = Fnv::new();
                        h.word(pairs.len() as u32);
                        for p in pairs {
                            hash_edges(&mut h, &[p.first, p.second]);
                            h.f(p.center.x);
                            h.f(p.center.y);
                            h.f(p.width);
                        }
                        out.push((
                            format!(
                                "pairs/{name}/a{angle}/l{half_len}/w{half_width}/s{step}/g{sigma}"
                            ),
                            h.0,
                        ));
                    }
                }
            }
        }
    }

    // Pixel types, border modes, the obliquity gate, an overhanging caliper.
    let img8 = bar(96, 96, 30.4, 61.9, 0.2, 3.0, 5);
    let img16 = Image::from_vec(
        96,
        96,
        img8.data().iter().map(|&v| u16::from(v) * 257).collect(),
    )
    .expect("valid");
    let imgf = Image::from_vec(
        96,
        96,
        img8.data().iter().map(|&v| f32::from(v) / 255.0).collect(),
    )
    .expect("valid");
    let geom = MeasureRect {
        center: Point2f::new(48.0, 48.0),
        angle: 0.2,
        half_len: 30.0,
        half_width: 4.0,
    };
    let base = MeasureConfig {
        threshold: 3.0,
        ..MeasureConfig::default()
    };
    out.push((
        "types/u16".into(),
        hash_result(
            Caliper::rect(
                geom,
                MeasureConfig {
                    threshold: 3.0 * 257.0,
                    ..base
                },
            )
            .measure(&img16.as_view()),
        ),
    ));
    out.push((
        "types/f32".into(),
        hash_result(
            Caliper::rect(
                geom,
                MeasureConfig {
                    threshold: 3.0 / 255.0,
                    ..base
                },
            )
            .measure(&imgf.as_view()),
        ),
    ));
    for (bname, border) in [
        ("clamp", BorderMode::Clamp),
        ("reflect", BorderMode::Reflect101),
        ("const", BorderMode::Constant(90.0)),
    ] {
        let over = MeasureRect {
            center: Point2f::new(8.0, 40.0),
            angle: 0.0,
            half_len: 30.0,
            half_width: 3.0,
        };
        out.push((
            format!("border/{bname}"),
            hash_result(
                Caliper::rect(over, MeasureConfig { border, ..base }).measure(&img8.as_view()),
            ),
        ));
    }
    for gate in [180.0f32, 60.0, 20.0] {
        let oblique = MeasureRect { angle: 1.1, ..geom };
        out.push((
            format!("obliquity/{gate}"),
            hash_result(
                Caliper::rect(
                    oblique,
                    MeasureConfig {
                        max_obliquity_deg: gate,
                        ..base
                    },
                )
                .measure(&img8.as_view()),
            ),
        ));
    }
    let flat = Image::from_vec(64, 64, vec![128u8; 64 * 64]).expect("valid");
    out.push((
        "reject/no_edge".into(),
        hash_result(Caliper::rect(geom, base).measure(&flat.as_view())),
    ));
    out.push((
        "reject/short".into(),
        hash_result(
            Caliper::rect(
                MeasureRect {
                    half_len: 0.4,
                    ..geom
                },
                base,
            )
            .measure(&img8.as_view()),
        ),
    ));

    // Arc and radial placements.
    let c = Point2f::new(64.0, 64.0);
    let wedge_img = wedge(128, 128, c);
    for (extent, hw, step) in [
        (70.0f32, 4.0f32, 1.0f32),
        (-70.0, 2.5, 0.6),
        (90.0, 0.0, 1.3),
    ] {
        let arc = MeasureArc {
            center: c,
            radius: 40.0,
            angle_start: if extent > 0.0 { -20.0f32 } else { 50.0 }.to_radians(),
            angle_extent: extent.to_radians(),
            half_width: hw,
        };
        let cfg = MeasureConfig { step, ..base };
        out.push((
            format!("arc/{extent}/{hw}/{step}"),
            hash_result(Caliper::arc(arc, cfg).measure(&wedge_img.as_view())),
        ));
    }
    let disc_img = disc(160, 160, Point2f::new(80.3, 79.6), 40.2);
    for angle in [0.0f32, 0.9, 2.4, -1.3] {
        for (hw, step) in [(0.0f32, 1.0f32), (5.0, 1.0), (10.0, 0.45)] {
            let radial = MeasureRadial {
                center: Point2f::new(80.0, 80.0),
                radius: 40.0,
                angle,
                half_len: 8.0,
                half_width: hw,
            };
            let cfg = MeasureConfig { step, ..base };
            out.push((
                format!("radial/{angle}/{hw}/{step}"),
                hash_result(Caliper::radial(radial, cfg).measure(&disc_img.as_view())),
            ));
        }
    }

    // The metrology model end to end: fit bits.
    let mut model = MetrologyModel::new();
    model.add(MetrologyObject::new(MetrologyShape::Circle {
        center: Point2f::new(80.0, 80.0),
        radius: 40.0,
        arc: None,
    }));
    model.add(MetrologyObject::new(MetrologyShape::Line {
        a: Point2f::new(50.0, 20.0),
        b: Point2f::new(110.0, 22.0),
    }));
    let fixture = Similarity2f::identity();
    let results = model.apply(&disc_img.as_view(), &fixture);
    let mut h = Fnv::new();
    for r in &results {
        match r {
            Ok(res) => {
                hash_edges(&mut h, &res.hits);
                match &res.fit {
                    MetrologyFit::Circle(f) => {
                        h.f(f.model.center.x);
                        h.f(f.model.center.y);
                        h.f(f.model.radius);
                        h.f(f.rms);
                        h.f(f.max_dev);
                    }
                    MetrologyFit::Line(f) => {
                        h.f(f.rms);
                        h.f(f.max_dev);
                    }
                }
            }
            Err(_) => h.word(0xbad),
        }
    }
    out.push(("model/apply".into(), h.0));
    out
}

/// Combined hash of every case, plus a per-case table printed on demand.
#[test]
fn caliper_output_is_bit_identical_to_the_pinned_implementation() {
    let cases = cases();
    let mut all = Fnv::new();
    for (_, h) in &cases {
        all.word(*h as u32);
        all.word((*h >> 32) as u32);
    }
    if std::env::var_os("CALIPER_PIN_PRINT").is_some() {
        for (name, h) in &cases {
            println!("{h:016x} {name}");
        }
        println!("cases={} combined={:016x}", cases.len(), all.0);
        return;
    }
    assert_eq!(cases.len(), EXPECTED_CASES, "the case list itself changed");
    assert_eq!(
        all.0, EXPECTED_COMBINED,
        "caliper output changed; rerun with CALIPER_PIN_PRINT=1 to see which cases"
    );
}

const EXPECTED_CASES: usize = 2132;
const EXPECTED_COMBINED: u64 = 0x5e9a_2663_38b6_7136;
