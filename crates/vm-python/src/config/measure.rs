//! Python-visible [`MeasureConfig`](vision_metrology::measure::MeasureConfig) mirror.

use std::num::NonZeroUsize;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vision_metrology::measure::{
    Derivative as NativeDerivative, EdgeSelect as NativeEdgeSelect,
    EdgeSequence as NativeEdgeSequence, Locate as NativeLocate,
    MeasureConfig as NativeMeasureConfig, OffImage as NativeOffImage,
    PolaritySelect as NativePolaritySelect, ProfileConfig as NativeProfileConfig,
};
use vm_primitives::{BorderMode, SubpixRefine};

/// How a caliper locates an edge on its profile — construct with
/// `Locate.gradient_peak(refine=...)`, `Locate.midpoint_crossing(...)` or
/// `Locate.half_contrast(...)`.
#[pyclass(frozen, get_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct Locate {
    /// "gradient_peak", "midpoint_crossing" or "half_contrast".
    pub kind: String,
    /// Subpixel refinement of a gradient peak: "none", "parabolic", "gaussian" or
    /// "centroid".
    pub refine: String,
    /// Half-width of the centroid window, in samples (only for `refine="centroid"`).
    pub centroid_radius: usize,
    /// `midpoint_crossing`: samples at each end whose median is that end's level.
    pub endpoint_samples: usize,
    /// `midpoint_crossing` and `half_contrast`: minimum difference between the two
    /// levels, on the input pixel scale.
    pub min_contrast: f32,
    /// `half_contrast`: inner and outer edges of each flank window, in pixels.
    pub flank_px: (f32, f32),
    /// `half_contrast`: convergence tolerance, in pixels.
    pub tol_px: f32,
    /// `half_contrast`: the most iterations per edge.
    pub max_iter: usize,
}

#[pymethods]
impl Locate {
    /// A local extremum of the derivative, refined to subpixel position.
    #[staticmethod]
    #[pyo3(signature = (refine="parabolic", centroid_radius=2))]
    pub fn gradient_peak(refine: &str, centroid_radius: usize) -> PyResult<Self> {
        check_name("refine", refine, REFINES)?;
        Ok(Self {
            refine: refine.into(),
            centroid_radius,
            ..Self::default()
        })
    }

    /// One edge where the smoothed profile crosses the mean of its two end levels (the
    /// medians of the first and last `endpoint_samples` samples), nearest the middle.
    #[staticmethod]
    #[pyo3(signature = (endpoint_samples=3, min_contrast=0.05))]
    pub fn midpoint_crossing(endpoint_samples: usize, min_contrast: f32) -> PyResult<Self> {
        positive("endpoint_samples", endpoint_samples)?;
        Ok(Self {
            kind: "midpoint_crossing".into(),
            endpoint_samples,
            min_contrast,
            ..Self::default()
        })
    }

    /// Gradient peaks, each moved to the crossing of its local half-contrast level: the
    /// mean of the medians `flank_px[0]` to `flank_px[1]` pixels either side of it.
    #[staticmethod]
    #[pyo3(signature = (flank_px=(3.0, 8.0), tol_px=0.01, max_iter=5, min_contrast=0.0))]
    pub fn half_contrast(
        flank_px: (f32, f32),
        tol_px: f32,
        max_iter: usize,
        min_contrast: f32,
    ) -> PyResult<Self> {
        positive("max_iter", max_iter)?;
        if !(flank_px.0 >= 0.0 && flank_px.0 <= flank_px.1) {
            return Err(PyValueError::new_err(format!(
                "flank_px must be (near, far) with 0 <= near <= far, got {flank_px:?}"
            )));
        }
        Ok(Self {
            kind: "half_contrast".into(),
            flank_px,
            tol_px,
            max_iter,
            min_contrast,
            ..Self::default()
        })
    }

    fn __repr__(&self) -> String {
        match self.kind.as_str() {
            "midpoint_crossing" => format!(
                "Locate.midpoint_crossing(endpoint_samples={}, min_contrast={})",
                self.endpoint_samples, self.min_contrast
            ),
            "half_contrast" => format!(
                "Locate.half_contrast(flank_px={:?}, tol_px={}, max_iter={}, min_contrast={})",
                self.flank_px, self.tol_px, self.max_iter, self.min_contrast
            ),
            _ => format!("Locate.gradient_peak(refine='{}')", self.refine),
        }
    }
}

/// `value` as a `NonZeroUsize`, or a `ValueError` naming `name`.
fn positive(name: &str, value: usize) -> PyResult<NonZeroUsize> {
    NonZeroUsize::new(value)
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be at least 1")))
}

// The names each enumerated string field accepts.
const REFINES: &[&str] = &["none", "parabolic", "gaussian", "centroid"];
const POLARITIES: &[&str] = &["any", "rising", "falling"];
const SELECTS: &[&str] = &["all", "first", "last", "strongest", "in_order"];
pub(super) const BORDER_MODES: &[&str] = &["clamp", "reflect101", "constant"];
pub(super) const DERIVATIVES: &[&str] = &["dog", "smooth_central"];
pub(super) const OFF_IMAGES: &[&str] = &["fill", "reject"];
/// Polarity names a `sequence` entry accepts; "either" and "any" both mean either.
const SEQUENCE_POLARITIES: &[&str] = &["rising", "falling", "either", "any"];

/// The `ValueError` for a `name` whose `value` is not one of `allowed`.
pub(super) fn not_one_of(name: &str, value: &str, allowed: &[&str]) -> PyErr {
    PyValueError::new_err(format!("{name} must be one of {allowed:?}, got '{value}'"))
}

/// `Ok` when `value` is one of `allowed`; otherwise the `ValueError` naming them.
pub(super) fn check_name(name: &str, value: &str, allowed: &[&str]) -> PyResult<()> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(not_one_of(name, value, allowed))
    }
}

impl Locate {
    pub fn to_native(&self) -> PyResult<NativeLocate> {
        Ok(match self.kind.as_str() {
            "gradient_peak" => NativeLocate::GradientPeak {
                refine: match self.refine.as_str() {
                    "none" => SubpixRefine::None,
                    "parabolic" => SubpixRefine::Parabolic3,
                    "gaussian" => SubpixRefine::Gaussian3,
                    "centroid" => SubpixRefine::Centroid {
                        radius: self.centroid_radius,
                    },
                    other => return Err(not_one_of("refine", other, REFINES)),
                },
            },
            "midpoint_crossing" => NativeLocate::MidpointCrossing {
                endpoint_samples: positive("endpoint_samples", self.endpoint_samples)?,
                min_contrast: self.min_contrast,
            },
            "half_contrast" => NativeLocate::HalfContrast {
                flank_near_px: self.flank_px.0,
                flank_far_px: self.flank_px.1,
                tol_px: self.tol_px,
                max_iter: positive("max_iter", self.max_iter)?,
                min_contrast: self.min_contrast,
            },
            other => {
                return Err(not_one_of(
                    "kind",
                    other,
                    &["gradient_peak", "midpoint_crossing", "half_contrast"],
                ));
            }
        })
    }
}

impl Locate {
    /// The Python mirror of a native locate mode.
    pub fn from_native(locate: NativeLocate) -> Self {
        let d = Self::default();
        match locate {
            NativeLocate::GradientPeak { refine } => {
                let (refine, centroid_radius) = match refine {
                    SubpixRefine::None => ("none", d.centroid_radius),
                    SubpixRefine::Parabolic3 => ("parabolic", d.centroid_radius),
                    SubpixRefine::Gaussian3 => ("gaussian", d.centroid_radius),
                    SubpixRefine::Centroid { radius } => ("centroid", radius),
                };
                Self {
                    refine: refine.into(),
                    centroid_radius,
                    ..d
                }
            }
            NativeLocate::MidpointCrossing {
                endpoint_samples,
                min_contrast,
            } => Self {
                kind: "midpoint_crossing".into(),
                endpoint_samples: endpoint_samples.get(),
                min_contrast,
                ..d
            },
            NativeLocate::HalfContrast {
                flank_near_px,
                flank_far_px,
                tol_px,
                max_iter,
                min_contrast,
            } => Self {
                kind: "half_contrast".into(),
                flank_px: (flank_near_px, flank_far_px),
                tol_px,
                max_iter: max_iter.get(),
                min_contrast,
                ..d
            },
        }
    }
}

impl Default for Locate {
    fn default() -> Self {
        Self {
            kind: "gradient_peak".into(),
            refine: "parabolic".into(),
            centroid_radius: 2,
            endpoint_samples: 3,
            min_contrast: 0.05,
            flank_px: (3.0, 8.0),
            tol_px: 0.01,
            max_iter: 5,
        }
    }
}

/// The native polarity for a name of `SEQUENCE_POLARITIES`, which includes every name of
/// `POLARITIES`.
fn polarity_from(name: &str) -> PyResult<NativePolaritySelect> {
    match name {
        "rising" => Ok(NativePolaritySelect::Rising),
        "falling" => Ok(NativePolaritySelect::Falling),
        "any" | "either" => Ok(NativePolaritySelect::Any),
        other => Err(not_one_of("polarity", other, SEQUENCE_POLARITIES)),
    }
}

/// Whether `sequence` holds at most two names of `SEQUENCE_POLARITIES`.
fn sequence_names_known(sequence: &[String]) -> bool {
    sequence.len() <= 2
        && sequence
            .iter()
            .all(|p| SEQUENCE_POLARITIES.contains(&p.as_str()))
}

/// The native sequence for `select="in_order"`: one or two known polarity names.
fn sequence_to_native(sequence: &[String]) -> PyResult<NativeEdgeSequence> {
    match sequence {
        [first, rest @ ..] if sequence_names_known(sequence) => Ok(NativeEdgeSequence {
            first: polarity_from(first)?,
            second: rest.first().map(|p| polarity_from(p)).transpose()?,
        }),
        _ => Err(PyValueError::new_err(format!(
            "select='in_order' needs a sequence of one or two of {SEQUENCE_POLARITIES:?}, \
             got {sequence:?}"
        ))),
    }
}

/// The flat Python fields of a native `ProfileConfig`, shared by `MeasureConfig` and
/// `BeadCaliper`.
pub(super) struct Profile<'a> {
    pub sigma: f32,
    pub step: f32,
    pub border_mode: &'a str,
    pub border_constant: f32,
    pub derivative: &'a str,
    pub kernel_radius_px: f32,
    pub off_image: &'a str,
}

impl Profile<'_> {
    /// The native profile; fails on a string field holding a name it does not accept.
    pub(super) fn to_native(&self) -> PyResult<NativeProfileConfig> {
        Ok(NativeProfileConfig {
            sigma: self.sigma,
            derivative: match self.derivative {
                "dog" => NativeDerivative::DerivativeOfGaussian,
                "smooth_central" => NativeDerivative::SmoothThenCentral {
                    radius_px: self.kernel_radius_px,
                },
                other => return Err(not_one_of("derivative", other, DERIVATIVES)),
            },
            step: self.step,
            border: match self.border_mode {
                "clamp" => BorderMode::Clamp,
                "reflect101" => BorderMode::Reflect101,
                "constant" => BorderMode::Constant(self.border_constant),
                other => return Err(not_one_of("border_mode", other, BORDER_MODES)),
            },
            off_image: match self.off_image {
                "fill" => NativeOffImage::Fill,
                "reject" => NativeOffImage::Reject,
                other => return Err(not_one_of("off_image", other, OFF_IMAGES)),
            },
        })
    }
}

/// The names a native profile's enumerated fields take in Python: `(border_mode,
/// border_constant, derivative, kernel_radius_px, off_image)`.
pub(super) fn profile_names(
    p: &NativeProfileConfig,
) -> (&'static str, f32, &'static str, f32, &'static str) {
    let (border_mode, border_constant) = match p.border {
        BorderMode::Clamp => ("clamp", 0.0),
        BorderMode::Reflect101 => ("reflect101", 0.0),
        BorderMode::Constant(c) => ("constant", c),
    };
    let (derivative, kernel_radius_px) = match p.derivative {
        NativeDerivative::DerivativeOfGaussian => ("dog", 3.0),
        NativeDerivative::SmoothThenCentral { radius_px } => ("smooth_central", radius_px),
    };
    let off_image = match p.off_image {
        NativeOffImage::Fill => "fill",
        NativeOffImage::Reject => "reject",
    };
    (
        border_mode,
        border_constant,
        derivative,
        kernel_radius_px,
        off_image,
    )
}

/// Mirrors `vision_metrology::measure::MeasureConfig`.
///
/// The string fields accept only their listed names: assigning anything else raises
/// `ValueError`, as the constructor does.
#[pyclass(get_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct MeasureConfig {
    /// Gaussian sigma of the profile smoothing, in pixels.
    #[pyo3(set)]
    pub sigma: f32,
    /// Minimum `|derivative response|` for an edge to be reported (unused by
    /// `Locate.midpoint_crossing`).
    #[pyo3(set)]
    pub threshold: f32,
    /// "any", "rising" or "falling".
    pub polarity: String,
    /// "all", "first", "last", "strongest" or "in_order".
    pub select: String,
    /// For `select="in_order"`: one or two polarities ("rising", "falling", "either"),
    /// found in scan order, each the strongest of its polarity after the previous one.
    pub sequence: Vec<String>,
    /// Profile sampling step along the scan axis, in pixels.
    #[pyo3(set)]
    pub step: f32,
    /// Maximum angle, in degrees, between scan direction and image gradient.
    /// `180.0` disables the obliquity gate.
    #[pyo3(set)]
    pub max_obliquity_deg: f32,
    /// "clamp", "reflect101" or "constant".
    pub border_mode: String,
    #[pyo3(set)]
    pub border_constant: f32,
    /// "dog" (derivative of Gaussian) or "smooth_central" (Gaussian, then central
    /// differences).
    pub derivative: String,
    /// Half-width of the smoothing kernel for `derivative="smooth_central"`, in pixels.
    #[pyo3(set)]
    pub kernel_radius_px: f32,
    /// How each edge is located on the profile.
    #[pyo3(set)]
    pub locate: Locate,
    /// "fill" (default) measures a caliper that overhangs the image, sampling the outside
    /// with `border_mode`, and reports "off_image" only when no edge is found; "reject"
    /// raises `MeasureRejected("off_image")` before looking for edges whenever any sample
    /// lies outside `[0, w - 1] x [0, h - 1]`.
    pub off_image: String,
}

#[pymethods]
impl MeasureConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        sigma=None,
        threshold=None,
        polarity=None,
        select=None,
        sequence=None,
        step=None,
        max_obliquity_deg=None,
        border_mode=None,
        border_constant=None,
        derivative=None,
        kernel_radius_px=None,
        locate=None,
        off_image=None
    ))]
    pub fn new(
        sigma: Option<f32>,
        threshold: Option<f32>,
        polarity: Option<String>,
        select: Option<String>,
        sequence: Option<Vec<String>>,
        step: Option<f32>,
        max_obliquity_deg: Option<f32>,
        border_mode: Option<String>,
        border_constant: Option<f32>,
        derivative: Option<String>,
        kernel_radius_px: Option<f32>,
        locate: Option<Locate>,
        off_image: Option<String>,
    ) -> PyResult<Self> {
        let d = Self::default();
        for (name, value, allowed) in [
            ("polarity", polarity.as_deref(), POLARITIES),
            ("select", select.as_deref(), SELECTS),
            ("border_mode", border_mode.as_deref(), BORDER_MODES),
            ("derivative", derivative.as_deref(), DERIVATIVES),
            ("off_image", off_image.as_deref(), OFF_IMAGES),
        ] {
            if let Some(v) = value {
                check_name(name, v, allowed)?;
            }
        }
        let select = select.unwrap_or(d.select);
        let sequence = sequence.unwrap_or_default();
        if select == "in_order" {
            sequence_to_native(&sequence)?;
        } else if !sequence.is_empty() {
            return Err(PyValueError::new_err(
                "sequence is only used with select='in_order'",
            ));
        }
        Ok(Self {
            sigma: sigma.unwrap_or(d.sigma),
            threshold: threshold.unwrap_or(d.threshold),
            polarity: polarity.unwrap_or(d.polarity),
            select,
            sequence,
            step: step.unwrap_or(d.step),
            max_obliquity_deg: max_obliquity_deg.unwrap_or(d.max_obliquity_deg),
            border_mode: border_mode.unwrap_or(d.border_mode),
            border_constant: border_constant.unwrap_or(d.border_constant),
            derivative: derivative.unwrap_or(d.derivative),
            kernel_radius_px: kernel_radius_px.unwrap_or(d.kernel_radius_px),
            locate: locate.unwrap_or(d.locate),
            off_image: off_image.unwrap_or(d.off_image),
        })
    }

    fn __repr__(&self) -> String {
        let sequence = if self.select == "in_order" {
            format!(", sequence={:?}", self.sequence)
        } else {
            String::new()
        };
        format!(
            "MeasureConfig(sigma={:.3}, threshold={:.3}, polarity='{}', select='{}'{sequence})",
            self.sigma, self.threshold, self.polarity, self.select
        )
    }

    #[setter]
    fn set_polarity(&mut self, value: String) -> PyResult<()> {
        check_name("polarity", &value, POLARITIES)?;
        self.polarity = value;
        Ok(())
    }

    #[setter]
    fn set_select(&mut self, value: String) -> PyResult<()> {
        check_name("select", &value, SELECTS)?;
        self.select = value;
        Ok(())
    }

    /// Each entry must be a known polarity name, at most two of them. Whether
    /// `select="in_order"` has a usable sequence is checked when the config is used, so
    /// the two fields can be assigned in either order.
    #[setter]
    fn set_sequence(&mut self, value: Vec<String>) -> PyResult<()> {
        if !sequence_names_known(&value) {
            return Err(PyValueError::new_err(format!(
                "sequence must hold at most two of {SEQUENCE_POLARITIES:?}, got {value:?}"
            )));
        }
        self.sequence = value;
        Ok(())
    }

    #[setter]
    fn set_border_mode(&mut self, value: String) -> PyResult<()> {
        check_name("border_mode", &value, BORDER_MODES)?;
        self.border_mode = value;
        Ok(())
    }

    #[setter]
    fn set_derivative(&mut self, value: String) -> PyResult<()> {
        check_name("derivative", &value, DERIVATIVES)?;
        self.derivative = value;
        Ok(())
    }

    #[setter]
    fn set_off_image(&mut self, value: String) -> PyResult<()> {
        check_name("off_image", &value, OFF_IMAGES)?;
        self.off_image = value;
        Ok(())
    }
}

impl Default for MeasureConfig {
    fn default() -> Self {
        let n = NativeMeasureConfig::default();
        Self {
            sigma: n.profile.sigma,
            threshold: n.threshold,
            polarity: "any".to_string(),
            select: "all".to_string(),
            sequence: Vec::new(),
            step: n.profile.step,
            max_obliquity_deg: n.max_obliquity_deg,
            border_mode: "clamp".to_string(),
            border_constant: 0.0,
            derivative: "dog".to_string(),
            kernel_radius_px: 3.0,
            locate: Locate::default(),
            off_image: "fill".to_string(),
        }
    }
}

impl MeasureConfig {
    /// The native config; fails when `select="in_order"` has no valid `sequence`, or when
    /// a string field holds a name it does not accept.
    pub fn to_native(&self) -> PyResult<NativeMeasureConfig> {
        check_name("polarity", &self.polarity, POLARITIES)?;
        Ok(NativeMeasureConfig {
            threshold: self.threshold,
            polarity: polarity_from(&self.polarity)?,
            select: match self.select.as_str() {
                "all" => NativeEdgeSelect::All,
                "first" => NativeEdgeSelect::First,
                "last" => NativeEdgeSelect::Last,
                "strongest" => NativeEdgeSelect::Strongest,
                "in_order" => {
                    NativeEdgeSelect::StrongestInOrder(sequence_to_native(&self.sequence)?)
                }
                other => return Err(not_one_of("select", other, SELECTS)),
            },
            locate: self.locate.to_native()?,
            max_obliquity_deg: self.max_obliquity_deg,
            profile: Profile {
                sigma: self.sigma,
                step: self.step,
                border_mode: &self.border_mode,
                border_constant: self.border_constant,
                derivative: &self.derivative,
                kernel_radius_px: self.kernel_radius_px,
                off_image: &self.off_image,
            }
            .to_native()?,
        })
    }
}
