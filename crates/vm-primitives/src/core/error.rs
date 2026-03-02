use core::fmt;

/// Library-wide error type.
///
/// All public fallible functions in `vm-*` crates return `Result<T, vm_core::Error>`.
/// Internal programmer-error invariants use `assert!` or `Option` instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Buffer or dimension mismatch between two operands.
    SizeMismatch {
        /// Expected size.
        expected: usize,
        /// Actual size encountered.
        actual: usize,
    },
    /// Index or coordinate is outside valid range.
    OutOfBounds,
    /// Stride value is inconsistent with image dimensions.
    InvalidStride,
    /// Fewer data points than required for the operation (e.g. fitting needs ≥ N samples).
    InsufficientData {
        /// Minimum number of samples required.
        need: usize,
        /// Number of samples actually provided.
        got: usize,
    },
    /// Degenerate geometric configuration (e.g. singular matrix, collinear points).
    Degenerate(&'static str),
    /// Parameter combination is invalid or out of range.
    InvalidConfig(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {expected}, got {actual}")
            }
            Self::OutOfBounds => write!(f, "out of bounds"),
            Self::InvalidStride => write!(f, "invalid stride"),
            Self::InsufficientData { need, got } => {
                write!(f, "insufficient data: need {need} points, got {got}")
            }
            Self::Degenerate(msg) => write!(f, "degenerate configuration: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn error_display() {
        assert_eq!(
            Error::SizeMismatch {
                expected: 10,
                actual: 5
            }
            .to_string(),
            "size mismatch: expected 10, got 5"
        );
        assert_eq!(
            Error::InsufficientData { need: 5, got: 3 }.to_string(),
            "insufficient data: need 5 points, got 3"
        );
        assert_eq!(
            Error::Degenerate("singular matrix").to_string(),
            "degenerate configuration: singular matrix"
        );
        assert_eq!(
            Error::InvalidConfig("sigma must be positive").to_string(),
            "invalid config: sigma must be positive"
        );
    }
}
