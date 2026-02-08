use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    SizeMismatch { expected: usize, actual: usize },
    OutOfBounds,
    InvalidStride,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {expected}, got {actual}")
            }
            Self::OutOfBounds => write!(f, "out of bounds"),
            Self::InvalidStride => write!(f, "invalid stride"),
        }
    }
}

impl std::error::Error for Error {}
