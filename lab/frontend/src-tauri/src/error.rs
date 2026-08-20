//! One error type for every command, serializable so it crosses the IPC boundary as a
//! plain string the frontend can show — Tauri commands return `Result<T, E>` where `E:
//! serde::Serialize`, and a bare `String` is the simplest choice that still lets each
//! command name what went wrong (mirrors the FastAPI routers' `HTTPException(status,
//! detail)`, minus the HTTP status code this transport has no use for).

use std::fmt;

#[derive(Debug, Clone)]
pub struct AppError(pub String);

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AppError {}

impl serde::Serialize for AppError {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.0)
    }
}

impl From<vm_primitives::Error> for AppError {
    fn from(e: vm_primitives::Error) -> Self {
        Self(e.to_string())
    }
}

impl From<String> for AppError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for AppError {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}

impl From<image::ImageError> for AppError {
    fn from(e: image::ImageError) -> Self {
        Self(e.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        Self(e.to_string())
    }
}

pub type AppResult<T> = Result<T, AppError>;

pub fn not_found(kind: &str, id: &str) -> AppError {
    AppError(format!("no such {kind}: {id}"))
}
