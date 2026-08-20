//! Adapts this crate's [`ImageView`] to corrmatch's own `ImageView`, and
//! maps corrmatch's error type onto [`Error`].
//!
//! corrmatch's `ImageView` borrows a `&[u8]` slice with an explicit stride;
//! our own [`ImageView`] never exposes that slice when it is strided
//! (`stride > width` — most commonly a ROI `subview`), so the zero-copy path
//! only applies when [`ImageView::as_contiguous_slice`] succeeds. A strided
//! caller falls back to one row-by-row copy — the same cost
//! `examples/common/corr.rs::roi_patch` already pays for the same reason.

use vm_primitives::{Error, ImageView, Rect2f};

/// Either a borrow of `img`'s own contiguous buffer, or an owned copy made
/// row by row when `img` is strided.
pub(super) enum SceneBuf<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl SceneBuf<'_> {
    pub(super) fn as_slice(&self) -> &[u8] {
        match self {
            SceneBuf::Borrowed(s) => s,
            SceneBuf::Owned(v) => v,
        }
    }
}

/// Zero-copy when `img` is contiguous, else one row-by-row copy.
pub(super) fn scene_buf<'a>(img: &ImageView<'a, u8>) -> SceneBuf<'a> {
    if let Some(s) = img.as_contiguous_slice() {
        SceneBuf::Borrowed(s)
    } else {
        let mut v = Vec::with_capacity(img.width() * img.height());
        for y in 0..img.height() {
            v.extend_from_slice(img.row(y));
        }
        SceneBuf::Owned(v)
    }
}

/// Wraps a contiguous `width x height` buffer as a corrmatch view.
pub(super) fn corr_view(
    buf: &[u8],
    width: usize,
    height: usize,
) -> Result<corrmatch::ImageView<'_, u8>, Error> {
    corrmatch::ImageView::from_slice(buf, width, height)
        .map_err(|_| Error::InvalidConfig("corr: invalid image dimensions"))
}

/// Rounds `rect` (nearest, then clamped to `[0, img_w) x [0, img_h)`) to an
/// integer pixel window, returning `(x0, y0, width, height)`.
///
/// # Errors
/// [`Error::InvalidConfig`] when the rounded rect does not overlap the
/// image at all.
pub(super) fn rect_bounds(
    img_w: usize,
    img_h: usize,
    rect: Rect2f,
) -> Result<(usize, usize, usize, usize), Error> {
    let x0 = rect.x.round().max(0.0) as usize;
    let y0 = rect.y.round().max(0.0) as usize;
    let x1 = ((rect.x + rect.width).round() as isize).clamp(0, img_w as isize) as usize;
    let y1 = ((rect.y + rect.height).round() as isize).clamp(0, img_h as isize) as usize;
    if x0 >= img_w || y0 >= img_h || x1 <= x0 || y1 <= y0 {
        return Err(Error::InvalidConfig(
            "corr: rect does not overlap the image",
        ));
    }
    Ok((x0, y0, x1 - x0, y1 - y0))
}

/// Copies the `width x height` window at `(x0, y0)` out of `img` into a
/// contiguous, owned buffer. Caller guarantees the window is in bounds
/// (e.g. via [`rect_bounds`]).
pub(super) fn copy_bounds_u8(
    img: &ImageView<'_, u8>,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(width * height);
    for y in y0..y0 + height {
        buf.extend_from_slice(&img.row(y)[x0..x0 + width]);
    }
    buf
}

/// Rounds `rect` to an integer window and copies it out of `img`.
pub(super) fn extract_rect_u8(
    img: &ImageView<'_, u8>,
    rect: Rect2f,
) -> Result<(Vec<u8>, usize, usize), Error> {
    let (x0, y0, w, h) = rect_bounds(img.width(), img.height(), rect)?;
    Ok((copy_bounds_u8(img, x0, y0, w, h), w, h))
}

/// Maps a corrmatch error onto this crate's [`Error`], keeping a fixed,
/// `'static` reason and dropping any dynamic detail — the same tradeoff
/// `metric::io` makes for `serde_json::Error` (see its module docs).
pub(super) fn map_corr_err(e: corrmatch::CorrMatchError) -> Error {
    use corrmatch::CorrMatchError as C;
    match e {
        C::InvalidDimensions { .. } => Error::InvalidConfig("corrmatch: invalid dimensions"),
        C::InvalidStride { .. } => Error::InvalidStride,
        C::BufferTooSmall { .. } => Error::InvalidConfig("corrmatch: buffer too small"),
        C::RoiOutOfBounds { .. } => Error::OutOfBounds,
        C::DegenerateTemplate { .. } => Error::Degenerate("corrmatch: degenerate template"),
        C::InvalidAngleGrid { .. } => Error::InvalidConfig("corrmatch: invalid angle grid"),
        C::IndexOutOfBounds { .. } => Error::OutOfBounds,
        C::NoCandidates { .. } => Error::Degenerate("corrmatch: no candidates found"),
        C::UnsupportedMetric { .. } => Error::InvalidConfig("corrmatch: unsupported metric"),
        C::RotationUnavailable { .. } => Error::InvalidConfig(
            "corrmatch: rotation search requires a template compiled with rotation support",
        ),
        C::ImageIo { .. } => Error::InvalidConfig("corrmatch: image io error"),
        C::ParallelUnavailable => {
            Error::InvalidConfig("corrmatch: parallel search requested without the rayon feature")
        }
        C::InvalidConfig { .. } => Error::InvalidConfig("corrmatch: invalid configuration"),
        _ => Error::InvalidConfig("corrmatch: error"),
    }
}
