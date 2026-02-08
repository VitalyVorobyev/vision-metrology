use crate::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct Image<T> {
    width: usize,
    height: usize,
    data: Vec<T>,
}

impl<T> Image<T> {
    pub fn from_vec(width: usize, height: usize, data: Vec<T>) -> Result<Self, Error> {
        let expected = width.checked_mul(height).ok_or(Error::SizeMismatch {
            expected: usize::MAX,
            actual: data.len(),
        })?;

        if data.len() != expected {
            return Err(Error::SizeMismatch {
                expected,
                actual: data.len(),
            });
        }

        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn as_view(&self) -> ImageView<'_, T> {
        ImageView {
            width: self.width,
            height: self.height,
            stride: self.width,
            data: &self.data,
        }
    }

    pub fn as_view_mut(&mut self) -> ImageViewMut<'_, T> {
        ImageViewMut {
            width: self.width,
            height: self.height,
            stride: self.width,
            data: &mut self.data,
        }
    }
}

impl<T: Clone> Image<T> {
    pub fn new_fill(width: usize, height: usize, value: T) -> Self {
        let len = width.checked_mul(height).expect("image size overflow");
        Self {
            width,
            height,
            data: vec![value; len],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImageView<'a, T> {
    width: usize,
    height: usize,
    stride: usize,
    data: &'a [T],
}

impl<'a, T> ImageView<'a, T> {
    pub fn from_slice(
        width: usize,
        height: usize,
        stride: usize,
        data: &'a [T],
    ) -> Result<Self, Error> {
        if stride < width {
            return Err(Error::InvalidStride);
        }

        let min_len = stride.checked_mul(height).ok_or(Error::SizeMismatch {
            expected: usize::MAX,
            actual: data.len(),
        })?;

        if data.len() < min_len {
            return Err(Error::SizeMismatch {
                expected: min_len,
                actual: data.len(),
            });
        }

        Ok(Self {
            width,
            height,
            stride,
            data,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn row(&self, y: usize) -> &'a [T] {
        assert!(y < self.height, "row index out of bounds");
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    pub fn get(&self, x: usize, y: usize) -> Option<&'a T> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = y * self.stride + x;
        self.data.get(idx)
    }

    /// Returns a pixel reference without bounds checks.
    ///
    /// # Safety
    /// Caller must guarantee `x < self.width()` and `y < self.height()`.
    pub unsafe fn get_unchecked(&self, x: usize, y: usize) -> &'a T {
        // SAFETY: Caller guarantees `x < width` and `y < height`. With view
        // invariants this implies `idx` is in bounds of `data`.
        unsafe { self.data.get_unchecked(y * self.stride + x) }
    }

    pub fn subview(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Result<ImageView<'a, T>, Error> {
        if x > self.width
            || y > self.height
            || width > (self.width - x)
            || height > (self.height - y)
        {
            return Err(Error::OutOfBounds);
        }

        let start = y
            .checked_mul(self.stride)
            .and_then(|v| v.checked_add(x))
            .ok_or(Error::OutOfBounds)?;
        let min_len = min_required_len(width, height, self.stride).ok_or(Error::OutOfBounds)?;
        let tail = self.data.get(start..).ok_or(Error::OutOfBounds)?;

        if tail.len() < min_len {
            return Err(Error::OutOfBounds);
        }

        Ok(ImageView {
            width,
            height,
            stride: self.stride,
            data: tail,
        })
    }

    pub fn is_contiguous(&self) -> bool {
        self.stride == self.width
    }

    pub fn as_contiguous_slice(&self) -> Option<&'a [T]> {
        if !self.is_contiguous() {
            return None;
        }
        let len = self.width * self.height;
        self.data.get(0..len)
    }
}

#[derive(Debug)]
pub struct ImageViewMut<'a, T> {
    width: usize,
    height: usize,
    stride: usize,
    data: &'a mut [T],
}

impl<'a, T> ImageViewMut<'a, T> {
    pub fn from_slice_mut(
        width: usize,
        height: usize,
        stride: usize,
        data: &'a mut [T],
    ) -> Result<Self, Error> {
        if stride < width {
            return Err(Error::InvalidStride);
        }

        let min_len = stride.checked_mul(height).ok_or(Error::SizeMismatch {
            expected: usize::MAX,
            actual: data.len(),
        })?;

        if data.len() < min_len {
            return Err(Error::SizeMismatch {
                expected: min_len,
                actual: data.len(),
            });
        }

        Ok(Self {
            width,
            height,
            stride,
            data,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn row(&self, y: usize) -> &[T] {
        assert!(y < self.height, "row index out of bounds");
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        assert!(y < self.height, "row index out of bounds");
        let start = y * self.stride;
        &mut self.data[start..start + self.width]
    }

    pub fn subview(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Result<ImageView<'_, T>, Error> {
        self.as_view().subview(x, y, width, height)
    }

    pub fn get(&self, x: usize, y: usize) -> Option<&T> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = y * self.stride + x;
        self.data.get(idx)
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = y * self.stride + x;
        self.data.get_mut(idx)
    }

    /// Returns a pixel reference without bounds checks.
    ///
    /// # Safety
    /// Caller must guarantee `x < self.width()` and `y < self.height()`.
    pub unsafe fn get_unchecked(&self, x: usize, y: usize) -> &T {
        // SAFETY: Caller guarantees `x < width` and `y < height`. With view
        // invariants this implies `idx` is in bounds of `data`.
        unsafe { self.data.get_unchecked(y * self.stride + x) }
    }

    /// Returns a mutable pixel reference without bounds checks.
    ///
    /// # Safety
    /// Caller must guarantee `x < self.width()` and `y < self.height()`.
    pub unsafe fn get_unchecked_mut(&mut self, x: usize, y: usize) -> &mut T {
        // SAFETY: Caller guarantees `x < width` and `y < height`. With view
        // invariants this implies `idx` is in bounds of `data`.
        unsafe { self.data.get_unchecked_mut(y * self.stride + x) }
    }

    pub fn subview_mut(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Result<ImageViewMut<'_, T>, Error> {
        if x > self.width
            || y > self.height
            || width > (self.width - x)
            || height > (self.height - y)
        {
            return Err(Error::OutOfBounds);
        }

        let start = y
            .checked_mul(self.stride)
            .and_then(|v| v.checked_add(x))
            .ok_or(Error::OutOfBounds)?;
        let min_len = min_required_len(width, height, self.stride).ok_or(Error::OutOfBounds)?;

        let (_, tail) = self.data.split_at_mut(start);
        if tail.len() < min_len {
            return Err(Error::OutOfBounds);
        }

        Ok(ImageViewMut {
            width,
            height,
            stride: self.stride,
            data: tail,
        })
    }

    pub fn as_view(&self) -> ImageView<'_, T> {
        ImageView {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: self.data,
        }
    }

    pub fn is_contiguous(&self) -> bool {
        self.stride == self.width
    }

    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        if !self.is_contiguous() {
            return None;
        }
        let len = self.width * self.height;
        self.data.get(0..len)
    }

    pub fn as_contiguous_slice_mut(&mut self) -> Option<&mut [T]> {
        if !self.is_contiguous() {
            return None;
        }
        let len = self.width * self.height;
        self.data.get_mut(0..len)
    }
}

fn min_required_len(width: usize, height: usize, stride: usize) -> Option<usize> {
    if width == 0 || height == 0 {
        return Some(0);
    }

    let rows_before_last = height.checked_sub(1)?;
    let base = rows_before_last.checked_mul(stride)?;
    base.checked_add(width)
}

pub fn to_f32(img: &ImageView<'_, u8>) -> Image<f32> {
    let mut out = Vec::with_capacity(img.width() * img.height());
    for y in 0..img.height() {
        for &px in img.row(y) {
            out.push(px as f32);
        }
    }

    Image {
        width: img.width(),
        height: img.height(),
        data: out,
    }
}

pub fn to_f32_u16(img: &ImageView<'_, u16>) -> Image<f32> {
    let mut out = Vec::with_capacity(img.width() * img.height());
    for y in 0..img.height() {
        for &px in img.row(y) {
            out.push(px as f32);
        }
    }

    Image {
        width: img.width(),
        height: img.height(),
        data: out,
    }
}

#[cfg(test)]
mod tests {
    use super::{Image, ImageView, ImageViewMut, to_f32, to_f32_u16};

    #[test]
    fn view_indexing_with_stride() {
        let data = vec![1u8, 2, 3, 99, 4, 5, 6, 88];
        let view = ImageView::from_slice(3, 2, 4, &data).expect("valid view");

        assert_eq!(view.row(0), &[1, 2, 3]);
        assert_eq!(view.row(1), &[4, 5, 6]);
        assert_eq!(view.get(0, 1), Some(&4));
        assert_eq!(view.get(2, 1), Some(&6));
        assert_eq!(view.get(3, 1), None);
        assert!(!view.is_contiguous());
        assert!(view.as_contiguous_slice().is_none());
    }

    #[test]
    fn subview_non_contiguous_parent() {
        let data = vec![
            10u8, 11, 12, 13, 99, // row 0
            20, 21, 22, 23, 98, // row 1
            30, 31, 32, 33, 97, // row 2
        ];
        let parent = ImageView::from_slice(4, 3, 5, &data).expect("valid parent");
        let sub = parent.subview(1, 1, 3, 2).expect("valid subview");

        assert_eq!(sub.width(), 3);
        assert_eq!(sub.height(), 2);
        assert_eq!(sub.stride(), 5);
        assert_eq!(sub.row(0), &[21, 22, 23]);
        assert_eq!(sub.row(1), &[31, 32, 33]);
        assert_eq!(sub.get(2, 1), Some(&33));
    }

    #[test]
    fn subview_mut_non_contiguous_parent() {
        let mut data = vec![
            1u8, 2, 3, 4, 0, // row 0
            5, 6, 7, 8, 0, // row 1
            9, 10, 11, 12, 0, // row 2
        ];

        let mut parent = ImageViewMut::from_slice_mut(4, 3, 5, &mut data).expect("valid parent");
        let mut sub = parent.subview_mut(1, 0, 2, 3).expect("valid subview");
        *sub.get_mut(0, 2).expect("in bounds") = 42;

        assert_eq!(sub.row(0), &[2, 3]);
        assert_eq!(sub.row(2), &[42, 11]);
        assert_eq!(sub.get(0, 2), Some(&42));
    }

    #[test]
    fn convert_to_f32_variants() {
        let img8 = Image::from_vec(2, 2, vec![1u8, 2, 3, 4]).expect("valid image");
        let out8 = to_f32(&img8.as_view());
        assert_eq!(out8.data(), &[1.0, 2.0, 3.0, 4.0]);

        let img16 = Image::from_vec(2, 2, vec![100u16, 200, 300, 400]).expect("valid image");
        let out16 = to_f32_u16(&img16.as_view());
        assert_eq!(out16.data(), &[100.0, 200.0, 300.0, 400.0]);
    }
}
