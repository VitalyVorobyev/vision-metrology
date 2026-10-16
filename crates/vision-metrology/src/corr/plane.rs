use vm_primitives::{Error, ImageView, Pixel};

#[derive(Default)]
pub(super) struct Plane {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
    pub valid: Vec<bool>,
    /// `window[j]`: every pixel of the 4×4 block `[x-1, x+2] × [y-1, y+2]` around `j` is
    /// in bounds and valid. Filled by `load`.
    pub window: Vec<bool>,
}

impl Plane {
    pub fn resize(&mut self, w: usize, h: usize) {
        self.w = w;
        self.h = h;
        self.data.resize(w * h, 0.0);
        self.valid.resize(w * h, false);
    }

    pub fn load<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        mask: Option<&ImageView<'_, u8>>,
    ) -> Result<(f64, f64, usize), Error> {
        self.resize(img.width(), img.height());
        let (mut mean, mut m2, mut n) = (0.0, 0.0, 0usize);
        for y in 0..self.h {
            for (x, pixel) in img.row(y).iter().enumerate() {
                let i = y * self.w + x;
                let v = pixel.to_f32();
                let valid = mask.is_none_or(|m| m.row(y)[x] != 0);
                if valid && !v.is_finite() {
                    return Err(Error::InvalidConfig(
                        "non-finite unmasked translation pixel",
                    ));
                }
                self.valid[i] = valid;
                self.data[i] = if valid { v } else { 0.0 };
                if valid {
                    n += 1;
                    let d = v as f64 - mean;
                    mean += d / n as f64;
                    m2 += d * (v as f64 - mean);
                }
            }
        }
        self.prepare_window();
        let scale = (m2 / n.max(1) as f64).sqrt();
        if scale > 0.0 {
            for (v, valid) in self.data.iter_mut().zip(&self.valid) {
                if *valid {
                    *v = ((*v as f64 - mean) / scale) as f32;
                }
            }
        }
        Ok((mean, scale, n))
    }

    pub fn downsample(&mut self, src: &Self) {
        self.resize(src.w / 2, src.h / 2);
        for y in 0..self.h {
            for x in 0..self.w {
                let j = 2 * y * src.w + 2 * x;
                let ids = [j, j + 1, j + src.w, j + src.w + 1];
                let i = y * self.w + x;
                self.valid[i] = ids.iter().all(|&k| src.valid[k]);
                self.data[i] = if self.valid[i] {
                    (ids.iter().map(|&k| src.data[k] as f64).sum::<f64>() * 0.25) as f32
                } else {
                    0.0
                };
            }
        }
    }

    /// Mark the pixels whose bilinear taps stay valid for every position within ±1 px:
    /// a sample at `(x, y)` with `window_valid(x, y)` stays valid for any `(x + dx, y + dy)`
    /// with `|dx|, |dy| < 1`. Separable 4-run erosion, two passes, no allocation once sized.
    fn prepare_window(&mut self) {
        let (w, h) = (self.w, self.h);
        self.window.resize(w * h, false);
        // Horizontal: window[j] holds whether x-1..=x+2 are all valid in this row.
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                self.window[row + x] =
                    x >= 1 && x + 2 < w && self.valid[row + x - 1..=row + x + 2].iter().all(|&v| v);
            }
        }
        // Vertical, in place: rows below are still horizontal; the row above is cached.
        for x in 0..w {
            let mut above = false;
            for y in 0..h {
                let here = self.window[y * w + x];
                let ok = y >= 1
                    && y + 2 < h
                    && above
                    && here
                    && self.window[(y + 1) * w + x]
                    && self.window[(y + 2) * w + x];
                above = here;
                self.window[y * w + x] = ok;
            }
        }
    }

    /// Whether the ±1 px tap window about `(x, y)` is valid (see `prepare_window`);
    /// `false` outside the image.
    pub fn window_valid(&self, x: f64, y: f64) -> bool {
        let (fx, fy) = (x.floor(), y.floor());
        if !(fx >= 0.0 && fy >= 0.0 && fx < self.w as f64 && fy < self.h as f64) {
            return false;
        }
        self.window[fy as usize * self.w + fx as usize]
    }

    /// Bilinear value and exact derivatives. All four taps must be valid, even
    /// at integer coordinates: an excluded pixel never enters a gradient.
    pub fn sample(&self, x: f64, y: f64) -> Option<(f64, f64, f64)> {
        if x < 0.0 || y < 0.0 || x >= (self.w - 1) as f64 || y >= (self.h - 1) as f64 {
            return None;
        }
        let (ix, iy) = (x.floor() as usize, y.floor() as usize);
        let j = iy * self.w + ix;
        if [j, j + 1, j + self.w, j + self.w + 1]
            .iter()
            .any(|&i| !self.valid[i])
        {
            return None;
        }
        let (u, v) = (x - ix as f64, y - iy as f64);
        let [a, b, c, d] = [j, j + 1, j + self.w, j + self.w + 1].map(|i| self.data[i] as f64);
        Some((
            (1.0 - v) * ((1.0 - u) * a + u * b) + v * ((1.0 - u) * c + u * d),
            (1.0 - v) * (b - a) + v * (d - c),
            (1.0 - u) * (c - a) + u * (d - b),
        ))
    }
}
