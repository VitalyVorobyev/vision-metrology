//! 2D edgel extraction at a single scale.
//!
//! Coordinate convention: pixel centers, so integer `(x, y)` is the center of
//! pixel index `(x, y)`.
//!
//! Normal direction: `n` is derived from image gradient `(gx, gy)` and points
//! from dark to bright (increasing intensity).
//!
//! Threshold behavior:
//! - If `high_thresh == 0.0` and `low_thresh == 0.0`, thresholds are chosen
//!   automatically as `high = 0.2 * max_nms`, `low = 0.1 * max_nms`.
//! - Otherwise provided thresholds are used as-is (with low/high ordering fixed
//!   if needed).
//!
//! This module is intentionally single-scale. Pyramid/multi-scale integration
//! is done in higher-level crates.

use vm_core::{BorderMode, Image, ImageView, Point2f, Vec2f, sample_bilinear_f32};

#[derive(Debug, Clone, PartialEq)]
pub struct Edgel {
    pub p: Point2f,
    pub n: Vec2f,
    pub strength: f32,
    pub idx: (usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subpix2D {
    None,
    ParabolicAlongNormal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothKind {
    None,
    Binomial3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edge2DConfig {
    pub pre_smooth: bool,
    pub smooth_kind: SmoothKind,
    pub low_thresh: f32,
    pub high_thresh: f32,
    pub border: BorderMode<f32>,
    pub subpix: Subpix2D,
}

impl Default for Edge2DConfig {
    fn default() -> Self {
        Self {
            pre_smooth: true,
            smooth_kind: SmoothKind::Binomial3,
            low_thresh: 0.0,
            high_thresh: 0.0,
            border: BorderMode::Clamp,
            subpix: Subpix2D::ParabolicAlongNormal,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Edge2DDetector {
    tmp: Image<f32>,
    gx: Image<f32>,
    gy: Image<f32>,
    mag: Image<f32>,
    nms: Image<f32>,
    strong: Vec<u8>,
    weak: Vec<u8>,
    visited: Vec<u8>,
    stack: Vec<usize>,
}

impl Edge2DDetector {
    pub fn new() -> Self {
        Self {
            tmp: Image::new_fill(0, 0, 0.0),
            gx: Image::new_fill(0, 0, 0.0),
            gy: Image::new_fill(0, 0, 0.0),
            mag: Image::new_fill(0, 0, 0.0),
            nms: Image::new_fill(0, 0, 0.0),
            strong: Vec::new(),
            weak: Vec::new(),
            visited: Vec::new(),
            stack: Vec::new(),
        }
    }

    pub fn detect_u8(&mut self, img: &ImageView<'_, u8>, cfg: &Edge2DConfig) -> Vec<Edgel> {
        self.ensure_dims(img.width(), img.height());
        copy_u8_to_tmp(img, self.tmp.data_mut(), img.width());
        self.detect_from_tmp(cfg)
    }

    pub fn detect_u16(&mut self, img: &ImageView<'_, u16>, cfg: &Edge2DConfig) -> Vec<Edgel> {
        self.ensure_dims(img.width(), img.height());
        copy_u16_to_tmp(img, self.tmp.data_mut(), img.width());
        self.detect_from_tmp(cfg)
    }

    pub fn detect_f32(&mut self, img: &ImageView<'_, f32>, cfg: &Edge2DConfig) -> Vec<Edgel> {
        self.ensure_dims(img.width(), img.height());
        copy_f32_to_tmp(img, self.tmp.data_mut(), img.width());
        self.detect_from_tmp(cfg)
    }

    fn ensure_dims(&mut self, w: usize, h: usize) {
        if self.tmp.width() != w || self.tmp.height() != h {
            self.tmp = Image::new_fill(w, h, 0.0);
            self.gx = Image::new_fill(w, h, 0.0);
            self.gy = Image::new_fill(w, h, 0.0);
            self.mag = Image::new_fill(w, h, 0.0);
            self.nms = Image::new_fill(w, h, 0.0);
        }

        let n = w.saturating_mul(h);
        if self.strong.len() != n {
            self.strong = vec![0; n];
            self.weak = vec![0; n];
            self.visited = vec![0; n];
        }
    }

    fn detect_from_tmp(&mut self, cfg: &Edge2DConfig) -> Vec<Edgel> {
        let w = self.tmp.width();
        let h = self.tmp.height();
        if w == 0 || h == 0 {
            return Vec::new();
        }

        if cfg.pre_smooth {
            match cfg.smooth_kind {
                SmoothKind::None => {}
                SmoothKind::Binomial3 => self.smooth_binomial3(),
            }
        }

        self.compute_scharr();
        self.non_max_suppression();
        let final_count = self.hysteresis(cfg);

        self.build_edgels(cfg, final_count)
    }

    fn smooth_binomial3(&mut self) {
        let w = self.tmp.width();
        let h = self.tmp.height();
        if w == 0 || h == 0 {
            return;
        }

        {
            let src = self.tmp.data();
            let dst = self.gx.data_mut();
            for y in 0..h {
                let row = y * w;
                for x in 0..w {
                    let xm1 = x.saturating_sub(1);
                    let xp1 = (x + 1).min(w - 1);
                    let s = src[row + xm1] + 2.0 * src[row + x] + src[row + xp1];
                    dst[row + x] = 0.25 * s;
                }
            }
        }

        {
            let src = self.gx.data();
            let dst = self.tmp.data_mut();
            for y in 0..h {
                let ym1 = y.saturating_sub(1);
                let yp1 = (y + 1).min(h - 1);
                let r0 = ym1 * w;
                let r1 = y * w;
                let r2 = yp1 * w;
                for x in 0..w {
                    let s = src[r0 + x] + 2.0 * src[r1 + x] + src[r2 + x];
                    dst[r1 + x] = 0.25 * s;
                }
            }
        }
    }

    fn compute_scharr(&mut self) {
        let w = self.tmp.width();
        let h = self.tmp.height();
        let src = self.tmp.data();

        let (gx_img, gy_img, mag_img) = (&mut self.gx, &mut self.gy, &mut self.mag);
        let gx = gx_img.data_mut();
        let gy = gy_img.data_mut();
        let mag = mag_img.data_mut();

        for y in 0..h {
            let ym1 = y.saturating_sub(1);
            let yp1 = (y + 1).min(h - 1);
            for x in 0..w {
                let xm1 = x.saturating_sub(1);
                let xp1 = (x + 1).min(w - 1);

                let p00 = src[ym1 * w + xm1];
                let p01 = src[ym1 * w + x];
                let p02 = src[ym1 * w + xp1];
                let p10 = src[y * w + xm1];
                let p12 = src[y * w + xp1];
                let p20 = src[yp1 * w + xm1];
                let p21 = src[yp1 * w + x];
                let p22 = src[yp1 * w + xp1];

                let gxx =
                    (3.0 * p02 + 10.0 * p12 + 3.0 * p22) - (3.0 * p00 + 10.0 * p10 + 3.0 * p20);
                let gyy =
                    (3.0 * p20 + 10.0 * p21 + 3.0 * p22) - (3.0 * p00 + 10.0 * p01 + 3.0 * p02);

                let idx = y * w + x;
                gx[idx] = gxx;
                gy[idx] = gyy;
                mag[idx] = (gxx * gxx + gyy * gyy).sqrt();
            }
        }
    }

    fn non_max_suppression(&mut self) {
        let w = self.tmp.width();
        let h = self.tmp.height();
        let gx = self.gx.data();
        let gy = self.gy.data();
        let mag = self.mag.data();
        let nms = self.nms.data_mut();

        nms.fill(0.0);
        if w < 3 || h < 3 {
            return;
        }

        const TAN22_5: f32 = 0.414_213_57;
        const TAN67_5: f32 = 2.414_213_7;

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = y * w + x;
                let m = mag[idx];
                if m <= 0.0 {
                    continue;
                }

                let gxx = gx[idx];
                let gyy = gy[idx];
                let ax = gxx.abs();
                let ay = gyy.abs();

                let (i1, i2) = if ay <= ax * TAN22_5 {
                    (idx - 1, idx + 1)
                } else if ay >= ax * TAN67_5 {
                    (idx - w, idx + w)
                } else if gxx * gyy > 0.0 {
                    (idx - w - 1, idx + w + 1)
                } else {
                    (idx - w + 1, idx + w - 1)
                };

                if m >= mag[i1] && m >= mag[i2] {
                    nms[idx] = m;
                }
            }
        }
    }

    fn hysteresis(&mut self, cfg: &Edge2DConfig) -> usize {
        let w = self.tmp.width();
        let h = self.tmp.height();
        let n = w * h;

        self.strong.fill(0);
        self.weak.fill(0);
        self.visited.fill(0);
        self.stack.clear();

        let mut low = cfg.low_thresh;
        let mut high = cfg.high_thresh;

        if low == 0.0 && high == 0.0 {
            let mut max_nms = 0.0f32;
            for &v in self.nms.data() {
                if v > max_nms {
                    max_nms = v;
                }
            }

            if max_nms <= 0.0 {
                return 0;
            }

            high = 0.2 * max_nms;
            low = 0.1 * max_nms;
        }

        if high < low {
            core::mem::swap(&mut high, &mut low);
        }

        for idx in 0..n {
            let v = self.nms.data()[idx];
            if v <= 0.0 {
                continue;
            }
            if v >= low {
                self.weak[idx] = 1;
            }
            if v >= high {
                self.strong[idx] = 1;
                self.visited[idx] = 1;
                self.stack.push(idx);
            }
        }

        let mut count = self.stack.len();

        while let Some(idx) = self.stack.pop() {
            let x = idx % w;
            let y = idx / w;

            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(h - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(w - 1);

            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let nidx = ny * w + nx;
                    if self.visited[nidx] == 0 && self.weak[nidx] != 0 {
                        self.visited[nidx] = 1;
                        self.stack.push(nidx);
                        count += 1;
                    }
                }
            }
        }

        count
    }

    fn build_edgels(&self, cfg: &Edge2DConfig, count_hint: usize) -> Vec<Edgel> {
        let w = self.tmp.width();
        let h = self.tmp.height();
        let gx = self.gx.data();
        let gy = self.gy.data();
        let mag = self.mag.data();
        let nms_data = self.nms.data();
        let nms_view = self.nms.as_view();

        let mut out = Vec::with_capacity(count_hint);

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if self.visited[idx] == 0 {
                    continue;
                }

                let m = mag[idx];
                if m <= 1e-12 {
                    continue;
                }

                let n = Vec2f {
                    x: gx[idx] / m,
                    y: gy[idx] / m,
                }
                .normalize();
                if n.norm() <= 1e-6 {
                    continue;
                }

                let mut t = 0.0f32;
                if cfg.subpix == Subpix2D::ParabolicAlongNormal {
                    let s0 = nms_data[idx];
                    let sp = sample_bilinear_f32(
                        &nms_view,
                        x as f32 + n.x,
                        y as f32 + n.y,
                        cfg.border.clone(),
                    );
                    let sm = sample_bilinear_f32(
                        &nms_view,
                        x as f32 - n.x,
                        y as f32 - n.y,
                        cfg.border.clone(),
                    );
                    let denom = sm - 2.0 * s0 + sp;
                    if denom.abs() > 1e-12 {
                        let tt = 0.5 * (sm - sp) / denom;
                        if tt.is_finite() {
                            t = tt.clamp(-1.0, 1.0);
                        }
                    }
                }

                let p = Point2f {
                    x: x as f32 + t * n.x,
                    y: y as f32 + t * n.y,
                };

                out.push(Edgel {
                    p,
                    n,
                    strength: nms_data[idx],
                    idx: (x, y),
                });
            }
        }

        out
    }
}

impl Default for Edge2DDetector {
    fn default() -> Self {
        Self::new()
    }
}

fn copy_u8_to_tmp(src: &ImageView<'_, u8>, dst: &mut [f32], dst_w: usize) {
    for y in 0..src.height() {
        let s = src.row(y);
        let d = &mut dst[y * dst_w..(y + 1) * dst_w];
        for (dv, &sv) in d.iter_mut().zip(s.iter()) {
            *dv = sv as f32;
        }
    }
}

fn copy_u16_to_tmp(src: &ImageView<'_, u16>, dst: &mut [f32], dst_w: usize) {
    for y in 0..src.height() {
        let s = src.row(y);
        let d = &mut dst[y * dst_w..(y + 1) * dst_w];
        for (dv, &sv) in d.iter_mut().zip(s.iter()) {
            *dv = sv as f32;
        }
    }
}

fn copy_f32_to_tmp(src: &ImageView<'_, f32>, dst: &mut [f32], dst_w: usize) {
    for y in 0..src.height() {
        let s = src.row(y);
        let d = &mut dst[y * dst_w..(y + 1) * dst_w];
        d.copy_from_slice(s);
    }
}

#[cfg(test)]
mod tests {
    use vm_core::Image;

    use crate::edge2d::{Edge2DConfig, Edge2DDetector, Edgel, Subpix2D};

    fn build_slanted_step(
        width: usize,
        height: usize,
        theta_deg: f32,
    ) -> (Vec<f32>, Vec<f32>, f32) {
        let th = theta_deg.to_radians();
        let n = vec![th.cos(), th.sin()];
        let t = n[0] * (0.5 * width as f32) + n[1] * (0.5 * height as f32);

        let mut img = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let d = n[0] * x as f32 + n[1] * y as f32 - t;
                img[y * width + x] = if d >= 0.0 { 1.0 } else { 0.0 };
            }
        }

        (img, n, t)
    }

    fn blur_binomial3(img: &mut [f32], w: usize, h: usize, passes: usize) {
        let mut tmp = vec![0.0f32; w * h];
        for _ in 0..passes {
            for y in 0..h {
                let row = y * w;
                for x in 0..w {
                    let xm1 = x.saturating_sub(1);
                    let xp1 = (x + 1).min(w - 1);
                    tmp[row + x] = 0.25 * (img[row + xm1] + 2.0 * img[row + x] + img[row + xp1]);
                }
            }
            for y in 0..h {
                let ym1 = y.saturating_sub(1);
                let yp1 = (y + 1).min(h - 1);
                let r0 = ym1 * w;
                let r1 = y * w;
                let r2 = yp1 * w;
                for x in 0..w {
                    img[r1 + x] = 0.25 * (tmp[r0 + x] + 2.0 * tmp[r1 + x] + tmp[r2 + x]);
                }
            }
        }
    }

    fn median_abs_distance(edgels: &[Edgel], n: &[f32], t: f32) -> f32 {
        let mut d: Vec<f32> = edgels
            .iter()
            .map(|e| (n[0] * e.p.x + n[1] * e.p.y - t).abs())
            .collect();
        d.sort_by(|a, b| a.partial_cmp(b).expect("finite compare"));
        d[d.len() / 2]
    }

    #[test]
    fn slanted_edge_subpixel_improves_median_error() {
        let (w, h) = (128usize, 96usize);
        let (mut img_f, n, t) = build_slanted_step(w, h, 20.0);
        blur_binomial3(&mut img_f, w, h, 2);

        let img_u8: Vec<u8> = img_f
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();
        let img = Image::from_vec(w, h, img_u8).expect("valid image");

        let mut det = Edge2DDetector::new();
        let cfg_none = Edge2DConfig {
            subpix: Subpix2D::None,
            ..Edge2DConfig::default()
        };
        let cfg_sub = Edge2DConfig {
            subpix: Subpix2D::ParabolicAlongNormal,
            ..Edge2DConfig::default()
        };

        let e_none = det.detect_u8(&img.as_view(), &cfg_none);
        let e_sub = det.detect_u8(&img.as_view(), &cfg_sub);

        assert!(e_none.len() > 100);
        assert!(e_sub.len() > 100);

        let m_none = median_abs_distance(&e_none, &n, t);
        let m_sub = median_abs_distance(&e_sub, &n, t);

        assert!(m_none <= 0.3);
        assert!(m_sub <= 0.2);
        assert!(m_sub <= m_none + 1e-4);
    }

    #[test]
    fn vertical_step_normals_point_brightward() {
        let (w, h) = (96usize, 64usize);
        let edge_x = 40.3f32;

        let mut img_f = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                img_f[y * w + x] = if (x as f32) >= edge_x { 1.0 } else { 0.0 };
            }
        }
        blur_binomial3(&mut img_f, w, h, 2);
        let img_u8: Vec<u8> = img_f
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();
        let img = Image::from_vec(w, h, img_u8).expect("valid image");

        let mut det = Edge2DDetector::new();
        let edgels = det.detect_u8(&img.as_view(), &Edge2DConfig::default());
        assert!(!edgels.is_empty());

        let pos = edgels.iter().filter(|e| e.n.x > 0.0).count();
        assert!(pos * 100 / edgels.len() >= 80);
    }

    #[test]
    fn threshold_levels_affect_edge_count() {
        let (w, h) = (128usize, 96usize);
        let (mut img_f, _, _) = build_slanted_step(w, h, 20.0);
        blur_binomial3(&mut img_f, w, h, 2);
        let img_u8: Vec<u8> = img_f
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();
        let img = Image::from_vec(w, h, img_u8).expect("valid image");

        let mut det = Edge2DDetector::new();

        let cfg_lo = Edge2DConfig {
            low_thresh: 25.0,
            high_thresh: 50.0,
            ..Edge2DConfig::default()
        };
        let cfg_hi = Edge2DConfig {
            low_thresh: 1.0e9,
            high_thresh: 1.0e9,
            ..Edge2DConfig::default()
        };

        let c_lo = det.detect_u8(&img.as_view(), &cfg_lo).len();
        let c_hi = det.detect_u8(&img.as_view(), &cfg_hi).len();

        assert!(c_lo > 0);
        assert!(c_hi < c_lo);
    }

    #[test]
    fn u16_and_f32_paths_work() {
        let (w, h) = (64usize, 48usize);
        let (mut img_f, _, _) = build_slanted_step(w, h, 15.0);
        blur_binomial3(&mut img_f, w, h, 1);

        let img_u16: Vec<u16> = img_f.iter().map(|&v| (v * 1024.0).round() as u16).collect();
        let img_f32 = img_f.clone();

        let img16 = Image::from_vec(w, h, img_u16).expect("valid image");
        let img32 = Image::from_vec(w, h, img_f32).expect("valid image");

        let mut det = Edge2DDetector::new();
        let e16 = det.detect_u16(&img16.as_view(), &Edge2DConfig::default());
        let e32 = det.detect_f32(&img32.as_view(), &Edge2DConfig::default());

        assert!(!e16.is_empty());
        assert!(!e32.is_empty());
    }
}
