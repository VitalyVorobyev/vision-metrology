//! Test fixture: speckled masks and a masked DoG band-pass, the conditions under which taps
//! enter and leave validity as a pose moves.
use vm_primitives::Image;

/// Excluded with probability `density`, from a deterministic hash of `(x, y, seed)`.
pub(super) fn speckle(w: usize, h: usize, density: f64, seed: u64) -> Image<u8> {
    Image::from_vec(
        w,
        h,
        (0..w * h)
            .map(|i| {
                let mut z = (i as u64 ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15))
                    .wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z ^= z >> 31;
                z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 29;
                u8::from((z >> 11) as f64 / (1u64 << 53) as f64 >= density)
            })
            .collect(),
    )
    .unwrap()
}

/// Separable Gaussian, truncated at 3σ, zero outside the image.
fn gaussian(values: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let r = (3.0 * sigma).ceil() as isize;
    let k: Vec<f64> = (-r..=r)
        .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
        .collect();
    let pass = |src: &[f64], dx: isize, dy: isize| -> Vec<f64> {
        let mut out = vec![0.0; w * h];
        for y in 0..h as isize {
            for x in 0..w as isize {
                let mut sum = 0.0;
                for (j, kj) in k.iter().enumerate() {
                    let t = j as isize - r;
                    let (xx, yy) = (x + t * dx, y + t * dy);
                    if xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize {
                        sum += kj * src[yy as usize * w + xx as usize];
                    }
                }
                out[y as usize * w + x as usize] = sum;
            }
        }
        out
    };
    pass(&pass(values, 1, 0), 0, 1)
}

/// Masked normalized difference of Gaussians (σ 1 and 4 px), zero where excluded:
/// excluded pixels carry no filter mass, as in the registration evaluators.
pub(super) fn bandpass(image: &Image<f32>, mask: &Image<u8>) -> Image<f32> {
    let (w, h) = (image.width(), image.height());
    let mass: Vec<f64> = mask.data().iter().map(|&m| f64::from(m != 0)).collect();
    let weighted: Vec<f64> = image
        .data()
        .iter()
        .zip(&mass)
        .map(|(&v, &m)| v as f64 * m)
        .collect();
    let smooth = |sigma| {
        let (n, d) = (
            gaussian(&weighted, w, h, sigma),
            gaussian(&mass, w, h, sigma),
        );
        n.iter()
            .zip(&d)
            .map(|(n, d)| n / d.max(1e-6))
            .collect::<Vec<_>>()
    };
    let (fine, coarse) = (smooth(1.0), smooth(4.0));
    Image::from_vec(
        w,
        h,
        (0..w * h)
            .map(|i| ((fine[i] - coarse[i]) * mass[i]) as f32)
            .collect(),
    )
    .unwrap()
}
