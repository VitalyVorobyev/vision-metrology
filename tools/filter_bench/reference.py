#!/usr/bin/env python3
"""Independent scipy reference for the filter_bench oracle (ADR-0023).

The Rust crate checks its f64 reference and its prototype against the golden files in
`golden/`. This script writes those golden files with scipy, which is an implementation
the Rust code does not share, so the oracle is checked by something other than itself.

The fixture is the same 64-bit LCG as `proto::test_image`, so both sides see identical
pixels. Clamp borders are scipy's `mode="nearest"`. The Gaussian truncation matches the
Rust radius `ceil(3 sigma)`: scipy uses `int(truncate*sigma + 0.5)`, which equals it for
the sigmas used here (1.5 gives radius 5 on both sides; 2 would give 6).

Usage (from this directory, with numpy and scipy installed):
    python3 reference.py          # regenerates golden/
    python3 reference.py --check  # verifies the golden files are current
"""

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

W, H = 24, 17
SIGMA = 1.5
BOX_RADIUS = 1
GOLDEN = Path(__file__).parent / "golden"
MASK = (1 << 64) - 1


def test_image(w: int, h: int) -> np.ndarray:
    """Mirror of `proto::test_image`: a 64-bit LCG, wrapping arithmetic."""
    state = 0x9E37_79B9_7F4A_7C15
    out = np.empty(w * h, dtype=np.int64)
    for i in range(w * h):
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK
        noise = state >> 59
        x, y = i % w, i // w
        out[i] = ((x * 7 + y * 5) % 256) ^ noise
    return out.reshape(h, w)


def box_exact(src: np.ndarray, r: int) -> np.ndarray:
    """Exact integer box mean, rounded half up: floor((sum + d/2) / d)."""
    n = 2 * r + 1
    d = n * n
    sums = ndimage.convolve(src, np.ones((n, n), dtype=np.int64), mode="nearest")
    return (sums + d // 2) // d


def gauss_f64(src: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(src.astype(np.float64), sigma, truncate=3.0, mode="nearest")


def write(path: Path, values: np.ndarray, fmt: str) -> None:
    path.write_text(" ".join(fmt % v for v in values.ravel()) + "\n")


def main() -> int:
    src = test_image(W, H)
    box = box_exact(src, BOX_RADIUS)
    gauss = gauss_f64(src, SIGMA)
    expected = {
        "src.txt": (src, "%d"),
        "box_r1.txt": (box, "%d"),
        "gauss_1_5.txt": (gauss, "%.17g"),
    }
    if "--check" in sys.argv:
        # Numeric comparison, not string comparison: scipy's last bits can differ between
        # platforms (macOS-generated goldens are checked on Linux CI), so floats get a
        # tolerance and integers must match exactly.
        bad = []
        for name, (vals, _fmt) in expected.items():
            p = GOLDEN / name
            if not p.exists():
                bad.append(name)
                continue
            got = np.array([float(t) for t in p.read_text().split()])
            want = vals.ravel().astype(np.float64)
            tol = 0.0 if name.startswith(("src", "box")) else 1e-9
            if got.shape != want.shape or not np.allclose(got, want, rtol=0.0, atol=tol):
                bad.append(name)
        if bad:
            print("golden out of date:", ", ".join(bad))
            return 1
        print("golden current")
        return 0
    GOLDEN.mkdir(exist_ok=True)
    for name, (vals, fmt) in expected.items():
        write(GOLDEN / name, vals, fmt)
        print("wrote", GOLDEN / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
