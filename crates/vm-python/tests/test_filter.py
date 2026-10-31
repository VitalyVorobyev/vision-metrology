"""Linear filters (ADR-0023): box and binomial are exact in integers; Gaussian is checked
against a direct f64 convolution."""

import numpy as np
import pytest

import vision_metrology as vm


def naive_box(a, r):
    h, w = a.shape
    p = np.pad(a.astype(np.int64), r, mode="edge")
    d = (2 * r + 1) ** 2
    out = np.zeros((h, w), np.uint8)
    for y in range(h):
        for x in range(w):
            s = p[y : y + 2 * r + 1, x : x + 2 * r + 1].sum()
            out[y, x] = (2 * s + d) // (2 * d)
    return out


def naive_separable(a, k, border="edge"):
    """Separable 1-D kernel `k` (odd length) applied in both directions, in f64."""
    r = len(k) // 2
    p = np.pad(a.astype(np.float64), r, mode=border)
    h, w = a.shape
    tmp = sum(k[j] * p[:, j : j + w] for j in range(len(k)))
    return sum(k[i] * tmp[i : i + h, :] for i in range(len(k)))


@pytest.fixture
def img():
    return np.random.default_rng(0).integers(0, 256, (30, 40), dtype=np.uint8)


@pytest.mark.parametrize("radius", [0, 1, 3])
def test_box_mean_is_exact(img, radius):
    assert np.array_equal(vm.box_mean(img, radius), naive_box(img, radius))


def test_binomial_order2_is_exact(img):
    k = np.array([1, 2, 1], dtype=np.int64)
    p = np.pad(img.astype(np.int64), 1, mode="edge")
    ref = sum(k[dy] * k[dx] * p[dy : dy + 30, dx : dx + 40] for dy in range(3) for dx in range(3))
    assert np.array_equal(vm.binomial(img, 2), ((ref + 8) // 16).astype(np.uint8))


def test_gaussian_matches_a_direct_convolution():
    rng = np.random.default_rng(1)
    f = (rng.random((30, 40)) * 255).astype(np.float32)
    sigma = 1.5
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    got = vm.gaussian(f, sigma)
    assert got.dtype == np.float32
    assert np.max(np.abs(got - naive_separable(f, k))) < 1e-3


def test_gaussian_large_sigma_is_within_the_stated_budget():
    # ADR-0023, Phase 3: the cascade is within 5 grey levels of the Gaussian on this fixture.
    rng = np.random.default_rng(2)
    f = (rng.random((48, 40)) * 255).astype(np.float32)
    sigma = 8.0
    r = int(np.ceil(8 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    got = vm.gaussian(f, sigma)
    assert np.max(np.abs(got - naive_separable(f, k))) < 5.0


def test_invalid_parameters_raise_value_error(img):
    with pytest.raises(ValueError):
        vm.binomial(img, 3)
    with pytest.raises(ValueError):
        vm.gaussian(img.astype(np.float32), 0.0)
    with pytest.raises(ValueError):
        vm.box_mean(img, 1, "bogus")


def test_constant_border_fills_the_outside(img):
    flat = np.full((5, 6), 9, dtype=np.uint8)
    assert np.array_equal(vm.box_mean(flat, 2, "constant", 9), flat)
