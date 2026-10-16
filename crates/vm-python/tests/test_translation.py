import numpy as np
import pytest
import vision_metrology as vm


def texture(shift=(0, 0), dtype=np.float32):
    y, x = np.mgrid[:97, :161].astype(np.float64)
    x -= shift[0]
    y -= shift[1]
    image = (100 + 25 * np.sin(x * .17 + y * .13)
             + 20 * np.cos(x * .08 - y * .21)
             + 15 * np.sin(x * .31 + y * .07)
             + 10 * np.cos(x * .03 + y * .39))
    return image.astype(dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_masked_translation_and_closure(dtype):
    estimator = vm.TranslationEstimator()
    a, b = texture(dtype=dtype), texture((.7, -.3), dtype=dtype)
    mask = np.ones(a.shape, dtype=np.uint8)
    mask[20:45, 30:100] = 0
    f = estimator.estimate(a, b, previous_mask=mask, current_mask=mask)
    assert f.status == "accepted"
    assert np.linalg.norm(np.array(f.shift) - [.7, -.3]) < .1
    assert f.support > 400
    assert f.score > .9
    assert f.margin > .02
    assert f.residual_rms >= 0
    assert f.residual_max >= f.residual_rms
    assert f.covariance[0] >= 0 and f.covariance[2] >= 0
    assert f.regions >= 2
    assert f.region_error < .5
    assert 0 <= f.region_chi2 <= vm.TranslationConfig().region_check.max_chi2
    r = estimator.estimate(b, a, previous_mask=mask, current_mask=mask)
    assert r.status == "accepted"
    assert np.linalg.norm(np.array(f.shift) + r.shift) < .05


def test_photometry_and_failures():
    a = texture()
    b = texture((1.2, -.4)) * 1.3 + 17
    estimator = vm.TranslationEstimator(vm.TranslationConfig(
        tuning=vm.TranslationTuning(iterations=16)))
    f = estimator.estimate(a, b, prediction=(1, 0))
    assert f.status == "accepted"
    assert abs(f.gain - 1 / 1.3) < .02
    assert abs(f.offset + 17 / 1.3) < 1
    flat = np.zeros(a.shape, dtype=np.float32)
    f = estimator.estimate(flat, flat)
    assert f.status == "low_texture" and f.shift is None
    f = estimator.estimate(a, b, previous_mask=np.zeros(a.shape, dtype=np.uint8))
    assert f.status == "insufficient_support"
    with pytest.raises(ValueError):
        estimator.estimate(a, b, current_mask=np.ones((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        estimator.estimate(a, b.astype(np.uint8))
    with pytest.raises(ValueError):
        estimator.estimate(a, b, prediction=(float("nan"), 0))
    with pytest.raises(ValueError):
        vm.TranslationEstimator(vm.TranslationConfig(search_radius=0)).estimate(a, b)


def test_region_check_config_and_rotation_rejection():
    cfg = vm.TranslationConfig()
    assert isinstance(cfg.region_check, vm.RegionCheck)
    assert cfg.region_check.max_chi2 > 0 and cfg.region_check.bias_floor_px >= 0
    y, x = np.mgrid[:97, :320].astype(np.float64)
    c, s = np.cos(.01), np.sin(.01)
    u, v = x - 160, y - 48
    a = (100 + 25*np.sin(x*.17+y*.13) + 20*np.cos(x*.08-y*.21)).astype(np.float32)
    xr, yr = c*u + s*v + 160 - 1.2, -s*u + c*v + 48 - .5
    b = (100 + 25*np.sin(xr*.17+yr*.13) + 20*np.cos(xr*.08-yr*.21)).astype(np.float32)
    f = vm.TranslationEstimator(vm.TranslationConfig(min_margin=0)).estimate(a, b)
    assert f.status == "model_mismatch" and f.region_chi2 > cfg.region_check.max_chi2
    f = vm.TranslationEstimator(vm.TranslationConfig(min_margin=0, region_check=None)).estimate(a, b)
    assert f.region_chi2 is None and f.region_error is None
    with pytest.raises(ValueError):
        vm.TranslationEstimator(vm.TranslationConfig(
            region_check=vm.RegionCheck(max_chi2=-1))).estimate(a, a)
