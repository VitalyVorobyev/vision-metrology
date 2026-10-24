import numpy as np
import pytest
import vision_metrology as vm
from vision_metrology import experimental


def matrix(parts):
    tx, ty, angle, scale = parts
    c, s = np.cos(angle)*scale, np.sin(angle)*scale
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.]])


def render(parts, dtype=np.float32, noise=0, seed=0):
    y, x = np.mgrid[:97, :161].astype(float)
    xy = np.linalg.inv(matrix(parts)) @ np.array([x.ravel(), y.ravel(), np.ones(x.size)])
    x, y = xy[:2]
    values = (100 + 25*np.sin(x*.17+y*.13) + 20*np.cos(x*.08-y*.21)
              + 15*np.sin(x*.31+y*.07) + 10*np.cos(x*.03+y*.39))
    values += np.random.default_rng(seed).uniform(-noise, noise, values.shape)
    return values.reshape(97, 161).astype(dtype)


def centered(dx=.7, dy=-.3, angle=.02, scale=1.01):
    linear = matrix((0, 0, angle, scale))[:2, :2]
    t = np.array([80., 48.])+[dx, dy] - linear @ [80., 48.]
    return *t, angle, scale


def corner_error(a, b):
    corners = np.array([[0, 160, 0, 160], [0, 0, 96, 96], [1, 1, 1, 1]])
    return np.linalg.norm(((matrix(a)-matrix(b)) @ corners)[:2], axis=0).max()


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32])
def test_similarity_and_prediction(dtype):
    truth = centered()
    a, b = render((0, 0, 0, 1), dtype), render(truth, dtype)
    mask = np.ones(a.shape, dtype=np.uint8)
    mask[25:55, 40:100] = 0
    f = experimental.SimilarityEstimator().estimate(a, b, previous_mask=mask, current_mask=mask)
    assert f.status == 'accepted'
    assert corner_error(f.transform, truth) < .1
    assert f.support > 400 and f.regions >= 2 and f.region_error < .5
    assert 0 <= f.region_chi2 <= experimental.SimilarityConfig().translation.region_check.max_chi2
    covariance = np.asarray(f.covariance)
    assert covariance.shape == (4, 4)
    assert np.linalg.eigvalsh(covariance).min() > -1e-9
    truth = centered(18.7, -.3, .03, 1.02)
    f = experimental.SimilarityEstimator().estimate(a, render(truth, dtype), prediction=centered(18, 0, .02, 1.01))
    assert f.status == 'accepted' and corner_error(f.transform, truth) < .1


def test_similarity_failures_photometry_and_configs():
    a, truth = render((0, 0, 0, 1)), centered()
    b = render(truth)*1.3+17
    e = experimental.SimilarityEstimator()
    f = e.estimate(a, b)
    assert f.status == 'accepted'
    assert abs(f.gain-1/1.3) < .02 and abs(f.offset+17/1.3) < 1
    assert e.estimate(a*0, a*0).status == 'low_texture'
    assert e.estimate(a, b, current_mask=np.zeros(a.shape, np.uint8)).status == 'insufficient_support'
    f = experimental.SimilarityEstimator(experimental.SimilarityConfig(max_angle=.005)).estimate(a, b)
    assert f.status == 'search_boundary'
    for prediction in [(0, 0, 0, 0), (0, 0, 0, -1), (np.nan, 0, 0, 1)]:
        with pytest.raises(ValueError):
            e.estimate(a, b, prediction=prediction)
    with pytest.raises(ValueError):
        e.estimate(a, b.astype(np.uint8))
    with pytest.raises(ValueError):
        e.estimate(a, b, current_mask=np.ones((4, 4), np.uint8))


def test_similarity_is_experimental_only():
    """The class lives under `vision_metrology.experimental`, not the top level, and its
    stub declares exactly what exists there."""
    import importlib.resources as res
    import re

    for name in ("Similarity", "SimilarityConfig", "SimilarityEstimator"):
        assert not hasattr(vm, name), f"{name} must not be in the top-level namespace"
        assert hasattr(experimental, name)
    stub = (res.files("vision_metrology") / "experimental.pyi").read_text()
    declared = re.findall(r"^class (\w+)", stub, re.MULTILINE)
    assert sorted(declared) == sorted(experimental.__all__)
