"""The synthetic false-reject study and the regional diagnosis; no private data needed."""
import argparse

import numpy as np

import diagnose
import false_reject
import run
from synthetic import generate
from manifest import Sequence


def args(**overrides):
    defaults = dict(motion='translation', trials=3, noise=1., contrast=12., mean=40.,
                    magnitude=None, preprocess='dog', no_check=False, seed=0,
                    max_chi2=None, bias_floor=None)
    defaults.update(overrides)
    ns = argparse.Namespace(**defaults)
    if ns.magnitude is None:
        ns.magnitude = .01 if ns.motion == 'rotation' else .005
    return ns


def mask():
    m = np.ones(false_reject.SHAPE, np.uint8)
    m[false_reject.BAND[0]:false_reject.BAND[1]] = 0
    return m


def test_texture_is_band_limited_and_deterministic():
    rng = np.random.default_rng(3)
    f = false_reject.dog_frequencies(rng, 5000)
    assert .04 < np.median(f) < .2 and f.max() < .45
    a = false_reject.Texture(np.random.default_rng(1))
    b = false_reject.Texture(np.random.default_rng(1))
    u, v = np.mgrid[:4, :5].astype(float)
    assert np.array_equal(a(u, v), b(u, v))


def test_true_translation_passes_and_rotation_is_rejected():
    rows = [false_reject.trial('translation', i, args(), mask()) for i in range(3)]
    summary = false_reject.summarize(rows)
    assert summary['outcomes'] == {'accepted': 3}, summary
    rows = [false_reject.trial('rotation', i, args(motion='rotation'), mask()) for i in range(3)]
    assert false_reject.summarize(rows)['model_mismatch_rate'] == 1.


def test_diagnosis_keeps_statistics_with_the_check_off(tmp_path):
    s = Sequence(generate(tmp_path / 'synthetic', frames=6))
    report = run.evaluate(s, 1, 'dog', check=False)
    out = diagnose.pairs_summary(report['pairs'], .5)
    assert out['pairs_with_regional_statistics'] >= 4
    assert out['share_above_limit'] == 0.
