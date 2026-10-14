"""Known-transform sequence acceptance: no cached algorithm serves as truth."""
import numpy as np
from vision_metrology.experimental import SimilarityEstimator
from anchors import AnchoredSequence, matrix, parts, corner_distance


def texture(transform, frame):
    y, x = np.mgrid[:97, :161].astype(float)
    xy = np.linalg.inv(transform) @ np.array([x.ravel(), y.ravel(), np.ones(x.size)])
    x, y = xy[:2]
    values = (100 + 25*np.sin(x*.17+y*.13) + 20*np.cos(x*.08-y*.21)
              + 15*np.sin(x*.31+y*.07) + 10*np.cos(x*.03+y*.39))
    values += np.random.default_rng(7919*frame+1).uniform(-2, 2, values.shape)
    return values.reshape(97, 161).astype(np.float32)


def test_native_long_sequence_anchors_reduce_known_truth_error():
    chain = AnchoredSequence(0, (97, 161))
    identity = np.eye(3)
    seed = texture(identity, 0)
    previous = seed
    e, anchors = SimilarityEstimator(), SimilarityEstimator()
    raw_errors, anchored_errors = [], []
    for frame in range(1, 65):
        truth = matrix((frame*.15, -frame*.025, frame*.00015, np.exp(frame*.00008)))
        current = texture(truth, frame)
        pair = e.estimate(previous, current)
        assert pair.status == 'accepted'
        chain.advance({'status': pair.status, 'transform': pair.transform})
        if frame % 8 == 0:
            anchor = anchors.estimate(seed, current, prediction=parts(chain.pose))
            assert anchor.status == 'accepted'
            correction = chain.constrain({'status': anchor.status, 'transform': anchor.transform})
            assert correction is not None
            raw_errors.append(corner_distance(chain.raw, truth, (97, 161)))
            anchored_errors.append(corner_distance(chain.pose, truth, (97, 161)))
        previous = current
    assert max(anchored_errors) < .1
    assert np.mean(anchored_errors) < np.mean(raw_errors), (raw_errors, anchored_errors)
    before = chain.pose.copy()
    assert chain.constrain({'status': 'insufficient_support', 'transform': None}) is None
    assert np.array_equal(chain.pose, before)


def test_unsupported_pair_cannot_create_pose():
    import pytest
    chain = AnchoredSequence(0, (97, 161))
    with pytest.raises(ValueError):
        chain.advance({'status': 'model_mismatch', 'transform': (0, 0, 0, 1)})
