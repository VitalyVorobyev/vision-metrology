"""HPatches evaluator geometry, on synthetic images; the dataset itself is not needed."""
import numpy as np

import hpatches as hp


def test_local_models_of_a_translation_and_a_similarity():
    t, parts, t_dev, s_dev = hp.local_models(np.array([[1, 0, 2.5], [0, 1, -1.], [0, 0, 1]]),
                                             (10, 20))
    assert np.allclose(t, [2.5, -1.]) and t_dev < 1e-9 and s_dev < 1e-9
    a = .01
    h = np.array([[np.cos(a), -np.sin(a), 3.], [np.sin(a), np.cos(a), 1.], [0, 0, 1]])
    _, parts, t_dev, s_dev = hp.local_models(h, (0, 0))
    assert t_dev > 1 and s_dev < 1e-9 and np.isclose(parts[2], a)


def test_rectified_crop_follows_the_known_map():
    y, x = np.mgrid[:300, :500].astype(float)
    image = 100 + 30*np.sin(x*.21 + y*.13) + 20*np.cos(x*.07 - y*.19)
    m = hp.known_map('none', np.random.default_rng(1), 0.)
    current, valid = hp.rectified(image, np.eye(3), (40, 60), m)
    assert valid.all()
    p = np.array([100., 50.])
    q = m[:2, :2] @ p + m[:2, 2]
    expected = image[int(p[1]) + 60, int(p[0]) + 40]
    assert abs(hp.map_coordinates(current, [[q[1]], [q[0]]], order=3)[0] - expected) < .5
