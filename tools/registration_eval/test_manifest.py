"""Manifest reader, synthetic generator and evaluator checks; no private data needed."""
import json

import numpy as np
import pytest
from PIL import Image
from scipy.special import erf

import run
import similarity_run
from manifest import Sequence, bandpass
from synthetic import BEAD_HALF, BEAD_Y, BLOCK, generate, pose


def write(tmp_path, doc, images=2, shape=(9, 12)):
    for i in range(images):
        Image.fromarray(np.full(shape, 10*i, np.uint8)).save(tmp_path / f'{i}.png')
    path = tmp_path / 'sequence.json'
    path.write_text(json.dumps(doc))
    return path


def test_reader_defaults_masks_groups_and_unassigned_frames(tmp_path):
    mask = np.zeros((9, 12), np.uint8)
    mask[2:7, 3:9] = 255
    Image.fromarray(mask).save(tmp_path / 'mask.png')
    np.save(tmp_path / 'f.npy', np.ones((9, 12), np.float32))
    path = write(tmp_path, {'frames': [{'image': '0.png'}, {'image': '1.png', 'mask': 'mask.png'},
                                       {'image': None, 'group': None}, {'image': 'f.npy', 'group': 'b'}]})
    s = Sequence(path)
    assert s.name == tmp_path.name and s.split is None and len(s) == 4 and s.shape == (9, 12)
    assert [s.group(i) for i in range(4)] == [0, 0, None, 'b']
    assert s.mask(0).all() and s.visible(1).all()
    assert s.mask(1).sum() == 30 and s.mask(1)[2, 3] and not s.mask(1)[1, 3]
    prepared, valid = s.prepare(1, 'none')
    assert prepared is s.image(1) and valid.dtype == np.uint8 and set(np.unique(valid)) == {0, 1}
    assert s.image(3).dtype == np.float32
    with pytest.raises(ValueError):
        s.image(2)
    assert s.truth_map(0, 1) is None and s.comparison_shift(0) is None


@pytest.mark.parametrize('doc, message', [
    ({'frames': []}, 'non-empty'),
    ({'frames': [{'image': '../0.png'}]}, 'leave'),
    ({'frames': [{'image': None}]}, 'unassigned'),
    ({'version': 2, 'frames': [{'image': '0.png'}]}, 'version'),
    ({'convention': 'pixel_corners', 'frames': [{'image': '0.png'}]}, 'pixel-centre'),
    ({'split': 'test', 'frames': [{'image': '0.png'}]}, 'split'),
    ({'frames': [{'image': '0.png'}], 'seed': {'frame': 3, 'polyline': [[0, 0], [1, 1]]}}, 'seed.frame'),
    ({'frames': [{'image': '0.png'}], 'seed': {'frame': 0, 'polyline': [[0, 0]]}}, 'polyline'),
    ({'frames': [{'image': '0.png'}], 'truth': {'motion': []}}, 'truth.motion'),
    ({'frames': [{'image': '0.png'}, {'image': '1.png'}], 'comparison': {'shifts': []}}, 'comparison'),
])
def test_reader_rejects_malformed_manifests(tmp_path, doc, message):
    with pytest.raises(ValueError, match=message):
        Sequence(write(tmp_path, doc))


def test_reader_rejects_mismatched_shapes(tmp_path):
    Image.fromarray(np.zeros((5, 5), np.uint8)).save(tmp_path / 'small.png')
    s = Sequence(write(tmp_path, {'frames': [{'image': '0.png'}, {'image': 'small.png'}]}))
    with pytest.raises(ValueError, match='shape'):
        s.image(1)


def test_normalized_filter_excludes_masked_values():
    image = np.sin(np.indices((41, 51))[1]*.1).astype(np.float32)
    valid = np.ones(image.shape, dtype=bool)
    valid[10:30, 15:35] = False
    other = image.copy()
    other[~valid] = np.nan
    assert np.array_equal(bandpass(image, valid), bandpass(other, valid))


def test_generator_writes_masks_truth_and_a_handover(tmp_path):
    s = Sequence(generate(tmp_path / 'synthetic', frames=8))
    assert len(s) == 8 and s.shape == (97, 161) and s.split == 'development'
    assert [s.group(i) for i in range(8)] == [0]*4 + [1]*4
    assert not s.mask(0)[BLOCK].any() and not s.visible(5)[BLOCK].any() and s.visible(5)[60:].all()
    assert not s.mask(0)[int(BEAD_Y)].any(), 'the bead must be excluded from registration'
    truth = s.truth_map(2, 3)
    assert np.isclose(np.arctan2(truth[1, 0], truth[0, 0]), pose(3)[2] - pose(2)[2])
    assert np.isclose(np.hypot(truth[0, 0], truth[1, 0]), pose(3)[3] / pose(2)[3])
    left, right = s.truth['boundaries'][0]
    assert np.allclose(np.asarray(right)[:, 1] - np.asarray(left)[:, 1], 2*BEAD_HALF)


def test_evaluators_recover_known_motion_and_report_the_handover(tmp_path):
    s = Sequence(generate(tmp_path / 'synthetic', frames=20))
    translation = run.evaluate(s, 1, 'dog')['summary']
    assert translation['pairs'] == 19 and translation['outcomes'].get('camera_handover') == 1
    summary = similarity_run.evaluate([s], None, 8, .25, 'dog')['summary']
    assert summary['pairs'] == 19 and summary['outcomes']['camera_handover'] == 1
    assert summary['accepted'] == 18, summary['outcomes']
    assert summary['truth_error_px_max'] < .1, summary
    assert summary['anchor_outcomes'].get('accepted', 0) >= 1, summary['anchor_outcomes']


def test_propagation_transports_the_seed_and_measures_both_walls(tmp_path):
    s = Sequence(generate(tmp_path / 'synthetic', frames=10))
    out = tmp_path / 'out'
    out.mkdir()
    result = run.propagate(s, out, 10, 'dog')
    forward = result['directions']['1']
    assert result['directions']['-1']['stop'] == 'sequence_end'
    assert forward['stop'] == 'camera_handover' and len(forward['frames']) == 4
    for row in forward['frames']:
        deviations = [abs(section[side]['normal_deviation_px']) for section in row['measured_sections']
                      for side in ('left', 'right') if 'point' in section[side]]
        assert len(deviations) >= 20 and max(deviations) < .3, deviations


def test_transport_preserves_independent_walls_and_reports_gaps():
    y, x = np.mgrid[:97, :161].astype(float)
    left = 42 + 4*np.sin(y*.07)
    right = 105 + 5*np.cos(y*.08)
    image = (25 + 170*.5*(erf((x-left)/1.2)-erf((x-right)/1.2))).astype(np.float32)
    image[43:54] = 25
    sections = []
    for row in (20, 35, 48, 65, 80):
        l, r = left[row, 0], right[row, 0]
        sections.append({'s': row, 'normal': [1, 0], 'left': [l, row],
                         'right': [r, row], 'center': [(l+r)/2, row]})
    seed = {'frame': 0, 'polyline': [s['center'] for s in sections], 'sections': sections}
    moved = run.transported(seed, [.7, -.3])
    assert moved['polyline'][0] != seed['polyline'][0]
    for original, section in zip(sections, moved['sections']):
        assert np.allclose(np.array(section['left'])-original['left'], [.7, -.3])
        assert np.allclose(np.array(section['right'])-section['left'],
                           np.array(original['right'])-original['left'])
    measured = run.local_sections(image, seed, np.ones(image.shape, dtype=bool))
    for i, section in enumerate(measured):
        for side in ('left', 'right'):
            if i == 2:
                assert section[side]['reject'] == 'no_edge'
            else:
                assert abs(section[side]['normal_deviation_px']) < .1
