"""Synthetic adapter checks; optional private-data read-only smoke test."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

from dataset import Dataset, bandpass, decode
from run import local_sections, transported


def test_transport_preserves_independent_walls_and_reports_gaps():
    y, x = np.mgrid[:97, :161].astype(float)
    left = 42 + 4*np.sin(y*.07)
    right = 105 + 5*np.cos(y*.08)
    image = (25 + 170*.5*(erf((x-left)/1.2)-erf((x-right)/1.2))).astype(np.float32)
    image[43:54] = 25
    sections = []
    for row in (20, 35, 48, 65, 80):
        l, r = left[row, 0], right[row, 0]
        sections.append({'s_px': row, 'normal': [1, 0], 'left': [l, row],
                         'right': [r, row], 'center': [(l+r)/2, row], 'status': 'measured'})
    ribbon = {'walls': [{'points': [s['left'] for s in sections]},
                        {'points': [s['right'] for s in sections]}],
              'derived': {'centerline': [s['center'] for s in sections], 'sections': sections}}
    translated = transported(ribbon, [.7, -.3])
    assert translated['walls'][0]['points'][0] != ribbon['walls'][0]['points'][0]
    for original, moved in zip(sections, translated['derived']['sections']):
        assert np.allclose(np.array(moved['left'])-original['left'], [.7, -.3])
        assert np.allclose(np.array(moved['right'])-original['right'], [.7, -.3])
        assert np.allclose(np.array(moved['right'])-moved['left'],
                           np.array(original['right'])-original['left'])
    measured = local_sections(image, ribbon, np.zeros(image.shape, dtype=bool))
    for i, section in enumerate(measured):
        for side in ('left', 'right'):
            if i == 2:
                assert section[side]['reject'] == 'no_edge'
            else:
                assert abs(section[side]['normal_deviation_px']) < .1


def test_normalized_filter_excludes_masked_values():
    image = np.sin(np.indices((41, 51))[1]*.1).astype(np.float32)
    valid = np.ones(image.shape, dtype=bool)
    valid[10:30, 15:35] = False
    other = image.copy()
    other[~valid] = np.nan
    assert np.array_equal(bandpass(image, valid), bandpass(other, valid))
    assert np.array_equal(decode([2, 3, 1], (2, 3)), [[False, False, True], [True, True, False]])


def test_private_adapter_is_read_only():
    root = os.environ.get('VM_RTVT_DATA')
    if not root or not (Path(root)/'pixel-manifest.json').exists():
        pytest.skip('private rtvt-pano dataset absent; set VM_RTVT_DATA to opt in')
    db = Path(root)/'pixel-labels/labels.sqlite'
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    dataset = Dataset(root, '15074')
    frame, camera, _, provenance = dataset.seed()
    a, mask = dataset.prepare(frame, camera)
    assert a.shape == mask.shape == (97, 320)
    assert mask.dtype == np.uint8
    assert provenance['geometry'] in ('reviewed_annotation', 'generated_cached_ribbon')
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
