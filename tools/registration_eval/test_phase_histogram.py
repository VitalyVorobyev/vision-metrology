"""The locking diagnostic on synthetic manifests; no private data needed."""
import json

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import phase_histogram as ph
from manifest import Sequence
from synthetic import BEAD_HALF, BEAD_Y, generate


def smooth_field(tmp_path, frames=10, blur=3.):
    """A manifest of smooth random fields: many edges at unrelated sub-pixel positions."""
    rng = np.random.default_rng(3)
    entries = []
    for i in range(frames):
        field = gaussian_filter(rng.normal(size=(97, 161)), blur)
        field = 128 + 90*field/np.abs(field).max() + rng.normal(scale=1, size=field.shape)
        Image.fromarray(np.clip(np.rint(field), 0, 255).astype(np.uint8)).save(tmp_path / f'{i}.png')
        entries.append({'image': f'{i}.png'})
    path = tmp_path / 'sequence.json'
    path.write_text(json.dumps({'frames': entries}))
    return Sequence(path)


def uniform_locator(pull):
    """Axis-aligned edgels at uniform positions, each pulled `pull` of the way to its pixel centre."""
    rng = np.random.default_rng(11)

    def locate(image):
        h, w = image.shape
        x, y = rng.uniform(8, w-8, 4000), rng.uniform(8, h-8, 4000)
        x = x + pull*(np.round(x) - x)
        return x, np.round(y), np.ones_like(x), np.zeros_like(x)
    return locate


def test_the_row_crossing_removes_the_oblique_normal_offset():
    angle = np.radians(20)
    x, y, nx, ny = map(np.array, ([10.3], [20.2], [np.cos(angle)], [np.sin(angle)]))
    coordinate, axis, keep = ph.axis_coordinates(x, y, nx, ny, 25.)
    assert keep.all() and axis.tolist() == [0]
    assert np.isclose(coordinate[0], 10.3 + .2*np.tan(angle))
    coordinate, axis, _ = ph.axis_coordinates(y, x, ny, nx, 25.)
    assert axis.tolist() == [1] and np.isclose(coordinate[0], 10.3 + .2*np.tan(angle))
    assert not ph.axis_coordinates(x, y, np.array([1.]), np.array([1.]), 25.)[2].any()


def test_a_locked_locator_gives_a_large_index_and_an_unlocked_one_near_one(tmp_path):
    s = smooth_field(tmp_path, frames=5)
    unlocked = ph.evaluate(s, 'none', locate=uniform_locator(0.))['pooled']
    locked = ph.evaluate(s, 'none', locate=uniform_locator(.3))['pooled']
    assert unlocked['edges'] == locked['edges'] and unlocked['locking_index'] < 1.15, unlocked
    # Pulling 30% of the way to the pixel centre empties the bins around ±0.5 px.
    assert locked['locking_index'] is None or locked['locking_index'] > 10, locked
    assert locked['normalized'][0] > 1.3 and locked['normalized'][4] < .1


def test_the_reference_and_the_detector_are_flat_on_smooth_data(tmp_path):
    s = smooth_field(tmp_path)
    reference = ph.evaluate(s, 'none', refine='dog')['pooled']
    assert reference['edges'] > 5000 and reference['locking_index'] < 1.3, reference
    # `Subpix2D::ParabolicAlongNormal` fits the pre-suppression gradient magnitude, so the
    # edgels leave their pixel centres (issue #99).
    detector = ph.evaluate(s, 'none')['pooled']
    assert detector['edges'] > 5000 and detector['locking_index'] < 1.3, detector


def test_the_library_sampler_agrees_with_the_reference(tmp_path):
    s = smooth_field(tmp_path, frames=4)
    reference = ph.evaluate(s, 'none', refine='dog')['pooled']
    sampler = ph.evaluate(s, 'none', refine='sampler')['pooled']
    assert sampler['locking_index'] < 1.3, sampler
    # The sampler drops what the reference reads from clamped pixels near the border.
    assert .9*reference['edges'] < sampler['edges'] <= reference['edges'], (reference, sampler)
    wide = ph.evaluate(s, 'none', refine='sampler', sigma=2.)['pooled']
    assert wide['edges'] < sampler['edges'], 'a wider support reaches the border sooner'


def test_masks_decide_which_edges_count_on_the_generator_sequence(tmp_path):
    s = Sequence(generate(tmp_path / 'synthetic', frames=4))

    def bead_walls(image):
        # One edgel on each bead wall, one on the substrate, one on the frame border.
        x = np.array([80., 80., 80., 0.])
        y = np.array([BEAD_Y - BEAD_HALF, BEAD_Y + BEAD_HALF, 20.3, 50.])
        return x, y, np.zeros(4), np.ones(4)

    visible = ph.evaluate(s, 'visibility', locate=bead_walls)
    valid = ph.evaluate(s, 'validity', locate=bead_walls)
    assert visible['frames'] == 4 and visible['y']['edges'] == 12 and visible['x']['edges'] == 0
    assert valid['y']['edges'] == 4, 'the registration mask excludes the bead and its walls'
    real = ph.evaluate(s, 'visibility')
    assert real['pooled']['edges'] > 1000 and len(real['pooled']['normalized']) == ph.BINS


def test_the_cli_writes_aggregates_only(tmp_path, monkeypatch):
    manifest = generate(tmp_path / 'seq', frames=4)
    out = tmp_path / 'out'
    monkeypatch.setattr('sys.argv', ['phase_histogram.py', '--manifest', str(manifest),
                                     '--out-dir', str(out), '--refine', 'dog'])
    ph.main()
    report = json.loads((out / 'phase_histogram.json').read_text())
    assert set(report['sequences']) == {'seq'} and report['settings']['refine'] == 'dog'
    assert report['aggregate']['counts'] == report['sequences']['seq']['counts']
    assert (out / 'summary.txt').read_text().startswith('locking diagnostic')
