"""Generator checks: determinism, the manifest contract, the truth and the self-check."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.special import expit

import generate
from presets import PRESETS, config
from render import Scenario, render
from scene import apply, ray_hit, se2
from validate import self_check

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'registration_eval'))
from manifest import Sequence  # noqa: E402


def digest(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp('tiny')
    path = generate.generate('tiny', out)
    return path, json.loads(path.read_text()), json.loads((out / 'truth.json').read_text())


def test_byte_identical_across_runs(tiny, tmp_path):
    path, _, _ = tiny
    again = generate.generate('tiny', tmp_path / 'tiny')
    first, second = digest(path.parent), digest(again.parent)
    assert first == second and len(first) > 20
    other = generate.generate('tiny', tmp_path / 'seed', seed=87)
    assert digest(other.parent)['frames/v0/0000.png'] != first['frames/v0/0000.png']


def test_manifest_reads_with_the_registration_evaluator(tiny):
    path, doc, truth = tiny
    seq = Sequence(path)
    views = [f['view'] for f in doc['frames']]
    assert views == sorted(views) and len(seq) == len(doc['frames'])
    dropped = [i for i, f in enumerate(doc['frames']) if f['image'] is None]
    assert dropped and all(seq.group(i) is None for i in dropped)
    assert [doc['frames'][i]['time'] for i in dropped] == [3]
    # View 0 is a similarity, so its image motion is known; view 1 is projective.
    motion = doc['truth']['motion']
    v0 = [i for i, f in enumerate(doc['frames']) if f['view'] == 0 and f['image'] is not None]
    v1 = [i for i, f in enumerate(doc['frames']) if f['view'] == 1]
    assert all(motion[i] is not None for i in v0) and all(motion[i] is None for i in v1)
    scn = Scenario(config('tiny'))
    a, b = v0[0], v0[1]
    p = np.array([[3., 4.], [-7., 2.]])
    ta, tb = doc['frames'][a]['time'], doc['frames'][b]['time']
    pa = apply(scn.image_map(ta, 0) @ se2(scn.poses[ta]), p)
    pb = apply(scn.image_map(tb, 0) @ se2(scn.poses[tb]), p)
    assert np.allclose(apply(seq.truth_map(a, b), pa), pb, atol=1e-4)
    assert seq.mask(v0[0]).sum() < seq.visible(v0[0]).sum()
    assert doc['rig']['poses'][0] == truth['poses'][0] and doc['rig']['poses'][2] != truth['poses'][2]


def test_truth_stations_visibility_and_annotations(tiny):
    path, doc, truth = tiny
    st = truth['stations']
    n = len(st['s'])
    assert all(len(row[v]) == n for row in truth['visibility'] for v in range(2) if row[v] is not None)
    assert truth['visibility'][0][1] is None and truth['visibility'][3][0] is None
    s = np.array(st['s'])
    laid = truth['laid']
    assert laid == sorted(laid) and laid[0] < laid[-1]
    for t, row in enumerate(truth['visibility']):
        for codes in filter(None, row):
            c = np.array(list(codes))
            assert set(c[s > laid[t]]) <= {'n'} and set(c[s <= laid[t]]) <= set('0123g')
            assert set(c[np.array(st['in_gap']) & (s <= laid[t])]) <= {'g'}
    assert truth['moves_on_substrate'] and truth['slips'][5][1] == pytest.approx(0.5)
    assert truth['occlusions'] and truth['image_offsets'][5][1] == [3.0, 0.0]
    # View 0 has scale 1, so the walls sit at their half widths from the centreline point.
    view0 = {i for i, f in enumerate(doc['frames']) if f['view'] == 0}
    checked = 0
    for a in doc['annotations']:
        if a['frame'] not in view0:
            continue
        point, normal, left, right = (np.array(a[k]) for k in ('point', 'normal', 'left', 'right'))
        k = a['station']
        assert np.allclose(right - point, st['right'][k]*normal, atol=1e-3)
        assert np.allclose(point - left, st['left'][k]*normal, atol=1e-3)
        checked += 1
    assert checked > 0


def test_fusion_single_matches_the_issue_profile(tmp_path):
    generate.generate('fusion-single', tmp_path)
    truth = json.loads((tmp_path / 'truth.json').read_text())
    codes = truth['visibility'][0][0]
    assert len(codes) == 60
    assert codes[25:29] == '0000' and set(codes[45:]) == {'2'}
    assert set(codes[:25] + codes[29:45]) == {'3'}
    image = np.asarray(Image.open(tmp_path / 'frames/v0/0000.png')).astype(float)

    def S(z):
        return expit(z/0.5)
    for k in (0, 17, 44):
        w = 12 + 0.04*k
        x = 30 + k
        u = np.linspace(-0.5, 0.5, 401)
        for y in range(4, 93):
            xi = -(y + u - 48)
            i = (20 + 180*S(xi + w)*S(w - xi) - 140*S(2*(xi - 2))*S(2*(5 - xi))
                 + 40*S((xi - 23)/4)*S((35 - xi)/4))
            assert abs(image[y, x] - i.mean()) <= 1.0, (k, y)


def test_every_preset_builds_and_renders(tmp_path):
    for name in PRESETS:
        cfg = config(name, quadrature=1)
        scn = Scenario(cfg)
        for v in range(len(scn.views)):
            t = int(np.flatnonzero(scn.recorded[:, v] & ~scn.dropped[:, v])[0])
            image, blank, valid, visible, _ = render(scn, t, v)
            assert image.shape == tuple(scn.views[v]['shape']) and (blank is None) == (not cfg['blank'])
            assert valid.sum() <= visible.sum()


def test_presets_carry_their_cases():
    fm = Scenario(config('fusion-multiview'))
    gap = fm.cfg['ribbon']['gap']
    assert gap[1] - gap[0] == pytest.approx(3*fm.cfg['speed'])
    assert fm.recorded[:, 1].sum() < fm.frames and not fm.dropped.any()
    bridged = Scenario(config('rig-bridged'))
    assert bridged.dropped[20].all() and bridged.dropped.sum() == 3
    single = Scenario(config('rig-single-view'))
    assert list(single.dropped[20]) == [False, True, True]
    gross = Scenario(config('rig-gross'))
    step = np.diff(gross.offsets[:, 2], axis=0)
    assert np.allclose(step[15 - 1], [2.5, -1.5]) and np.allclose(step[:14], 0) and np.allclose(step[24:], 0)
    th = [p[2] for p in gross.poses]
    assert 0 < abs(th[-1] - th[0]) < np.radians(30) and np.ptp(np.diff(th)) > 0
    moved = Scenario(config('fusion-moved'))
    assert moved.slips[-1][1] > 5
    reg = Scenario(config('fusion-regerr'))
    assert reg.given[0] == reg.poses[0] and reg.given[10] != reg.poses[10]
    disc = Scenario(config('cap-half-disc'))
    rim = disc.ribbon.rim('end', disc.laid[0])
    centre = disc.curve.point(disc.laid[0])
    for a in np.radians([10, 60, 90, 150]):
        assert ray_hit(rim, centre, [np.sin(a), -np.cos(a)]) == pytest.approx(10, abs=1e-3)
    flat = Scenario(config('cap-flat'))
    assert np.ptp(flat.ribbon.rim('end', flat.laid[0])[:, 0]) < 1e-9


def test_generator_self_check():
    try:
        import vision_metrology
        sampler = hasattr(vision_metrology, 'ProfileSampler')
    except ImportError:
        sampler = False
    errors = self_check(use_sampler=sampler)
    for (view, kind, method, wall), (worst, rms, count) in errors.items():
        assert count > 100
        bound = {('float', 'numpy'): 0.002, ('float', 'sampler'): 0.01}.get((kind, method), 0.05)
        assert worst < bound, (view, kind, method, wall, worst)


def test_output_inside_the_repository_is_refused():
    with pytest.raises(SystemExit):
        generate.check_out_dir(generate.REPO / 'tools' / 'out')
    generate.check_out_dir(generate.REPO / 'data' / 'bench')
