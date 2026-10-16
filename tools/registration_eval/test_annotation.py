"""Cross-section annotation: stations, profiles, blind passes, storage and evaluation.

Everything runs headlessly on the synthetic sequence, whose walls are known.
"""
import json
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

import annotate
import annotation
import annotations_eval
from manifest import Sequence
from synthetic import BEAD_HALF, BEAD_Y, generate


def crossing(station, normal, polyline):
    """The signed offset along the normal where the line meets the polyline."""
    p, n = np.asarray(station, float), np.asarray(normal, float)
    points = np.asarray(polyline, float)
    for a, b in zip(points[:-1], points[1:]):
        d = b - a
        system = np.array([[n[0], -d[0]], [n[1], -d[1]]])
        if abs(np.linalg.det(system)) < 1e-12:
            continue
        t, u = np.linalg.solve(system, a - p)
        if -1e-9 <= u <= 1 + 1e-9:
            return float(t)
    raise AssertionError('no crossing')


def truth_walls(sequence, entry):
    left, right = sequence.truth['boundaries'][entry['frame']]
    return (crossing(entry['station'], entry['normal'], left),
            crossing(entry['station'], entry['normal'], right))


@pytest.fixture
def synthetic(tmp_path):
    """A synthetic manifest that names its annotations file, and a curve on frames 0–3."""
    path = generate(tmp_path / 'seq', frames=8)
    doc = json.loads(path.read_text())
    doc['annotations'] = 'annotations.json'
    path.write_text(json.dumps(doc))
    sequence = Sequence(path)
    curves = []
    for frame in range(4):
        left, right = (np.asarray(b) for b in sequence.truth['boundaries'][frame])
        curves.append({'frame': frame, 'polyline': ((left + right)/2).tolist()})
    curves_path = tmp_path / 'seq' / 'curves.json'
    curves_path.write_text(json.dumps({'curves': curves}))
    return sequence, annotation.load_curves(sequence, curves_path)


def annotate_all(sequence, session, annotator='a', noise=lambda entry, pass_: (0., 0.)):
    """Answer every task with the known walls plus `noise`, in the session's order."""
    while (task := session.next(annotator)) is not None:
        index, pass_ = task
        entry = session.sections[index]
        (left, right), (dl, dr) = truth_walls(sequence, entry), noise(entry, pass_)
        session.annotate(annotator, index, pass_, left + dl, right + dr)


def test_stations_are_uniform_seeded_and_normal_to_the_curve(synthetic):
    sequence, curves = synthetic
    seed_curve = annotation.load_curves(sequence)
    assert list(seed_curve) == [0]
    first = annotation.uniform_sections(seed_curve, 10, seed=3)
    assert first == annotation.uniform_sections(seed_curve, 10, seed=3)
    assert first != annotation.uniform_sections(seed_curve, 10, seed=4)
    length = annotation.arc_lengths(seed_curve[0])[-1]
    for k, entry in enumerate(first):
        assert k*length/10 <= entry['s'] < (k + 1)*length/10
        assert np.allclose(entry['normal'], [0, 1]) and np.isclose(entry['station'][1], BEAD_Y)
    spread = annotation.uniform_sections(curves, 40, seed=0)
    assert sorted({e['frame'] for e in spread}) == [0, 1, 2, 3]
    for entry in spread:
        polyline = curves[entry['frame']]
        s = annotation.project(polyline, entry['station'])
        assert abs(s - entry['s']) < 1e-3
        tangent = annotation.point_at(polyline, s + 1) - annotation.point_at(polyline, s - 1)
        assert abs(np.dot(tangent, entry['normal'])) < 1e-6 and np.isclose(np.hypot(*entry['normal']), 1)
    span = annotation.span_sections(curves, 2, 30., 40., 4, seed=0)
    assert len(span) == 4 and all(30 <= e['s'] <= 40 and e['source'] == 'span' for e in span)


def test_profile_shows_both_walls_and_marks_hidden_samples(synthetic):
    sequence, _ = synthetic
    image, visible = sequence.image(0), sequence.visible(0)
    for method in ('cubic', 'sampler'):
        p = annotation.profile(image, visible, [60., BEAD_Y], [0., 1.], method=method)
        assert np.allclose(np.diff(p['offsets']), annotation.STEP) and p['valid'].all()
        t, d = p['offsets'], p['derivative']
        rising, falling = t[np.argmax(np.where(t < 0, d, -np.inf))], t[np.argmin(np.where(t > 0, d, np.inf))]
        assert abs(rising + BEAD_HALF) < .2 and abs(falling - BEAD_HALF) < .2, (method, rising, falling)
    # Upwards from below the stationary block, which visibility excludes.
    p = annotation.profile(image, visible, [130., 30.], [0., -1.], half_length=8)
    assert p['valid'][p['offsets'] < 2].all() and not p['valid'][p['offsets'] > 6].any()
    assert np.isnan(p['values'][~p['valid']]).all()
    pixels, origin = annotation.crop(image, [60., BEAD_Y], 10)
    assert pixels.shape == (21, 21) and pixels.dtype == np.uint8 and origin == (50, 38)


def test_session_runs_two_blind_shuffled_passes_and_resumes(synthetic, tmp_path):
    sequence, curves = synthetic
    path = sequence.annotations_path
    session = annotation.Session.create(path, annotation.uniform_sections(curves, 6, 1), {'seed': 1})
    with pytest.raises(FileExistsError):
        annotation.Session.create(path, [], {'seed': 1})
    assert session.order(1) != session.order(2)
    app = annotate.App(sequence, session, 'a', curves)
    seen = []
    for _ in range(12):
        task = app.task()
        assert not {'left', 'right', 'pass', 'annotations'} & set(task), 'a task shows no earlier mark'
        seen.append(task['task'])
        index, pass_ = (int(v) for v in task['task'].split('.'))
        if len(seen) == 3:
            with pytest.raises(ValueError, match='pending'):
                session.annotate('a', index, 3, 0., 1.)
            pending = next(i for i in range(6) if (i, 1) not in session.done('a'))
            with pytest.raises(ValueError, match='follows'):
                session.annotate('a', pending, 2, 0., 1.)
        if len(seen) == 7:
            resumed = annotation.Session.open(path, len(sequence))
            assert resumed.next('a') == session.next('a') and resumed.progress('a') == (6, 12)
        left, right = truth_walls(sequence, session.sections[index])
        app.submit({'task': task['task'], 'left': left, 'right': None if index == 0 else right})
    assert [t.endswith('.1') for t in seen] == [True]*6 + [False]*6
    assert sorted(seen[:6]) == sorted(t[:-1] + '1' for t in seen[6:])
    assert app.task()['done'] and app.task()['progress'] == [12, 12]
    assert annotation.Session.open(path).progress('b') == (0, 12), 'progress is per annotator'
    order = session.order(2)
    assert session.add(annotation.section(1, curves[1], 33., 'user'))
    assert not session.add(annotation.section(1, curves[1], 33., 'user'))
    assert [i for i in session.order(2) if i < 6] == order, 'adding keeps the order'
    assert session.next('a') == (6, 1) and annotation.Session.open(path).progress('a') == (12, 14)


def test_annotations_round_trip_through_the_manifest(synthetic, tmp_path):
    sequence, curves = synthetic
    assert sequence.annotations == [], 'a named file that does not exist yet reads as empty'
    session = annotation.Session.create(sequence.annotations_path,
                                        annotation.uniform_sections(curves, 5, 0), {'seed': 0})
    annotate_all(sequence, session)
    reread = Sequence(sequence.path)
    assert reread.annotations == session.doc['annotations'] and len(reread.annotations) == 10
    for entry in reread.annotations:
        assert set(entry) == set(annotation.FIELDS) and entry['left'] < entry['right']
    inline = json.loads(sequence.path.read_text())
    inline['annotations'] = reread.annotations
    sequence.path.write_text(json.dumps(inline))
    assert Sequence(sequence.path).annotations == reread.annotations
    exact = {'frame': 0, 'station': 3, 'point': [60., BEAD_Y], 'normal': [0., 1.],
             'left': [60., BEAD_Y - BEAD_HALF], 'right': [60., BEAD_Y + BEAD_HALF]}
    inline['annotations'] = reread.annotations + [exact]
    sequence.path.write_text(json.dumps(inline))
    mixed = Sequence(sequence.path)
    assert mixed.annotations[-1] == exact and mixed.annotated == reread.annotations, \
        'a generator cross-section is carried through, not scored'
    good = reread.annotations[0]
    for bad, message in [({'left': 5.}, 'before right'), ({'left_visible': False}, 'hidden'),
                         ({'pass': 3}, 'pass'), ({'normal': [1., 1.]}, 'unit'),
                         ({'frame': 99}, 'frame index')]:
        inline['annotations'] = [{**good, **bad}]
        sequence.path.write_text(json.dumps(inline))
        with pytest.raises(ValueError, match=message):
            Sequence(sequence.path)
    inline['annotations'] = '../elsewhere.json'
    sequence.path.write_text(json.dumps(inline))
    with pytest.raises(ValueError, match='leave'):
        Sequence(sequence.path)


def results_from(sequence, annotations, shift=(0., 0.), as_offsets=False):
    """An operator that reports the known walls, moved by `shift` along the normal."""
    rows = []
    for ref in annotations_eval.references(annotations).values():
        left, right = truth_walls(sequence, ref)
        left, right = left + shift[0], right + shift[1]
        if as_offsets:
            rows.append({'frame': ref['frame'], 'station': ref['station'], 'normal': ref['normal'],
                         'left': left, 'right': right})
        else:
            p, n = np.asarray(ref['station']), np.asarray(ref['normal'])
            rows.append({'frame': ref['frame'], 'station': (p + [.1, 0]).tolist(),
                         'left': (p + left*n).tolist(), 'right': (p + right*n).tolist()})
    return rows


def test_evaluation_is_exact_on_perfect_annotations_and_finds_shifts(synthetic):
    sequence, curves = synthetic
    session = annotation.Session.create(sequence.annotations_path,
                                        annotation.uniform_sections(curves, 20, 2), {'seed': 2})
    annotate_all(sequence, session)
    annotations = session.doc['annotations']
    report = annotations_eval.evaluate(annotations, results_from(sequence, annotations))
    assert report['sections'] == report['matched'] == 20
    for row in (report['walls']['left'], report['walls']['right'], report['walls']['both'], report['width']):
        assert row['bias'] == 0 and row['rms'] == 0 and row['sigma'] == 0
    assert report['walls']['both']['n'] == 40 and report['coverage']['left']['rate'] == 1
    assert report['annotator']['pairs'] == 20 and report['annotator']['sigma'] == 0
    shifted = annotations_eval.evaluate(annotations, results_from(sequence, annotations, (.3, .3), True))
    assert shifted['walls']['left']['bias'] == shifted['walls']['right']['bias'] == .3
    assert shifted['walls']['both']['sigma'] == 0 and shifted['width']['bias'] == 0
    wider = annotations_eval.evaluate(annotations, results_from(sequence, annotations, (-.2, .25)))
    assert wider['width']['bias'] == .45 and wider['walls']['left']['bias'] == -.2
    far = annotations_eval.evaluate(annotations, results_from(sequence, annotations), match_px=.05)
    assert far['matched'] == 0 and far['walls']['both']['n'] == 0


def test_evaluation_reports_the_annotator_sigma_and_coverage(synthetic):
    sequence, curves = synthetic
    session = annotation.Session.create(sequence.annotations_path,
                                        annotation.uniform_sections(curves, 30, 4), {'seed': 4})
    rng = np.random.default_rng(0)
    noise = {}

    def jitter(entry, pass_):
        if pass_ == 1:
            return 0., 0.
        noise[annotation.key(entry)] = d = tuple(rng.normal(0, .1, 2))
        return d
    annotate_all(sequence, session, noise=jitter)
    annotations = session.doc['annotations']
    d = np.array(list(noise.values()))
    spread = annotations_eval.annotator_spread(annotations)
    assert np.isclose(spread['sigma'], np.std(d.ravel(), ddof=1)/np.sqrt(2), atol=1e-3)
    assert spread['visibility_agreement'] == 1
    # One right wall hidden in both passes, one disputed; the operator misses one left wall.
    first, second = annotation.key(session.sections[0]), annotation.key(session.sections[1])
    for entry in annotations:
        if annotation.key(entry) == first or (annotation.key(entry) == second and entry['pass'] == 2):
            entry['right'], entry['right_visible'] = None, False
    rows = results_from(sequence, annotations)
    for row in rows:
        if np.allclose(np.asarray(row['station']) - [.1, 0], session.sections[2]['station']):
            row['left'] = None
    report = annotations_eval.evaluate(annotations, rows)
    assert report['coverage']['right'] == {'visible': 28, 'found': 28, 'rate': 1., 'hidden': 1,
                                           'spurious': 1, 'disputed': 1}
    assert report['coverage']['left']['visible'] == 30 and report['coverage']['left']['found'] == 29
    assert report['walls']['both']['n'] == 57
    # The reference is the mean of two passes, so it carries half the pass difference.
    errors = np.r_[-d[:, 0], -d[:, 1]]/2
    assert abs(report['walls']['both']['rms'] - np.sqrt(np.mean(errors**2))) < .02
    assert report['wall_sigma_less_reference_noise'] < report['walls']['both']['sigma']
    assert 'per_section' not in report
    assert len(annotations_eval.evaluate(annotations, rows, per_section=True)['per_section']) == 30


def test_the_page_round_trips_through_loopback(synthetic):
    sequence, curves = synthetic
    session = annotation.Session.create(sequence.annotations_path,
                                        annotation.uniform_sections(curves, 2, 0), {'seed': 0})
    httpd = annotate.server(annotate.App(sequence, session, 'a', curves))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{httpd.server_port}'

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            return response.read().decode()
    try:
        assert 'Cross-section annotation' in call('/')
        task = json.loads(call('/api/task'))
        assert task['progress'] == [0, 4] and len(task['offsets']) == len(task['values'])
        assert task['crop'].startswith('data:image/png;base64,')
        with pytest.raises(urllib.error.HTTPError) as rejected:
            call('/api/annotate', {'task': task['task'], 'left': 2., 'right': 1.})
        assert rejected.value.code == 400
        after = json.loads(call('/api/annotate', {'task': task['task'], 'left': -4., 'right': None}))
        assert after['progress'] == [1, 4]
        overview = json.loads(call('/api/overview?frame=2'))
        assert overview['frames'] == [0, 1, 2, 3]
        added = json.loads(call('/api/station', {'frame': 2, 'point': [70., BEAD_Y + 1]}))
        assert added['added'] and added['progress'] == [1, 6]
    finally:
        httpd.shutdown()
    stored = Sequence(sequence.path).annotations
    assert len(stored) == 1 and stored[0]['left'] == -4 and not stored[0]['right_visible']
    assert annotation.read(sequence.annotations_path)['sections'][-1]['source'] == 'user'


def test_the_evaluation_command_writes_aggregates_and_stations(synthetic, tmp_path, monkeypatch, capsys):
    sequence, curves = synthetic
    session = annotation.Session.create(sequence.annotations_path,
                                        annotation.uniform_sections(curves, 4, 0), {'seed': 0})
    annotate_all(sequence, session)
    stations, results, out = tmp_path / 'stations.json', tmp_path / 'results.json', tmp_path / 'report.json'
    argv = ['annotations_eval.py', '--manifest', str(sequence.path), '--stations-out', str(stations)]
    monkeypatch.setattr(sys, 'argv', argv)
    annotations_eval.main()
    blank = json.loads(stations.read_text())
    assert len(blank['sections']) == 4 and all(r['left'] is None for r in blank['sections'])
    results.write_text(json.dumps({'operator': 'truth', 'sections': results_from(sequence, session.doc['annotations'])}))
    monkeypatch.setattr(sys, 'argv', argv[:3] + ['--results', str(results), '--out', str(out)])
    annotations_eval.main()
    report = json.loads(out.read_text())
    assert report['operator'] == 'truth' and report['walls']['both']['bias'] == 0
    assert 'per_section' not in report and 'station' not in json.dumps(report)
    assert 'annotator σ 0.0' in capsys.readouterr().out
    monkeypatch.setattr(sys, 'argv', argv[:3] + ['--results', str(results), '--source', 'uniform'])
    annotations_eval.main()
    assert '4/4 sections matched' in capsys.readouterr().out
    monkeypatch.setattr(sys, 'argv', argv[:3] + ['--results', str(results), '--source', 'span'])
    with pytest.raises(SystemExit):
        annotations_eval.main()
    doc = json.loads(sequence.path.read_text())
    sequence.path.write_text(json.dumps({**doc, 'split': 'held_out'}))
    monkeypatch.setattr(sys, 'argv', argv[:3] + ['--results', str(results)])
    with pytest.raises(SystemExit):
        annotations_eval.main()
    assert 'held-out' in capsys.readouterr().err
