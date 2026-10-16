"""Translation evaluation over neutral sequence manifests, and seed-segment propagation."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import vision_metrology as vm

from anchors import corner_distance
from manifest import Sequence
from settings import describe_translation, translation_config

FIELDS = ('shift', 'score', 'margin', 'support', 'residual_rms', 'residual_max',
          'gain', 'offset', 'covariance', 'regions', 'region_error', 'status')


def shift_map(shift):
    return np.array([[1., 0., shift[0]], [0., 1., shift[1]], [0., 0., 1.]])


def estimate(estimator, sequence, previous, current, preprocess):
    start = perf_counter()
    a, am = sequence.prepare(previous, preprocess)
    b, bm = sequence.prepare(current, preprocess)
    prepared_ms = 1000 * (perf_counter() - start)
    start = perf_counter()
    result = estimator.estimate(a, b, previous_mask=am, current_mask=bm)
    elapsed_ms = 1000 * (perf_counter() - start)
    row = {k: getattr(result, k) for k in FIELDS} | {
        'previous': previous, 'current': current, 'group': sequence.group(previous),
        'prepare_ms': prepared_ms, 'estimate_python_ms': elapsed_ms}
    truth = sequence.truth_map(previous, current)
    if truth is not None and row['shift'] is not None:
        row['truth_error_px'] = corner_distance(shift_map(row['shift']), truth, sequence.shape)
    return row


def summary(rows):
    accepted = [r for r in rows if r['status'] == 'accepted']
    def percentile(values, q):
        return float(np.percentile(values, q)) if values else None
    return {
        'pairs': len(rows), 'accepted': len(accepted),
        'coverage': len(accepted) / len(rows) if rows else 0,
        'outcomes': dict(Counter(r['status'] for r in rows)),
        'estimate_python_ms_median': percentile([r['estimate_python_ms'] for r in rows if 'estimate_python_ms' in r], 50),
        'estimate_python_ms_p95': percentile([r['estimate_python_ms'] for r in rows if 'estimate_python_ms' in r], 95),
        'residual_filtered_gl_median': percentile([r['residual_rms'] for r in accepted], 50),
        'closure_px_median': percentile([r['closure_px'] for r in accepted if r.get('closure_px') is not None], 50),
        'closure_px_p95': percentile([r['closure_px'] for r in accepted if r.get('closure_px') is not None], 95),
        'comparison_px_median': percentile([r['comparison_px'] for r in accepted if r.get('comparison_px') is not None], 50),
        'truth_error_px_median': percentile([r['truth_error_px'] for r in accepted if 'truth_error_px' in r], 50),
        'truth_error_px_max': percentile([r['truth_error_px'] for r in accepted if 'truth_error_px' in r], 100),
    }


def evaluate(sequence, stride, preprocess, check=True):
    estimator = vm.TranslationEstimator(translation_config(check))
    rows = []
    for frame in range(0, len(sequence)-1, stride):
        group = sequence.group(frame)
        if group is None or sequence.group(frame+1) != group:
            rows.append({'previous': frame, 'current': frame+1, 'status': 'camera_handover'})
            continue
        row = estimate(estimator, sequence, frame, frame+1, preprocess)
        comparison = sequence.comparison_shift(frame)
        if comparison is not None:
            row['comparison_shift'] = comparison
            row['comparison_px'] = (float(np.linalg.norm(np.array(row['shift'])-comparison))
                                    if row['shift'] else None)
        if row['status'] == 'accepted':
            reverse = estimate(estimator, sequence, frame+1, frame, preprocess)
            row['reverse_status'] = reverse['status']
            row['closure_px'] = (float(np.linalg.norm(np.array(row['shift'])+reverse['shift']))
                                 if reverse['status'] == 'accepted' else None)
        rows.append(row)
    result = {'summary': summary(rows), 'pairs': rows}
    if sequence.comparison is not None:
        result['comparison_provenance'] = sequence.comparison.get(
            'provenance', 'another estimator; not motion ground truth')
    return result


def transported(seed, shift):
    """Transport every geometry representation; preserve normals and arc identity."""
    moved = copy.deepcopy(seed)
    def points(values):
        return (np.asarray(values) + shift).tolist() if values else []
    moved['polyline'] = points(moved['polyline'])
    moved['walls'] = [points(wall) for wall in moved.get('walls', [])]
    for section in moved.get('sections', []):
        for key in ('center', 'left', 'right'):
            if section.get(key) is not None:
                section[key] = (np.asarray(section[key]) + shift).tolist()
    return moved


def visible_path(points, visible, shape):
    points = np.asarray(points, dtype=np.float32)
    valid = ((points[:, 0] >= 0) & (points[:, 0] <= shape[1]-1)
             & (points[:, 1] >= 0) & (points[:, 1] <= shape[0]-1))
    for i in np.flatnonzero(valid):
        x, y = np.rint(points[i]).astype(int)
        valid[i] = visible[y, x]
    runs = np.split(np.arange(len(points)), np.flatnonzero(np.diff(valid.astype(int)))+1)
    runs = [run for run in runs if len(run) >= 2 and valid[run[0]]]
    return np.ascontiguousarray(points[max(runs, key=len)]) if runs else None


def local_sections(image, seed, visible):
    """Independent single-edge calipers ±4 px about each transported wall point."""
    results = []
    for section in seed.get('sections', []):
        entry = {'s': section.get('s')}
        normal = np.asarray(section['normal'])
        for side in ('left', 'right'):
            point = section.get(side)
            if point is None or section.get('left') is None or section.get('right') is None:
                entry[side] = {'reject': 'unsupported_reference'}
                continue
            point = np.asarray(point)
            start, end = point - 4*normal, point + 4*normal
            samples = np.linspace(start, end, 17)
            xy = np.rint(samples).astype(int)
            if np.any(xy < 0) or np.any(xy[:, 0] >= image.shape[1]) or np.any(xy[:, 1] >= image.shape[0]):
                entry[side] = {'reject': 'off_image'}
                continue
            if not np.all(visible[xy[:, 1], xy[:, 0]]):
                entry[side] = {'reject': 'occluded'}
                continue
            order = float(np.dot(np.asarray(section['right'])-section['left'], normal))
            rising = (side == 'left') == (order >= 0)
            config = vm.MeasureConfig(threshold=2., select='all',
                                      polarity='rising' if rising else 'falling', off_image='reject')
            caliper = vm.Caliper.strip(tuple(start), tuple(end), half_width=0., config=config)
            try:
                edges = caliper.measure(image)
                if len(edges) != 1:
                    entry[side] = {'reject': 'ambiguous', 'candidates': len(edges)}
                    continue
                edge = edges[0]
                entry[side] = {'point': [edge.x, edge.y],
                               'normal_deviation_px': float(np.dot(np.array([edge.x, edge.y])-point, normal))}
            except vm.MeasureRejected as rejected:
                entry[side] = {'reject': str(rejected.args[0])}
        results.append(entry)
    return results


def walls_of(seed):
    if seed.get('walls'):
        return seed['walls']
    sections = seed.get('sections', [])
    return [[s[side] for s in sections if s.get(side) is not None] for side in ('left', 'right')]


def overlay(sequence, frame, seed, measured, out):
    image = Image.fromarray(sequence.image(frame)).convert('RGB')
    draw = ImageDraw.Draw(image)
    for wall in walls_of(seed):
        if len(wall) > 1:
            draw.line([tuple(p) for p in wall], fill=(255, 190, 40), width=1)
    for section in measured:
        for side in ('left', 'right'):
            if 'point' in section[side]:
                x, y = section[side]['point']
                draw.ellipse((x-1, y-1, x+1, y+1), fill=(70, 255, 140))
    image.resize((3*image.width, 3*image.height)).save(out)


def section_width(section):
    if section.get('width') is not None:
        return section['width']
    if section.get('left') is None or section.get('right') is None:
        return None
    return float(np.linalg.norm(np.asarray(section['right'])-section['left']))


def propagate(sequence, out, max_frames, preprocess):
    seed = sequence.seed
    if seed is None:
        raise ValueError(f'{sequence.name}: propagation needs seed geometry in the manifest')
    start_frame = seed['frame']
    group = sequence.group(start_frame)
    widths = [w for w in map(section_width, seed.get('sections', [])) if w is not None and w > 0]
    if not widths:
        raise ValueError('seed has no measured sections')
    tracker = vm.BeadTracker(vm.BeadConfig(
        polarity='light', min_width=max(1., min(widths)*.5), max_width=max(widths)*1.5,
        spacing=2., min_margin=.1,
        track=vm.BeadCaliper(max_offset=3., half_width=1., threshold=2.),
        measure=vm.BeadCaliper(max_offset=3., half_width=1., threshold=2., off_image='reject')))
    seed_sections = local_sections(sequence.image(start_frame), seed, sequence.visible(start_frame))
    overlay(sequence, start_frame, seed, seed_sections, out / f'seed-{start_frame:05}.png')
    result = {'sequence': sequence.name, 'seed': start_frame, 'group': group,
              'provenance': seed.get('provenance'), 'directions': {}}
    for direction in (-1, 1):
        estimator = vm.TranslationEstimator()
        cumulative = np.zeros(2)
        previous = start_frame
        rows = []
        stop = 'frame_budget'
        for step in range(1, max_frames+1):
            frame = start_frame+direction*step
            if not 0 <= frame < len(sequence):
                stop = 'sequence_end'
                break
            if sequence.group(frame) != group:
                stop = 'camera_handover'
                break
            row = estimate(estimator, sequence, previous, frame, preprocess)
            rows.append(row)
            if row['status'] != 'accepted':
                stop = row['status']
                break
            cumulative += row['shift']
            predicted = transported(seed, cumulative)
            visible = sequence.visible(frame)
            path = visible_path(predicted['polyline'], visible, sequence.shape)
            if path is None:
                stop = 'segment_left_view'
                break
            image = sequence.image(frame)
            start = perf_counter()
            measured = tracker.track(image, path)
            sections = local_sections(image, predicted, visible)
            row['measurement_ms'] = 1000*(perf_counter()-start)
            row['predicted_geometry'] = predicted | {'walls': walls_of(predicted)}
            row['measured_sections'] = sections
            row['tracker'] = {'support': measured.support, 'rejects': measured.rejects,
                              'center_rms': measured.center_rms, 'stop': measured.stop,
                              'regularized_curve': measured.centerline.tolist(),
                              'stations': [
                                  {'first': first.tolist() if np.isfinite(first).all() else None,
                                   'second': second.tolist() if np.isfinite(second).all() else None,
                                   'reject': rejected}
                                  for first, second, rejected in zip(
                                      measured.first, measured.second, measured.reject)]}
            # Tracking refinements never alter the immutable seed or registration.
            if step == 1 or step % 10 == 0:
                overlay(sequence, frame, predicted, sections, out / f'{direction:+d}-{frame:05}.png')
            previous = frame
        result['directions'][str(direction)] = {'stop': stop, 'summary': summary(rows), 'frames': rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, nargs='+', required=True,
                        help='one sequence.json per sequence')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--propagate', action='store_true')
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--preprocess', choices=('dog', 'none'), default='dog')
    parser.add_argument('--region-check', choices=('on', 'off'), default='on',
                        help='off reports the regional statistics without rejecting on them')
    args = parser.parse_args()
    if args.stride < 1 or args.max_frames < 1:
        parser.error('stride and max-frames must be positive')
    if args.region_check == 'off' and args.propagate:
        parser.error('propagation uses the regional check; --region-check off is for diagnosis')
    sequences = [Sequence(path) for path in args.manifest]
    if len({s.name for s in sequences}) != len(sequences):
        parser.error('sequence names must be unique')
    if args.propagate and any(s.split == 'held_out' for s in sequences):
        parser.error('held-out sequences evaluate registration only; propagation is not allowed')
    out = args.out_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    if out.is_relative_to(repo) or any(out.is_relative_to(s.root) for s in sequences):
        parser.error('outputs must be outside the repository and the manifest directories')
    out.mkdir(parents=True, exist_ok=True)
    report = {'preprocessing': ('masked normalized DoG sigma 1/4 px' if args.preprocess == 'dog'
                                else 'none'),
              'configuration': 'TranslationConfig defaults', 'region_check': args.region_check,
              'sequences': {}}
    report['settings'] = describe_translation(translation_config(args.region_check == 'on'))
    for sequence in sequences:
        evaluated = evaluate(sequence, args.stride, args.preprocess, args.region_check == 'on')
        evaluated['split'] = sequence.split
        evaluated['provenance'] = sequence.provenance
        if args.propagate:
            sequence_out = out / sequence.name
            sequence_out.mkdir(exist_ok=True)
            evaluated['propagation'] = propagate(sequence, sequence_out, args.max_frames, args.preprocess)
            evaluated['propagation']['overlay_directory'] = str(sequence_out)
        report['sequences'][sequence.name] = evaluated
        print(sequence.name, json.dumps(evaluated['summary']), flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
