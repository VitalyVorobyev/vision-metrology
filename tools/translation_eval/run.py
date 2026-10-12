"""Local sequence registration evaluation and annotated-segment propagation."""
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

from dataset import DEVELOPMENT, HELD_OUT, Dataset

FIELDS = ('shift', 'score', 'margin', 'support', 'residual_rms', 'residual_max',
          'gain', 'offset', 'covariance', 'regions', 'region_error', 'status')


def estimate(estimator, dataset, previous, current, camera):
    start = perf_counter()
    a, am = dataset.prepare(previous, camera)
    b, bm = dataset.prepare(current, camera)
    prepared_ms = 1000 * (perf_counter() - start)
    start = perf_counter()
    result = estimator.estimate(a, b, previous_mask=am, current_mask=bm)
    elapsed_ms = 1000 * (perf_counter() - start)
    return {k: getattr(result, k) for k in FIELDS} | {
        'previous': previous, 'current': current, 'camera': camera,
        'prepare_ms': prepared_ms, 'estimate_python_ms': elapsed_ms}


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
    }


def evaluate(dataset, stride):
    estimator = vm.TranslationEstimator()
    rows = []
    roles = dataset.analysis['roles']['rear']
    for frame in range(0, dataset.analysis['bead_frames']-1, stride):
        camera = roles[frame]
        if camera is None or roles[frame+1] != camera:
            rows.append({'previous': frame, 'current': frame+1, 'status': 'camera_handover'})
            continue
        row = estimate(estimator, dataset, frame, frame+1, camera)
        cached = dataset.analysis['cameras'][camera]['bead_motion'][frame]
        row['cached_motion'] = cached['raw']
        row['cached_motion_status'] = cached['status']
        row['comparison_px'] = (float(np.linalg.norm(np.array(row['shift'])-cached['raw']))
                                if row['shift'] else None)
        if row['status'] == 'accepted':
            reverse = estimate(estimator, dataset, frame+1, frame, camera)
            row['reverse_status'] = reverse['status']
            row['closure_px'] = (float(np.linalg.norm(np.array(row['shift'])+reverse['shift']))
                                 if reverse['status'] == 'accepted' else None)
        rows.append(row)
    return {'summary': summary(rows), 'pairs': rows,
            'comparison_provenance': 'private sequence dataset generated raw estimates, not motion ground truth'}


def transported(ribbon, shift):
    """Transport every geometry representation; preserve normals and arc identity."""
    moved = copy.deepcopy(ribbon)
    def points(values):
        return (np.asarray(values) + shift).tolist() if values else []
    for wall in moved['walls']:
        wall['points'] = points(wall['points'])
    for end in moved.get('ends', []):
        end['outline'] = points(end.get('outline', []))
    derived = moved['derived']
    derived['centerline'] = points(derived['centerline'])
    for section in derived['sections']:
        for key in ('center', 'left', 'right'):
            if section.get(key) is not None:
                section[key] = (np.asarray(section[key]) + shift).tolist()
    return moved


def visible_path(points, static, shape):
    points = np.asarray(points, dtype=np.float32)
    valid = ((points[:, 0] >= 0) & (points[:, 0] <= shape[1]-1)
             & (points[:, 1] >= 0) & (points[:, 1] <= shape[0]-1))
    for i in np.flatnonzero(valid):
        x, y = np.rint(points[i]).astype(int)
        valid[i] = not static[y, x]
    runs = np.split(np.arange(len(points)), np.flatnonzero(np.diff(valid.astype(int)))+1)
    runs = [run for run in runs if len(run) >= 2 and valid[run[0]]]
    return np.ascontiguousarray(points[max(runs, key=len)]) if runs else None


def local_sections(image, ribbon, static):
    results = []
    for section in ribbon['derived']['sections']:
        entry = {'s_px': section['s_px'], 's_material_px': section.get('s_material_px')}
        normal = np.asarray(section['normal'])
        for side in ('left', 'right'):
            point = section.get(side)
            if point is None or section.get('status') != 'measured':
                entry[side] = {'reject': 'unsupported_reference'}
                continue
            point = np.asarray(point)
            start, end = point - 4*normal, point + 4*normal
            samples = np.linspace(start, end, 17)
            xy = np.rint(samples).astype(int)
            if np.any(xy < 0) or np.any(xy[:, 0] >= image.shape[1]) or np.any(xy[:, 1] >= image.shape[0]):
                entry[side] = {'reject': 'off_image'}
                continue
            if np.any(static[xy[:, 1], xy[:, 0]]):
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


def overlay(dataset, frame, camera, ribbon, measured, out):
    image = Image.fromarray(dataset.strip(frame, camera)).convert('RGB')
    draw = ImageDraw.Draw(image)
    for wall in ribbon['walls']:
        if len(wall['points']) > 1:
            draw.line([tuple(p) for p in wall['points']], fill=(255, 190, 40), width=1)
    for section in measured:
        for side in ('left', 'right'):
            if 'point' in section[side]:
                x, y = section[side]['point']
                draw.ellipse((x-1, y-1, x+1, y+1), fill=(70, 255, 140))
    image.resize((960, 291)).save(out)


def propagate(dataset, out, requested, max_frames):
    seed, camera, ribbon, provenance = dataset.seed(requested)
    widths = [s['width_px'] for s in ribbon['derived']['sections']
              if s.get('status') == 'measured' and s.get('width_px', 0) > 0]
    if not widths:
        raise ValueError('seed has no measured sections')
    tracker = vm.BeadTracker(vm.BeadConfig(
        polarity='light', min_width=max(1., min(widths)*.5), max_width=max(widths)*1.5,
        spacing=2., min_margin=.1,
        track=vm.BeadCaliper(max_offset=3., half_width=1., threshold=2.),
        measure=vm.BeadCaliper(max_offset=3., half_width=1., threshold=2., off_image='reject')))
    static = dataset.static_exclusions(camera)
    seed_sections = local_sections(dataset.strip(seed, camera), ribbon, static)
    overlay(dataset, seed, camera, ribbon, seed_sections, out / f'seed-{seed:05}.png')
    result = {'seed': seed, 'camera': camera, 'provenance': provenance, 'directions': {}}
    for direction in (-1, 1):
        estimator = vm.TranslationEstimator()
        cumulative = np.zeros(2)
        previous = seed
        rows = []
        stop = 'frame_budget'
        for step in range(1, max_frames+1):
            frame = seed+direction*step
            if not 0 <= frame < dataset.analysis['bead_frames']:
                stop = 'sequence_end'
                break
            if dataset.analysis['roles']['rear'][frame] != camera:
                stop = 'camera_handover'
                break
            row = estimate(estimator, dataset, previous, frame, camera)
            rows.append(row)
            if row['status'] != 'accepted':
                stop = row['status']
                break
            cumulative += row['shift']
            predicted = transported(ribbon, cumulative)
            path = visible_path(predicted['derived']['centerline'], static, dataset.shape)
            if path is None:
                stop = 'segment_left_view'
                break
            image = dataset.strip(frame, camera)
            start = perf_counter()
            measured = tracker.track(image, path)
            sections = local_sections(image, predicted, static)
            row['measurement_ms'] = 1000*(perf_counter()-start)
            row['predicted_geometry'] = predicted
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
                overlay(dataset, frame, camera, predicted, sections, out / f'{direction:+d}-{frame:05}.png')
            previous = frame
        result['directions'][str(direction)] = {'stop': stop, 'summary': summary(rows), 'frames': rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'held-out'), default='development')
    parser.add_argument('--sequence', choices=DEVELOPMENT+HELD_OUT)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--propagate', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-frames', type=int, default=100)
    args = parser.parse_args()
    if args.stride < 1 or args.max_frames < 1:
        parser.error('stride and max-frames must be positive')
    inspections = DEVELOPMENT if args.split == 'development' else HELD_OUT
    if args.sequence and args.sequence not in inspections:
        parser.error('sequence must belong to the requested split')
    if args.propagate and args.split == 'held-out':
        parser.error('held-out mode evaluates registration only; seed geometry is not available')
    out = args.out_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    if out.is_relative_to(repo) or out.is_relative_to(args.data_root.resolve()):
        parser.error('outputs must be outside the repository and source dataset')
    out.mkdir(parents=True, exist_ok=True)
    report = {'split': args.split, 'preprocessing': 'masked normalized DoG sigma 1/4 px',
              'configuration': 'TranslationConfig defaults; no held-out tuning', 'sequences': {}}
    defaults = vm.TranslationConfig()
    report['settings'] = {name: getattr(defaults, name) for name in
                          ('search_radius', 'min_support', 'min_score', 'min_margin', 'max_region_error')}
    report['settings']['tuning'] = {name: getattr(defaults.tuning, name) for name in
                                   ('max_levels', 'iterations', 'tolerance', 'huber', 'min_texture')}
    for inspection in ([args.sequence] if args.sequence else inspections):
        dataset = Dataset(args.data_root, inspection)
        evaluated = evaluate(dataset, args.stride)
        if args.propagate:
            sequence_out = out / inspection
            sequence_out.mkdir(exist_ok=True)
            evaluated['propagation'] = propagate(dataset, sequence_out, args.seed, args.max_frames)
            evaluated['propagation']['overlay_directory'] = str(sequence_out)
        report['sequences'][inspection] = evaluated
        print(inspection, json.dumps(evaluated['summary']), flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
