"""Evaluate native similarity and caller-owned anchors over neutral sequence manifests.

Writes a report, per-pair overlay PNGs and a standalone HTML viewer into `--out-dir`.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image, ImageDraw
import vision_metrology as vm
from vision_metrology.experimental import SimilarityEstimator
from anchors import AnchoredSequence, matrix, parts, corner_distance
from manifest import Sequence
from settings import describe_similarity, similarity_config, translation_config

FIELDS = ('transform', 'score', 'margin', 'support', 'residual_rms', 'residual_max',
          'gain', 'offset', 'covariance', 'regions', 'region_error', 'region_chi2', 'status')


def estimate(estimator, sequence, previous, current, preprocess, prediction=None):
    start = perf_counter()
    a, am = sequence.prepare(previous, preprocess)
    b, bm = sequence.prepare(current, preprocess)
    preparation = 1000*(perf_counter()-start)
    start = perf_counter()
    result = estimator.estimate(a, b, previous_mask=am, current_mask=bm, prediction=prediction)
    row = {field: getattr(result, field, None) for field in FIELDS}
    return row | {'source': sequence.name, 'previous': previous, 'current': current,
                  'group': sequence.group(previous),
                  'prepare_ms': preparation, 'estimate_python_ms': 1000*(perf_counter()-start)}


def evaluate_sequence(sequence, count, anchor_every, max_closure, preprocess, check=True):
    """Adjacent pairs within a group; a group change is a handover and ends a segment."""
    rows = []
    estimator = SimilarityEstimator(similarity_config(check))
    anchor_estimator = SimilarityEstimator(similarity_config(check))
    translation = vm.TranslationEstimator(translation_config(check))
    chain = None
    for frame in range(count-1):
        group = sequence.group(frame)
        if group is None or sequence.group(frame+1) != group:
            if group is not None:
                rows.append({'source': sequence.name, 'previous': frame, 'current': frame+1,
                             'group': group, 'status': 'camera_handover'})
            chain = None
            continue
        row = estimate(estimator, sequence, frame, frame+1, preprocess)
        truth = sequence.truth_map(frame, frame+1)
        if truth is not None and row['transform'] is not None:
            row['truth_error_px'] = corner_distance(matrix(row['transform']), truth, sequence.shape)
        a, am = sequence.prepare(frame, preprocess)
        b, bm = sequence.prepare(frame+1, preprocess)
        start = perf_counter()
        t = translation.estimate(a, b, previous_mask=am, current_mask=bm)
        row['translation'] = {field: getattr(t, field, None) for field in FIELDS if field != 'transform'} | {
            'shift': t.shift, 'estimate_python_ms': 1000*(perf_counter()-start)}
        if t.status == 'accepted':
            reverse_t = translation.estimate(b, a, previous_mask=bm, current_mask=am)
            row['translation']['reverse_status'] = reverse_t.status
            row['translation']['closure_px'] = (float(np.linalg.norm(np.array(t.shift)+reverse_t.shift))
                                                if reverse_t.status == 'accepted' else None)
        if row['status'] == 'accepted':
            reverse = estimate(estimator, sequence, frame+1, frame, preprocess)
            row['reverse_status'] = reverse['status']
            row['closure_px'] = (corner_distance(matrix(reverse['transform']) @ matrix(row['transform']),
                                                 np.eye(3), sequence.shape) if reverse['status'] == 'accepted' else None)
            row['sequence_status'] = ('reverse_' + reverse['status'] if reverse['status'] != 'accepted'
                                      else 'inconsistent_closure' if row['closure_px'] > max_closure else 'accepted')
            if row['sequence_status'] != 'accepted':
                chain = None
                row['anchor'] = {'status': 'unavailable_reciprocity'}
                rows.append(row)
                continue
            if chain is None:
                chain = AnchoredSequence(frame, sequence.shape)
            chain.advance(row)
            age = frame+1-chain.seed
            if age % anchor_every == 0:
                anchor = estimate(anchor_estimator, sequence, chain.seed, frame+1, preprocess, parts(chain.pose))
                anchor['native_status'] = anchor['status']
                if anchor['status'] == 'accepted':
                    reverse_anchor = estimate(anchor_estimator, sequence, frame+1, chain.seed, preprocess,
                                              parts(np.linalg.inv(matrix(anchor['transform']))))
                    anchor['reverse_status'] = reverse_anchor['status']
                    anchor['closure_px'] = (corner_distance(matrix(reverse_anchor['transform']) @ matrix(anchor['transform']), np.eye(3), sequence.shape)
                                            if reverse_anchor['status'] == 'accepted' else None)
                    if reverse_anchor['status'] != 'accepted':
                        anchor['status'] = 'reverse_' + reverse_anchor['status']
                    elif anchor['closure_px'] > max_closure:
                        anchor['status'] = 'inconsistent_closure'
                row['anchor'] = anchor
                row['anchor_correction_px'] = chain.constrain(anchor)
            else:
                row['anchor'] = {'status': 'not_attempted'}
                row['anchor_correction_px'] = None
            row['sequence'] = chain.diagnostics()
        else:
            chain = None  # No geometry is transported across a loss of evidence.
        rows.append(row)
    return rows


def summarize(rows):
    accepted = [r for r in rows if r['status'] == 'accepted']
    closure = [r['closure_px'] for r in accepted if r.get('closure_px') is not None]
    times = [r['estimate_python_ms'] for r in rows if 'estimate_python_ms' in r]
    truth = [r['truth_error_px'] for r in accepted if 'truth_error_px' in r]
    return {'pairs': len(rows), 'accepted': len(accepted),
            'coverage': len(accepted)/len(rows) if rows else 0,
            'outcomes': dict(Counter(r['status'] for r in rows)),
            'translation_outcomes': dict(Counter(r['translation']['status'] for r in rows if 'translation' in r)),
            'closure_pairs': len(closure),
            'sequence_outcomes': dict(Counter(r.get('sequence_status', r['status']) for r in rows)),
            'reverse_outcomes': dict(Counter(r['reverse_status'] for r in accepted)),
            'closure_px_median': float(np.median(closure)) if closure else None,
            'closure_px_p95': float(np.percentile(closure, 95)) if closure else None,
            'anchor_outcomes': dict(Counter(r['anchor']['status'] for r in accepted)),
            'truth_error_px_median': float(np.median(truth)) if truth else None,
            'truth_error_px_max': float(np.max(truth)) if truth else None,
            'estimate_python_ms_median': float(np.median(times)) if times else None,
            'estimate_python_ms_p95': float(np.percentile(times, 95)) if times else None}


def evaluate(sequences, limit, anchor_every, max_closure, preprocess, check=True):
    rows = []
    for sequence in sequences:
        count = min(len(sequence), limit) if limit else len(sequence)
        rows += evaluate_sequence(sequence, count, anchor_every, max_closure, preprocess, check)
    return {'summary': summarize(rows), 'pairs': rows,
            'anchors': {'every': anchor_every, 'max_closure_px': max_closure,
                        'policy': 'fixed first keyframe per reciprocal supported segment; failed constraints unavailable'}}


def overlay(sequence, row, out):
    a = Image.fromarray(sequence.image(row['previous'])).convert('RGB')
    b = Image.fromarray(sequence.image(row['current'])).convert('RGB')
    if row.get('transform') is not None:
        # PIL's affine resampler expects output-to-input mapping.
        inverse = np.linalg.inv(matrix(row['transform']))
        a = a.transform(b.size, Image.Transform.AFFINE, tuple(inverse[:2].ravel()), Image.Resampling.BILINEAR)
    image = Image.blend(b, a, .45)
    draw = ImageDraw.Draw(image)
    draw.text((5, 5), f"group {row['group']} {row['previous']} > {row['current']} {row['status']}",
              fill=(70, 255, 140) if row['status'] == 'accepted' else (255, 190, 40))
    image.resize((3*image.width, 3*image.height)).save(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, nargs='+', required=True,
                        help='one sequence.json per sequence; their pairs are summarized together')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--propagation-report', type=Path,
                        help='a run.py --propagate report whose seed geometry the viewer overlays')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-closure', type=float, default=.25)
    parser.add_argument('--anchor-every', type=int, default=8)
    parser.add_argument('--preprocess', choices=('dog', 'none'), default='dog')
    parser.add_argument('--region-check', choices=('on', 'off'), default='on',
                        help='off reports the regional statistics without rejecting on them')
    args = parser.parse_args()
    if args.anchor_every <= 0 or not np.isfinite(args.max_closure) or args.max_closure <= 0:
        parser.error('choose a positive anchor interval and closure bound')
    sequences = [Sequence(path) for path in args.manifest]
    if len({s.name for s in sequences}) != len(sequences):
        parser.error('sequence names must be unique')
    out = args.out_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    if out.is_relative_to(repo) or any(out.is_relative_to(s.root) for s in sequences):
        parser.error('output must be outside the repository and the manifest directories')
    out.mkdir(parents=True, exist_ok=True)
    check = args.region_check == 'on'
    report = evaluate(sequences, args.limit, args.anchor_every, args.max_closure, args.preprocess, check)
    report['provenance'] = {s.name: s.provenance for s in sequences}
    report['region_check'] = args.region_check
    report['settings'] = describe_similarity(similarity_config(check))
    by_name = {s.name: s for s in sequences}
    for sequence in sequences:
        for group in dict.fromkeys(r['group'] for r in report['pairs'] if r['source'] == sequence.name):
            candidates = [r for r in report['pairs'] if r['source'] == sequence.name
                          and r['group'] == group and r.get('transform') is not None]
            for row in candidates[::max(1, len(candidates)//8)]:
                overlay(sequence, row, out/f"{sequence.name}-{group}-{row['current']:05}.png")
    from viewer import create
    propagation = None
    if args.propagation_report:
        saved = json.loads(args.propagation_report.read_text())
        propagation = [s['propagation'] for s in saved['sequences'].values()
                       if s.get('propagation') and s['propagation']['sequence'] in by_name]
    create(by_name, report, out/'index.html', propagation)
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['summary'], indent=2))
    print(out/'index.html')


if __name__ == '__main__':
    main()
