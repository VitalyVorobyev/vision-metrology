"""Wall accuracy against annotated cross-sections, with the annotator's σ beside it.

Given an operator's results file and a manifest's annotations, it reports, per wall and
pooled, the bias and σ of the operator's boundaries, the width error and the coverage. The
reference at each section is the mean of every visible annotation of it (all annotators,
both passes). The annotator's σ comes from pass 1 against pass 2 of the same section: a
single annotation's σ is `std(pass 2 − pass 1)/√2`. Only aggregates are written unless
`--per-section` asks for the rows, which carry geometry and stay local.

A results file is `{"operator": name, "sections": [{frame, station: [x, y],
normal?: [nx, ny], left, right}, ...]}`. Each wall is `null` (not found), an `[x, y]`
point, or a signed offset along the row's own normal from its station. A result row
matches the annotated section of the same frame whose station lies within `--match-px`.
`--stations-out` writes the annotated stations as an empty results file for an operator
to fill.

    python tools/registration_eval/annotations_eval.py --manifest seq/sequence.json \\
        --results results.json [--out report.json] [--held-out]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import annotation
from annotation import key
from manifest import Sequence

SIDES = ('left', 'right')
MATCH_PX = .5


def statistics(errors):
    """n, bias, σ (sample), RMS and the largest magnitude of a list of errors (px)."""
    e = np.asarray(errors, dtype=float)
    if e.size == 0:
        return {'n': 0, 'bias': None, 'sigma': None, 'rms': None, 'max_abs': None}
    return {'n': int(e.size), 'bias': round(float(e.mean()), 4),
            'sigma': round(float(e.std(ddof=1)), 4) if e.size > 1 else None,
            'rms': round(float(np.sqrt(np.mean(e*e))), 4), 'max_abs': round(float(np.abs(e).max()), 4)}


def references(annotations, annotator=None):
    """Per section (keyed by frame and station): its geometry, the mean visible boundaries,
    each side's visibility (`visible`, `hidden` or `disputed`) and its annotation count."""
    grouped = defaultdict(list)
    for entry in annotations:
        if annotator is None or entry['annotator'] == annotator:
            grouped[key(entry)].append(entry)
    result = {}
    for k, entries in grouped.items():
        row = {'frame': entries[0]['frame'], 'station': entries[0]['station'],
               'normal': entries[0]['normal'], 'annotations': len(entries)}
        for side in SIDES:
            flags = {e[f'{side}_visible'] for e in entries}
            row[f'{side}_state'] = ('visible' if flags == {True} else
                                    'hidden' if flags == {False} else 'disputed')
            row[side] = float(np.mean([e[side] for e in entries])) if flags == {True} else None
        result[k] = row
    return result


def annotator_spread(annotations, annotator=None):
    """Pass 1 against pass 2: the inter-pass bias, a single annotation's σ per wall and
    for the width, and how often the two passes agree on visibility."""
    passes = defaultdict(dict)
    for entry in annotations:
        if annotator is None or entry['annotator'] == annotator:
            passes[(entry['annotator'], key(entry))][entry['pass']] = entry
    walls, widths, agree, sides = [], [], 0, 0
    pairs = [p for p in passes.values() if 1 in p and 2 in p]
    for pair in pairs:
        first, second = pair[1], pair[2]
        for side in SIDES:
            sides += 1
            agree += first[f'{side}_visible'] == second[f'{side}_visible']
            if first[side] is not None and second[side] is not None:
                walls.append(second[side] - first[side])
        if None not in (first['left'], first['right'], second['left'], second['right']):
            widths.append((second['right'] - second['left']) - (first['right'] - first['left']))

    def sigma(d):
        return round(float(np.std(d, ddof=1)/np.sqrt(2)), 4) if len(d) > 1 else None
    return {'pairs': len(pairs), 'wall_pairs': len(walls),
            'pass_difference_bias': statistics(walls)['bias'],
            'sigma': sigma(walls), 'width_sigma': sigma(widths),
            'visibility_agreement': round(agree/sides, 4) if sides else None}


def wall_offset(value, row, reference):
    """The operator's wall as a signed offset along the reference normal, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if 'normal' not in row:
            raise ValueError('a wall given as an offset needs its row normal')
        point = np.asarray(row['station'], float) + value*np.asarray(row['normal'], float)
    else:
        point = np.asarray(value, dtype=float)
    return float(np.dot(point - np.asarray(reference['station']), reference['normal']))


def match(results, refs, match_px=MATCH_PX):
    """{section key: result row}, the nearest row of the same frame within `match_px`."""
    by_frame = defaultdict(list)
    for row in results:
        by_frame[int(row['frame'])].append(row)
    matched = {}
    for k, ref in refs.items():
        best, distance = None, match_px
        for row in by_frame.get(ref['frame'], []):
            d = float(np.hypot(*(np.asarray(row['station'], float) - ref['station'])))
            if d <= distance:
                best, distance = row, d
        if best is not None:
            matched[k] = best
    return matched


def evaluate(annotations, results, match_px=MATCH_PX, annotator=None, per_section=False):
    """The aggregate report of one operator's results against the annotations."""
    refs = references(annotations, annotator)
    spread = annotator_spread(annotations, annotator)
    matched = match(results, refs, match_px)
    errors = {side: [] for side in SIDES}
    width_errors, counts, rows = [], defaultdict(int), []
    reference_variance = []
    for k, ref in refs.items():
        row = matched.get(k)
        if row is None:
            continue
        found = {side: wall_offset(row.get(side), row, ref) for side in SIDES}
        entry = {'frame': ref['frame'], 'station': ref['station']}
        for side in SIDES:
            state = ref[f'{side}_state']
            counts[f'{side}_{state}'] += 1
            if state == 'visible':
                if found[side] is None:
                    counts[f'{side}_missed'] += 1
                else:
                    errors[side].append(found[side] - ref[side])
                    entry[f'{side}_error'] = round(errors[side][-1], 4)
            elif state == 'hidden' and found[side] is not None:
                counts[f'{side}_spurious'] += 1
        if None not in (ref['left'], ref['right'], found['left'], found['right']):
            width_errors.append((found['right'] - found['left']) - (ref['right'] - ref['left']))
            entry['width_error'] = round(width_errors[-1], 4)
            if spread['sigma'] is not None:
                reference_variance.append(spread['sigma']**2/ref['annotations'])
        rows.append(entry)
    walls = {side: statistics(errors[side]) for side in SIDES}
    walls['both'] = statistics(errors['left'] + errors['right'])
    coverage = {}
    for side in SIDES:
        visible = counts[f'{side}_visible']
        coverage[side] = {'visible': visible, 'found': visible - counts[f'{side}_missed'],
                          'rate': round((visible - counts[f'{side}_missed'])/visible, 4) if visible else None,
                          'hidden': counts[f'{side}_hidden'], 'spurious': counts[f'{side}_spurious'],
                          'disputed': counts[f'{side}_disputed']}
    sigma = walls['both']['sigma']
    corrected = None
    if sigma is not None and reference_variance:
        corrected = round(float(np.sqrt(max(sigma**2 - np.mean(reference_variance), 0.))), 4)
    report = {'sections': len(refs), 'matched': len(matched), 'walls': walls,
              'width': statistics(width_errors), 'coverage': coverage,
              'wall_sigma_less_reference_noise': corrected, 'annotator': spread}
    if per_section:
        report['per_section'] = rows
    return report


def stations(annotations):
    """The annotated sections as an unfilled results file."""
    refs = references(annotations)
    return {'operator': None, 'sections': [
        {'frame': r['frame'], 'station': r['station'], 'normal': r['normal'], 'left': None, 'right': None}
        for r in refs.values()]}


def text_summary(name, report):
    w, width, a = report['walls']['both'], report['width'], report['annotator']
    cov = report['coverage']
    return (f"{name}: {report['matched']}/{report['sections']} sections matched; walls n={w['n']} "
            f"bias {w['bias']} σ {w['sigma']} px; width n={width['n']} bias {width['bias']} "
            f"σ {width['sigma']} px; coverage left {cov['left']['rate']} right {cov['right']['rate']}; "
            f"annotator σ {a['sigma']} px over {a['pairs']} repeated sections")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, help='an operator results file')
    parser.add_argument('--out', type=Path, help='write the report as JSON here')
    parser.add_argument('--stations-out', type=Path, help='write the annotated stations here')
    parser.add_argument('--annotator', help='only this annotator (default: all)')
    parser.add_argument('--source', nargs='+', choices=('uniform', 'span', 'user', 'stratified'),
                        help='only sections from these sources (default: all)')
    parser.add_argument('--match-px', type=float, default=MATCH_PX)
    parser.add_argument('--per-section', action='store_true',
                        help='include per-section errors (geometry: keep local)')
    parser.add_argument('--held-out', action='store_true',
                        help='allow a held-out sequence: run once, with frozen settings')
    args = parser.parse_args()
    sequence = Sequence(args.manifest)
    if sequence.split == 'held_out' and not args.held_out:
        parser.error('a held-out sequence needs --held-out; freeze settings on development first')
    annotations = sequence.annotated
    if args.source:
        if sequence.annotations_path is None:
            parser.error('--source needs an annotations file, which records each section\'s source')
        sources = {key(s): s.get('source') for s in annotation.read(sequence.annotations_path)['sections']}
        annotations = [a for a in annotations if sources.get(key(a)) in args.source]
    if not annotations:
        parser.error('no annotations to evaluate')
    repo = Path(__file__).resolve().parents[2]
    for out in (args.out, args.stations_out):
        if out is not None and out.resolve().is_relative_to(repo):
            parser.error('outputs must be outside the repository')
    if args.stations_out:
        args.stations_out.write_text(json.dumps(stations(annotations), indent=1) + '\n')
    if args.results is None:
        return
    doc = json.loads(args.results.read_text())
    report = evaluate(annotations, doc.get('sections', []), args.match_px, args.annotator,
                      args.per_section)
    report = {'sequence': sequence.name, 'split': sequence.split, 'operator': doc.get('operator'),
              'sources': args.source or 'all', 'match_px': args.match_px, **report}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(text_summary(sequence.name, report))


if __name__ == '__main__':
    main()
