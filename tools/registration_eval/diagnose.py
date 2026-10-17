"""Aggregate regional disagreement against reciprocal closure and anchor corrections.

Reads `report.json` files written with `--region-check off`, so every pair that reached
the regional check is accepted and its statistics are kept. Prints aggregates only:
- for translation (`run.py`), `region_error` percentiles, the share above a px limit,
  the Spearman correlation with forward/backward closure, and closure p95 on each side
  of the limit;
- for similarity (`similarity_run.py`), the same for its pairs, plus each accepted
  anchor's correction against the summed `region_error` of the steps it corrects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def percentiles(values, qs=(50, 90, 95)):
    return {f'p{q}': float(np.percentile(values, q)) for q in qs} if len(values) else None


def rank_correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[keep], y[keep]).statistic) if keep.sum() >= 5 else None


def pairs_summary(pairs, limit):
    stats = [p for p in pairs if p.get('region_error') is not None]
    error = np.array([p['region_error'] for p in stats])
    closure = np.array([np.nan if p.get('closure_px') is None else p['closure_px'] for p in stats])
    chi2 = [p['region_chi2'] for p in stats if p.get('region_chi2') is not None]
    below, above = closure[error <= limit], closure[error > limit]
    return {
        'pairs_with_regional_statistics': len(stats),
        'region_error_px': percentiles(error),
        'region_chi2': percentiles(chi2),
        'share_above_limit': float((error > limit).mean()) if len(error) else None,
        'spearman_region_error_closure': rank_correlation(error, closure),
        'closure_p95_px_at_or_below_limit': percentiles(below[np.isfinite(below)], (95,)),
        'closure_p95_px_above_limit': percentiles(above[np.isfinite(above)], (95,)),
    }


def anchor_summary(pairs):
    """Each accepted anchor against the summed regional error since the segment's seed."""
    summed, corrections, running, seed = [], [], 0., None
    for p in pairs:
        sequence = p.get('sequence')
        if sequence is None:
            running, seed = 0., None
            continue
        if sequence['seed'] != seed:
            running, seed = 0., sequence['seed']
        running += p.get('region_error') or 0.
        if p.get('anchor_correction_px') is not None:
            summed.append(running)
            corrections.append(p['anchor_correction_px'])
            running = 0.
    return {'anchors': len(corrections), 'correction_px': percentiles(corrections),
            'spearman_summed_region_error_correction': rank_correlation(summed, corrections)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--translation', type=Path, nargs='*', default=[],
                        help='run.py reports')
    parser.add_argument('--similarity', type=Path, nargs='*', default=[],
                        help='similarity_run.py reports')
    parser.add_argument('--limit', type=float, default=.5, help='px limit for the split')
    args = parser.parse_args()
    out = {'limit_px': args.limit, 'translation': {}, 'similarity': {}}
    for path in args.translation:
        report = json.loads(path.read_text())
        for name, sequence in report['sequences'].items():
            out['translation'][name] = pairs_summary(sequence['pairs'], args.limit)
    for path in args.similarity:
        report = json.loads(path.read_text())
        for name in dict.fromkeys(p['source'] for p in report['pairs']):
            pairs = [p for p in report['pairs'] if p['source'] == name]
            out['similarity'][name] = pairs_summary(pairs, args.limit) | anchor_summary(pairs)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
