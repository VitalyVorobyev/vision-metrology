"""Regional-check evidence on HPatches: real texture under real illumination and viewpoint.

HPatches (MIT; `tools/datasets/registry.json`, fetched into the git-ignored `data/`) gives
six images per planar scene and the homographies `H_1_k` from image 1 to image k. The
split is by sequence (`tools/datasets/splits/hpatches.json`): tune on `dev`, run `test`
once with the settings frozen.

Two kinds of crop pair, each 320x97 on a grid over image 1:
- `natural`: the same crop of image k, with the homography's own motion as truth and its
  prediction. For `i_` sequences (fixed camera, changing light) H is near the identity;
  a crop counts as near-translation (near-similarity) when H departs from its local
  translation (similarity) by less than `--near` px anywhere on the crop.
- `rectified`: image k resampled through H into image 1's crop frame, then moved by a
  known map M, so `current(M(p)) = image_k(H(p))`. M is a translation (a small
  similarity for `--model similarity`) for false rejects, or adds a rotation
  (`--residual rotation`, rad) or an anisotropic scale (`affine`), which an estimator
  must reject. The appearance change is HPatches' own: illumination for `i_`, viewpoint
  resampling and parallax for `v_`. H's own error at crop scale is part of the truth
  error, so rejections here include real disagreement.
- `self`: image 1 resampled by M against itself (H is the identity): real texture with an
  exact model, for the false-reject rate; `--noise` adds read noise to both crops.

Prints aggregates only: outcome counts, the regional-check reject rate, and the corner
error of accepted maps against the truth.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import vision_metrology as vm
from vision_metrology.experimental import SimilarityEstimator

from anchors import corner_distance, matrix
from manifest import bandpass
from settings import similarity_config, translation_config

CROP = (97, 320)
REPO = Path(__file__).resolve().parents[2]


def gray(path):
    return np.asarray(Image.open(path).convert('L'), np.float64)


def homography(path):
    return np.loadtxt(path)


def apply(h, x, y):
    w = h[2, 0]*x + h[2, 1]*y + h[2, 2]
    return (h[0, 0]*x + h[0, 1]*y + h[0, 2]) / w, (h[1, 0]*x + h[1, 1]*y + h[1, 2]) / w


def local_models(h, origin):
    """The crop-local map of H (previous crop px to current crop px, same origin), its
    best translation and similarity (least squares over a grid), and their departures."""
    gy, gx = np.mgrid[0:CROP[0]:8, 0:CROP[1]:8].astype(float)
    gx, gy = gx.ravel(), gy.ravel()
    qx, qy = apply(h, gx + origin[0], gy + origin[1])
    qx, qy = qx - origin[0], qy - origin[1]
    t = np.array([np.mean(qx - gx), np.mean(qy - gy)])
    translation_dev = np.hypot(qx - gx - t[0], qy - gy - t[1]).max()
    design = np.zeros((2*gx.size, 4))
    design[0::2] = np.c_[gx, -gy, np.ones_like(gx), np.zeros_like(gx)]
    design[1::2] = np.c_[gy, gx, np.zeros_like(gx), np.ones_like(gx)]
    target = np.empty(2*gx.size)
    target[0::2], target[1::2] = qx, qy
    (a, b, tx, ty), *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ np.array([a, b, tx, ty])
    similarity_dev = np.hypot(*(target - fitted).reshape(-1, 2).T).max()
    parts = (tx, ty, float(np.arctan2(b, a)), float(np.hypot(a, b)))
    return t, parts, translation_dev, similarity_dev


def known_map(kind, rng, magnitude, similarity=False):
    """Crop-local M about the crop centre, as a 3x3 matrix: a translation, a small
    similarity for the similarity model, and the requested residual."""
    c = np.array([(CROP[1]-1)/2, (CROP[0]-1)/2])
    shift = rng.uniform(-3, 3, 2)
    linear = np.eye(2)
    if similarity:
        a, s = rng.uniform(-2e-3, 2e-3), np.exp(rng.uniform(-1e-3, 1e-3))
        linear = s*np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    if kind == 'rotation':
        a = magnitude*rng.choice([-1, 1])
        linear = linear @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    elif kind == 'affine':
        e = magnitude*rng.choice([-1, 1])
        linear = linear @ np.diag([1+e, 1-e])
    m = np.eye(3)
    m[:2, :2] = linear
    m[:2, 2] = c + shift - linear @ c
    return m


def rectified(image_k, h, origin, m):
    """current(q) = image_k(H(origin + M^-1 q)); validity where all taps are inside."""
    y, x = np.mgrid[:CROP[0], :CROP[1]].astype(float)
    inverse = np.linalg.inv(m)
    px = inverse[0, 0]*x + inverse[0, 1]*y + inverse[0, 2] + origin[0]
    py = inverse[1, 0]*x + inverse[1, 1]*y + inverse[1, 2] + origin[1]
    sx, sy = apply(h, px, py)
    valid = (sx >= 0) & (sy >= 0) & (sx <= image_k.shape[1]-2) & (sy <= image_k.shape[0]-2)
    values = map_coordinates(image_k, [sy, sx], order=3, mode='nearest')
    return values, valid


def origins(shape, step):
    ys = range(0, shape[0]-CROP[0]+1, step[0])
    xs = range(0, shape[1]-CROP[1]+1, step[1])
    return [(x, y) for y in ys for x in xs]


def prepare(image, valid, preprocess):
    valid = np.ascontiguousarray(valid, np.uint8)
    if preprocess == 'dog':
        return bandpass(image.astype(np.float32), valid > 0), valid
    return np.ascontiguousarray(np.clip(np.rint(image), 0, 255).astype(np.uint8)), valid


def evaluate(args):
    split = json.loads(args.split_file.read_text())
    names = [n for n in split[args.split] if n.startswith(tuple(f'{k}_' for k in args.kinds))]
    overrides = (args.max_chi2, args.bias_floor)
    translation = vm.TranslationEstimator(translation_config(not args.no_check, *overrides))
    similarity = SimilarityEstimator(similarity_config(not args.no_check, *overrides))
    rng = np.random.default_rng(args.seed)
    rows = []
    for name in names:
        root = args.root / name
        first = gray(root / '1.ppm')
        for k in range(2, 7):
            if args.pairs == 'self':  # image 1 against itself: five draws of M per crop
                h, other = np.eye(3), first
            else:
                h, other = homography(root / f'H_1_{k}'), gray(root / f'{k}.ppm')
            for origin in origins(first.shape, (args.step, 2*args.step)):
                t, parts, t_dev, s_dev = local_models(h, origin)
                previous = first[origin[1]:origin[1]+CROP[0], origin[0]:origin[0]+CROP[1]]
                row = {'sequence': name, 'kind': name[0], 'k': k,
                       'translation_departure_px': float(t_dev),
                       'similarity_departure_px': float(s_dev)}
                if args.pairs == 'natural':
                    if t_dev > args.near and args.model == 'translation':
                        continue
                    if s_dev > args.near and args.model == 'similarity':
                        continue
                    x0, y0 = origin
                    if y0+CROP[0] > other.shape[0] or x0+CROP[1] > other.shape[1]:
                        continue
                    current = other[y0:y0+CROP[0], x0:x0+CROP[1]]
                    valid = np.ones(CROP, bool)
                    truth = matrix(parts) if args.model == 'similarity' else matrix((*t, 0., 1.))
                else:
                    m = known_map(args.residual, rng, args.magnitude,
                                  args.model == 'similarity')
                    current, valid = rectified(other, h, origin, m)
                    if valid.mean() < .9:
                        continue
                    truth = m
                if args.noise > 0:
                    previous = previous + rng.normal(0, args.noise, CROP)
                    current = current + rng.normal(0, args.noise, CROP)
                a, am = prepare(previous, np.ones(CROP, bool), args.preprocess)
                b, bm = prepare(current, valid, args.preprocess)
                if args.model == 'translation':
                    prediction = tuple(map(float, t)) if args.pairs == 'natural' else None
                    r = translation.estimate(a, b, previous_mask=am, current_mask=bm,
                                             prediction=prediction)
                    estimate = None if r.shift is None else matrix((*r.shift, 0., 1.))
                else:
                    prediction = parts if args.pairs == 'natural' else None
                    r = similarity.estimate(a, b, previous_mask=am, current_mask=bm,
                                            prediction=prediction)
                    estimate = None if r.transform is None else matrix(r.transform)
                row |= {'status': r.status, 'region_error': r.region_error,
                        'region_chi2': getattr(r, 'region_chi2', None)}
                if estimate is not None and r.status == 'accepted':
                    row['truth_error_px'] = corner_distance(estimate, truth, CROP)
                rows.append(row)
    return rows


def summarize(rows):
    out = {}
    for kind in sorted({r['kind'] for r in rows}):
        group = [r for r in rows if r['kind'] == kind]
        outcomes = Counter(r['status'] for r in group)
        checked = sum(outcomes[s] for s in ('accepted', 'model_mismatch'))
        errors = [r['truth_error_px'] for r in group if 'truth_error_px' in r]
        out[f'{kind}_'] = {
            'pairs': len(group), 'sequences': len({r['sequence'] for r in group}),
            'outcomes': dict(outcomes),
            'model_mismatch_of_all': outcomes['model_mismatch'] / len(group),
            'model_mismatch_of_checked': outcomes['model_mismatch'] / checked if checked else None,
            'accepted_truth_error_px': ({f'p{q}': float(np.percentile(errors, q)) for q in (50, 95)}
                                        if errors else None)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path,
                        default=REPO / 'data/hpatches/extracted/hpatches-sequences-release')
    parser.add_argument('--split-file', type=Path,
                        default=REPO / 'tools/datasets/splits/hpatches.json')
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--kinds', nargs='+', choices=('i', 'v'), default=['i', 'v'])
    parser.add_argument('--model', choices=('translation', 'similarity'), default='translation')
    parser.add_argument('--pairs', choices=('natural', 'rectified', 'self'), default='natural')
    parser.add_argument('--residual', choices=('none', 'rotation', 'affine'), default='none')
    parser.add_argument('--magnitude', type=float,
                        help='rotation (rad, default 0.01) or anisotropic scale (default 0.005)')
    parser.add_argument('--near', type=float, default=.1,
                        help='natural pairs: largest departure from the model, px')
    parser.add_argument('--step', type=int, default=97, help='crop grid step, rows')
    parser.add_argument('--preprocess', choices=('dog', 'none'), default='dog')
    parser.add_argument('--noise', type=float, default=0., help='added read noise, DN')
    parser.add_argument('--no-check', action='store_true')
    parser.add_argument('--max-chi2', type=float, help='override the default RegionCheck')
    parser.add_argument('--bias-floor', type=float, help='override the default RegionCheck, px')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, help='per-pair JSON report (keep it local)')
    args = parser.parse_args()
    if args.magnitude is None:
        args.magnitude = .01 if args.residual == 'rotation' else .005
    if args.pairs == 'natural' and args.residual != 'none':
        parser.error('a known residual needs --pairs rectified')
    rows = evaluate(args)
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                if k not in ('root', 'split_file', 'out')}
    report = {'settings': settings, 'summary': summarize(rows)}
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report | {'rows': rows}, indent=1))


if __name__ == '__main__':
    main()
