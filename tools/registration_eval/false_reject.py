"""Synthetic false-reject study of the regional consistency check.

Each trial renders a fresh band-limited texture (a sum of plane waves whose radial
frequencies follow the power of the masked DoG band, sigma 1/4 px, with pixel-area
integration) at 320x97, moves it by a known transform, adds Gaussian read noise and
quantizes to 8 bits. A horizontal band is excluded from both masks, as a tracked object
would be. The estimator then runs with its default configuration, and the study reports
how often the regional check alone rejects a motion its model represents exactly.

Motions:
- `translation`: a true translation; `TranslationEstimator`.
- `similarity`: a true small similarity about the image centre; `SimilarityEstimator`.
- `rotation`: a translation plus `--magnitude` radians of rotation (default 0.01), which
  `TranslationEstimator` must reject.
- `affine`: a similarity plus an anisotropic scale of +/-`--magnitude` (default 0.005),
  which `SimilarityEstimator` must reject.

`--no-check` disables the check's decision while still computing its statistics, so a
threshold can be chosen from the recorded distribution.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import vision_metrology as vm
from vision_metrology.experimental import SimilarityEstimator

from manifest import DOG_SIGMAS, bandpass
from settings import similarity_config, translation_config

SHAPE = (97, 320)
BAND = (38, 59)
WAVES = 300


def dog_frequencies(rng, count, sigmas=DOG_SIGMAS):
    """Radial frequencies, cycles/px, drawn from the 2-D power of the DoG band."""
    grid = np.linspace(.004, .45, 2000)
    gain = np.exp(-2*(np.pi*sigmas[0]*grid)**2) - np.exp(-2*(np.pi*sigmas[1]*grid)**2)
    pdf = gain**2 * grid
    cdf = np.cumsum(pdf) / pdf.sum()
    return np.interp(rng.uniform(size=count), cdf, grid)


class Texture:
    """A continuous random texture, evaluated exactly at mapped coordinates."""

    def __init__(self, rng, waves=WAVES):
        f = dog_frequencies(rng, waves)
        theta = rng.uniform(0, np.pi, waves)
        self.k = 2*np.pi*np.c_[f*np.cos(theta), f*np.sin(theta)]
        self.phase = rng.uniform(0, 2*np.pi, waves)
        # Pixel-area integration: the box filter's transfer function per component.
        self.amplitude = np.sinc(f*np.cos(theta)) * np.sinc(f*np.sin(theta))

    def __call__(self, u, v):
        out = np.zeros(u.shape)
        for k, phase, amplitude in zip(self.k, self.phase, self.amplitude):
            out += amplitude*np.cos(k[0]*u + k[1]*v + phase)
        return out


def linear_map(angle=0., log_scale=0., anisotropy=0.):
    s = np.exp(log_scale)
    c, n = np.cos(angle), np.sin(angle)
    return s*np.array([[c, -n], [n, c]]) @ np.diag([1+anisotropy, 1-anisotropy])


def render(texture, linear, shift, rng, noise, mean, contrast, norm):
    """uint8 frame of `texture` mapped by q = L (p - c) + c + shift (previous to current)."""
    h, w = SHAPE
    y, x = np.mgrid[:h, :w].astype(float)
    c = np.array([(w-1)/2, (h-1)/2])
    q = np.stack([x.ravel()-c[0]-shift[0], y.ravel()-c[1]-shift[1]])
    p = np.linalg.solve(linear, q) + c[:, None]
    values = texture(p[0].reshape(SHAPE), p[1].reshape(SHAPE))
    values = mean + contrast*values/norm + rng.normal(0, noise, SHAPE)
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def motion(kind, rng, magnitude):
    shift = rng.uniform(-3, 3, 2)
    if kind == 'translation':
        return np.eye(2), shift, {}
    if kind == 'rotation':
        angle = magnitude*rng.choice([-1, 1])
        return linear_map(angle), shift, {'angle': angle}
    angle, log_scale = rng.uniform(-2e-3, 2e-3), rng.uniform(-1e-3, 1e-3)
    anisotropy = magnitude*rng.choice([-1, 1]) if kind == 'affine' else 0.
    return linear_map(angle, log_scale, anisotropy), shift, {
        'angle': angle, 'log_scale': log_scale, 'anisotropy': anisotropy}


def estimator(kind, check, max_chi2=None, bias_floor_px=None):
    """Defaults, optionally overridden; `check=False` keeps the statistics without their
    decision."""
    translation = kind in ('translation', 'rotation')
    cfg = translation_config(check) if translation else similarity_config(check)
    inner = cfg if translation else cfg.translation
    if max_chi2 is not None or bias_floor_px is not None:
        region = inner.region_check
        if max_chi2 is not None and check:
            region.max_chi2 = max_chi2
        if bias_floor_px is not None:
            region.bias_floor_px = bias_floor_px
        inner.region_check = region
    if translation:
        return vm.TranslationEstimator(cfg)
    cfg.translation = inner
    return SimilarityEstimator(cfg)


def trial(kind, index, args, mask):
    rng = np.random.default_rng([args.seed, index])
    texture = Texture(rng)
    probe = texture(*np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)[::-1])
    norm = probe.std()
    linear, shift, extra = motion(kind, rng, args.magnitude)
    a = render(texture, np.eye(2), np.zeros(2), rng, args.noise, args.mean, args.contrast, norm)
    b = render(texture, linear, shift, rng, args.noise, args.mean, args.contrast, norm)
    if args.preprocess == 'dog':
        a, b = bandpass(a, mask > 0), bandpass(b, mask > 0)
    result = estimator(kind, not args.no_check, args.max_chi2, args.bias_floor).estimate(a, b, previous_mask=mask, current_mask=mask)
    row = {'trial': index, 'shift': shift.tolist(), 'status': result.status,
           'regions': result.regions, 'region_error': result.region_error} | extra
    if hasattr(result, 'region_chi2'):
        row['region_chi2'] = result.region_chi2
    if result.covariance is not None:
        c = np.asarray(result.covariance, float)
        variances = (c[0], c[2]) if c.ndim == 1 else (c[0, 0], c[1, 1])
        row['sigma_px'] = float(np.sqrt(max(variances)))
    return row


def summarize(rows):
    outcomes = Counter(r['status'] for r in rows)
    errors = [r['region_error'] for r in rows if r['region_error'] is not None]
    out = {'trials': len(rows), 'outcomes': dict(outcomes),
           'model_mismatch_rate': outcomes['model_mismatch'] / len(rows),
           'regional_check_available': len(errors) / len(rows)}
    for name, values in (('region_error', errors),
                         ('region_chi2', [r['region_chi2'] for r in rows
                                          if r.get('region_chi2') is not None]),
                         ('sigma_px', [r['sigma_px'] for r in rows if 'sigma_px' in r])):
        if values:
            out[name] = {f'p{q}': float(np.percentile(values, q)) for q in (50, 90, 95, 99, 100)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--motion', choices=('translation', 'similarity', 'rotation', 'affine'),
                        default='translation')
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--noise', type=float, default=1., help='read noise sigma, DN')
    parser.add_argument('--contrast', type=float, default=12., help='texture std, DN')
    parser.add_argument('--mean', type=float, default=40., help='mean level, DN')
    parser.add_argument('--magnitude', type=float,
                        help='rotation (rad) or anisotropic scale of the negative controls')
    parser.add_argument('--preprocess', choices=('dog', 'none'), default='dog')
    parser.add_argument('--no-check', action='store_true')
    parser.add_argument('--max-chi2', type=float, help='override the default RegionCheck')
    parser.add_argument('--bias-floor', type=float, help='override the default RegionCheck, px')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--jobs', type=int, default=1, help='worker processes')
    parser.add_argument('--out', type=Path, help='per-trial JSON report')
    args = parser.parse_args()
    if args.magnitude is None:
        args.magnitude = .01 if args.motion == 'rotation' else .005
    mask = np.ones(SHAPE, np.uint8)
    mask[BAND[0]:BAND[1]] = 0
    run = partial(trial, args.motion, args=args, mask=mask)
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = list(pool.map(run, range(args.trials), chunksize=8))
    else:
        rows = [run(i) for i in range(args.trials)]
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                if k not in ('out', 'jobs')}
    report = {'settings': settings,
              'shape': SHAPE, 'excluded_rows': BAND, 'summary': summarize(rows)}
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report | {'rows': rows}, indent=1))


if __name__ == '__main__':
    main()
