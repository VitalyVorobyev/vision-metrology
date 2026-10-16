"""The generator's self-check: rendered walls against the truth, measured independently.

The `validate-straight` preset renders a straight ribbon with asymmetric walls on a flat
substrate, in two views: at 7° and scale 1, and at −4° and scale 1.3. Two measurements
that share no code with the renderer locate every wall:

- **numpy**: per image column, the centroid of the column's first differences within
  ±6 px of the truth crossing, converted to a distance along the image normal;
- **sampler**: the bindings' `ProfileSampler` (σ = 1 px) along each station's image
  normal on a 0.05 px grid, its derivative extremum refined by a parabola.

Both run on the unquantized pixel means and on the written 8-bit images. A symmetric
edge's centroid and derivative extremum are both the edge itself, so the residual is the
generator's own error: quadrature, the box integral and, for 8-bit, the rounding.

    python tools/sequence_bench/validate.py
"""
from __future__ import annotations

import numpy as np

from generate import station_geometry
from presets import config
from render import Scenario, render, render_float
from scene import apply, apply_direction

WINDOW = 6.0


def _column_walls(image, m, st, wall, sign):
    """Errors along the image normal of a wall measured per column by first-difference centroids."""
    h, w = image.shape
    a = apply(m, st[wall][0])
    b = apply(m, st[wall][-1])
    direction = (b - a)/np.linalg.norm(b - a)
    cos = abs(direction[0])
    errors = []
    lo, hi = sorted((a[0], b[0]))
    for x in range(max(8, int(np.ceil(lo))), min(w - 8, int(hi) + 1)):
        y_true = a[1] + (x - a[0])*direction[1]/direction[0]
        if not 8 <= y_true <= h - 9:
            continue
        g = sign*np.diff(image[:, x].astype(float))
        y = np.arange(h - 1) + 0.5
        near = np.abs(y - y_true) <= WINDOW
        g = np.where(near, np.clip(g, 0, None), 0.)
        errors.append(((y*g).sum()/g.sum() - y_true)*cos)
    return np.asarray(errors)


def _sampler_walls(image, m, st, scale, sign_left):
    """Errors along the image normal of both walls from ProfileSampler derivative extrema."""
    import vision_metrology as vm
    h, w = image.shape
    centre = apply(m, st['point'])
    normal = apply_direction(m, st['point'], st['normal'])
    inside = ((centre[:, 0] > 12) & (centre[:, 0] < w - 13) & (centre[:, 1] > 0) & (centre[:, 1] < h - 1))
    reach = scale*max(st['left'].max(), st['right'].max()) + 5
    offsets = np.arange(-reach, reach + 1e-9, 0.05).astype(np.float32)
    _, deriv, valid = vm.ProfileSampler(1.0).sample(
        np.ascontiguousarray(image, dtype=np.float32), centre[inside].astype(np.float32),
        normal[inside].astype(np.float32), offsets)
    out = {}
    for wall, truth, sign in (('left', -scale*st['left'][inside], sign_left),
                              ('right', scale*st['right'][inside], -sign_left)):
        errors = []
        for row, ok, d0 in zip(deriv, valid, truth):
            near = ok & (np.abs(offsets - d0) <= 3)
            if near.sum() < 5:
                continue
            k = np.flatnonzero(near)[np.argmax(sign*row[near])]
            y0, y1, y2 = sign*row[k - 1], sign*row[k], sign*row[k + 1]
            frac = 0.5*(y0 - y2)/(y0 - 2*y1 + y2)
            errors.append(offsets[k] + frac*0.05 - d0)
        out[wall] = np.asarray(errors)
    return out


def self_check(use_sampler=True):
    """{(view, image kind, method, wall): (max |error|, rms, count)} in image px along the normal."""
    scn = Scenario(config('validate-straight'))
    st = station_geometry(scn)
    rising = 1. if scn.cfg['ribbon']['level'] > scn.cfg['substrate']['mean'] else -1.
    out = {}
    for v, view in enumerate(scn.views):
        m = scn.ribbon_to_image(0, v)
        scale = view.get('scale', 1.)
        images = {'float': render_float(scn, 0, v), 'uint8': render(scn, 0, v)[0]}
        for kind, image in images.items():
            found = {('numpy', 'left'): _column_walls(image, m, st, 'left_point', rising),
                     ('numpy', 'right'): _column_walls(image, m, st, 'right_point', -rising)}
            if use_sampler:
                for wall, errors in _sampler_walls(image, m, st, scale, rising).items():
                    found[('sampler', wall)] = errors
            for (method, wall), e in found.items():
                out[(v, kind, method, wall)] = (float(np.abs(e).max()), float(np.sqrt((e**2).mean())), len(e))
    return out


def main():
    try:
        import vision_metrology  # noqa: F401
        use_sampler = hasattr(vision_metrology, 'ProfileSampler')
    except ImportError:
        use_sampler = False
    if not use_sampler:
        print('vision_metrology.ProfileSampler is not installed: numpy measurement only')
    print('view  image  method   wall   max|err| px  rms px  n')
    for (v, kind, method, wall), (worst, rms, n) in self_check(use_sampler).items():
        print(f'{v:4}  {kind:5}  {method:7}  {wall:5}  {worst:11.4f}  {rms:6.4f}  {n}')


if __name__ == '__main__':
    main()
