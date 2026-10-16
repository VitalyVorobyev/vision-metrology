"""Sub-pixel locking diagnostic, needing no ground truth: fractional edge coordinates.

A locator that locks to the pixel grid puts located positions at particular sub-pixel
phases. Over many edges at unrelated positions an unlocked locator's fractional parts are
uniform, so their histogram is flat. A flat histogram is necessary, not sufficient, for an
unbiased locator.

For every frame of each manifest, edges come from the bindings' 2-D sub-pixel detector
(`detect_edges`, `EdgeConfig` defaults: binomial pre-smoothing, automatic hysteresis,
`parabolic_along_normal`). An edge counts when its pixel lies inside the frame's mask,
eroded by `--erode` px so that no excluded pixel reaches the detector's support, and when
its normal is within `--max-angle` degrees of an image axis. Its coordinate along that
axis is where its tangent crosses its own pixel row (or column): an edgel moves from its
pixel centre along the normal, so at an oblique normal its raw `x` would cover only part
of each pixel even when it is unbiased.

`--refine dog` re-locates each counted edge along its normal with a continuous
derivative of Gaussian (σ 1 px, tapered by `(1 − r²/R²)²` to zero at R = 4σ), evaluated
at the continuous point, as a reference that does not lock: the same edges, another
locator.

The report holds, per sequence and pooled, the 10-bin histogram normalized to the uniform
rate (1.0 everywhere when flat) and the locking index, its max/min bin ratio. Only
aggregates; no positions, frames or pixels.

    python tools/registration_eval/phase_histogram.py --manifest seq/sequence.json \\
        --out-dir /tmp/phase-out [--mask visibility] [--refine dog]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
import vision_metrology as vm

from manifest import Sequence

BINS = 10
MASKS = ('visibility', 'validity', 'both', 'none')
DOG_SIGMA = 1.
DOG_RADIUS = 4*DOG_SIGMA


def frame_mask(sequence, i, which, erode):
    """Where an edgel's pixel may lie: the chosen mask, eroded by `erode` px."""
    keep = np.ones(sequence.shape, dtype=bool)
    if which in ('visibility', 'both'):
        keep &= sequence.visible(i)
    if which in ('validity', 'both'):
        keep &= sequence.mask(i)
    # The frame's own border is excluded too: the detector clamps there.
    keep[[0, -1], :] = keep[:, [0, -1]] = False
    if erode > 0:
        keep = binary_erosion(keep, np.ones((2*erode+1,)*2, dtype=bool), border_value=0)
    return keep


def edgels(image):
    """(x, y, nx, ny) arrays of the detector's edgels on one frame."""
    found = vm.detect_edges(np.ascontiguousarray(image), vm.EdgeConfig())
    if not found:
        return np.zeros((4, 0))
    return np.array([(e.x, e.y, e.nx, e.ny) for e in found], dtype=float).T


def axis_coordinates(x, y, nx, ny, max_angle):
    """(coordinate, axis) of the edgels whose normal is within `max_angle` of an axis.

    `axis` is 0 for x (normal near ±x) and 1 for y. The coordinate is where the edgel's
    tangent crosses its pixel row (column): x + (y − round(y))·ny/nx.
    """
    cos = np.cos(np.radians(max_angle))
    norm = np.hypot(nx, ny)
    ax, ay = np.abs(nx)/norm, np.abs(ny)/norm
    along_x, along_y = ax >= cos, ay >= cos
    with np.errstate(divide='ignore', invalid='ignore'):
        cx = x + (y - np.round(y))*ny/nx
        cy = y + (x - np.round(x))*nx/ny
    coordinate = np.where(along_x, cx, cy)
    keep = along_x | along_y
    return coordinate[keep], np.where(along_x, 0, 1)[keep], keep


def dog_relocate(image, x, y, nx, ny, sigma=DOG_SIGMA, radius=DOG_RADIUS):
    """Each edgel moved along its normal to the peak of a continuous, tapered DoG response."""
    image = np.asarray(image, dtype=float)
    norm = np.hypot(nx, ny)
    nx, ny = nx/norm, ny/norm
    r = int(np.ceil(radius))
    oy, ox = np.mgrid[-r:r+1, -r:r+1]
    ox, oy = ox.ravel(), oy.ravel()
    qx = np.clip(np.round(x)[:, None] + ox, 0, image.shape[1]-1).astype(int)
    qy = np.clip(np.round(y)[:, None] + oy, 0, image.shape[0]-1).astype(int)
    values = image[qy, qx]

    def response(s):
        dx = x[:, None] + s[:, None]*nx[:, None] - qx
        dy = y[:, None] + s[:, None]*ny[:, None] - qy
        d2 = dx*dx + dy*dy
        taper = np.clip(1 - d2/radius**2, 0, None)**2
        along = dx*nx[:, None] + dy*ny[:, None]
        return np.abs(np.sum(values*along*np.exp(-d2/(2*sigma*sigma))*taper, axis=1))

    shift = np.zeros_like(x)
    for step, span in ((.25, 4), (.05, 2)):
        offsets = np.arange(-span, span+1)*step
        rows = np.array([response(shift + o) for o in offsets]).T
        best = np.clip(np.argmax(rows, axis=1), 1, len(offsets)-2)
        k = np.arange(len(x))
        a, b, c = rows[k, best-1], rows[k, best], rows[k, best+1]
        denominator = a - 2*b + c
        with np.errstate(divide='ignore', invalid='ignore'):
            vertex = np.where(denominator < 0, .5*(a - c)/denominator, 0.)
        shift = shift + offsets[best] + np.clip(vertex, -1, 1)*step
    return x + shift*nx, y + shift*ny


def histogram(coordinates):
    """Counts of the fractional parts in BINS equal bins."""
    fraction = coordinates - np.floor(coordinates)
    return np.bincount(np.minimum((fraction*BINS).astype(int), BINS-1), minlength=BINS)


def summarize(counts):
    """The normalized histogram (1.0 = uniform rate) and the locking index (max/min)."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return {'edges': 0, 'normalized': None, 'locking_index': None}
    normalized = counts*BINS/total
    index = float(counts.max()/counts.min()) if counts.min() > 0 else None
    return {'edges': int(total), 'normalized': [round(float(v), 4) for v in normalized],
            'normalized_range': [round(float(normalized.min()), 4), round(float(normalized.max()), 4)],
            'locking_index': None if index is None else round(index, 4)}


def evaluate(sequence, mask='visibility', erode=5, max_angle=25., refine='none', stride=1,
             locate=None):
    """Per-axis and pooled counts over one sequence's frames.

    `locate(image) -> (x, y, nx, ny)` replaces the detector (for tests).
    """
    locate = locate or edgels
    counts = np.zeros((2, BINS), dtype=np.int64)
    frames = 0
    for i in range(0, len(sequence), stride):
        if sequence.frames[i]['image'] is None:
            continue
        image = sequence.image(i)
        x, y, nx, ny = locate(image)
        keep = frame_mask(sequence, i, mask, erode)
        h, w = keep.shape
        px, py = np.round(x).astype(int), np.round(y).astype(int)
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        inside[inside] = keep[py[inside], px[inside]]
        x, y, nx, ny = x[inside], y[inside], nx[inside], ny[inside]
        coordinate, axis, near_axis = axis_coordinates(x, y, nx, ny, max_angle)
        if refine == 'dog' and near_axis.any():
            rx, ry = dog_relocate(image, x[near_axis], y[near_axis], nx[near_axis], ny[near_axis])
            coordinate, axis, _ = axis_coordinates(rx, ry, nx[near_axis], ny[near_axis], max_angle)
        for a in (0, 1):
            counts[a] += histogram(coordinate[axis == a])
        frames += 1
    return {'frames': frames, 'counts': counts.tolist(),
            'x': summarize(counts[0]), 'y': summarize(counts[1]), 'pooled': summarize(counts.sum(0))}


def text_summary(report):
    lines = [f"locking diagnostic: detector {report['settings']['detector']}, "
             f"refine {report['settings']['refine']}, mask {report['settings']['mask']}"]
    for name, row in list(report['sequences'].items()) + [('all', report['aggregate'])]:
        pooled = row['pooled']
        index = pooled['locking_index']
        lines.append(f"{name}: {pooled['edges']} edges, locking index "
                     f"{'inf' if index is None else f'{index:.3f}'}, normalized range "
                     f"{pooled.get('normalized_range')}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, nargs='+', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--mask', choices=MASKS, default='visibility',
                        help='visibility (default), validity (the registration mask), both, or none')
    parser.add_argument('--erode', type=int, default=5, help='mask erosion, px')
    parser.add_argument('--max-angle', type=float, default=25.)
    parser.add_argument('--refine', choices=('none', 'dog'), default='none')
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    if args.stride < 1 or args.erode < 0 or not 0 < args.max_angle < 45:
        parser.error('stride ≥ 1, erode ≥ 0 and 0 < max-angle < 45')
    sequences = [Sequence(path) for path in args.manifest]
    if len({s.name for s in sequences}) != len(sequences):
        parser.error('sequence names must be unique')
    out = args.out_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    if out.is_relative_to(repo) or any(out.is_relative_to(s.root) for s in sequences):
        parser.error('outputs must be outside the repository and the manifest directories')
    out.mkdir(parents=True, exist_ok=True)
    report = {'settings': {'detector': 'detect_edges, EdgeConfig defaults', 'refine': args.refine,
                           'mask': args.mask, 'erode_px': args.erode, 'max_angle_deg': args.max_angle,
                           'stride': args.stride, 'bins': BINS},
              'sequences': {}}
    total = np.zeros((2, BINS), dtype=np.int64)
    for sequence in sequences:
        row = evaluate(sequence, args.mask, args.erode, args.max_angle, args.refine, args.stride)
        row['split'] = sequence.split
        report['sequences'][sequence.name] = row
        total += np.asarray(row['counts'])
    report['aggregate'] = {'counts': total.tolist(), 'x': summarize(total[0]), 'y': summarize(total[1]),
                           'pooled': summarize(total.sum(0))}
    (out / 'phase_histogram.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    text = text_summary(report)
    (out / 'summary.txt').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
