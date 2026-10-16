"""Write a synthetic `sequence.json`: a textured substrate under known similarity motion.

A light bead with two boundaries rides on the substrate. Each frame's mask excludes the
bead (dilated by 3 px) and a stationary block, so registration sees only substrate. The
visibility mask excludes only the stationary block. The sequence splits into two groups,
so its middle pair is a handover. Truth holds every frame's pose relative to frame 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import erf

from anchors import matrix

SHAPE = (97, 161)
BEAD_Y, BEAD_HALF = 48., 4.
BLOCK = (slice(8, 26), slice(118, 146))


def pose(frame):
    """Frame 0 to frame `frame`, as (tx, ty, angle, scale)."""
    return (frame*.15, -frame*.025, frame*.00015, float(np.exp(frame*.00008)))


def render(frame, shape=SHAPE):
    """(uint8 image, bool registration mask) of one frame."""
    h, w = shape
    y, x = np.mgrid[:h, :w].astype(float)
    xy = np.linalg.inv(matrix(pose(frame))) @ np.array([x.ravel(), y.ravel(), np.ones(x.size)])
    u, v = xy[0].reshape(shape), xy[1].reshape(shape)
    values = (100 + 25*np.sin(u*.17+v*.13) + 20*np.cos(u*.08-v*.21)
              + 15*np.sin(u*.31+v*.07) + 10*np.cos(u*.03+v*.39))
    bead = .5*(erf((v-(BEAD_Y-BEAD_HALF))/.8) - erf((v-(BEAD_Y+BEAD_HALF))/.8))
    values = values*(1-bead) + 215*bead
    values += np.random.default_rng(7919*frame+1).uniform(-2, 2, values.shape)
    values[BLOCK] = 235.
    valid = np.abs(v-BEAD_Y) > BEAD_HALF+3
    valid[BLOCK] = False
    return np.clip(np.rint(values), 0, 255).astype(np.uint8), valid


def transform(points, frame):
    points = np.asarray(points, dtype=float)
    return (matrix(pose(frame)) @ np.c_[points, np.ones(len(points))].T)[:2].T.tolist()


def generate(out_dir, frames=20, shape=SHAPE):
    """Write the frames, masks and `sequence.json` into `out_dir`; return the manifest path."""
    out = Path(out_dir)
    (out / 'frames').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir(exist_ok=True)
    visibility = np.ones(shape, dtype=np.uint8)*255
    visibility[BLOCK] = 0
    Image.fromarray(visibility).save(out / 'visibility.png')
    entries = []
    for frame in range(frames):
        image, valid = render(frame, shape)
        Image.fromarray(image).save(out / 'frames' / f'{frame:05}.png')
        Image.fromarray(valid.astype(np.uint8)*255).save(out / 'masks' / f'{frame:05}.png')
        entries.append({'image': f'frames/{frame:05}.png', 'mask': f'masks/{frame:05}.png',
                        'visibility': 'visibility.png', 'group': 0 if frame < frames//2 else 1})
    xs = np.arange(10., shape[1]-10, 10.)
    left = [[x, BEAD_Y-BEAD_HALF] for x in xs]
    right = [[x, BEAD_Y+BEAD_HALF] for x in xs]
    manifest = {
        'version': 1, 'name': out.name, 'convention': 'pixel_centres', 'split': 'development',
        'provenance': {'source': 'synthetic', 'generator': 'tools/registration_eval/synthetic.py'},
        'frames': entries,
        'seed': {'frame': 0, 'polyline': [[x, BEAD_Y] for x in xs],
                 'provenance': 'synthetic truth at frame 0',
                 'walls': [left, right],
                 'sections': [{'s': float(x-xs[0]), 'normal': [0., 1.], 'left': l, 'right': r,
                               'center': [x, BEAD_Y], 'width': 2*BEAD_HALF}
                              for x, l, r in zip(xs, left, right)]},
        'truth': {'motion': [list(pose(frame)) for frame in range(frames)],
                  'boundaries': [[transform(left, f), transform(right, f)] for f in range(frames)]},
    }
    path = out / 'sequence.json'
    path.write_text(json.dumps(manifest, indent=1))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=20)
    args = parser.parse_args()
    if args.frames < 4:
        parser.error('at least four frames')
    print(generate(args.out_dir, args.frames))


if __name__ == '__main__':
    main()
