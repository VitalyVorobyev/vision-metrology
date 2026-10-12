"""Read-only adapter for private private sequence dataset data; never import its mutable Store."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates

DEVELOPMENT = ('15074', '15077', '15082', '15516')
HELD_OUT = ('15090', '15521')


def decode(runs, shape):
    values = np.repeat(np.arange(len(runs), dtype=np.uint32) % 2, runs)
    if values.size != np.prod(shape):
        raise ValueError('RLE mask dimensions disagree with cached analysis')
    return values.reshape(shape).astype(bool)


def bandpass(image, valid):
    """Caller-owned normalized DoG; excluded pixels contribute no filter mass.

    Mirrors the application preprocessing in private sequence dataset. The library estimator
    itself accepts original pixel units and does not select a filtering policy.
    """
    mass = valid.astype(np.float32)
    weighted = np.where(valid, image, 0).astype(np.float32)
    smooth = []
    for sigma in (1., 4.):
        denominator = gaussian_filter(mass, sigma, mode='nearest')
        numerator = gaussian_filter(weighted, sigma, mode='nearest')
        smooth.append(numerator / np.maximum(denominator, 1e-6))
    return np.ascontiguousarray((smooth[0] - smooth[1]) * mass, dtype=np.float32)


class Dataset:
    def __init__(self, root, inspection):
        self.root = Path(root).resolve()
        self.inspection = inspection
        self.sequence = 'r1/' + inspection
        self.manifest = json.loads((self.root / 'pixel-manifest.json').read_text())
        if self.manifest['coordinate_convention'] != 'pixel_centres':
            raise ValueError('unsupported dataset coordinate convention')
        if self.manifest.get('source_flip_x') not in (None, False):
            raise ValueError('flipped source convention requires an explicit adapter')
        self.sequences = {s['id']: s for s in self.manifest['sequences']}
        self.analysis = json.loads((self.root / 'pixel-labels/cache' /
                                   f'analysis-r1_{inspection}.json').read_text())
        self.shape = tuple(reversed(self.analysis['dims']))
        self.annotations = {}
        db = self.root / 'pixel-labels/labels.sqlite'
        if db.exists():
            # URI read-only mode: no migration, journal change, or application jobs.
            with sqlite3.connect('file:' + quote(str(db)) + '?mode=ro', uri=True) as conn:
                for frame, camera, metadata, mask in conn.execute(
                        'SELECT frame,camera,metadata,mask FROM annotations WHERE sequence=?',
                        (self.sequence,)):
                    item = json.loads(metadata)
                    item['mask'] = np.frombuffer(zlib.decompress(mask), dtype=np.uint8).reshape(self.shape).copy()
                    self.annotations[frame, camera] = item
        labels = self.root / 'pixel-labels/cache' / f'labels-r1_{inspection}.json'
        self.labels = json.loads(labels.read_text()) if labels.exists() else None
        self.images = {}
        self.prepared = {}

    def strip(self, frame, camera, blank=False):
        key = (blank, frame, camera)
        if key not in self.images:
            sequence = ('r1wo/' if blank else 'r1/') + self.inspection
            info = self.sequences[sequence]['frames'][frame]['image']
            path = (self.root / info['path']).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('manifest image path leaves the dataset')
            if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
                raise ValueError(f'image hash mismatch: {sequence}, frame {frame}')
            array = np.asarray(Image.open(path))
            h, w = self.shape
            if array.shape != (3*h, w):
                raise ValueError('unexpected stacked-camera image dimensions')
            self.images[key] = np.ascontiguousarray(array[camera*h:(camera+1)*h])
        return self.images[key]

    def static_exclusions(self, camera):
        c = self.analysis['cameras'][camera]
        return decode(c['silhouette_rle'], self.shape) | decode(c['unlit_rle'], self.shape)

    def difference(self, frame, camera):
        c = self.analysis['cameras'][camera]
        reg = c['registration'][frame]
        if reg['shift_status'] != 'ok' or reg['blank_idx'] is None:
            return None, None
        # Use the cached registered correspondence, including fallback methods
        # where vendor stamps freeze; never pair by equal frame numbers.
        blank = self.strip(reg['blank_idx'], camera, blank=True)
        y, x = np.indices(self.shape, dtype=np.float64)
        static = self.static_exclusions(camera)
        dx, dy = reg['d']
        sx, sy = np.where(static, x, x+dx), np.where(static, y, y+dy)
        valid = (sx >= 0) & (sy >= 0) & (sx < self.shape[1]-1) & (sy < self.shape[0]-1) & ~static
        registered = map_coordinates(blank.astype(np.float32), [sy, sx], order=1, mode='constant', cval=0)
        difference = self.strip(frame, camera).astype(np.float32) - (reg['gain'] * registered + reg['offset'])
        foreground = (difference > max(25., 5*reg['sigma_gl'])) & valid
        return foreground, difference

    def prepare(self, frame, camera):
        key = frame, camera
        if key not in self.prepared:
            c = self.analysis['cameras'][camera]
            foreground = decode(c['support_rle'][frame], self.shape)
            annotation = self.annotations.get(key)
            if annotation and annotation.get('checked') and not annotation.get('stale'):
                foreground |= annotation['mask'] != 0
            difference, _ = self.difference(frame, camera)
            if difference is not None:
                foreground |= difference
            excluded = self.static_exclusions(camera) | binary_dilation(foreground, iterations=3)
            valid = np.ascontiguousarray(~excluded, dtype=np.uint8)
            filtered = bandpass(self.strip(frame, camera), valid != 0)
            self.prepared[key] = filtered, valid
        return self.prepared[key]

    def seed(self, requested=None):
        if not self.labels:
            raise ValueError('no saved ribbon geometry; propagation needs a seed model')
        candidates = []
        for f in self.labels['frames']:
            if requested is not None and f['frame'] != requested:
                continue
            cam = f['camera']
            if cam != self.analysis['roles']['rear'][f['frame']]:
                continue
            a = self.annotations.get((f['frame'], cam), {})
            reviewed = a.get('checked') and not a.get('stale') and a.get('ribbons')
            ribbons = a['ribbons'] if reviewed else f['ribbons']
            if not ribbons or not ribbons[0]['derived']['sections']:
                continue
            if requested is None and not reviewed and f['qa']['grade'] != 'green':
                continue
            candidates.append((0 if reviewed else 1,
                               abs(f['frame'] - self.analysis['bead_frames']//2), f, ribbons, a))
        if not candidates:
            raise ValueError('no usable saved seed geometry')
        _, _, f, ribbons, a = min(candidates, key=lambda c: (c[0], c[1], c[2]['frame']))
        return f['frame'], f['camera'], ribbons[0], {
            'geometry': 'reviewed_annotation' if a.get('ribbons') and a.get('checked') and not a.get('stale') else 'generated_cached_ribbon',
            'reviewed_mask': bool(a.get('checked') and not a.get('stale')),
            'label_version': self.labels.get('label_version'),
        }
