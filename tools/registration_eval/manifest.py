"""Reader for the neutral sequence manifest (`sequence.json`), and caller preprocessing.

The format is documented in this directory's README. A manifest names frames, optional
validity and visibility masks, optional groups (a group change is a handover), and
optional seed geometry, truth, comparison estimates and annotations. Paths resolve
relative to the manifest's own directory and may not leave it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import annotation
from anchors import matrix

VERSION = 1
DOG_SIGMAS = (1., 4.)


def bandpass(image, valid, sigmas=DOG_SIGMAS):
    """Masked normalized difference of Gaussians; excluded pixels contribute no mass.

    This is one caller-side preprocessing choice for textured substrates. The library
    estimators accept original pixel units and do not select a filtering policy.
    """
    mass = valid.astype(np.float32)
    weighted = np.where(valid, image, 0).astype(np.float32)
    smooth = []
    for sigma in sigmas:
        denominator = gaussian_filter(mass, sigma, mode='nearest')
        numerator = gaussian_filter(weighted, sigma, mode='nearest')
        smooth.append(numerator / np.maximum(denominator, 1e-6))
    return np.ascontiguousarray((smooth[0] - smooth[1]) * mass, dtype=np.float32)


def load_array(path):
    """A single-channel image: any Pillow-readable file, or a 2-D `.npy` array."""
    if path.suffix == '.npy':
        array = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f'{path.name}: expected a single-channel image, got shape {array.shape}')
    return np.ascontiguousarray(array)


class Sequence:
    """One manifest. Frames are indexed by their position in `frames`."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        doc = json.loads(self.path.read_text())
        if doc.get('version', VERSION) != VERSION:
            raise ValueError(f'unsupported manifest version {doc.get("version")}')
        if doc.get('convention', 'pixel_centres') != 'pixel_centres':
            raise ValueError('only pixel-centre coordinates are supported')
        self.name = str(doc.get('name') or self.root.name)
        self.split = doc.get('split')
        if self.split not in (None, 'development', 'held_out'):
            raise ValueError(f'unknown split {self.split!r}')
        self.provenance = doc.get('provenance', {})
        frames = doc.get('frames')
        if not isinstance(frames, list) or not frames:
            raise ValueError('a manifest needs a non-empty frames list')
        self.frames = []
        for i, frame in enumerate(frames):
            group = frame.get('group', 0)
            image = frame.get('image')
            if image is None and group is not None:
                raise ValueError(f'frame {i}: only an unassigned frame (group null) may omit its image')
            self.frames.append({
                'image': self._resolve(image), 'group': group,
                'mask': self._resolve(frame.get('mask')),
                'visibility': self._resolve(frame.get('visibility')),
            })
        first = next((f['image'] for f in self.frames if f['image'] is not None), None)
        if first is None:
            raise ValueError('a manifest needs at least one image')
        self.shape = load_array(first).shape
        self.seed = doc.get('seed')
        if self.seed is not None:
            self._check_seed(self.seed)
        self.truth = doc.get('truth') or {}
        motion = self.truth.get('motion')
        if motion is not None and len(motion) != len(self.frames):
            raise ValueError('truth.motion needs one pose (or null) per frame')
        self.comparison = doc.get('comparison')
        if self.comparison is not None and len(self.comparison.get('shifts', [])) != len(self.frames)-1:
            raise ValueError('comparison.shifts needs one entry (or null) per adjacent pair')
        annotations = doc.get('annotations', [])
        self.annotations_path = None
        if isinstance(annotations, str):
            self.annotations_path = self._resolve(annotations)
            annotations = (annotation.read(self.annotations_path, len(self.frames))['annotations']
                           if self.annotations_path.exists() else [])
        elif isinstance(annotations, list):
            # An entry with no annotator is an exact cross-section from a generator, with
            # wall points; it is carried through unchanged and never scored as an annotation.
            for entry in annotations:
                if 'annotator' in entry:
                    annotation.check(entry, len(self.frames))
        else:
            raise ValueError('annotations is a list or the path of an annotations file')
        self.annotations = annotations
        self._images, self._masks, self._visible, self._prepared = {}, {}, {}, {}

    @property
    def annotated(self):
        """The hand annotations: the `annotations` entries that name an annotator."""
        return [entry for entry in self.annotations if 'annotator' in entry]

    def __len__(self):
        return len(self.frames)

    def _resolve(self, relative):
        if relative is None:
            return None
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f'{relative}: manifest paths may not leave the manifest directory')
        return path

    def _check_seed(self, seed):
        if not 0 <= seed.get('frame', -1) < len(self.frames):
            raise ValueError('seed.frame is not a frame index')
        if self.frames[seed['frame']]['group'] is None:
            raise ValueError('seed.frame belongs to no group')
        if len(seed.get('polyline', [])) < 2:
            raise ValueError('seed.polyline needs at least two points')

    def group(self, i):
        return self.frames[i]['group']

    def image(self, i):
        if i not in self._images:
            path = self.frames[i]['image']
            if path is None:
                raise ValueError(f'frame {i} has no image')
            array = load_array(path)
            if array.shape != self.shape:
                raise ValueError(f'frame {i}: shape {array.shape} differs from {self.shape}')
            self._images[i] = array
        return self._images[i]

    def _bool(self, i, key, cache):
        if i not in cache:
            path = self.frames[i][key]
            if path is None:
                cache[i] = np.ones(self.shape, dtype=bool)
            else:
                array = load_array(path)
                if array.shape != self.shape:
                    raise ValueError(f'frame {i}: {key} shape {array.shape} differs from {self.shape}')
                cache[i] = array != 0
        return cache[i]

    def mask(self, i):
        """Registration validity: True where the pixel may serve as substrate evidence."""
        return self._bool(i, 'mask', self._masks)

    def visible(self, i):
        """Geometry visibility: True where seed geometry may be observed."""
        return self._bool(i, 'visibility', self._visible)

    def prepare(self, i, preprocess='dog'):
        """(filtered float32 or original image, uint8 0/1 validity) for the estimators."""
        key = i, preprocess
        if key not in self._prepared:
            valid = self.mask(i)
            image = self.image(i)
            if preprocess == 'dog':
                prepared = bandpass(image, valid)
            elif preprocess == 'none':
                prepared = image
            else:
                raise ValueError(f'unknown preprocessing {preprocess!r}')
            self._prepared[key] = prepared, np.ascontiguousarray(valid, dtype=np.uint8)
        return self._prepared[key]

    def truth_map(self, previous, current):
        """The known previous-to-current matrix, or None."""
        motion = self.truth.get('motion')
        if motion is None or motion[previous] is None or motion[current] is None:
            return None
        return matrix(motion[current]) @ np.linalg.inv(matrix(motion[previous]))

    def comparison_shift(self, previous):
        """Another method's shift for the pair (previous, previous+1); never truth."""
        if self.comparison is None:
            return None
        return self.comparison['shifts'][previous]
