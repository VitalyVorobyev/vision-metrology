"""Two-edge cross-section annotation: stations, magnified profiles, blind passes, storage.

Pure logic with no UI. `annotate.py` serves it as a local page and the tests drive it
headlessly. The README's protocol section says how to use it.

A *section* is a station on a frame's curve and the unit normal there. Its two boundaries
are signed offsets along that normal, in px from the station: `left` is the boundary on
the negative side, `right` the one on the positive side, so `left < right`. A boundary
the annotator cannot see is `None` with its `*_visible` flag false. The normal is the
curve's tangent turned by +90° in image coordinates, `(-ty, tx)`.

Every section is annotated twice by each annotator. Pass 1 visits all sections in one
shuffled order, pass 2 in another, and a task never shows an earlier mark, so the two
passes are blind to each other and their difference gives the annotator's σ.

The annotations file is JSON:
`{version, protocol, sections: [{frame, station, normal, s, source}], annotations: [...]}`.
A manifest names it (`"annotations": "annotations.json"`), or carries the `annotations`
list inline.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

VERSION = 1
STEP = .05            # profile sampling step, px
HALF_LENGTH = 12.     # profile reach either side of the station, px
TANGENT_SPAN = 2.     # arc length either side of a station that sets its normal, px
DIGITS = 4            # stored station coordinates, decimals
FIELDS = ('frame', 'station', 'normal', 'left', 'right', 'left_visible', 'right_visible',
          'annotator', 'pass', 'timestamp')


# Curves -------------------------------------------------------------------------------

def arc_lengths(polyline):
    points = np.asarray(polyline, dtype=float)
    return np.r_[0., np.cumsum(np.hypot(*np.diff(points, axis=0).T))]


def point_at(polyline, s):
    """The point at arc length `s`, clamped to the curve."""
    points = np.asarray(polyline, dtype=float)
    lengths = arc_lengths(points)
    s = float(np.clip(s, 0, lengths[-1]))
    return np.array([np.interp(s, lengths, points[:, 0]), np.interp(s, lengths, points[:, 1])])


def normal_at(polyline, s, span=TANGENT_SPAN):
    """The unit normal `(-ty, tx)` at arc length `s`, from the chord over `s ± span`."""
    tangent = point_at(polyline, s + span) - point_at(polyline, s - span)
    norm = np.hypot(*tangent)
    if norm == 0:
        raise ValueError('a curve with no extent has no normal')
    return np.array([-tangent[1], tangent[0]]) / norm


def project(polyline, point):
    """The arc length of the curve point nearest to `point`."""
    points = np.asarray(polyline, dtype=float)
    a, b = points[:-1], points[1:]
    d = b - a
    length2 = np.maximum(np.sum(d*d, axis=1), 1e-12)
    t = np.clip(np.sum((np.asarray(point, dtype=float) - a)*d, axis=1) / length2, 0, 1)
    distance = np.hypot(*(a + t[:, None]*d - point).T)
    k = int(np.argmin(distance))
    return float(arc_lengths(points)[k] + t[k]*np.sqrt(length2[k]))


def load_curves(sequence, path=None):
    """`{frame: polyline}`: the manifest's seed polyline, or a curves file.

    A curves file is `{"curves": [{"frame": i, "polyline": [[x, y], ...]}, ...]}`, for
    example a tracked curve per frame, in the frame's own pixel-centre coordinates.
    """
    if path is None:
        if sequence.seed is None:
            raise ValueError('the manifest has no seed; pass a curves file')
        return {int(sequence.seed['frame']): sequence.seed['polyline']}
    curves = {}
    for row in json.loads(Path(path).read_text()).get('curves', []):
        frame = row.get('frame')
        if not isinstance(frame, int) or not 0 <= frame < len(sequence):
            raise ValueError(f'curve frame {frame!r} is not a frame index')
        if len(row.get('polyline') or []) < 2 or arc_lengths(row['polyline'])[-1] == 0:
            raise ValueError(f'frame {frame}: a curve needs two distinct points')
        if frame in curves:
            raise ValueError(f'frame {frame}: more than one curve')
        curves[frame] = row['polyline']
    if not curves:
        raise ValueError('the curves file has no curves')
    return curves


# Sections -----------------------------------------------------------------------------

def section(frame, polyline, s, source):
    station, normal = point_at(polyline, s), normal_at(polyline, s)
    return {'frame': int(frame), 'station': [round(float(v), DIGITS) for v in station],
            'normal': [round(float(v), 6) + 0. for v in normal], 's': round(float(s), DIGITS),
            'source': source}


def stratified(total, count, rng):
    """`count` positions in `[0, total)`, one uniformly at random in each equal stratum."""
    return (np.arange(count) + rng.uniform(size=count)) * total / count


def uniform_sections(curves, count, seed):
    """`count` sections spread uniformly over the curves' joint arc length (seeded)."""
    frames = sorted(curves)
    lengths = np.array([arc_lengths(curves[f])[-1] for f in frames])
    starts = np.r_[0., np.cumsum(lengths)]
    sections = []
    for u in stratified(starts[-1], count, np.random.default_rng(seed)):
        k = min(int(np.searchsorted(starts, u, side='right')) - 1, len(frames) - 1)
        sections.append(section(frames[k], curves[frames[k]], u - starts[k], 'uniform'))
    return sections


def span_sections(curves, frame, s0, s1, count, seed):
    """`count` sections spread uniformly over `[s0, s1]` of one frame's curve: a hard span."""
    if frame not in curves:
        raise ValueError(f'frame {frame} has no curve')
    if not 0 <= s0 < s1:
        raise ValueError('a span needs 0 <= s0 < s1')
    rng = np.random.default_rng([seed, frame, 1])
    return [section(frame, curves[frame], s0 + u, 'span') for u in stratified(s1 - s0, count, rng)]


def key(entry):
    """The identity of a section or annotation: its frame and station."""
    x, y = entry['station']
    return int(entry['frame']), round(float(x), DIGITS), round(float(y), DIGITS)


# Profiles -----------------------------------------------------------------------------

def offsets(half_length=HALF_LENGTH, step=STEP):
    n = int(round(half_length / step))
    return np.arange(-n, n + 1) * step


def profile(image, visible, station, normal, half_length=HALF_LENGTH, step=STEP,
            method='cubic', sigma=1.):
    """The magnified profile along the normal: `{offsets, values, derivative, valid}`.

    `cubic` interpolates the original pixels with a cubic spline; `sampler` reads the
    bindings' `ProfileSampler` (a continuous derivative of Gaussian of `sigma` px). A
    sample outside the frame, or on a pixel that `visible` excludes, is invalid (NaN).
    """
    t = offsets(half_length, step)
    station, normal = np.asarray(station, dtype=float), np.asarray(normal, dtype=float)
    points = station + t[:, None]*normal
    h, w = image.shape
    nearest = np.rint(points).astype(int)
    valid = (nearest[:, 0] >= 0) & (nearest[:, 0] < w) & (nearest[:, 1] >= 0) & (nearest[:, 1] < h)
    valid[valid] = visible[nearest[valid, 1], nearest[valid, 0]]
    if method == 'cubic':
        margin = 4
        x0, y0 = (np.floor(points.min(axis=0)) - margin).astype(int)
        x1, y1 = (np.ceil(points.max(axis=0)) + margin + 1).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        window = np.asarray(image[y0:y1, x0:x1], dtype=float)
        values = map_coordinates(window, [points[:, 1] - y0, points[:, 0] - x0], order=3,
                                 mode='nearest')
        derivative = np.gradient(values, step)
    elif method == 'sampler':
        import vision_metrology as vm
        source = np.ascontiguousarray(image)
        if source.dtype not in (np.uint8, np.uint16, np.float32):
            source = source.astype(np.float32)
        mask = np.ascontiguousarray(visible, dtype=np.uint8)*255
        values, derivative, ok = vm.ProfileSampler(sigma=sigma).sample(
            source, station[None].astype(np.float32), normal[None].astype(np.float32),
            t.astype(np.float32), mask=mask)
        values, derivative = values[0].astype(float), derivative[0].astype(float)
        valid &= ok[0]
    else:
        raise ValueError(f'unknown profile method {method!r}')
    values, derivative = values.copy(), derivative.copy()
    values[~valid] = derivative[~valid] = np.nan
    return {'offsets': t, 'values': values, 'derivative': derivative, 'valid': valid}


def crop(image, station, radius):
    """(uint8 crop stretched to its 1–99 % range, (x0, y0) of its top-left pixel)."""
    h, w = image.shape
    cx, cy = (int(round(v)) for v in station)
    x0, y0 = max(cx - radius, 0), max(cy - radius, 0)
    window = np.asarray(image[y0:min(cy + radius + 1, h), x0:min(cx + radius + 1, w)], dtype=float)
    if window.size == 0:
        raise ValueError('the station lies outside the frame')
    lo, hi = np.percentile(window, (1, 99))
    stretched = np.clip((window - lo) / max(hi - lo, 1e-9), 0, 1)
    return np.rint(stretched*255).astype(np.uint8), (x0, y0)


# Records and the file -----------------------------------------------------------------

def record(entry, left, right, annotator, pass_, timestamp=None):
    """One annotation of the section `entry`; `None` marks a boundary not visible."""
    result = {'frame': entry['frame'], 'station': list(entry['station']),
              'normal': list(entry['normal']),
              'left': None if left is None else round(float(left), DIGITS),
              'right': None if right is None else round(float(right), DIGITS),
              'left_visible': left is not None, 'right_visible': right is not None,
              'annotator': str(annotator), 'pass': int(pass_),
              'timestamp': timestamp or datetime.now(timezone.utc).isoformat(timespec='seconds')}
    check(result)
    return result


def check(entry, frames=None):
    """Raise ValueError unless `entry` is a well-formed annotation."""
    missing = [f for f in FIELDS if f not in entry]
    if missing:
        raise ValueError(f'annotation lacks {", ".join(missing)}')
    if not isinstance(entry['frame'], int) or (frames is not None and not 0 <= entry['frame'] < frames):
        raise ValueError(f'annotation frame {entry["frame"]!r} is not a frame index')
    if len(entry['station']) != 2 or len(entry['normal']) != 2:
        raise ValueError('annotation station and normal are [x, y] pairs')
    if abs(np.hypot(*entry['normal']) - 1) > 1e-3:
        raise ValueError('annotation normal is not a unit vector')
    if entry['pass'] not in (1, 2):
        raise ValueError('annotation pass is 1 or 2')
    if not entry['annotator']:
        raise ValueError('annotation needs an annotator')
    for side in ('left', 'right'):
        if entry[f'{side}_visible'] != (entry[side] is not None):
            raise ValueError(f'annotation {side}: a visible boundary has a position, a hidden one none')
    if entry['left'] is not None and entry['right'] is not None and not entry['left'] < entry['right']:
        raise ValueError('annotation left must lie before right along the normal')


def read(path, frames=None):
    """The annotations file, validated."""
    doc = json.loads(Path(path).read_text())
    if doc.get('version', VERSION) != VERSION:
        raise ValueError(f'unsupported annotations version {doc.get("version")}')
    for entry in doc.setdefault('annotations', []):
        check(entry, frames)
    doc.setdefault('sections', [])
    doc.setdefault('protocol', {})
    return doc


def write(path, doc):
    """Replace the annotations file atomically."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    os.replace(temporary, path)


# Sessions -----------------------------------------------------------------------------

class Session:
    """The blind two-pass queue of one annotations file, persisted after every change."""

    def __init__(self, path, doc):
        self.path, self.doc = Path(path), doc
        self.seed = int(doc['protocol'].get('seed', 0))

    @classmethod
    def create(cls, path, sections, protocol):
        if Path(path).exists():
            raise FileExistsError(f'{path} exists; open it to resume')
        doc = {'version': VERSION, 'protocol': dict(protocol), 'sections': [], 'annotations': []}
        session = cls(path, doc)
        for entry in sections:
            session._append(entry)
        write(path, doc)
        return session

    @classmethod
    def open(cls, path, frames=None):
        return cls(path, read(path, frames))

    @property
    def sections(self):
        return self.doc['sections']

    def _append(self, entry):
        if key(entry) in {key(s) for s in self.sections}:
            return False
        self.sections.append(entry)
        return True

    def add(self, entry):
        """Add a section (a user station). Its two passes join the queue."""
        added = self._append(entry)
        if added:
            write(self.path, self.doc)
        return added

    def order(self, pass_):
        """The section indices in pass `pass_`'s shuffled order.

        Each section's rank is a hash of the seed, the pass and the section, so adding a
        section inserts it somewhere at random without reordering the others.
        """
        def rank(i):
            text = f'{self.seed}:{pass_}:{key(self.sections[i])}'.encode()
            return hashlib.sha256(text).hexdigest()
        return sorted(range(len(self.sections)), key=rank)

    def done(self, annotator):
        index = {key(s): i for i, s in enumerate(self.sections)}
        return {(index[key(a)], a['pass']) for a in self.doc['annotations']
                if a['annotator'] == annotator and key(a) in index}

    def next(self, annotator):
        """(section index, pass) of the next task, or None when both passes are complete."""
        done = self.done(annotator)
        for pass_ in (1, 2):
            for i in self.order(pass_):
                if (i, pass_) not in done:
                    return i, pass_
        return None

    def progress(self, annotator):
        return len(self.done(annotator)), 2*len(self.sections)

    def annotate(self, annotator, index, pass_, left, right):
        """Record one task's boundaries: a pending task, and pass 2 only after pass 1."""
        done = self.done(annotator)
        if not 0 <= index < len(self.sections) or pass_ not in (1, 2) or (index, pass_) in done:
            raise ValueError('that task is not pending')
        if pass_ == 2 and (index, 1) not in done:
            raise ValueError('pass 2 of a section follows its pass 1')
        entry = record(self.sections[index], left, right, annotator, pass_)
        self.doc['annotations'].append(entry)
        write(self.path, self.doc)
        return entry
