"""Caller-owned similarity composition and independently supported seed anchors."""
import numpy as np

IDENTITY = (0., 0., 0., 1.)


def matrix(parts):
    x, y, angle, scale = parts
    c, s = scale*np.cos(angle), scale*np.sin(angle)
    return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]])


def parts(transform):
    return (float(transform[0, 2]), float(transform[1, 2]),
            float(np.arctan2(transform[1, 0], transform[0, 0])),
            float(np.hypot(transform[0, 0], transform[1, 0])))


def corner_distance(a, b, shape):
    h, w = shape
    points = np.array([[0, w-1, 0, w-1], [0, 0, h-1, h-1], [1, 1, 1, 1]])
    return float(np.linalg.norm(((a-b) @ points)[:2], axis=0).max())


class AnchoredSequence:
    """One supported segment; the fixed first keyframe is never silently replaced.

    `raw` is the uncorrected adjacent composition, `pose` accepts direct constraints.
    Failed pair/anchor/camera transitions are handled explicitly by the caller.
    """
    def __init__(self, seed, shape):
        self.seed = seed
        self.shape = shape
        self.raw = np.eye(3)
        self.pose = np.eye(3)

    def advance(self, pair):
        if pair['status'] != 'accepted':
            raise ValueError('cannot compose an unsupported registration')
        relative = matrix(pair['transform'])
        self.raw = relative @ self.raw
        self.pose = relative @ self.pose

    def constrain(self, anchor):
        if anchor['status'] != 'accepted':
            return None
        anchored = matrix(anchor['transform'])
        correction = corner_distance(self.pose, anchored, self.shape)
        self.pose = anchored
        return correction

    def diagnostics(self):
        return {'seed': self.seed, 'chained_pose': parts(self.raw),
                'anchored_pose': parts(self.pose),
                'chain_anchor_departure_px': corner_distance(self.raw, self.pose, self.shape)}
