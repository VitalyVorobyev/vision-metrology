"""Scene geometry for the sequence benchmark: the path, the ribbon, poses and views.

Three frames, each with x right and y down:
- **ribbon** coordinates, where the bead's centreline, walls and caps are defined;
- **substrate** coordinates, fixed to the moving part; the ribbon is fixed to it, except
  in the slip variant, where a per-frame `slip` pose maps ribbon to substrate;
- **plane** coordinates, fixed to the rig; a per-frame `pose` maps substrate to plane.
A view's `plane_to_image` maps plane to image pixels, where integer coordinates are pixel
centres. Lengths are in substrate units (su); a view's scale is in px/su.

The centreline is a chain of lines and circular arcs, continued by tangent half-lines past
both ends, parametrised by arc length `s`. `t(s)` is the unit tangent and
`n(s) = (-t_y, t_x)`, so on screen +n points to the right of travel. A point's ribbon
coordinates `(s, d)` are its nearest centreline location and its signed offset along
`n(s)`. Within the curves' radius of curvature this is exact: the station normal at `s`
is the set of points whose nearest location is `s`. A positive arc sweep turns clockwise
on screen.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, ndtr


def edge(z, soft, kind='erf'):
    """Soft step from 0 to 1 at z = 0: Φ(z/soft) for 'erf', 1/(1+exp(-z/soft)) for 'logistic'."""
    if kind == 'logistic':
        return expit(z / soft)
    return ndtr(z / soft)


def box(d, lo, hi, soft, kind='erf'):
    """Soft indicator of lo < d < hi."""
    return edge(d - lo, soft, kind) * edge(hi - d, soft, kind)


def se2(pose):
    """3×3 matrix of an SE(2) pose (tx, ty, θ)."""
    tx, ty, th = pose
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, tx], [s, c, ty], [0., 0., 1.]])


def apply(matrix, points):
    """Map (..., 2) points through a 3×3 homogeneous matrix."""
    points = np.asarray(points, dtype=float)
    flat = points.reshape(-1, 2)
    q = flat @ matrix[:2, :2].T + matrix[:2, 2]
    w = flat @ matrix[2, :2] + matrix[2, 2]
    return (q / w[:, None]).reshape(points.shape)


def apply_direction(matrix, points, directions):
    """Unit image directions of `directions` at `points` under a 3×3 map (finite difference)."""
    h = 1e-4
    a = apply(matrix, points)
    b = apply(matrix, np.asarray(points) + h*np.asarray(directions))
    v = b - a
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Curve:
    """Lines and arcs from `start` with initial `heading_deg`; segments are
    `['line', length]` or `['arc', radius, sweep_deg]`."""

    def __init__(self, start, heading_deg, segments):
        p = np.asarray(start, dtype=float)
        phi = np.radians(heading_deg)
        s = 0.
        self.pieces = []
        for seg in segments:
            if seg[0] == 'line':
                length = float(seg[1])
                t = np.array([np.cos(phi), np.sin(phi)])
                self.pieces.append({'kind': 'line', 's0': s, 'len': length, 'p': p.copy(), 'phi': phi})
                p = p + length*t
            elif seg[0] == 'arc':
                radius, sweep = float(seg[1]), np.radians(seg[2])
                sign = 1. if sweep > 0 else -1.
                n = np.array([-np.sin(phi), np.cos(phi)])
                centre = p + sign*radius*n
                a0 = phi - sign*np.pi/2
                length = radius*abs(sweep)
                self.pieces.append({'kind': 'arc', 's0': s, 'len': length, 'c': centre, 'r': radius,
                                    'a0': a0, 'sign': sign, 'phi': phi})
                phi = phi + sweep
                p = centre + radius*np.array([np.cos(a0 + sweep), np.sin(a0 + sweep)])
            else:
                raise ValueError(f'unknown segment {seg!r}')
            s += length
        self.length = s
        self.start = np.asarray(start, dtype=float)
        self.phi0 = np.radians(heading_deg)
        self.end, self.phi1 = p, phi

    def _piece(self, s):
        for piece in self.pieces:
            if s <= piece['s0'] + piece['len']:
                return piece
        return self.pieces[-1]

    def heading(self, s):
        s = float(s)
        if s <= 0:
            return self.phi0
        if s >= self.length:
            return self.phi1
        piece = self._piece(s)
        if piece['kind'] == 'line':
            return piece['phi']
        return piece['phi'] + piece['sign']*(s - piece['s0'])/piece['r']

    def frame(self, s):
        """(point, unit tangent, unit normal) at arc length `s`, tangent lines past the ends."""
        s = float(s)
        phi = self.heading(s)
        t = np.array([np.cos(phi), np.sin(phi)])
        n = np.array([-t[1], t[0]])
        if s <= 0:
            return self.start + s*t, t, n
        if s >= self.length:
            return self.end + (s - self.length)*t, t, n
        piece = self._piece(s)
        u = s - piece['s0']
        if piece['kind'] == 'line':
            return piece['p'] + u*t, t, n
        a = piece['a0'] + piece['sign']*u/piece['r']
        return piece['c'] + piece['r']*np.array([np.cos(a), np.sin(a)]), t, n

    def point(self, s, d=0.):
        p, _, n = self.frame(s)
        return p + d*n

    def nearest(self, xy):
        """Ribbon coordinates (s, d) of (N, 2) points."""
        xy = np.asarray(xy, dtype=float)
        best = np.full(len(xy), np.inf)
        s_out, d_out = np.zeros(len(xy)), np.zeros(len(xy))
        lines = [(self.start, self.phi0, -np.inf, 0., 0.)]
        for piece in self.pieces:
            if piece['kind'] == 'line':
                lines.append((piece['p'], piece['phi'], 0., piece['len'], piece['s0']))
        lines.append((self.end, self.phi1, 0., np.inf, self.length))
        for p, phi, lo, hi, s0 in lines:
            t = np.array([np.cos(phi), np.sin(phi)])
            v = xy - p
            along = v @ t
            u = np.clip(along, lo, hi)
            dist = np.hypot(v[:, 0] - u*t[0], v[:, 1] - u*t[1])
            take = dist < best
            best = np.where(take, dist, best)
            s_out = np.where(take, s0 + u, s_out)
            d_out = np.where(take, v[:, 1]*t[0] - v[:, 0]*t[1], d_out)
        for piece in self.pieces:
            if piece['kind'] != 'arc':
                continue
            v = xy - piece['c']
            radius = np.hypot(v[:, 0], v[:, 1])
            sweep = piece['len']/piece['r']
            rel = np.mod(piece['sign']*(np.arctan2(v[:, 1], v[:, 0]) - piece['a0']), 2*np.pi)
            over = rel > sweep
            rel = np.where(over & (rel - sweep < 2*np.pi - rel), sweep, np.where(over, 0., rel))
            a = piece['a0'] + piece['sign']*rel
            foot = piece['c'] + piece['r']*np.stack([np.cos(a), np.sin(a)], axis=1)
            dist = np.hypot(*(xy - foot).T)
            take = dist < best
            best = np.where(take, dist, best)
            s_out = np.where(take, piece['s0'] + piece['r']*rel, s_out)
            d_out = np.where(take, piece['sign']*(piece['r'] - radius), d_out)
        return s_out, d_out


def half_width(spec, s, s_ref):
    """A wall's distance from the centreline: base + slope·(s − s_ref) + amp·sin(2π(s − s_ref)/period + phase)."""
    u = np.asarray(s, dtype=float) - s_ref
    w = spec['half'] + spec.get('slope', 0.)*u
    if spec.get('amp', 0.):
        w = w + spec['amp']*np.sin(2*np.pi*u/spec['period'] + spec.get('phase', 0.))
    return w


class Ribbon:
    """The bead: walls, caps, a gap and interior stripes, in ribbon coordinates."""

    def __init__(self, curve, cfg, s_ref):
        self.curve, self.cfg, self.s_ref = curve, cfg, s_ref
        self.kind = cfg.get('edge', 'erf')
        self.start = float(cfg['start'])
        caps = cfg['caps']
        self.cap_soft = float(caps.get('soft', cfg['left']['soft']))

    def walls(self, s):
        """(left, right) half widths at s; the left wall is at d = −left, the right at d = +right."""
        return (half_width(self.cfg['left'], s, self.s_ref),
                half_width(self.cfg['right'], s, self.s_ref))

    def _cap(self, kind, beyond, d, cap_s, p):
        """(factor, None) for a cut cap, or (None, disc) for a half disc; `beyond` is the arc
        length past the cap station, positive outside the ribbon."""
        left, right = self.walls(cap_s)
        mid, h = (right - left)/2, (left + right)/2
        soft, k = self.cap_soft, self.kind
        if kind in ('flat', 'skewed'):
            beta = np.radians(self.cfg['caps'].get('skew_deg', 0.)) if kind == 'skewed' else 0.
            return edge((np.tan(beta)*(d - mid) - beyond)*np.cos(beta), soft, k), None
        if kind == 'half_disc':
            e, _, n = self.curve.frame(cap_s)
            r = np.hypot(*(p - (e + mid*n)).T)
            return None, edge(h - r, soft, k)
        raise ValueError(f'unknown cap {kind!r}')

    def membership(self, p, end_s):
        """(material fraction m in [0, 1], ribbon offset d, arc length s) of (N, 2) ribbon points."""
        cfg, k = self.cfg, self.kind
        s, d = self.curve.nearest(p)
        left, right = self.walls(s)
        m = edge(d + left, cfg['left']['soft'], k) * edge(right - d, cfg['right']['soft'], k)
        for which, cap_s, beyond in (('start', self.start, self.start - s), ('end', end_s, s - end_s)):
            kind = cfg['caps'][which]
            factor, disc = self._cap(kind, beyond, d, cap_s, p)
            if disc is None:
                m = m*factor
            else:
                m = np.where(beyond > 0, disc, m)
        gap = cfg.get('gap')
        if gap:
            g0, g1 = gap
            m = m*(1 - edge(s - g0, self.cap_soft, k)*edge(g1 - s, self.cap_soft, k))
        return m, d, s

    def level(self, d):
        """Bead intensity at offset d, with its interior stripes."""
        value = np.full(d.shape, float(self.cfg['level']))
        for stripe in self.cfg.get('interior', []):
            value += stripe['contrast']*box(d, stripe['offset'][0], stripe['offset'][1],
                                            stripe['soft'], stripe.get('edge', self.kind))
        return value

    def rim(self, which, end_s, step=0.25):
        """The cap boundary in ribbon coordinates, from the left wall's end point to the right's."""
        kind = self.cfg['caps'][which]
        e, t, n = self.curve.frame(end_s)
        sign = 1. if which == 'end' else -1.
        left, right = self.walls(end_s)
        mid, h = (right - left)/2, (left + right)/2
        if kind == 'half_disc':
            count = max(int(np.ceil(np.pi*h/step)), 8)
            a = np.linspace(0., np.pi, count + 1)
            points = e + mid*n + h*(-np.cos(a)[:, None]*n + sign*np.sin(a)[:, None]*t)
        else:
            beta = np.radians(self.cfg['caps'].get('skew_deg', 0.)) if kind == 'skewed' else 0.
            dd = np.linspace(-left, right, max(int(np.ceil((left + right)/step)), 2) + 1)
            points = np.array([self.curve.point(end_s + sign*np.tan(beta)*(x - mid), x) for x in dd])
        return points


def view_matrix(view):
    """plane→image of a view: image = c_px + scale·R(angle)·A·(q − centre), A = [[aspect, shear], [0, 1]],
    divided by 1 + projective·(q − centre)."""
    h, w = view['shape']
    centre = np.asarray(view['centre'], dtype=float)
    th = np.radians(view.get('angle_deg', 0.))
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    affine = np.array([[view.get('aspect', 1.), view.get('shear', 0.)], [0., 1.]])
    m = view.get('scale', 1.)*rot @ affine
    shift = np.eye(3)
    shift[:2, 2] = -centre
    core = np.eye(3)
    core[:2, :2] = m
    core[2, :2] = view.get('projective', [0., 0.])
    to_px = np.eye(3)
    to_px[:2, 2] = [(w - 1)/2, (h - 1)/2]
    return to_px @ core @ shift


def ray_hit(polyline, origin, direction):
    """Distance from `origin` along unit `direction` to its first crossing of `polyline`, or None."""
    p = np.asarray(polyline, dtype=float)
    a, b = p[:-1], p[1:]
    e = b - a
    o = np.asarray(origin, dtype=float)
    dvec = np.asarray(direction, dtype=float)
    den = dvec[0]*e[:, 1] - dvec[1]*e[:, 0]
    ok = np.abs(den) > 1e-12
    w = a - o
    t = np.where(ok, (w[:, 0]*e[:, 1] - w[:, 1]*e[:, 0])/np.where(ok, den, 1.), np.inf)
    u = np.where(ok, (w[:, 0]*dvec[1] - w[:, 1]*dvec[0])/np.where(ok, den, 1.), -1.)
    hit = ok & (t >= 0) & (u >= 0) & (u <= 1)
    return float(t[hit].min()) if hit.any() else None
