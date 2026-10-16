"""A scenario built from its configuration, and its box-integrated frame renderer.

A pixel's value is the mean of the scene over the unit square around its centre, by
tensor Gauss–Legendre quadrature (`quadrature` nodes per axis). Every random draw comes
from `numpy.random.default_rng([seed, stream, time, view])`, so a scenario is a pure
function of its configuration (invariant 12).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates, spline_filter

from scene import Curve, Ribbon, apply, box, edge, se2, view_matrix

TEXTURE, NOISE, BLANK_NOISE, OCCLUSION, REGISTRATION = 1, 2, 3, 4, 5


def rng(seed, stream, time=0, view=0):
    return np.random.default_rng([seed, stream, time, view])


class Scenario:
    """Everything about a configuration that does not depend on rendering a pixel."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = int(cfg['seed'])
        self.frames = int(cfg['frames'])
        path = cfg['path']
        self.curve = Curve(path['start'], path['heading_deg'], path['segments'])
        st = cfg['stations']
        rib = cfg['ribbon']
        self.s_from = float(rib['start'] if st['from'] is None else st['from'])
        self.s_to = float(rib['end'] if st['to'] is None else st['to'])
        self.ribbon = Ribbon(self.curve, rib, self.s_from)
        self.times = np.arange(self.frames)
        self.tool = np.asarray(cfg['tool']['plane'], dtype=float)
        self.s_tool = cfg['tool']['start_s'] + cfg['speed']*self.times
        if cfg['growth']:
            self.laid = np.minimum(rib['end'], self.s_tool - cfg['tool']['lag'])
        else:
            self.laid = np.full(self.frames, float(rib['end']))
        self.poses = [self._pose(s) for s in self.s_tool]
        rate = (cfg['slip'] or {}).get('rate', [0., 0.])
        self.slips = [(rate[0]*t, rate[1]*t, 0.) for t in self.times]
        self.views = cfg['views']
        self.nominal = [view_matrix(v) for v in self.views]
        self.offsets = np.zeros((self.frames, len(self.views), 2))
        for shake in cfg['shake']:
            steps = np.clip(self.times - shake['from'] + 1, 0, shake['to'] - shake['from'] + 1)
            self.offsets[:, shake['view']] += steps[:, None]*np.asarray(shake['px'], dtype=float)
        self.recorded = np.zeros((self.frames, len(self.views)), dtype=bool)
        for v, view in enumerate(self.views):
            first, last = view.get('window') or (0, self.frames - 1)
            self.recorded[first:last + 1, v] = True
        self.dropped = np.zeros_like(self.recorded)
        for drop in cfg['drops']:
            views = drop['views'] if drop['views'] is not None else range(len(self.views))
            for v in views:
                self.dropped[drop['time'], v] = self.recorded[drop['time'], v]
        self.given = self._given_poses()
        n = int(np.floor((self.s_to - self.s_from)/st['spacing'] + 1e-9)) + 1
        self.stations = self.s_from + st['spacing']*np.arange(n)
        self._texture()

    def _pose(self, s):
        """Substrate→plane pose putting the tool at curve(s), the tangent along plane +x."""
        p, t, _ = self.curve.frame(s)
        th = -np.arctan2(t[1], t[0])
        c, sn = np.cos(th), np.sin(th)
        q = np.array([c*p[0] - sn*p[1], sn*p[0] + c*p[1]])
        return (float(self.tool[0] - q[0]), float(self.tool[1] - q[1]), float(th))

    def _given_poses(self):
        """The poses handed to consumers: exact, or with injected error. Frame 0 is exact."""
        err = self.cfg['registration_error']
        if not err:
            return [tuple(p) for p in self.poses]
        scale = self.views[0].get('scale', 1.)
        lever = 0.5*np.hypot(*self.views[0]['shape'])
        draws = rng(self.seed, REGISTRATION).normal(size=(self.frames, 3))
        given = []
        for t, (tx, ty, th) in enumerate(self.poses):
            if t == 0:
                given.append((tx, ty, th))
                continue
            sigma = err.get('sigma_px', 0.)
            dx = (sigma*draws[t, 0] + err.get('drift_px', 0.)*t)/scale
            dy = sigma*draws[t, 1]/scale
            given.append((tx + dx, ty + dy, th + sigma*draws[t, 2]/lever))
        return given

    def image_map(self, t, v):
        """The actual plane→image matrix of view v at time t, with its camera motion."""
        shift = np.eye(3)
        shift[:2, 2] = self.offsets[t, v]
        return shift @ self.nominal[v]

    def ribbon_to_image(self, t, v):
        return self.image_map(t, v) @ se2(self.poses[t]) @ se2(self.slips[t])

    def _texture(self):
        """A DoG-band texture on a 1 su lattice covering every view's footprint."""
        sub = self.cfg['substrate']
        corners = []
        for t in self.times:
            for v, view in enumerate(self.views):
                h, w = view['shape']
                c = np.array([[-1, -1], [w, -1], [-1, h], [w, h]], dtype=float)
                inv = np.linalg.inv(self.image_map(t, v) @ se2(self.poses[t]))
                corners.append(apply(inv, c))
        corners = np.concatenate(corners)
        pad = 3*max(sub['band']) + 4
        self.origin = np.floor(corners.min(axis=0)) - pad
        size = (np.ceil(corners.max(axis=0)) + pad - self.origin).astype(int) + 1
        self.texture_coeffs = None
        if sub['texture'] > 0:
            noise = rng(self.seed, TEXTURE).normal(size=(size[1], size[0]))
            lo, hi = sub['band']
            band = gaussian_filter(noise, lo) - gaussian_filter(noise, hi)
            band *= sub['texture']/band.std()
            self.texture_coeffs = spline_filter(band, order=3)

    def substrate(self, p, d_nominal):
        """Substrate intensity at (N, 2) substrate points, before the bead covers it."""
        cfg = self.cfg
        value = np.full(len(p), float(cfg['substrate']['mean']))
        if self.texture_coeffs is not None:
            q = p - self.origin
            value += map_coordinates(self.texture_coeffs, [q[:, 1], q[:, 0]], order=3,
                                     prefilter=False, mode='nearest')
        par = cfg['parallel']
        if par:
            value += par['contrast']*box(d_nominal, par['offset'][0], par['offset'][1],
                                         par['soft'], par.get('edge', 'erf'))
        for dis in cfg['distractors']:
            _, tangent, _ = self.curve.frame(dis['s'])
            centre = self.curve.point(dis['s'], dis['d'])
            a = np.arctan2(tangent[1], tangent[0]) + np.radians(dis.get('angle_deg', 0.))
            u = (p - centre) @ np.array([np.cos(a), np.sin(a)])
            w = (p - centre) @ np.array([-np.sin(a), np.cos(a)])
            q = (edge(dis['size'][0]/2 - np.abs(u), dis['soft'])
                 * edge(dis['size'][1]/2 - np.abs(w), dis['soft']))
            value = value*(1 - q) + dis['level']*q
        return value

    def occluders(self, t, v):
        """[(kind, geometry, level)] in image pixels covering view v at time t."""
        out = []
        for occ in self.cfg['occluders']:
            if occ.get('views') not in (None, []) and v not in occ['views']:
                continue
            if occ.get('times') is not None and t not in occ['times']:
                continue
            kind = 'rect' if 'rect' in occ else 'ellipse'
            out.append((kind, occ[kind], occ.get('level', 0.)))
        spec = self.cfg['occlusions']
        if spec:
            draw = rng(self.seed, OCCLUSION, t, v)
            h, w = self.views[v]['shape']
            if draw.uniform() < spec['rate']:
                lo, hi = spec['radius']
                geometry = [draw.uniform(0, w - 1), draw.uniform(0, h - 1),
                            draw.uniform(lo, hi), draw.uniform(lo, hi), draw.uniform(0, 180)]
                out.append(('ellipse', [float(x) for x in geometry], spec['level']))
        return out

    def outside(self, v):
        return [o['rect'] for o in self.cfg['outside']
                if o.get('views') in (None, []) or v in o['views']]


def _inside(kind, geometry, x, y):
    if kind == 'rect':
        x0, y0, x1, y1 = geometry
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    cx, cy, a, b, angle = geometry
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    u, w = (x - cx)*c + (y - cy)*s, -(x - cx)*s + (y - cy)*c
    return (u/a)**2 + (w/b)**2 <= 1


def _samples(scn, t, v):
    """Quadrature points of view v at time t and the scene's geometry there."""
    h, w = scn.views[v]['shape']
    k = int(scn.cfg['quadrature'])
    nodes, weights = np.polynomial.legendre.leggauss(k)
    ny, nx = np.meshgrid(nodes/2, nodes/2, indexing='ij')
    wq = np.outer(weights/2, weights/2).ravel()
    yy, xx = np.mgrid[:h, :w].astype(float)
    x = (xx[..., None] + nx.ravel()).ravel()
    y = (yy[..., None] + ny.ravel()).ravel()
    sub = apply(np.linalg.inv(scn.image_map(t, v) @ se2(scn.poses[t])), np.stack([x, y], axis=1))
    rib = apply(np.linalg.inv(se2(scn.slips[t])), sub)
    m, d, _ = scn.ribbon.membership(rib, scn.laid[t])
    d_nominal = scn.curve.nearest(sub)[1] if scn.cfg['slip'] else d
    return x, y, sub, m, d, d_nominal, lambda a: a.reshape(h, w, k*k) @ wq


def render(scn, t, v):
    """(image, blank or None, registration mask, visibility mask, occluders) of view v at time t.

    Images are uint8. The visibility mask is False on every pixel an occluder or an outside
    region touches. The registration mask is False there dilated by 1 px, and on the
    material dilated by 3 px."""
    cfg = scn.cfg
    h, w = scn.views[v]['shape']
    x, y, sub, m, d, d_nominal, pixels = _samples(scn, t, v)
    base = scn.substrate(sub, d_nominal)
    bead = scn.ribbon.level(d)
    hl = cfg['highlight']
    if hl and v in hl['views']:
        (cx, cy), (sx, sy) = hl['centre'], hl['sigma']
        bead = bead + hl['amplitude']*np.exp(-0.5*(((x - cx)/sx)**2 + ((y - cy)/sy)**2))
    gain = np.ones_like(x)
    sh = cfg['shading']
    if sh:
        cx, cy = (w - 1)/2, (h - 1)/2
        gx, gy = sh.get('gradient', [0., 0.])
        r2 = ((x - cx)**2 + (y - cy)**2)/(cx**2 + cy**2)
        gain = 1 + gx*(x - cx)/w + gy*(y - cy)/h - sh.get('vignette', 0.)*r2
    value = (base*(1 - m) + bead*m)*gain
    blank = base*gain if cfg['blank'] else None
    hidden = np.zeros(len(x), dtype=bool)
    occluders = scn.occluders(t, v)
    for kind, geometry, level in occluders:
        inside = _inside(kind, geometry, x, y)
        value = np.where(inside, level, value)
        hidden |= inside
    for rect in scn.outside(v):
        inside = _inside('rect', rect, x, y)
        value = np.where(inside, 0., value)
        if blank is not None:
            blank = np.where(inside, 0., blank)
        hidden |= inside

    def finish(a, stream):
        a = pixels(a)
        if cfg['noise'] > 0:
            a = a + rng(scn.seed, stream, t, v).normal(0., cfg['noise'], a.shape)
        if cfg['invert']:
            a = 255. - a
        return np.clip(np.rint(a), 0, 255).astype(np.uint8)

    visible = ~(pixels(hidden.astype(float)) > 0)
    valid = ~binary_dilation(~visible, iterations=1) & ~binary_dilation(pixels(m) > 0.01, iterations=3)
    return (finish(value, NOISE), None if blank is None else finish(blank, BLANK_NOISE),
            valid, visible, occluders)


def render_float(scn, t, v):
    """Noise-free, unquantized pixel means of the material over the substrate (no camera
    fields, occluders or inversion), for the self-check."""
    _, _, sub, m, d, d_nominal, pixels = _samples(scn, t, v)
    return pixels(scn.substrate(sub, d_nominal)*(1 - m) + scn.ribbon.level(d)*m)
