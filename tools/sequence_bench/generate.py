"""Write a synthetic sequence benchmark: `sequence.json`, frames, masks and `truth.json`.

    python tools/sequence_bench/generate.py --list
    python tools/sequence_bench/generate.py --preset fusion-multiview --out-dir /tmp/bench/fm
    python tools/sequence_bench/generate.py --all --out-dir /tmp/bench

The output is a pure function of the preset and the overrides: two runs write identical
bytes. The README documents the presets, the manifest's extra fields and the truth file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from presets import PRESETS, config
from render import Scenario, render
from scene import apply, apply_direction, se2

VERSION = 1
REPO = Path(__file__).resolve().parents[2]


def rounded(value, digits=6):
    """JSON-ready copy with floats rounded, so the files do not carry the last ulp."""
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return rounded(value.tolist(), digits)
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits) + 0.
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def station_geometry(scn):
    rib = scn.ribbon
    frames = [scn.curve.frame(s) for s in scn.stations]
    point = np.array([f[0] for f in frames])
    normal = np.array([f[2] for f in frames])
    left, right = rib.walls(scn.stations)
    gap = rib.cfg.get('gap')
    in_gap = np.zeros(len(scn.stations), dtype=bool) if not gap else \
        (scn.stations >= gap[0]) & (scn.stations <= gap[1])
    return {'s': scn.stations, 'point': point, 'normal': normal, 'left': left, 'right': right,
            'left_point': point - left[:, None]*normal, 'right_point': point + right[:, None]*normal,
            'in_gap': in_gap}


def observable(img, visible):
    """True where an image point lies in a visible pixel: inside the image, not occluded."""
    h, w = visible.shape
    ix, iy = np.floor(img[:, 0] + 0.5).astype(int), np.floor(img[:, 1] + 0.5).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    out = np.zeros(len(img), dtype=bool)
    out[ok] = visible[iy[ok], ix[ok]]
    return out


def visibility_codes(scn, st, t, v, visible):
    """One character per station: 'n' not laid, 'g' in the gap, else '0'–'3' (bit 1: left
    wall observable, bit 2: right wall observable)."""
    m = scn.ribbon_to_image(t, v)
    left = observable(apply(m, st['left_point']), visible)
    right = observable(apply(m, st['right_point']), visible)
    codes = np.char.mod('%d', left.astype(int) + 2*right.astype(int))
    laid = (st['s'] >= scn.ribbon.start) & (st['s'] <= scn.laid[t])
    codes = np.where(st['in_gap'], 'g', codes)
    return ''.join(np.where(laid, codes, 'n'))


def runs(mask):
    """[start, stop) runs of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def similarity(m):
    """[tx, ty, angle, scale] of a 3×3 similarity, or None."""
    a = m[:2, :2]
    scale = np.sqrt(abs(np.linalg.det(a)))
    angle = np.arctan2(a[1, 0], a[0, 0])
    c, s = scale*np.cos(angle), scale*np.sin(angle)
    if np.abs(m[2] - [0, 0, 1]).max() > 1e-12 or np.abs(a - [[c, -s], [s, c]]).max() > 1e-9*max(scale, 1):
        return None
    return [m[0, 2], m[1, 2], angle, scale]


def caps_in_frame(scn, t, v, visible):
    """Image-space rims of the caps that lie near view v at time t."""
    rib = scn.ribbon
    h, w = visible.shape
    m = scn.ribbon_to_image(t, v)
    out = []
    for which, s in (('start', rib.start), ('end', scn.laid[t])):
        if which == 'end' and s <= rib.start:
            continue
        rim = apply(m, rib.rim(which, s))
        if (rim[:, 0].max() < -0.5 or rim[:, 0].min() > w - 0.5
                or rim[:, 1].max() < -0.5 or rim[:, 1].min() > h - 0.5):
            continue
        out.append({'which': which, 'kind': rib.cfg['caps'][which], 's': s,
                    'chord': [rim[0], rim[-1]], 'rim': rim,
                    'observable': observable(rim, visible).astype(int)})
    return out


def generate(preset, out_dir, **overrides):
    """Write one scenario into `out_dir`; return the manifest path."""
    cfg = config(preset, **overrides)
    scn = Scenario(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = station_geometry(scn)
    every, bevery = cfg['stations']['annotate_every'], cfg['stations']['boundary_every']
    entries, motion, boundaries, annotations, caps, index = [], [], [], [], [], []
    visibility = [[None]*len(scn.views) for _ in scn.times]
    occlusions, seed = [], None
    reference = None
    for v in range(len(scn.views)):
        for t in scn.times:
            if not scn.recorded[t, v]:
                continue
            fi = len(entries)
            index.append([int(t), v])
            if scn.dropped[t, v]:
                entries.append({'image': None, 'group': None, 'view': v, 'time': int(t)})
                motion.append(None)
                boundaries.append([])
                caps.append([])
                continue
            image, blank, valid, visible, occluders = render(scn, t, v)
            name = f'v{v}/{t:04}.png'
            entry = {'image': f'frames/{name}', 'mask': f'masks/{name}',
                     'visibility': f'visibility/{name}', 'group': v, 'view': v, 'time': int(t)}
            for folder, array in (('frames', image), ('masks', valid), ('visibility', visible),
                                  ('blanks', blank)):
                if array is None:
                    continue
                (out / folder / f'v{v}').mkdir(parents=True, exist_ok=True)
                data = array if array.dtype == np.uint8 else array.astype(np.uint8)*255
                Image.fromarray(data).save(out / folder / name)
            if blank is not None:
                entry['blank'] = f'blanks/{name}'
            entries.append(entry)
            occlusions += [{'time': int(t), 'view': v, 'kind': k, 'geometry': g, 'level': lv}
                           for k, g, lv in occluders]
            codes = visibility_codes(scn, st, t, v, visible)
            visibility[t][v] = codes
            m = scn.ribbon_to_image(t, v)
            sub_to_image = scn.image_map(t, v) @ se2(scn.poses[t])
            if reference is None:
                reference = np.linalg.inv(sub_to_image)
            motion.append(similarity(sub_to_image @ reference))
            code = np.array(list(codes))
            frame_lines = []
            for wall, bits in (('left_point', ('1', '3')), ('right_point', ('2', '3'))):
                pts = apply(m, st[wall])
                ok = np.isin(code, bits)
                for a, b in runs(ok):
                    keep = list(range(a, b, bevery)) + ([b - 1] if (b - 1 - a) % bevery else [])
                    if len(keep) >= 2:
                        frame_lines.append(pts[keep])
            boundaries.append(frame_lines)
            both = np.flatnonzero(code == '3')
            centre = apply(m, st['point'])
            left, right = apply(m, st['left_point']), apply(m, st['right_point'])
            normal = apply_direction(m, st['point'], st['normal'])
            for k in both[both % every == 0]:
                annotations.append({'frame': fi, 'station': int(k), 's': st['s'][k], 'point': centre[k],
                                    'normal': normal[k], 'left': left[k], 'right': right[k]})
            sampled = both[both % every == 0]
            if seed is None and len(sampled) >= 2:
                seed = {'frame': fi, 'provenance': 'truth: exact centreline and walls',
                        'polyline': centre[sampled], 'walls': [left[sampled], right[sampled]],
                        'sections': [{'s': st['s'][k], 'normal': normal[k], 'left': left[k],
                                      'right': right[k], 'center': centre[k],
                                      'width': float(np.linalg.norm(right[k] - left[k]))}
                                     for k in sampled]}
            caps.append(caps_in_frame(scn, t, v, visible))
    rib = scn.ribbon
    bright = (rib.cfg['level'] > cfg['substrate']['mean']) != bool(cfg['invert'])
    truth = {
        'version': VERSION, 'preset': preset, 'issue': PRESETS[preset][0], 'config': cfg,
        'units': 'su', 'times': scn.frames, 'frames': index,
        'tool': scn.tool, 's_tool': scn.s_tool, 'laid': scn.laid,
        'poses': scn.poses, 'poses_given': scn.given, 'slips': scn.slips,
        'moves_on_substrate': bool(cfg['slip']),
        'views': [{'shape': view['shape'], 'plane_to_image': scn.nominal[v]}
                  for v, view in enumerate(scn.views)],
        'image_offsets': scn.offsets,
        'dropped': np.argwhere(scn.dropped).tolist(),
        'ribbon': {'polarity': 'bright' if bright else 'dark', 'start': rib.start,
                   'end': rib.cfg['end'], 'gap': rib.cfg.get('gap'),
                   'caps': {w: {'kind': rib.cfg['caps'][w], 's': s, 'rim': rib.rim(w, s)}
                            for w, s in (('start', rib.start), ('end', float(scn.laid[-1])))}},
        'stations': st,
        'visibility': visibility,
        'occlusions': occlusions,
    }
    (out / 'truth.json').write_text(json.dumps(rounded(truth), separators=(',', ':')))
    manifest = {
        'version': 1, 'name': preset + ('-inverted' if cfg['invert'] else ''), 'convention': 'pixel_centres', 'split': 'development',
        'provenance': {'source': 'synthetic', 'generator': 'tools/sequence_bench/generate.py',
                       'preset': preset, 'seed': cfg['seed']},
        'frames': entries,
        'rig': {'units': 'su', 'times': scn.frames, 'tool': scn.tool, 'poses': scn.given,
                'views': [{'id': v, 'shape': view['shape'], 'plane_to_image': scn.nominal[v],
                           'window': view.get('window') or [0, scn.frames - 1]}
                          for v, view in enumerate(scn.views)]},
        'truth': {'file': 'truth.json', 'motion': motion if motion and motion[0] is not None else None,
                  'boundaries': boundaries, 'poses': scn.poses, 'caps': caps},
        'annotations': annotations,
    }
    if seed is not None:
        manifest['seed'] = seed
    path = out / 'sequence.json'
    path.write_text(json.dumps(rounded(manifest), separators=(',', ':')))
    return path


def summary(path):
    """One line per view: frames written, dropped, and the share of laid stations observable."""
    truth = json.loads((path.parent / 'truth.json').read_text())
    lines = [f"{path.parent.name}: {truth['preset']} ({truth['issue']}), {truth['times']} times"]
    for v in range(len(truth['views'])):
        codes = [row[v] for row in truth['visibility'] if row[v] is not None]
        joined = ''.join(codes)
        laid = sum(joined.count(c) for c in '0123')
        both = joined.count('3')
        one = joined.count('1') + joined.count('2')
        share = f'{both/laid:.0%} both, {one/laid:.0%} one wall' if laid else 'no laid stations'
        lines.append(f'  view {v}: {len(codes)} frames, {share}')
    return '\n'.join(lines)


def check_out_dir(path):
    path = Path(path).resolve()
    if path.is_relative_to(REPO) and not path.is_relative_to(REPO / 'data'):
        raise SystemExit(f'{path}: write outside the repository (or under its ignored data/)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=sorted(PRESETS))
    parser.add_argument('--all', action='store_true', help='every preset, one subdirectory each')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--invert', action='store_true', help='255 − I: the opposite polarity')
    parser.add_argument('--reg-sigma', type=float, help='injected pose error σ, view-0 px')
    parser.add_argument('--reg-drift', type=float, help='injected pose drift, view-0 px per frame')
    args = parser.parse_args()
    if args.list:
        for name, (issue, line, _) in PRESETS.items():
            print(f'{name:18} {issue:9} {line}')
        return
    if args.out_dir is None or (args.preset is None) == (not args.all):
        parser.error('give --out-dir and exactly one of --preset or --all')
    check_out_dir(args.out_dir)
    overrides = {'seed': args.seed, 'frames': args.frames, 'invert': True if args.invert else None}
    if args.reg_sigma is not None or args.reg_drift is not None:
        overrides['registration_error'] = {'sigma_px': args.reg_sigma or 0., 'drift_px': args.reg_drift or 0.}
    names = sorted(PRESETS) if args.all else [args.preset]
    for name in names:
        out = args.out_dir / name if args.all else args.out_dir
        print(summary(generate(name, out, **overrides)))


if __name__ == '__main__':
    main()
