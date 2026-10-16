"""Named scenarios. Each is a partial configuration merged over `DEFAULTS` (dicts merge
recursively; lists and scalars replace). The README documents every key and preset."""
from __future__ import annotations

import copy

DEFAULTS = {
    'seed': 86,
    'frames': 40,
    'speed': 3.0,                       # tool speed along the path, su per frame
    'path': {'start': [0., 0.], 'heading_deg': 0., 'segments': [['line', 400.]]},
    'tool': {'plane': [0., 0.], 'start_s': 60., 'lag': 0.},
    'growth': False,                    # lay material up to the tool, frame by frame
    'ribbon': {
        'start': 0., 'end': 400.,
        'edge': 'erf',
        'level': 200.,
        'left': {'half': 10., 'soft': 0.8},
        'right': {'half': 10., 'soft': 0.8},
        'caps': {'start': 'half_disc', 'end': 'half_disc', 'skew_deg': 30., 'soft': 0.8},
        'interior': [],                 # [{'offset': [a, b], 'contrast': c, 'soft': σ}]
        'gap': None,                    # [s0, s1]: no material there
    },
    'slip': None,                       # {'rate': [vx, vy]} su per frame, ribbon→substrate
    'substrate': {'mean': 110., 'texture': 15., 'band': [1.5, 4.0]},
    'views': [{'shape': [97, 320], 'centre': [-120., 0.], 'angle_deg': 0., 'scale': 1.}],
    'noise': 2.0,
    'quadrature': 4,
    'invert': False,
    'blank': False,
    'shading': None,                    # {'gradient': [gx, gy], 'vignette': v}
    'highlight': None,                  # {'views': [...], 'centre': [x, y], 'sigma': [sx, sy], 'amplitude': a}
    'occlusions': None,                 # {'rate': p, 'radius': [lo, hi], 'level': v}
    'occluders': [],                    # [{'views', 'times', 'rect' | 'ellipse', 'level'}]
    'outside': [],                      # [{'views', 'rect'}]
    'parallel': None,                   # {'offset': [a, b], 'contrast': c, 'soft': σ}
    'distractors': [],                  # [{'s', 'd', 'size': [along, across], 'angle_deg', 'level', 'soft'}]
    'drops': [],                        # [{'time': t, 'views': [...] | None}]
    'shake': [],                        # [{'view': v, 'from': t0, 'to': t1, 'px': [dx, dy]}]
    'registration_error': None,         # {'sigma_px': σ, 'drift_px': δ}
    'stations': {'from': None, 'to': None, 'spacing': 1., 'annotate_every': 10, 'boundary_every': 2},
}

_FUSION = {
    'frames': 48, 'speed': 3., 'growth': True, 'blank': True,
    'path': {'segments': [['line', 120.], ['arc', 150., 35.], ['line', 300.]]},
    'tool': {'start_s': 100.},
    'ribbon': {'end': 500., 'level': 195.,
               'left': {'half': 14., 'amp': 0.5, 'period': 70., 'soft': 0.8},
               'right': {'half': 16., 'amp': 0.4, 'period': 45., 'phase': 1., 'soft': 1.1},
               'gap': [160., 169.]},
    'views': [{'shape': [96, 160], 'centre': [-60., 0.]},
              {'shape': [96, 160], 'centre': [-175., 45.], 'angle_deg': 25., 'scale': 0.95,
               'window': [6, 47]},
              {'shape': [96, 160], 'centre': [110., 0.], 'scale': 1.05}],
    'highlight': {'views': [0, 1], 'centre': [80., 50.], 'sigma': [14., 3.], 'amplitude': 45.},
    'shading': {'gradient': [0.06, 0.03], 'vignette': 0.08},
    'distractors': [{'s': 140., 'd': 30., 'size': [60., 10.], 'angle_deg': 0., 'level': 45., 'soft': 0.8}],
    'occlusions': {'rate': 0.25, 'radius': [6., 14.], 'level': 25.},
}

_RIG = {
    'frames': 40, 'speed': 2.5, 'growth': True,
    'path': {'segments': [['line', 60.], ['arc', 300., 30.], ['line', 300.]]},
    'tool': {'start_s': 40.},
    'ribbon': {'end': 500.},
    'views': [{'shape': [97, 160], 'centre': [0., -60.], 'angle_deg': 0., 'scale': 1.},
              {'shape': [97, 160], 'centre': [-52., 30.], 'angle_deg': 120., 'scale': 0.9},
              {'shape': [97, 160], 'centre': [52., 30.], 'angle_deg': 240., 'scale': 1.1,
               'shear': 0.05}],
}

_CAP = {
    'frames': 1, 'speed': 0.,
    'path': {'segments': [['line', 400.]]},
    'tool': {'start_s': 0.},
    'ribbon': {'start': -50., 'end': 100.},
    'substrate': {'texture': 8.},
    'views': [{'shape': [97, 160], 'centre': [69.5, 0.]}],
}

#: name -> (issue, one line, partial config)
PRESETS = {
    'tiny': ('CI', 'every nuisance at once on two 64×40 views over six frames', {
        'frames': 6, 'speed': 4., 'growth': True, 'blank': True,
        'path': {'segments': [['line', 30.], ['arc', 40., 30.], ['line', 100.]]},
        'tool': {'start_s': 30.},
        'ribbon': {'end': 200., 'left': {'half': 5., 'amp': 0.4, 'period': 20.},
                   'right': {'half': 6., 'soft': 1.0}, 'gap': [38., 41.],
                   'interior': [{'offset': [-1., 1.], 'contrast': -40., 'soft': 0.4}]},
        'slip': {'rate': [0., 0.1]},
        'views': [{'shape': [40, 64], 'centre': [-25., 0.]},
                  {'shape': [40, 64], 'centre': [-30., 5.], 'angle_deg': 20., 'scale': 0.9,
                   'shear': 0.1, 'projective': [0.0005, 0.], 'window': [1, 5]}],
        'quadrature': 2,
        'highlight': {'views': [0], 'centre': [30., 20.], 'sigma': [6., 2.], 'amplitude': 30.},
        'shading': {'gradient': [0.05, 0.], 'vignette': 0.05},
        'occlusions': {'rate': 0.5, 'radius': [3., 6.], 'level': 20.},
        'outside': [{'views': [1], 'rect': [-0.5, -0.5, 8.5, 39.5]}],
        'parallel': {'offset': [9., 10.5], 'contrast': -40., 'soft': 0.6},
        'distractors': [{'s': 35., 'd': -12., 'size': [12., 4.], 'angle_deg': 10., 'level': 40., 'soft': 0.7}],
        'drops': [{'time': 3, 'views': [0]}],
        'shake': [{'view': 1, 'from': 4, 'to': 5, 'px': [1.5, 0.]}],
        'registration_error': {'sigma_px': 0.3, 'drift_px': 0.05},
    }),
    'fusion-single': ('#73', "the single-observation profile case, rendered as one frame", {
        'frames': 1, 'speed': 0., 'noise': 0.,
        'path': {'segments': [['line', 140.]]}, 'tool': {'start_s': 0.},
        'ribbon': {'start': -20., 'end': 160., 'edge': 'logistic', 'level': 200.,
                   'left': {'half': 12., 'slope': 0.04, 'soft': 0.5},
                   'right': {'half': 12., 'slope': 0.04, 'soft': 0.5},
                   'caps': {'soft': 0.5},
                   'interior': [{'offset': [-5., -2.], 'contrast': -140., 'soft': 0.25}]},
        'substrate': {'mean': 20., 'texture': 0.},
        'views': [{'shape': [97, 120], 'centre': [59.5, 0.]}],
        'parallel': {'offset': [-35., -23.], 'contrast': 40., 'soft': 2., 'edge': 'logistic'},
        'occluders': [{'rect': [54.5, -0.5, 58.5, 96.5], 'level': 0.}],
        'outside': [{'rect': [74.5, -0.5, 119.5, 37.5]}],
        'stations': {'from': 30., 'to': 89., 'annotate_every': 1, 'boundary_every': 1},
    }),
    'fusion-multiview': ('#73', 'three 160×96 views, a 30 su bead with a highlight, a blank-present '
                         'distractor, a view with no bead and a three-frame gap', _FUSION),
    'fusion-parallel': ('#73', 'fusion-multiview with a substrate groove parallel to the right wall', {
        **_FUSION, 'parallel': {'offset': [21., 23.], 'contrast': -50., 'soft': 0.6}}),
    'fusion-moved': ('#73', 'fusion-multiview whose bead slides on its substrate: a mismatch', {
        **_FUSION, 'slip': {'rate': [0., 0.15]}}),
    'fusion-regerr': ('#73', 'fusion-multiview with registration error in the given poses', {
        **_FUSION, 'registration_error': {'sigma_px': 0.5, 'drift_px': 0.03}}),
    'fusion-junction': ('#73', 'a dark bead crossing a blank-present structure of its own level', {
        **_FUSION, 'ribbon': {**_FUSION['ribbon'], 'level': 50.},
        'highlight': None,
        'distractors': [{'s': 150., 'd': 0., 'size': [8., 90.], 'angle_deg': 0., 'level': 50., 'soft': 0.8}]}),
    'rig-bridged': ('#74', 'three affine views 120° apart; every view drops one frame', {
        **_RIG, 'drops': [{'time': 20, 'views': None}]}),
    'rig-gross': ('#74', 'three affine views; one camera moves 3 px per frame for ten steps', {
        **_RIG, 'shake': [{'view': 2, 'from': 15, 'to': 24, 'px': [2.5, -1.5]}]}),
    'rig-single-view': ('#74', 'three affine views; two drop one frame, leaving one view', {
        **_RIG, 'drops': [{'time': 20, 'views': [1, 2]}]}),
    'cap-half-disc': ('#78', 'a 20 su bright stripe closed by a half disc of radius 10', _CAP),
    'cap-flat': ('#78', 'the stripe cut square', {**_CAP, 'ribbon': {**_CAP['ribbon'], 'caps': {'end': 'flat'}}}),
    'cap-skewed': ('#78', 'the stripe cut at 30° to its normal', {
        **_CAP, 'ribbon': {**_CAP['ribbon'], 'caps': {'end': 'skewed', 'skew_deg': 30.}}}),
    'cap-clipped': ('#78', 'a half disc whose rim leaves the image', {
        **_CAP, 'ribbon': {**_CAP['ribbon'], 'end': 143.}}),
    'cap-distractor': ('#78', 'a half disc with a blank-present dark edge 6 su past its rim', {
        **_CAP, 'blank': True,
        'distractors': [{'s': 126., 'd': 0., 'size': [20., 80.], 'angle_deg': 0., 'level': 50., 'soft': 0.8}]}),
    'validate-straight': ('generator', 'a straight ribbon on a flat substrate seen at 7° and −4°, for the self-check', {
        'frames': 1, 'speed': 0., 'noise': 0.,
        'path': {'segments': [['line', 400.]]},
        'tool': {'start_s': 0.},
        'ribbon': {'start': -150., 'end': 400., 'left': {'half': 9.3, 'soft': 0.8}, 'right': {'half': 11.6, 'soft': 1.0}},
        'substrate': {'mean': 60., 'texture': 0.},
        'views': [{'shape': [97, 320], 'centre': [0.37, 0.21], 'angle_deg': 7.},
                  {'shape': [97, 320], 'centre': [0.37, 0.21], 'angle_deg': -4., 'scale': 1.3}],
        'stations': {'from': -100., 'to': 100., 'annotate_every': 5},
    }),
}


def merge(base, over):
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def config(name, **overrides):
    """The full configuration of a preset, with top-level overrides."""
    if name not in PRESETS:
        raise KeyError(f'unknown preset {name!r}; known: {", ".join(PRESETS)}')
    cfg = merge(DEFAULTS, PRESETS[name][2])
    return merge(cfg, {k: v for k, v in overrides.items() if v is not None})
