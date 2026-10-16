"""Estimator configurations shared by the evaluators.

`check=False` keeps the regional statistics but disables their decision, for diagnosis:
the χ² limit becomes unreachable, so `region_error` and `region_chi2` are still reported.
"""
import vision_metrology as vm
from vision_metrology.experimental import SimilarityConfig

UNREACHABLE = 1e30


def _disable(cfg):
    if hasattr(cfg, 'region_check'):
        check = cfg.region_check or vm.RegionCheck()
        check.max_chi2 = UNREACHABLE
        cfg.region_check = check
    else:  # bindings older than RegionCheck: a px limit
        cfg.max_region_error = UNREACHABLE
    return cfg


def translation_config(check=True):
    cfg = vm.TranslationConfig()
    return cfg if check else _disable(cfg)


def similarity_config(check=True):
    cfg = SimilarityConfig()
    if not check:
        cfg.translation = _disable(cfg.translation)
    return cfg


def describe_translation(cfg):
    names = ('search_radius', 'min_support', 'min_score', 'min_margin')
    out = {name: getattr(cfg, name) for name in names}
    if hasattr(cfg, 'region_check'):
        c = cfg.region_check
        out['region_check'] = None if c is None else {'max_chi2': c.max_chi2,
                                                      'bias_floor_px': c.bias_floor_px}
    else:
        out['max_region_error'] = cfg.max_region_error
    out['tuning'] = {name: getattr(cfg.tuning, name) for name in
                     ('max_levels', 'iterations', 'tolerance', 'huber', 'min_texture')}
    return out


def describe_similarity(cfg):
    return {'max_angle': cfg.max_angle, 'max_log_scale': cfg.max_log_scale,
            'translation': describe_translation(cfg.translation)}
