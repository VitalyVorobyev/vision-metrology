"""Estimator configurations shared by the evaluators.

`check=False` keeps the regional statistics but disables their decision, for diagnosis:
the threshold becomes unreachable, so `region_error` is still reported.
"""
import vision_metrology as vm
from vision_metrology.experimental import SimilarityConfig

UNREACHABLE_PX = 1e9


def translation_config(check=True):
    cfg = vm.TranslationConfig()
    if not check:
        cfg.max_region_error = UNREACHABLE_PX
    return cfg


def similarity_config(check=True):
    cfg = SimilarityConfig()
    if not check:
        translation = cfg.translation
        translation.max_region_error = UNREACHABLE_PX
        cfg.translation = translation
    return cfg


def describe_translation(cfg):
    out = {name: getattr(cfg, name) for name in
           ('search_radius', 'min_support', 'min_score', 'min_margin', 'max_region_error')}
    out['tuning'] = {name: getattr(cfg.tuning, name) for name in
                     ('max_levels', 'iterations', 'tolerance', 'huber', 'min_texture')}
    return out


def describe_similarity(cfg):
    return {'max_angle': cfg.max_angle, 'max_log_scale': cfg.max_log_scale,
            'translation': describe_translation(cfg.translation)}
