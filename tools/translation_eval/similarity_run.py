"""Evaluate native similarity and caller-owned anchors; produce private overlays/viewers."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image, ImageDraw
import vision_metrology as vm
from anchors import AnchoredSequence, IDENTITY, matrix, parts, corner_distance
from dataset import Dataset, bandpass

FIELDS = ('transform', 'score', 'margin', 'support', 'residual_rms', 'residual_max',
          'gain', 'offset', 'covariance', 'regions', 'region_error', 'status')


class RawStrips:
    """No annotated roles, blank pairs or reviewed masks are invented for raw BMPs."""
    def __init__(self, root, roi):
        self.root, self.roi = Path(root).resolve(), roi
        self.paths = sorted(self.root.glob('Image_*.bmp'))
        if not self.paths:
            raise ValueError('no raw Image_*.bmp frames')
        self.frames = [int(p.stem.split('_')[-1]) for p in self.paths]
        if self.frames != list(range(len(self.paths))):
            raise ValueError('raw frame numbers must be contiguous; timestamps are unavailable')
        shape = np.asarray(Image.open(self.paths[0])).shape
        if len(shape) != 2 or shape[0] % 3:
            raise ValueError('expected three stacked monochrome strips')
        self.shape = (shape[0]//3, shape[1])
        x, y, w, h = roi
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > self.shape[1] or y+h > self.shape[0]:
            raise ValueError('invalid valid-texture ROI')
        self.prepared = {}
        self.provenance = {'pixels': 'private raw BMPs', 'camera_roles': 'unavailable; each strip independent',
                           'masks': 'caller-selected texture ROI, not reviewed annotations',
                           'valid_roi': roi, 'foreground_and_blank_pairs': 'unavailable', 'motion_truth': 'unavailable'}

    def strip(self, frame, camera):
        pixels = np.asarray(Image.open(self.paths[frame]))
        h, w = self.shape
        if pixels.shape != (3*h, w):
            raise ValueError('raw frame dimensions changed')
        return pixels[camera*h:(camera+1)*h].copy()

    def prepare(self, frame, camera):
        key = frame, camera
        if key not in self.prepared:
            image = self.strip(frame, camera)
            valid = np.zeros(self.shape, bool)
            x, y, w, h = self.roi
            valid[y:y+h, x:x+w] = True
            self.prepared[key] = bandpass(image, valid), valid.astype(np.uint8)
        return self.prepared[key]


def estimate(estimator, dataset, previous, current, camera, prediction=None):
    start = perf_counter()
    a, am = dataset.prepare(previous, camera)
    b, bm = dataset.prepare(current, camera)
    preparation = 1000*(perf_counter()-start)
    start = perf_counter()
    result = estimator.estimate(a, b, previous_mask=am, current_mask=bm, prediction=prediction)
    row = {field: getattr(result, field) for field in FIELDS}
    return row | {'previous': previous, 'current': current, 'camera': camera,
                  'prepare_ms': preparation, 'estimate_python_ms': 1000*(perf_counter()-start)}


def evaluate(dataset, cameras, count, anchor_every, max_closure):
    rows = []
    for camera in cameras:
        estimator, anchor_estimator = vm.SimilarityEstimator(), vm.SimilarityEstimator()
        translation = vm.TranslationEstimator()
        chain = None
        for frame in range(count-1):
            roles = None if isinstance(dataset, RawStrips) else dataset.analysis['roles']['rear']
            if roles is not None and (roles[frame] != camera or roles[frame+1] != camera):
                if roles[frame] == camera:
                    rows.append({'previous': frame, 'current': frame+1, 'camera': camera, 'status': 'camera_handover'})
                chain = None
                continue
            row = estimate(estimator, dataset, frame, frame+1, camera)
            a, am = dataset.prepare(frame, camera)
            b, bm = dataset.prepare(frame+1, camera)
            start = perf_counter()
            t = translation.estimate(a, b, previous_mask=am, current_mask=bm)
            row['translation'] = {field: getattr(t, field) for field in FIELDS if field != 'transform'} | {
                'shift': t.shift, 'estimate_python_ms': 1000*(perf_counter()-start)}
            if t.status == 'accepted':
                reverse_t = translation.estimate(b, a, previous_mask=bm, current_mask=am)
                row['translation']['reverse_status'] = reverse_t.status
                row['translation']['closure_px'] = (float(np.linalg.norm(np.array(t.shift)+reverse_t.shift))
                                                    if reverse_t.status == 'accepted' else None)
            if row['status'] == 'accepted':
                reverse = estimate(estimator, dataset, frame+1, frame, camera)
                row['reverse_status'] = reverse['status']
                row['closure_px'] = (corner_distance(matrix(reverse['transform']) @ matrix(row['transform']),
                                                      np.eye(3), dataset.shape) if reverse['status'] == 'accepted' else None)
                row['sequence_status'] = ('reverse_' + reverse['status'] if reverse['status'] != 'accepted'
                                          else 'inconsistent_closure' if row['closure_px'] > max_closure else 'accepted')
                if row['sequence_status'] != 'accepted':
                    chain = None
                    row['anchor'] = {'status': 'unavailable_reciprocity'}
                    rows.append(row)
                    continue
                if chain is None:
                    chain = AnchoredSequence(frame, dataset.shape)
                chain.advance(row)
                age = frame+1-chain.seed
                if age % anchor_every == 0:
                    anchor = estimate(anchor_estimator, dataset, chain.seed, frame+1, camera, parts(chain.pose))
                    anchor['native_status'] = anchor['status']
                    if anchor['status'] == 'accepted':
                        reverse_anchor = estimate(anchor_estimator, dataset, frame+1, chain.seed, camera,
                                                  parts(np.linalg.inv(matrix(anchor['transform']))))
                        anchor['reverse_status'] = reverse_anchor['status']
                        anchor['closure_px'] = (corner_distance(matrix(reverse_anchor['transform']) @ matrix(anchor['transform']), np.eye(3), dataset.shape)
                                                if reverse_anchor['status'] == 'accepted' else None)
                        if reverse_anchor['status'] != 'accepted':
                            anchor['status'] = 'reverse_' + reverse_anchor['status']
                        elif anchor['closure_px'] > max_closure:
                            anchor['status'] = 'inconsistent_closure'
                    row['anchor'] = anchor
                    row['anchor_correction_px'] = chain.constrain(anchor)
                else:
                    row['anchor'] = {'status': 'not_attempted'}
                    row['anchor_correction_px'] = None
                row['sequence'] = chain.diagnostics()
            else:
                chain = None  # No geometry is transported across a loss of evidence.
            rows.append(row)
    accepted = [r for r in rows if r['status'] == 'accepted']
    closure = [r['closure_px'] for r in accepted if r.get('closure_px') is not None]
    times = [r['estimate_python_ms'] for r in rows if 'estimate_python_ms' in r]
    return {'summary': {'pairs': len(rows), 'accepted': len(accepted),
                       'coverage': len(accepted)/len(rows) if rows else 0,
                       'outcomes': dict(Counter(r['status'] for r in rows)),
                       'translation_outcomes': dict(Counter(r['translation']['status'] for r in rows if 'translation' in r)),
                       'closure_pairs': len(closure),
                       'sequence_outcomes': dict(Counter(r.get('sequence_status', r['status']) for r in rows)),
                       'reverse_outcomes': dict(Counter(r['reverse_status'] for r in accepted)),
                       'closure_px_median': float(np.median(closure)) if closure else None,
                       'closure_px_p95': float(np.percentile(closure, 95)) if closure else None,
                       'anchor_outcomes': dict(Counter(r['anchor']['status'] for r in accepted)),
                       'estimate_python_ms_median': float(np.median(times)) if times else None,
                       'estimate_python_ms_p95': float(np.percentile(times, 95)) if times else None},
            'pairs': rows, 'anchors': {'every': anchor_every, 'max_closure_px': max_closure, 'policy': 'fixed first keyframe per reciprocal supported camera segment; failed constraints unavailable'}}


def overlay(dataset, row, out):
    a = Image.fromarray(dataset.strip(row['previous'], row['camera'])).convert('RGB')
    b = Image.fromarray(dataset.strip(row['current'], row['camera'])).convert('RGB')
    if row.get('transform') is not None:
        # PIL's affine resampler expects output-to-input mapping.
        inverse = np.linalg.inv(matrix(row['transform']))
        a = a.transform(b.size, Image.Transform.AFFINE, tuple(inverse[:2].ravel()), Image.Resampling.BILINEAR)
    image = Image.blend(b, a, .45)
    draw = ImageDraw.Draw(image)
    draw.text((5, 5), f"camera {row['camera']} {row['previous']} > {row['current']} {row['status']}",
              fill=(70, 255, 140) if row['status'] == 'accepted' else (255, 190, 40))
    image.resize((960, 291)).save(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--sequence', default='15074')
    parser.add_argument('--valid-roi', type=int, nargs=4, default=[5, 5, 100, 85])
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--propagation-report', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-closure', type=float, default=.25)
    parser.add_argument('--anchor-every', type=int, default=8)
    args = parser.parse_args()
    if bool(args.raw_data) == bool(args.data_root) or args.anchor_every <= 0 or not np.isfinite(args.max_closure) or args.max_closure <= 0:
        parser.error('choose exactly one dataset input and a positive anchor interval/closure bound')
    root = (args.raw_data or args.data_root).resolve()
    out = args.out_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    if out.is_relative_to(root) or out.is_relative_to(repo):
        parser.error('output must be outside the source dataset and repository')
    out.mkdir(parents=True, exist_ok=True)
    if args.raw_data:
        dataset = RawStrips(root, args.valid_roi)
        cameras, count, provenance = range(3), len(dataset.paths), dataset.provenance
    else:
        dataset = Dataset(root, args.sequence)
        cameras, count = range(3), dataset.analysis['bead_frames']
        provenance = {'camera_roles': 'saved rear roles', 'masks_and_pairs': 'private sequence dataset manifest, annotation and registration metadata', 'motion_truth': 'unavailable'}
    if args.limit:
        count = min(count, args.limit)
    report = evaluate(dataset, cameras, count, args.anchor_every, args.max_closure)
    report['provenance'] = provenance
    cfg = vm.SimilarityConfig()
    report['settings'] = {'max_angle': cfg.max_angle, 'max_log_scale': cfg.max_log_scale,
                          'translation': {k: getattr(cfg.translation, k) for k in ('search_radius', 'min_support', 'min_score', 'min_margin', 'max_region_error')},
                          'tuning': {k: getattr(cfg.translation.tuning, k) for k in ('iterations', 'max_levels', 'tolerance', 'huber', 'min_texture')}}
    for camera in cameras:
        candidates = [r for r in report['pairs'] if r['camera'] == camera and r.get('transform') is not None]
        if candidates:
            for row in candidates[::max(1, len(candidates)//8)]:
                overlay(dataset, row, out/f"camera-{camera}-{row['current']:05}.png")
    from viewer import create
    propagation = None
    if args.propagation_report:
        saved = json.loads(args.propagation_report.read_text())
        propagation = saved['sequences'][args.sequence].get('propagation')
    create(dataset, report, out/'index.html', propagation)
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['summary'], indent=2))
    print(out/'index.html')


if __name__ == '__main__':
    main()
