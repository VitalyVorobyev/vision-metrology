"""Rig poses on a generated sequence: per-view translations, then `RigPoseEstimator`.

Each step and view registers the previous frame against the current one with
`TranslationEstimator` over both frames' registration masks, and lifts it at the
centroid of the previous frame's mask. The registration runs with its defaults except
`max_region_error=None`: the substrate rotates in every view on the arc, so the regions
of one view disagree by design, and the cross-view solve is the consistency check here. The rig estimator runs with its
defaults, anchored at the given frame-0 pose. Against `truth.poses` it reports, per step,
the increment's translation error (su) and rotation error (degrees) as bias, σ and count,
the provenance of the steps the preset is about, and the path-length ratio of the
substrate origin.

    python tools/sequence_bench/generate.py --preset rig-gross --out-dir /tmp/bench/rig-gross
    python tools/sequence_bench/rig_eval.py /tmp/bench/rig-gross
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import vision_metrology as vm


def _load(root: Path, rel: str | None):
    return None if rel is None else np.asarray(Image.open(root / rel))


def _increment(a, b):
    """The plane-frame increment from pose a to pose b, as (tx, ty, theta)."""
    ta, tb = np.asarray(a[:2]), np.asarray(b[:2])
    th = b[2] - a[2]
    c, s = np.cos(th), np.sin(th)
    t = tb - np.array([[c, -s], [s, c]]) @ ta
    return np.array([t[0], t[1], th])


def measure(root: Path):
    manifest = json.loads((root / 'sequence.json').read_text())
    truth = json.loads((root / manifest['truth']['file']).read_text())
    rig = manifest['rig']
    times, views = rig['times'], rig['views']
    frames = {(f['view'], f['time']): f for f in manifest['frames']}
    n, steps = len(views), times - 1
    d = np.full((steps, n, 2), np.nan)
    cov = np.ones((steps, n, 3))
    cov[:, :, 1] = 0.0
    cen = np.zeros((steps, n, 2))
    acc = np.zeros((steps, n), dtype=bool)
    est = vm.TranslationEstimator(vm.TranslationConfig(max_region_error=None))
    for k in range(steps):
        for v in range(n):
            a, b = frames.get((v, k)), frames.get((v, k + 1))
            if a is None or b is None or a['image'] is None or b['image'] is None:
                continue
            ma, mb = _load(root, a['mask']), _load(root, b['mask'])
            ys, xs = np.nonzero(ma)
            if len(xs) == 0:
                continue
            cen[k, v] = [xs.mean(), ys.mean()]
            r = est.estimate(_load(root, a['image']), _load(root, b['image']),
                             previous_mask=(ma > 0).astype(np.uint8),
                             current_mask=(mb > 0).astype(np.uint8))
            if r.shift is None:
                continue
            d[k, v] = r.shift
            if r.covariance is not None:
                cov[k, v] = r.covariance
            acc[k, v] = r.status == 'accepted'
    h = np.array([np.asarray(view['plane_to_image'], dtype=np.float64) for view in views])
    cfg = vm.RigPoseConfig(anchor=tuple(float(x) for x in rig['poses'][0]))
    traj = vm.RigPoseEstimator(cfg).estimate(h, d, cov, cen, acc)
    poses = np.asarray(truth['poses'], dtype=np.float64)
    offsets = np.asarray(truth['image_offsets'], dtype=np.float64)
    return traj, poses, _registration_error(h, poses, offsets, d, cen, acc)


def _registration_error(h, poses, offsets, d, cen, acc):
    """Each accepted shift minus the exact image motion of its centroid, camera motion
    included, in px: what the registration got wrong, apart from the rig model."""
    errs = []
    for k in range(d.shape[0]):
        inc = _increment(poses[k], poses[k + 1])
        c, s = np.cos(inc[2]), np.sin(inc[2])
        for v in range(d.shape[1]):
            if not acc[k, v]:
                continue
            q = np.linalg.solve(h[v], np.append(cen[k, v] - offsets[k][v], 1.0))
            p0 = q[:2] / q[2]
            p1 = np.array([[c, -s], [s, c]]) @ p0 + inc[:2]
            i1 = h[v] @ np.append(p1, 1.0)
            errs.append(d[k, v] - (i1[:2] / i1[2] + offsets[k + 1][v] - cen[k, v]))
    return np.array(errs).reshape(-1, 2)


def report(name: str, traj, poses, reg) -> dict:
    inc = traj.increments[1:]
    true = np.array([_increment(poses[k], poses[k + 1]) for k in range(len(poses) - 1)])
    err = inc - true
    err[:, 2] = np.degrees(np.angle(np.exp(1j * err[:, 2])))
    out = {'preset': name, 'steps': len(err)}
    out['registration_px'] = {'bias': reg.mean(axis=0).tolist(), 'sigma': reg.std(axis=0).tolist(),
                              'count': len(reg)}
    for i, axis in enumerate(['tx_su', 'ty_su', 'theta_deg']):
        out[axis] = {'bias': float(err[:, i].mean()), 'sigma': float(err[:, i].std()),
                     'max': float(np.abs(err[:, i]).max())}
    measured = [k for k, s in enumerate(traj.steps[1:]) if s == 'measured']
    out['measured_only'] = {
        axis: {'bias': float(err[measured, i].mean()), 'sigma': float(err[measured, i].std())}
        for i, axis in enumerate(['tx_su', 'ty_su', 'theta_deg'])}
    out['measured_only']['count'] = len(measured)
    pose_err = traj.poses - poses
    out['final_pose_error'] = {'translation_su': float(np.hypot(*pose_err[-1, :2])),
                               'rotation_deg': float(np.degrees(pose_err[-1, 2]))}
    out['counts'] = {'measured': traj.measured, 'single_view': traj.single_view,
                     'bridged': traj.bridged}
    out['steps_by_kind'] = {kind: [k for k, s in enumerate(traj.steps) if s == kind]
                            for kind in ('single_view', 'bridged')}
    w = traj.view_weights
    out['view_weights_15_24'] = [[round(float(x), 3) for x in w[k]] for k in range(14, 24)]
    out['outcomes_15_24'] = traj.view_outcomes[14:24]
    out['residual_median'] = traj.residual_median_mm
    out['residual_scale_median'] = float(np.median(traj.residual_scale[1:]))
    out['path_length_ratio'] = traj.path_length_ratio((0.0, 0.0), poses[:, :2].copy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('roots', nargs='+', type=Path)
    args = parser.parse_args()
    for root in args.roots:
        traj, poses, reg = measure(root)
        print(json.dumps(report(root.name, traj, poses, reg), indent=1))


if __name__ == '__main__':
    main()
