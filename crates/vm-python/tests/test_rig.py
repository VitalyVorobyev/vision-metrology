"""Planar rig substrate poses: three affine views 120° apart, known increments."""
import numpy as np
import pytest
import vision_metrology as vm

N, SCALE, RADIUS = 3, 20.0, 30.0


def views():
    hs = []
    for v in range(N):
        a = 2 * np.pi * v / N
        c, s = np.cos(a), np.sin(a)
        cx, cy = RADIUS * c, RADIUS * s
        A = SCALE * np.array([[c, -s], [s, c]])
        h = np.eye(3)
        h[:2, :2] = A
        h[:2, 2] = np.array([640.0, 512.0]) - A @ [cx, cy]
        hs.append(h)
    return np.array(hs)


def sequence(steps, theta=2e-4, t=(0.4, 0.05), sigma=0.0, seed=1):
    rng = np.random.default_rng(seed)
    hs = views()
    d = np.zeros((steps, N, 2))
    centroid = np.tile([640.0, 512.0], (steps, N, 1))
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for v in range(N):
        p0 = np.linalg.solve(hs[v], [640.0, 512.0, 1.0])
        p0 = p0[:2] / p0[2]
        q = hs[v] @ np.append(R @ p0 + t, 1.0)
        d[:, v] = q[:2] / q[2] - [640.0, 512.0]
    d += sigma * rng.standard_normal(d.shape)
    var = max(sigma, 1e-3) ** 2
    cov = np.tile([var, 0.0, var], (steps, N, 1))
    accepted = np.ones((steps, N), dtype=bool)
    return hs, d, cov, centroid, accepted


def test_recovers_increments_with_provenance():
    hs, d, cov, cen, acc = sequence(20)
    d[5] = np.nan                      # step 5: nothing measured -> bridged
    acc[10, 1:] = False                # step 10: one view -> single_view
    d[15, 2] += [3.0, -1.0]            # step 15: a gross view
    traj = vm.RigPoseEstimator().estimate(hs, d, cov, cen, acc)
    assert traj.poses.shape == (21, 3)
    assert traj.covariances.shape == (21, 3, 3)
    assert traj.steps[0] == "anchor"
    assert traj.steps[6] == "bridged" and traj.views_used[6] == 0
    assert traj.steps[11] == "single_view"
    assert traj.view_outcomes[10] == ["used", "not_accepted", "not_accepted"]
    assert traj.view_outcomes[5] == ["missing"] * 3
    assert traj.view_outcomes[15][2] == "rejected"
    assert traj.view_weights.shape == (20, 3)
    assert traj.view_levers_mm.shape == (20, 3, 2)
    assert np.isnan(traj.view_levers_mm[5]).all()
    assert (traj.measured, traj.single_view, traj.bridged) == (18, 1, 1)
    inc = traj.increments
    good = [k for k in range(1, 21) if k not in (6, 11)]
    assert np.abs(inc[good, 0] - 0.4).max() < 1e-4
    assert np.abs(inc[good, 2] - 2e-4).max() < 1e-6
    np.testing.assert_allclose(inc[11, 2], inc[10, 2], atol=1e-9)   # held at the prior
    np.testing.assert_allclose(inc[6], inc[5], atol=1e-9)           # constant velocity
    assert np.isnan(traj.residual_rms_mm[0])
    assert traj.residual_median_mm < 1e-3


def test_config_hold_zero_and_smoothing():
    hs, d, cov, cen, acc = sequence(40, sigma=0.1)
    acc[20, 1:] = False
    cfg = vm.RigPoseConfig(rotation_hold="zero", tukey_c=None)
    est = vm.RigPoseEstimator(cfg)
    traj = est.estimate(hs, d, cov, cen, acc)
    assert traj.increments[21, 2] == 0.0
    sm = est.smooth(traj, vm.RigSmootherConfig(
        motion_noise=vm.MotionNoise(translation_mm=1e-3, rotation_rad=1e-5)))
    assert sm.poses.shape == (41, 3)
    assert len(sm.gated) == 41 and sm.gated_count == sum(sm.gated)
    raw = np.abs(traj.increments[1:, 0] - 0.4).mean()
    smooth = np.abs(sm.increments[1:, 0] - 0.4).mean()
    assert smooth < raw
    ext = traj.poses[:, :2].copy()
    assert abs(traj.path_length_ratio((0.0, 0.0), ext) - 1.0) < 1e-6
    assert sm.path_length_ratio((0.0, 0.0), np.zeros((41, 2))) is None


def test_bad_input_raises():
    hs, d, cov, cen, acc = sequence(4)
    est = vm.RigPoseEstimator()
    with pytest.raises(ValueError):
        est.estimate(hs[:2], d, cov, cen, acc)
    with pytest.raises(ValueError):
        est.estimate(hs, d, cov[:, :, :2], cen, acc)
    with pytest.raises(ValueError):
        vm.RigPoseEstimator(vm.RigPoseConfig(rotation_hold="sideways")).estimate(
            hs, d, cov, cen, acc)
