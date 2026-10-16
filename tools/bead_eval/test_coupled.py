"""Tests for `coupled.py`: both solvers against brute force on small problems, and both
variants on synthetic bars whose walls are known exactly.

    python -m pytest tools/bead_eval/test_coupled.py

The solver tests need only numpy; the image tests need the `vision_metrology` bindings and
are skipped without them.
"""

from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.special import ndtr  # noqa: E402

import coupled  # noqa: E402


# ---------------------------------------------------------------------------
# Brute force
# ---------------------------------------------------------------------------
def tiny_profiles(rng: np.random.Generator, n: int, edges: list | None = None) -> coupled.Profiles:
    """`n` stations 2 px apart, profiles of 9 samples 0.5 px apart, random responses."""
    offsets = 0.5 * np.arange(-4, 5, dtype=np.float64)
    pts = np.stack([2.0 * np.arange(n), np.zeros(n)], axis=1)
    nrm = np.tile([0.0, 1.0], (n, 1))
    resp = rng.normal(0.0, 8.0, (n, len(offsets)))
    edges = edges if edges is not None else [np.zeros((0, 3)) for _ in range(n)]
    return coupled.Profiles(pts, nrm, 2.0, offsets, resp, edges)


TINY = coupled.CoupledConfig(
    polarity="dark", min_width=1.0, max_width=2.0, reach=1.0, step=0.5, g0=10.0, absent_level=4.0,
    wall_stiffness=0.3, width_stiffness=0.4, max_wall_slope=0.5, max_width_slope=0.5,
    offset_prior=0.2, switch_cost=1.5, reacquire_cost=4.0, max_bridge=4.0, tol=0.6, extra_tols=(),
)


def dense_states(pr: coupled.Profiles, cfg: coupled.CoupledConfig) -> list[tuple]:
    """(present, l, r, unary at each station) for every dense state."""
    sl, sr = coupled.wall_signs(cfg.polarity)
    t0 = (len(pr.offsets) - 1) // 2
    out = []
    for c in np.arange(-cfg.reach, cfg.reach + 1e-9, cfg.step):
        for w in np.arange(cfg.min_width, cfg.max_width + 1e-9, 2 * cfg.step):
            il, ir = t0 + round((c - w / 2) / cfg.step), t0 + round((c + w / 2) / cfg.step)
            data = -coupled.sat(sl * pr.response[:, il], cfg.g0) - coupled.sat(sr * pr.response[:, ir], cfg.g0)
            u = pr.h * (data + cfg.offset_prior * (c / cfg.reach) ** 2)
            out.append((True, c - w / 2, c + w / 2, u))
            out.append((False, c - w / 2, c + w / 2, np.full(len(pr.points), pr.h * coupled.absent_cost(cfg))))
    return out


def dense_trans(a: tuple, b: tuple, h: float, cfg: coupled.CoupledConfig) -> float:
    dl, dr = b[1] - a[1], b[2] - a[2]
    dc, dw = 0.5 * (dl + dr), dr - dl
    if abs(dc) > cfg.max_wall_slope * h + 1e-9 or abs(dw) > cfg.max_width_slope * h + 1e-9:
        return np.inf
    return (cfg.wall_stiffness * (dl**2 + dr**2) + cfg.width_stiffness * dw**2) / h + cfg.switch_cost * (a[0] != b[0])


def test_dense_matches_brute_force():
    rng = np.random.default_rng(1)
    for trial in range(3):
        n = 3
        pr = tiny_profiles(rng, n)
        cfg = TINY
        ex = coupled.dense_solve(pr, cfg, None)
        st = dense_states(pr, cfg)
        costs = {}
        for path in itertools.product(range(len(st)), repeat=n):
            c = sum(st[k][3][i] for i, k in enumerate(path))
            c += sum(dense_trans(st[path[i]], st[path[i + 1]], pr.h, cfg) for i in range(n - 1))
            costs[path] = c
        best_path = min(costs, key=costs.get)
        best = costs[best_path]
        assert abs(ex.cost - best) < 1e-9, f"trial {trial}: DP {ex.cost} vs brute force {best}"
        for i in range(n):
            assert ex.present[i] == st[best_path[i]][0]
            if ex.present[i]:
                assert abs(ex.left[i] - st[best_path[i]][1]) < 1e-9 and abs(ex.right[i] - st[best_path[i]][2]) < 1e-9
                far = [v for p, v in costs.items() if st[p[i]][0]
                       and max(abs(st[p[i]][1] - ex.left[i]), abs(st[p[i]][2] - ex.right[i])) > cfg.tol]
                want = min(far) - best if far else np.inf
                assert abs(ex.margin[i] - want) < 1e-9, f"station {i}: margin {ex.margin[i]} vs {want}"
                flip = min(v for p, v in costs.items() if not st[p[i]][0]) - best
                assert abs(ex.presence_margin[i] - flip) < 1e-9


def sparse_brute(pr: coupled.Profiles, cfg: coupled.CoupledConfig) -> tuple[float, dict]:
    """Every choice of absent or one candidate pair per station, costed as the DAG of
    `sparse_solve` defines it."""
    n, h = len(pr.points), pr.h
    cand = coupled.pairs(pr, cfg)
    ab, s, rq = h * coupled.absent_cost(cfg), cfg.switch_cost, cfg.reacquire_cost
    kmax = max(1, int(np.floor(cfg.max_bridge / h)) + 1)
    costs = {}
    for choice in itertools.product(*[range(-1, len(c)) for c in cand]):
        on = [i for i, k in enumerate(choice) if k >= 0]
        if not on:
            costs[choice] = n * ab
            continue
        c = sum(h * cand[i][choice[i], 2] for i in on) + (n - len(on)) * ab
        c += s * (on[0] > 0) + s * (on[-1] < n - 1)
        for i, j in itertools.pairwise(on):
            gap = j - i - 1
            li, ri = cand[i][choice[i], :2]
            lj, rj = cand[j][choice[j], :2]
            d = (j - i) * h
            dl, dr = lj - li, rj - ri
            ok = j - i <= kmax and max(abs(dl), abs(dr)) <= cfg.max_wall_slope * d and abs(dr - dl) <= cfg.max_width_slope * d
            link = (cfg.wall_stiffness * (dl**2 + dr**2) + cfg.width_stiffness * (dr - dl) ** 2) / d + 2 * s * (gap > 0) if ok else np.inf
            if gap > 0:
                link = min(link, 2 * s + rq)
            c += link
        costs[choice] = c
    return min(costs.values()), costs


def test_sparse_matches_brute_force():
    rng = np.random.default_rng(2)
    cfg = dataclasses.replace(TINY, max_width=4.0, reach=2.0, max_bridge=4.0, max_wall_slope=0.4)
    for trial in range(4):
        n = 5
        edges = []
        for _ in range(n):
            k = rng.integers(0, 3)
            fall = np.stack([rng.uniform(-3, 0, k), rng.uniform(3, 20, k), -np.ones(k)], axis=1)
            rise = np.stack([rng.uniform(0, 3, 2), rng.uniform(3, 20, 2), np.ones(2)], axis=1)
            edges.append(np.concatenate([fall, rise]))
        pr = tiny_profiles(rng, n, edges)
        ex = coupled.sparse_solve(pr, cfg, None)
        best, costs = sparse_brute(pr, cfg)
        assert abs(ex.cost - best) < 1e-9, f"trial {trial}: Viterbi {ex.cost} vs brute force {best}"
        cand = coupled.pairs(pr, cfg)
        chosen = min(costs, key=costs.get)
        assert list(ex.present) == [k >= 0 for k in chosen]
        for i in np.nonzero(ex.present)[0]:
            far = [v for p, v in costs.items() if p[i] >= 0
                   and max(abs(cand[i][p[i], 0] - ex.left[i]), abs(cand[i][p[i], 1] - ex.right[i])) > cfg.tol]
            want = min(far) - best if far else np.inf
            assert (np.isinf(want) and np.isinf(ex.margin[i])) or abs(ex.margin[i] - want) < 1e-9


# ---------------------------------------------------------------------------
# Synthetic bars
# ---------------------------------------------------------------------------
vm = pytest.importorskip("vision_metrology")


def bars(walls: list[tuple[float, float]], polarity: str = "dark", gap: tuple | None = None, seed: int = 0) -> np.ndarray:
    """Horizontal bars between the given rows (walls at those `y`), blurred by 1 px,
    120 DN background and 60 DN contrast, noise 1 DN; `gap` blanks columns `[a, b)`."""
    yy = np.mgrid[0:120, 0:240][0].astype(np.float64)
    sign = -1.0 if polarity == "dark" else 1.0
    img = np.full(yy.shape, 120.0)
    for a, b in walls:
        bar = ndtr((yy - a) / 1.0) - ndtr((yy - b) / 1.0)
        if gap is not None:
            bar[:, gap[0]:gap[1]] = 0.0
        img += sign * 60.0 * bar
    img += np.random.default_rng(seed).normal(0.0, 1.0, img.shape)
    return img.astype(np.float32)


PRIOR = np.array([[20.0, 60.0], [220.0, 60.0]])  # +n points to +y


@pytest.mark.parametrize("method", ["sparse", "dense"])
@pytest.mark.parametrize("polarity", ["dark", "light"])
def test_asymmetric_walls_are_found_subpixel(method, polarity):
    img = bars([(55.3, 61.8)], polarity)
    cfg = coupled.CoupledConfig(polarity=polarity, min_width=2.0, max_width=12.0, threshold=5.0)
    _, ex, _ = coupled.extract(img, PRIOR, cfg, method)
    assert ex.support == 1.0
    for got, want in ((ex.left, 55.3 - 60.0), (ex.right, 61.8 - 60.0)):
        err = got - want
        assert abs(np.mean(err)) < 0.02 and np.max(np.abs(err)) < 0.1, f"wall error mean {np.mean(err)}, max {np.max(np.abs(err))}"
    assert ex.snapped.all(), "every wall comes from a caliper edge"


def test_result_does_not_depend_on_station_spacing():
    img = bars([(55.3, 61.8), (68.0, 72.0)])
    out = []
    for spacing in (1.0, 2.0, 4.0):
        cfg = coupled.CoupledConfig(min_width=2.0, max_width=12.0, threshold=5.0, spacing=spacing)
        pr, ex, _ = coupled.extract(img, PRIOR + [0.0, 3.0], cfg, "dense")
        out.append((ex.support, np.median(ex.left), np.median(ex.right), ex.cost / (pr.h * (len(pr.points) - 1))))
    for o in out[1:]:
        assert o[0] == out[0][0] and abs(o[1] - out[0][1]) < 0.02 and abs(o[2] - out[0][2]) < 0.02
        assert abs(o[3] - out[0][3]) / abs(out[0][3]) < 0.05, f"cost per px {o[3]} vs {out[0][3]}"


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_a_parallel_line_lowers_the_whole_call_margin(method):
    # Two identical bars 14 px apart; a prior on one of them, then one midway.
    img = bars([(52.0, 60.0), (66.0, 74.0)])
    cfg = coupled.CoupledConfig(min_width=2.0, max_width=12.0, threshold=5.0)
    _, on, first_on = coupled.extract(img, np.array([[20.0, 56.0], [220.0, 56.0]]), cfg, method)
    _, mid, first_mid = coupled.extract(img, np.array([[20.0, 63.0], [220.0, 63.0]]), cfg, method)
    assert on.support == mid.support == 1.0
    clear, ambiguous = first_on.ambiguity(0.25, 5.0), first_mid.ambiguity(0.25, 5.0)
    assert ambiguous < 0.25 * clear, f"ambiguity midway {ambiguous} vs on the bar {clear}"


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_a_gap_is_absent_and_bridged(method):
    img = bars([(56.0, 64.0)], gap=(100, 140))
    cfg = coupled.CoupledConfig(min_width=2.0, max_width=12.0, threshold=5.0)
    pr, ex, _ = coupled.extract(img, PRIOR + [0.0, 2.0], cfg, method)
    x = pr.points[:, 0]
    inside, outside = (x > 104) & (x < 136), (x < 96) | (x > 144)
    assert not ex.present[inside].any(), "no bead in the gap"
    assert ex.present[outside].all()
    assert np.all(np.abs(ex.center + 2.0) < 0.1), "the centre is bridged across the gap"
