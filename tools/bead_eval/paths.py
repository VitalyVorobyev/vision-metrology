"""Reference centrelines from the DamSegment crack masks.

Per image: crack pixels of the mask -> skeleton -> the skeleton graph, with short spurs
pruned, split at junctions and endpoints into non-branching paths -> each path ordered,
trimmed back from junctions, smoothed along its length and resampled at 1 px -> paths of
at least `--min-length` px -> the local width from the distance transform,
`2 * EDT - 1`, sampled on the path and smoothed.

The geometry is pixel-level: the masks are hand-drawn polygons, rasterised.

    python -I tools/bead_eval/paths.py --data-dir data/damsegment/extracted
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Isolated mode (-I) leaves the script's own directory off sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

import common  # noqa: E402

OFFSETS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def adjacency(skel: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """8-connected neighbours of each skeleton pixel `(row, col)`, without a diagonal
    step where a 4-connected route through a shared neighbour exists. Without that rule
    every staircase corner would count as a junction."""
    pix = {(int(r), int(c)) for r, c in zip(*np.nonzero(skel))}
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in pix:
        nb = []
        for dr, dc in OFFSETS8:
            q = (r + dr, c + dc)
            if q not in pix:
                continue
            if dr and dc and ((r + dr, c) in pix or (r, c + dc) in pix):
                continue
            nb.append(q)
        adj[(r, c)] = nb
    return adj


def branches(adj: dict) -> tuple[list[list[tuple[int, int]]], int]:
    """Non-branching chains between graph nodes (pixels whose degree is not 2), and the
    number of closed loops (chains that return to where they started, or components
    with no node at all), which are skipped."""
    nodes = sorted(p for p, nb in adj.items() if len(nb) != 2)
    node_set = set(nodes)
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    visited: set[tuple[int, int]] = set(nodes)
    chains = []
    loops = 0
    for n in nodes:
        for q in sorted(adj[n]):
            if (n, q) in seen:
                continue
            chain = [n, q]
            seen.update({(n, q), (q, n)})
            prev, cur = n, q
            while cur not in node_set:
                visited.add(cur)
                nxt = [x for x in adj[cur] if x != prev][0]
                seen.update({(cur, nxt), (nxt, cur)})
                chain.append(nxt)
                prev, cur = cur, nxt
            if chain[0] == chain[-1]:
                loops += 1
            else:
                chains.append(chain)
    # Components made only of degree-2 pixels are closed loops.
    rest = set(adj) - visited
    while rest:
        loops += 1
        stack = [rest.pop()]
        while stack:
            for x in adj[stack.pop()]:
                if x in rest:
                    rest.remove(x)
                    stack.append(x)
    return chains, loops


def chain_length(chain: list[tuple[int, int]]) -> float:
    a = np.asarray(chain, dtype=float)
    return float(np.hypot(*np.diff(a, axis=0).T).sum())


def prune_spurs(skel: np.ndarray, spur_px: float, rounds: int = 3) -> np.ndarray:
    """Remove dangling branches shorter than `spur_px` (a tip joined to a junction):
    the bumps of a hand-drawn outline grow them, and each would split a crack in two."""
    skel = skel.copy()
    for _ in range(rounds):
        adj = adjacency(skel)
        chains, _ = branches(adj)
        changed = False
        for chain in chains:
            da, db = len(adj[chain[0]]), len(adj[chain[-1]])
            if da == 1 and db >= 3:
                drop = chain[:-1]
            elif db == 1 and da >= 3:
                drop = chain[1:]
            else:
                continue
            if chain_length(chain) < spur_px:
                for r, c in drop:
                    skel[r, c] = False
                changed = True
        if not changed:
            break
    return skel


def smooth_path(p: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along a 1 px resampled path. The ends are padded by point
    reflection, which keeps a straight end straight instead of curling it."""
    k = int(np.ceil(4 * sigma))
    if len(p) <= 2 * k + 2:
        k = max(1, (len(p) - 2) // 2)
    head = 2 * p[0] - p[k:0:-1]
    tail = 2 * p[-1] - p[-2 : -k - 2 : -1]
    padded = np.concatenate([head, p, tail])
    out = ndimage.gaussian_filter1d(padded, sigma, axis=0, mode="nearest")
    return out[k : k + len(p)]


def trim(p: np.ndarray, start: float, end: float) -> np.ndarray:
    """Cut `start` px of arc length off the front and `end` px off the back of a 1 px
    resampled path."""
    i0 = int(round(start))
    i1 = len(p) - int(round(end))
    return p[i0:i1]


def end_kind(px: tuple[int, int], degree: int, shape: tuple[int, int]) -> str:
    r, c = px
    if r <= 1 or c <= 1 or r >= shape[0] - 2 or c >= shape[1] - 2:
        return "border"
    return "junction" if degree >= 3 else "tip"


def image_paths(mask: np.ndarray, args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Reference paths of one crack mask, and the skeleton's bookkeeping."""
    skel = prune_spurs(skeletonize(mask), args.spur_px)
    adj = adjacency(skel)
    chains, loops = branches(adj)
    edt = ndimage.distance_transform_edt(mask)
    out = []
    for chain in chains:
        if chain_length(chain) < args.min_length:
            continue
        ends = [
            end_kind(chain[0], len(adj[chain[0]]), mask.shape),
            end_kind(chain[-1], len(adj[chain[-1]]), mask.shape),
        ]
        xy = np.asarray([(c, r) for r, c in chain], dtype=float)
        p = common.resample(xy, 1.0)
        p = trim(
            p,
            args.junction_trim if ends[0] == "junction" else 0.0,
            args.junction_trim if ends[1] == "junction" else 0.0,
        )
        if len(p) < 3:
            continue
        p = common.resample(smooth_path(p, args.smooth_px), 1.0)
        length = float(common.arc_length(p)[-1])
        if length < args.min_length:
            continue
        d = ndimage.map_coordinates(edt, [p[:, 1], p[:, 0]], order=1, mode="nearest")
        width = ndimage.gaussian_filter1d(np.maximum(2.0 * d - 1.0, 1.0), 5.0, mode="nearest")
        out.append(
            {
                "length": round(length, 2),
                "ends": ends,
                "width_median": round(float(np.median(width)), 3),
                "points": np.round(p, 3).tolist(),
                "width": np.round(width, 3).tolist(),
            }
        )
    info = {
        "crack_pixels": int(mask.sum()),
        "skeleton_pixels": int(skel.sum()),
        "branches": len(chains),
        "loops_skipped": loops,
    }
    return out, info


def main() -> None:
    parser = common.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    parser.add_argument("--min-length", type=float, default=80.0, help="px (default 80)")
    parser.add_argument("--spur-px", type=float, default=12.0, help="prune shorter spurs (default 12)")
    parser.add_argument(
        "--junction-trim", type=float, default=6.0, help="px cut back from a junction (default 6)"
    )
    parser.add_argument("--smooth-px", type=float, default=3.0, help="Gaussian sigma (default 3)")
    parser.add_argument("--limit", type=int, default=None, help="first N images per difficulty")
    args = parser.parse_args()
    root = common.dataset_root(args.data_dir)
    out_dir = common.resolve_out_dir(args) / "paths"

    params = {
        "min_length": args.min_length,
        "spur_px": args.spur_px,
        "junction_trim": args.junction_trim,
        "smooth_px": args.smooth_px,
        "width": "2 * EDT - 1 of the crack mask, bilinear on the path, Gaussian sigma 5 px",
    }
    index = {"params": params, "difficulties": {}, "paths": []}
    for diff in common.DIFFICULTIES:
        stems = common.image_stems(root, diff)[: args.limit]
        n_paths = n_cracked = 0
        for stem in stems:
            mask = common.load_crack_mask(common.mask_path(root, diff, stem))
            paths, info = image_paths(mask, args)
            n_cracked += int(info["crack_pixels"] > 0)
            for i, p in enumerate(paths):
                p["id"] = f"{diff}/{stem}/{i}"
            rel = f"{diff}/{stem}.json"
            common.write_json(
                out_dir / rel,
                {
                    "difficulty": diff,
                    "stem": stem,
                    "image": f"{diff}/Images/{stem}.jpg",
                    "mask": f"{diff}/Labels/Mask/{stem}_mask.png",
                    "size": [int(mask.shape[1]), int(mask.shape[0])],
                    "params": params,
                    "skeleton": info,
                    "paths": paths,
                },
            )
            for p in paths:
                index["paths"].append(
                    {
                        "id": p["id"],
                        "difficulty": diff,
                        "stem": stem,
                        "file": rel,
                        "length": p["length"],
                        "width_median": p["width_median"],
                        "ends": p["ends"],
                    }
                )
            n_paths += len(paths)
        index["difficulties"][diff] = {
            "images": len(stems),
            "images_with_cracks": n_cracked,
            "paths": n_paths,
        }
        print(f"{diff}: {len(stems)} images, {n_cracked} with cracks, {n_paths} paths")
    common.write_json(out_dir / "index.json", index)
    print(f"wrote {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
