/**
 * The 166 contours, as something you can reason about.
 *
 * `teach_preview` returns every candidate edge inside the ROI with the two facts that
 * decide whether it belongs to the part or to the bench it is sitting on — arc `length` and
 * `mean_strength` — and the lab displayed neither. All that reached the screen was one
 * cyan colour and a count, so curating meant clicking contours one at a time and hoping,
 * and "7365 points" was a number with no way to look at it.
 *
 * This is the arithmetic behind an inventory: per-contour facts, an order, a filter, and a
 * selection model with the four modes a list-plus-canvas needs. Pure, because a selection
 * bug is invisible — the wrong contour highlighted looks exactly like the right one.
 *
 * Two states per contour, kept deliberately separate:
 *
 *   - **kept** — will be in the model (`ModelCreateRequest.keep_contours`);
 *   - **selected** — the current working set, which the keep actions operate on.
 *
 * The old UI conflated them: a click *was* a drop, so there was no way to look at a contour
 * without changing the model, and no way to act on several at once.
 */

import type { ContourOut } from "../api/backend";

export interface Bounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** One contour, with the facts derived once instead of per render. */
export interface ContourStat {
  id: number;
  /** Vertices, i.e. `points.length / 2`. */
  points: number;
  /** Arc length in pixels, from the backend. */
  length: number;
  /** Mean gradient magnitude along the contour, from the backend. */
  strength: number;
  closed: boolean;
  bounds: Bounds;
}

export type SortKey = "id" | "length" | "strength" | "points";
export type KeepFilter = "all" | "kept" | "dropped";
/** How a gesture combines with the selection already in hand. */
export type SelectMode = "replace" | "toggle" | "add" | "range";

const EMPTY_BOUNDS: Bounds = { x: 0, y: 0, width: 0, height: 0 };

/** Per-contour facts, in the backend's own order. One pass over the points, memoised by the caller. */
export function describeContours(contours: ContourOut[]): ContourStat[] {
  return contours.map((contour) => ({
    id: contour.id,
    points: contour.points.length / 2,
    length: contour.length,
    strength: contour.mean_strength,
    closed: contour.closed,
    bounds: boundsOfPoints(contour.points),
  }));
}

export function boundsOfPoints(points: number[]): Bounds {
  if (points.length < 2) return EMPTY_BOUNDS;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i + 1 < points.length; i += 2) {
    const x = points[i]!;
    const y = points[i + 1]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

/** The box around a set of contours — what "frame the selection" needs. */
export function boundsOf(stats: ContourStat[]): Bounds | null {
  if (stats.length === 0) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const stat of stats) {
    minX = Math.min(minX, stat.bounds.x);
    minY = Math.min(minY, stat.bounds.y);
    maxX = Math.max(maxX, stat.bounds.x + stat.bounds.width);
    maxY = Math.max(maxY, stat.bounds.y + stat.bounds.height);
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

/**
 * Sorted for reading: quantities descend, because the question a list of contours answers
 * is "which are the substantial ones", and id ascends, because that is an index, not a
 * measurement. Ties break on id so the order is total — an unstable order makes
 * keyboard iteration jump around between renders.
 */
export function sortContours(stats: ContourStat[], key: SortKey): ContourStat[] {
  const sorted = [...stats];
  sorted.sort((a, b) => {
    switch (key) {
      case "length":
        return b.length - a.length || a.id - b.id;
      case "strength":
        return b.strength - a.strength || a.id - b.id;
      case "points":
        return b.points - a.points || a.id - b.id;
      case "id":
        return a.id - b.id;
    }
  });
  return sorted;
}

export function filterContours(
  stats: ContourStat[],
  filter: KeepFilter,
  kept: ReadonlySet<number>,
): ContourStat[] {
  if (filter === "all") return stats;
  const wanted = filter === "kept";
  return stats.filter((stat) => kept.has(stat.id) === wanted);
}

/**
 * Apply a selection gesture.
 *
 * `order` is the list's *current* order, so a shift-click range means the rows between the
 * two clicks as the user sees them — a range over ids would select something else entirely
 * whenever the list is sorted by length or strength, which is most of the time.
 *
 * Returns the new anchor as well: `range` extends from the anchor without moving it, so
 * shift-clicking twice re-ranges from the same origin rather than walking the selection
 * down the list.
 */
export function applySelect(
  current: ReadonlySet<number>,
  anchor: number | null,
  ids: number[],
  mode: SelectMode,
  order: number[],
): { selected: Set<number>; anchor: number | null } {
  if (mode === "replace") {
    return { selected: new Set(ids), anchor: ids[ids.length - 1] ?? null };
  }
  if (mode === "add") {
    const selected = new Set(current);
    for (const id of ids) selected.add(id);
    return { selected, anchor: ids[ids.length - 1] ?? anchor };
  }
  if (mode === "toggle") {
    const selected = new Set(current);
    for (const id of ids) {
      if (selected.has(id)) selected.delete(id);
      else selected.add(id);
    }
    return { selected, anchor: ids[ids.length - 1] ?? anchor };
  }

  // range
  const target = ids[ids.length - 1];
  if (target === undefined) return { selected: new Set(current), anchor };
  const from = anchor === null ? target : anchor;
  const a = order.indexOf(from);
  const b = order.indexOf(target);
  if (a < 0 || b < 0) return { selected: new Set([target]), anchor: target };
  const [lo, hi] = a <= b ? [a, b] : [b, a];
  return { selected: new Set(order.slice(lo, hi + 1)), anchor: from };
}

/** Keep or drop a set of ids, leaving everything else alone. */
export function applyKeep(
  kept: ReadonlySet<number>,
  ids: Iterable<number>,
  keep: boolean,
): Set<number> {
  const next = new Set(kept);
  for (const id of ids) {
    if (keep) next.add(id);
    else next.delete(id);
  }
  return next;
}

export function invertKeep(kept: ReadonlySet<number>, all: ContourStat[]): Set<number> {
  return new Set(all.filter((stat) => !kept.has(stat.id)).map((stat) => stat.id));
}

/**
 * Which contours a rubber band caught: **any** vertex inside the box.
 *
 * The predecessor tested one vertex — the middle one — so a long contour crossing the band
 * was missed, which is the case a sweep is most often aimed at.
 */
export function contoursInBox(contours: ContourOut[], box: Bounds): number[] {
  const right = box.x + box.width;
  const bottom = box.y + box.height;
  const hits: number[] = [];
  for (const contour of contours) {
    const points = contour.points;
    for (let i = 0; i + 1 < points.length; i += 2) {
      const x = points[i]!;
      const y = points[i + 1]!;
      if (x >= box.x && x <= right && y >= box.y && y <= bottom) {
        hits.push(contour.id);
        break;
      }
    }
  }
  return hits;
}

/** Total vertices over a set of ids — the "5940 of 7365 points" half of the summary. */
export function pointsIn(stats: ContourStat[], ids: ReadonlySet<number>): number {
  let total = 0;
  for (const stat of stats) if (ids.has(stat.id)) total += stat.points;
  return total;
}

/** The next id in the list order, wrapping — how `↑`/`↓` walk the inventory. */
export function stepThrough(order: number[], current: number | null, delta: 1 | -1): number | null {
  if (order.length === 0) return null;
  if (current === null) return delta > 0 ? order[0]! : order[order.length - 1]!;
  const index = order.indexOf(current);
  if (index < 0) return order[0]!;
  const next = (index + delta + order.length) % order.length;
  return order[next]!;
}
