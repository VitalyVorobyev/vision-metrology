/**
 * The match inventory's arithmetic: per-match facts, the box each match occupies, an order,
 * and the overlay state each match is drawn in.
 *
 * A search returns its matches as poses. The list shows the pose and the two numbers that
 * say how good it is (`score`, and `support`, the model points that agreed), and the canvas
 * draws the model at each pose. Both act on the state defined here, which is pure for the
 * same reason the contour inventory's is: the wrong match highlighted looks exactly like the
 * right one.
 *
 * A match is named by its **position in the search result** (`index`), never by its place
 * in the sorted list: that is what `highlightedMatch` holds, what the overlay ids carry
 * (`match-<index>`), and what Verify's rectified crop of instance `index` is keyed on.
 */

import type { OverlayState, Point, Rect } from "@vitavision/stage2d";

import type { BatchFindItem, MatchOut, ModelGeometryOut, ModelOut } from "../api/backend";
import { boxBounds, pickBox, type RotatedBox } from "./rotatedBox";

/** One match, with the facts derived once instead of per render. */
export interface MatchStat {
  /** Position in the search result. */
  index: number;
  score: number;
  x: number;
  y: number;
  /** Degrees, for reading; the pose itself stays in radians on `MatchOut`. */
  angle: number;
  scale: number;
  /** Model points that agreed with the image. */
  support: number;
  /** The model's extent at the found pose. */
  box: RotatedBox;
  bounds: Rect;
}

export type MatchSortKey = "index" | "score" | "x" | "y" | "angle" | "scale" | "support";

export interface MatchSort {
  key: MatchSortKey;
  descending: boolean;
}

/** Best first: the order a search reports its matches in, and the question a list answers. */
export const DEFAULT_MATCH_SORT: MatchSort = { key: "score", descending: true };

/**
 * The model's extent in the frame a match pose maps, and the point the pose turns about.
 *
 * A pose `(x, y, angle, scale)` puts the model's origin at `(x, y)` and rotates and scales
 * about it, so `pose · p = (x, y) + s·R(angle)·(p − origin)` for a point `p` of this frame.
 */
export interface ModelFrame {
  rect: Rect;
  origin: [number, number];
}

/**
 * The extent to draw a match with.
 *
 * The model's own points, in the frame a pose consumes, when the shell can read them (the
 * desktop): their bounds are the part, exactly, at any reference angle. Otherwise the
 * rectangle the model was taught from, which is the same frame because the browser build
 * teaches without a reference angle.
 */
export function modelFrame(model: ModelOut | null, geometry: ModelGeometryOut | null): ModelFrame | null {
  if (geometry !== null && geometry.points.length >= 4) {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i + 3 < geometry.points.length; i += 4) {
      const x = geometry.points[i]!;
      const y = geometry.points[i + 1]!;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
    return { rect: { x: minX, y: minY, width: maxX - minX, height: maxY - minY }, origin: geometry.origin };
  }
  if (model === null) return null;
  const [x, y, width, height] = model.roi;
  return { rect: { x, y, width, height }, origin: model.origin };
}

/** Where a pose puts a model frame's rectangle. */
export function matchBox(match: MatchOut, frame: ModelFrame | null): RotatedBox {
  if (frame === null) return { cx: match.x, cy: match.y, width: 0, height: 0, angle: match.angle };
  const { rect, origin } = frame;
  const ux = rect.x + rect.width / 2 - origin[0];
  const uy = rect.y + rect.height / 2 - origin[1];
  const c = Math.cos(match.angle) * match.scale;
  const s = Math.sin(match.angle) * match.scale;
  return {
    cx: match.x + c * ux - s * uy,
    cy: match.y + s * ux + c * uy,
    width: rect.width * match.scale,
    height: rect.height * match.scale,
    angle: match.angle,
  };
}

/** Per-match facts, in the search's own order. */
export function describeMatches(matches: MatchOut[], frame: ModelFrame | null): MatchStat[] {
  return matches.map((match, index) => {
    const box = matchBox(match, frame);
    return {
      index,
      score: match.score,
      x: match.x,
      y: match.y,
      angle: (match.angle * 180) / Math.PI,
      scale: match.scale,
      support: match.support,
      box,
      bounds: boxBounds(box),
    };
  });
}

/** A total order, ties broken on the index, so `↑`/`↓` cannot jump about between renders. */
export function sortMatches(stats: MatchStat[], sort: MatchSort): MatchStat[] {
  const sign = sort.descending ? -1 : 1;
  return [...stats].sort((a, b) => sign * (a[sort.key] - b[sort.key]) || a.index - b.index);
}

/**
 * The sort a header click asks for: the same column flips direction; a new one starts
 * where reading it starts, quantities that mean "better" (score, support, scale) highest
 * first and positions lowest first.
 */
export function nextSort(current: MatchSort, key: MatchSortKey): MatchSort {
  if (current.key === key) return { key, descending: !current.descending };
  return { key, descending: key === "score" || key === "support" || key === "scale" };
}

/** The match under a canvas point, or `null`. `tolerance` is in image pixels. */
export function pickMatch(stats: readonly MatchStat[], p: Point, tolerance = 0): number | null {
  return pickBox(stats, (stat) => stat.box, p, tolerance)?.index ?? null;
}

/** The overlay id of match `index`: what links its primitives to its row. */
export function matchId(index: number): string {
  return `match-${index}`;
}

/** The match an overlay id names, or `null` for anything else. */
export function matchIndexOf(id: string | null): number | null {
  if (id === null || !id.startsWith("match-")) return null;
  const index = Number(id.slice("match-".length));
  return Number.isInteger(index) && index >= 0 ? index : null;
}

/** The overlay grammar's state for one match: selection outranks hover. */
export function matchState(index: number, selected: number | null, hovered: number | null): OverlayState {
  if (index === selected) return "selected";
  if (index === hovered) return "hover";
  return "default";
}

/** What a batch run said about one frame: `null` when the batch did not cover it. */
export type FrameVerdict = "found" | "not-found" | "error";

export function frameVerdict(item: BatchFindItem | undefined): FrameVerdict | null {
  if (item === undefined) return null;
  if (item.error !== null) return "error";
  return item.matches.length > 0 ? "found" : "not-found";
}

/** How many of a batch's frames had at least one match. */
export function framesFound(items: Iterable<BatchFindItem>): { found: number; total: number } {
  let found = 0;
  let total = 0;
  for (const item of items) {
    total += 1;
    if (frameVerdict(item) === "found") found += 1;
  }
  return { found, total };
}
