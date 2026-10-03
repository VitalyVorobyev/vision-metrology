/**
 * Rotated boxes on the image: what a found match or a caliper occupies, and which one a
 * pointer is over.
 *
 * `MeasureOverlay` draws results without taking pointer events, so an inventory that links
 * its rows to the canvas resolves the pointer itself. Both inventories that need it (Find's
 * matches, Measure's calipers) describe their items as a rotated box, the same shape as
 * stage2d's caliper primitive: a centre, a size along and across its own axis, and the
 * axis's angle.
 */

import { caliperCorners, type Point, type Rect } from "@vitavision/stage2d";

/** A rectangle `width` (along its own axis) by `height`, centred at `(cx, cy)`, turned by `angle` radians. */
export interface RotatedBox {
  cx: number;
  cy: number;
  width: number;
  height: number;
  angle: number;
}

/** Whether `p` lies inside `box`, or within `tolerance` image pixels of it. */
export function boxContains(box: RotatedBox, p: Point, tolerance = 0): boolean {
  const dx = p.x - box.cx;
  const dy = p.y - box.cy;
  const c = Math.cos(box.angle);
  const s = Math.sin(box.angle);
  // The point in the box's own frame: `u` along its axis, `v` across it.
  const u = dx * c + dy * s;
  const v = -dx * s + dy * c;
  return Math.abs(u) <= box.width / 2 + tolerance && Math.abs(v) <= box.height / 2 + tolerance;
}

/** The axis-aligned box around a rotated one: what "frame this" hands the stage. */
export function boxBounds(box: RotatedBox): Rect {
  const corners = caliperCorners(box.cx, box.cy, box.width, box.height, box.angle);
  const xs = corners.map((corner) => corner.x);
  const ys = corners.map((corner) => corner.y);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y };
}

/**
 * The item under `p`: of the boxes that contain it, the one whose centre is nearest.
 *
 * Nearest-centre rather than first-found, because boxes overlap: two matches of a part
 * that sit side by side share a margin, and the calipers of a small circle overlap near
 * its centre. The pointer then means the one it is closer to the middle of.
 */
export function pickBox<T>(
  items: readonly T[],
  boxOf: (item: T) => RotatedBox,
  p: Point,
  tolerance = 0,
): T | null {
  let best: T | null = null;
  let bestDistance = Infinity;
  for (const item of items) {
    const box = boxOf(item);
    if (!boxContains(box, p, tolerance)) continue;
    const distance = Math.hypot(p.x - box.cx, p.y - box.cy);
    if (distance < bestDistance) {
      best = item;
      bestDistance = distance;
    }
  }
  return best;
}
