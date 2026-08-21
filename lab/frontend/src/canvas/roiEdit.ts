/**
 * Editing a box that already exists.
 *
 * The lab could only ever draw a *new* ROI: every press started again from a corner, so
 * nudging a box that was almost right meant redrawing it, and a box drawn at fit could not
 * be refined by zooming in first. This is the arithmetic that turns it into an object —
 * eight handles and an interior — kept pure so the corner cases that actually bite (a drag
 * that inverts the box through itself, a handle pulled past the image edge, a box collapsed
 * to nothing) are testable without a pointer.
 *
 * Everything here is in **source-image pixels**, the coordinate the backend's `Roi` is in.
 */

import type { Roi } from "../api/backend";

export type RoiHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
/** A handle, or the interior — what a press on the ROI can grab. */
export type RoiGrab = RoiHandle | "move";

export interface Point {
  x: number;
  y: number;
}

export interface Size {
  width: number;
  height: number;
}

/** Below this a box is a mis-click rather than a region. */
export const MIN_ROI = 4;

export const ROI_HANDLES: RoiHandle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

/** The CSS cursor for each handle, so the affordance is legible before the press. */
export const HANDLE_CURSOR: Record<RoiGrab, string> = {
  nw: "nwse-resize",
  n: "ns-resize",
  ne: "nesw-resize",
  e: "ew-resize",
  se: "nwse-resize",
  s: "ns-resize",
  sw: "nesw-resize",
  w: "ew-resize",
  move: "move",
};

/** Where a handle sits, as the centre of its grab square. */
export function handlePoint(roi: Roi, handle: RoiHandle): Point {
  const [x, y, w, h] = roi;
  const mx = x + w / 2;
  const my = y + h / 2;
  switch (handle) {
    case "nw":
      return { x, y };
    case "n":
      return { x: mx, y };
    case "ne":
      return { x: x + w, y };
    case "e":
      return { x: x + w, y: my };
    case "se":
      return { x: x + w, y: y + h };
    case "s":
      return { x: mx, y: y + h };
    case "sw":
      return { x, y: y + h };
    case "w":
      return { x, y: my };
  }
}

/** Two dragged corners as a positive-area box — the same normalisation `rectToRoi` does. */
export function roiFromCorners(a: Point, b: Point): Roi {
  return [
    Math.min(a.x, b.x),
    Math.min(a.y, b.y),
    Math.abs(b.x - a.x),
    Math.abs(b.y - a.y),
  ];
}

/** A box held inside the image, and no smaller than `MIN_ROI` on either side. */
export function clampRoi(roi: Roi, image: Size): Roi {
  const w = Math.min(Math.max(roi[2], MIN_ROI), image.width);
  const h = Math.min(Math.max(roi[3], MIN_ROI), image.height);
  const x = Math.min(Math.max(roi[0], 0), image.width - w);
  const y = Math.min(Math.max(roi[1], 0), image.height - h);
  return [x, y, w, h];
}

/**
 * Drag a handle to `to`.
 *
 * The dragged edge follows the pointer and the opposite edge stays put, which is what makes
 * a handle feel attached to the corner it is drawn on. Dragging an edge through its
 * opposite flips the box rather than clamping it to zero: a clamp leaves the pointer moving
 * and the box not, which reads as the drag having died.
 */
export function resizeRoi(roi: Roi, handle: RoiHandle, to: Point, image: Size): Roi {
  const [x, y, w, h] = roi;
  let left = x;
  let top = y;
  let right = x + w;
  let bottom = y + h;

  const px = clamp(to.x, 0, image.width);
  const py = clamp(to.y, 0, image.height);

  if (handle.includes("w")) left = px;
  if (handle.includes("e")) right = px;
  if (handle.includes("n")) top = py;
  if (handle.includes("s")) bottom = py;

  return clampRoi(roiFromCorners({ x: left, y: top }, { x: right, y: bottom }), image);
}

/** Move the whole box, clipped to the image without changing its size. */
export function moveRoi(roi: Roi, dx: number, dy: number, image: Size): Roi {
  const [, , w, h] = roi;
  return [
    clamp(roi[0] + dx, 0, Math.max(0, image.width - w)),
    clamp(roi[1] + dy, 0, Math.max(0, image.height - h)),
    w,
    h,
  ];
}

/**
 * What a press at `p` grabs, given a handle radius in image pixels, or `null` for a press
 * that missed the box entirely.
 *
 * The **nearest** handle in range wins, not the first one found: on a box small enough for
 * its handles to overlap — which is exactly the box a user is trying to fix — first-match
 * hands back a neighbouring edge instead of the corner under the cursor. Handles are all
 * tested before the interior for the same reason.
 */
export function grabAt(roi: Roi, p: Point, radius: number): RoiGrab | null {
  let best: RoiHandle | null = null;
  let bestDistance = Infinity;
  for (const handle of ROI_HANDLES) {
    const centre = handlePoint(roi, handle);
    const dx = Math.abs(p.x - centre.x);
    const dy = Math.abs(p.y - centre.y);
    if (dx > radius || dy > radius) continue;
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {
      best = handle;
      bestDistance = distance;
    }
  }
  if (best !== null) return best;

  const [x, y, w, h] = roi;
  if (p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h) return "move";
  return null;
}

/** Whether two ROIs are the same box — how a preview knows its extraction is still current. */
export function sameRoi(a: Roi | null, b: Roi | null): boolean {
  if (a === null || b === null) return a === b;
  return a.every((value, index) => Math.abs(value - b[index]!) < 1e-6);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
