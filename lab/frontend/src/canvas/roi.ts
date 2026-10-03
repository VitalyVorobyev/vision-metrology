/**
 * The backend's `Roi` tuple and stage2d's `Rect`, which are the same box.
 *
 * The region is edited on the canvas by stage2d's `RectRoiEditor` and typed into the panel as
 * four numbers; both go through the same arithmetic (`clampRect`, `sameRect`), so a typed box
 * and a dragged one obey one set of limits. Everything is in **source-image pixels**: a start
 * pixel and a size, held inside `0..width × 0..height`.
 */

import { clampRect, sameRect, type Rect } from "@vitavision/stage2d";

import type { Roi } from "../api/backend";

/** Below this a box is a mis-click rather than a region. */
export const MIN_ROI = 4;

export function roiToRect(roi: Roi): Rect {
  return { x: roi[0], y: roi[1], width: roi[2], height: roi[3] };
}

export function rectToRoi(rect: Rect): Roi {
  return [rect.x, rect.y, rect.width, rect.height];
}

/** A box held inside the image, and no smaller than `MIN_ROI` on either side. */
export function clampRoi(roi: Roi, image: { width: number; height: number }): Roi {
  return rectToRoi(clampRect(roiToRect(roi), { x: 0, y: 0, ...image }, MIN_ROI));
}

/** Whether two ROIs are the same box — how a preview knows its extraction is still current. */
export function sameRoi(a: Roi | null, b: Roi | null): boolean {
  return sameRect(a && roiToRect(a), b && roiToRect(b));
}
