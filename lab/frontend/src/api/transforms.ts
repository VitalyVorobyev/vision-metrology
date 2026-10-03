// Pure logic pulled out of the tab components so it's unit-testable without rendering
// anything — see transforms.test.ts.

import type { Roi } from "./backend";

/** A `ZoomPanCanvas` content-space drag rectangle (screen/content pixels, top-left +
 * size, possibly dragged in any direction) into the backend's `Roi` tuple — always
 * positive width/height, corner-normalized. */
export function rectToRoi(x0: number, y0: number, x1: number, y1: number): Roi {
  const x = Math.min(x0, x1);
  const y = Math.min(y0, y1);
  const w = Math.abs(x1 - x0);
  const h = Math.abs(y1 - y0);
  return [x, y, w, h];
}

export type MeasureUnit = "px" | "mm";

/** Format a measurement value for the given unit, `"—"` when the value isn't available —
 * the `mm` branch specifically for a value that is `null`/`undefined` because no
 * calibration was selected, or because the ray for that point missed the measurement
 * plane (see `vm_lab.routers.measure._pixel_to_plane_mm`). */
export function formatMeasurement(
  unit: MeasureUnit,
  pxValue: number | null | undefined,
  mmValue: number | null | undefined,
  digits = 3,
): string {
  const value = unit === "mm" ? mmValue : pxValue;
  return value === null || value === undefined ? "—" : `${value.toFixed(digits)} ${unit}`;
}
