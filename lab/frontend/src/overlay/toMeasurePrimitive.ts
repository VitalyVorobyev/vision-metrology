/**
 * Backend overlay primitives → `MeasurePrimitive`.
 *
 * `OverlayPrimitiveOut` mirrors lab-ui's `MeasurePrimitive` union exactly —
 * same field names, same units, angles in radians — so the geometry never has
 * to be recomputed on this side (see `lab/contract/openapi.json`'s own note on
 * that schema). The only difference is that JSON round-tripping turns absent
 * fields into `null`, which the lab-ui union does not carry, and that the
 * generated type widens `kind` across the whole union instead of discriminating
 * on it.
 */

import type { MeasurePrimitive } from "@vitavision/lab-ui";

import type { OverlayPrimitiveOut } from "../api/backend";

export function toMeasurePrimitive(p: OverlayPrimitiveOut): MeasurePrimitive {
  return Object.fromEntries(
    Object.entries(p)
      .filter(([, v]) => v !== null)
      .map(([k, v]) => [k, v ?? undefined]),
  ) as unknown as MeasurePrimitive;
}

export function toMeasurePrimitives(ps: OverlayPrimitiveOut[]): MeasurePrimitive[] {
  return ps.map(toMeasurePrimitive);
}
