/**
 * The caliper inventory's arithmetic: one row per caliper of every measured object, a
 * filter, the box each caliper occupies, and the overlay state each is drawn in.
 *
 * The measure response already carries everything per caliper: the verdict (hit, or why
 * not), the edge with its amplitude, its residual against the object's fit, its profile, and
 * an overlay whose box and edge mark carry the caliper's id (`caliper-<object>-<index>`).
 * This module only joins them, so the list and the canvas name the same caliper.
 */

import type { EdgeMark, Marker, ProfileSeries } from "@vitavision/charts";
import type { MeasurePrimitive, OverlayState, Point, Rect } from "@vitavision/stage2d";
import type { MeasureTone } from "@vitavision/ui";

import type { CaliperResultOut, MeasureObjectResultOut } from "../api/backend";
import { boxBounds, pickBox, type RotatedBox } from "./rotatedBox";

/** One caliper of one object, with what the list shows. */
export interface CaliperRow {
  /** `caliper-<object>-<index>`: the id its box and edge mark carry. */
  id: string;
  object: number;
  index: number;
  status: "hit" | "rejected";
  /** Why it was rejected (`no_edge`, `too_oblique`, …); `null` for a hit. */
  reason: string | null;
  /** Edge position along the caliper, px, signed from its centre (the nominal edge). */
  edge: number | null;
  /** Signed distance of the edge from the object's fit, px. */
  residual: number | null;
  /** The edge's local contrast. */
  amplitude: number | null;
  /** Where the caliper sat, from its overlay box; `null` if the overlay has none. */
  box: RotatedBox | null;
  bounds: Rect | null;
  caliper: CaliperResultOut;
}

export type CaliperFilter = "all" | "hits" | "rejected";

/** The id a caliper's overlay primitives carry (the backend's `caliper_id`). */
export function caliperId(object: number, index: number): string {
  return `caliper-${object}-${index}`;
}

/** Every caliper of every object, object by object, in caliper order. */
export function describeCalipers(objects: readonly MeasureObjectResultOut[]): CaliperRow[] {
  const rows: CaliperRow[] = [];
  objects.forEach((object, o) => {
    const boxes = new Map<string, RotatedBox>();
    for (const p of object.overlay ?? []) {
      if (p.kind !== "caliper" || !p.id) continue;
      boxes.set(p.id, {
        cx: p.cx ?? 0,
        cy: p.cy ?? 0,
        width: p.width ?? 0,
        height: p.height ?? 0,
        angle: p.angle ?? 0,
      });
    }
    for (const caliper of object.calipers ?? []) {
      const id = caliperId(o, caliper.index);
      const box = boxes.get(id) ?? null;
      const edge = caliper.profile.edges[0];
      rows.push({
        id,
        object: o,
        index: caliper.index,
        status: caliper.status,
        reason: caliper.reason ?? null,
        edge: edge?.pos_px ?? null,
        residual: caliper.residual ?? null,
        amplitude: edge?.amplitude ?? null,
        box,
        bounds: box === null ? null : boxBounds(box),
        caliper,
      });
    }
  });
  return rows;
}

export function filterCalipers(rows: CaliperRow[], filter: CaliperFilter): CaliperRow[] {
  if (filter === "all") return rows;
  const wanted = filter === "hits" ? "hit" : "rejected";
  return rows.filter((row) => row.status === wanted);
}

/** How many calipers each filter would show, for its label. */
export function filterCounts(rows: readonly CaliperRow[]): Record<CaliperFilter, number> {
  const hits = rows.filter((row) => row.status === "hit").length;
  return { all: rows.length, hits, rejected: rows.length - hits };
}

/** The caliper under a canvas point, among `rows`, or `null`. `tolerance` is in image pixels. */
export function pickCaliper(rows: readonly CaliperRow[], p: Point, tolerance = 0): string | null {
  const withBox = rows.filter((row) => row.box !== null);
  return pickBox(withBox, (row) => row.box!, p, tolerance)?.id ?? null;
}

/**
 * The overlay with each caliper's primitives in its state: selected, hovered, dimmed when
 * the filter hides its row, otherwise as the backend drew it. The fit and anything else
 * without an id is left alone.
 */
export function withCaliperStates(
  overlay: readonly MeasurePrimitive[],
  selected: string | null,
  hovered: string | null,
  shown: ReadonlySet<string>,
): MeasurePrimitive[] {
  return overlay.map((p) => {
    if (p.id === undefined) return p;
    const state: OverlayState =
      p.id === selected ? "selected" : p.id === hovered ? "hover" : shown.has(p.id) ? "default" : "dimmed";
    return state === "default" ? p : { ...p, state };
  });
}

/** `no_edge` → "no edge": the reason as a reader says it. */
export function reasonText(reason: string | null): string {
  return reason === null ? "" : reason.replaceAll("_", " ");
}

/**
 * One caliper's profile as `LineProfile` draws it.
 *
 * `x` is the position along the caliper in the frame its edge's `pos_px` is in (signed from
 * the caliper's centre, the nominal edge), so the edge rule lands on the step in the trace.
 * The backend reports where the first and last samples sit; a response without them falls
 * back to `step_px` from zero, the older reading.
 *
 * The edge is drawn twice on purpose: as a rule through the plot (`edges`), and as a tick on
 * the axis (`markers`) beside the nominal position, so how far the edge sits from where it
 * was expected reads off the axis.
 */
export function caliperProfile(caliper: CaliperResultOut): {
  series: ProfileSeries[];
  edges: EdgeMark[];
  markers: Marker[];
  xDomain: [number, number] | undefined;
} {
  const { values, step_px, edges, start_px, end_px } = caliper.profile;
  const n = values.length;
  const spanned = start_px !== null && start_px !== undefined && end_px !== null && end_px !== undefined;
  const at = (i: number) =>
    spanned ? start_px + ((end_px - start_px) * i) / Math.max(n - 1, 1) : i * step_px;
  const tone: MeasureTone = caliper.status === "hit" ? "signal" : "defect";
  const markers: Marker[] = spanned ? [{ position: 0, tone: "muted", label: "nominal edge" }] : [];
  for (const edge of edges) {
    markers.push({ position: edge.pos_px, tone, label: `edge at ${edge.pos_px.toFixed(2)} px` });
  }
  return {
    series: [{ name: `caliper ${caliper.index}`, points: values.map((y, i) => ({ x: at(i), y })) }],
    edges: edges.map((edge) => ({ position: edge.pos_px, label: edge.polarity, tone })),
    markers,
    xDomain: spanned && n > 1 ? [start_px, end_px] : undefined,
  };
}
