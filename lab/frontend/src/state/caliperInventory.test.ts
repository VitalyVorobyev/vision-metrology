import { describe, expect, it } from "vitest";

import type { CaliperResultOut, MeasureObjectResultOut, OverlayPrimitiveOut } from "../api/backend";
import { toMeasurePrimitives } from "../overlay/toMeasurePrimitive";
import {
  caliperId,
  caliperProfile,
  describeCalipers,
  filterCalipers,
  filterCounts,
  pickCaliper,
  reasonText,
  withCaliperStates,
} from "./caliperInventory";

/** A caliper 20 px long, sampled at five points from −10 to +10 about its centre. */
function caliper(index: number, hit: boolean, extra: Partial<CaliperResultOut> = {}): CaliperResultOut {
  return {
    index,
    status: hit ? "hit" : "rejected",
    reason: hit ? null : "no_edge",
    residual: hit ? 0.02 * (index + 1) : null,
    profile: {
      values: [200, 200, 110, 20, 20],
      step_px: 5,
      start_px: -10,
      end_px: 10,
      edges: hit ? [{ pos_px: 0.4, polarity: "falling", amplitude: 64 }] : [],
    },
    ...extra,
  };
}

/** A box `width` along its axis, at `(cx, cy)`, turned `angle`. */
function box(id: string, cx: number, cy: number, angle = 0): OverlayPrimitiveOut {
  return { kind: "caliper", id, cx, cy, width: 20, height: 6, angle, tone: "signal" };
}

/** Two objects: a circle with three calipers (one rejected), a line with one. */
const OBJECTS: MeasureObjectResultOut[] = [
  {
    kind: "circle",
    calipers: [caliper(0, true), caliper(1, false), caliper(2, true)],
    overlay: [
      box("caliper-0-0", 100, 50),
      { kind: "point", id: "caliper-0-0", x: 100.4, y: 50, cross: true },
      box("caliper-0-1", 150, 100, Math.PI / 2),
      box("caliper-0-2", 100, 150),
      { kind: "point", id: "caliper-0-2", x: 100.4, y: 150, cross: true },
      { kind: "circle", cx: 100, cy: 100, r: 50 },
    ],
  },
  {
    kind: "line",
    calipers: [caliper(0, true)],
    overlay: [box("caliper-1-0", 300, 300)],
  },
];

describe("describeCalipers", () => {
  const rows = describeCalipers(OBJECTS);

  it("lists every caliper of every object, object by object", () => {
    expect(rows.map((r) => r.id)).toEqual(["caliper-0-0", "caliper-0-1", "caliper-0-2", "caliper-1-0"]);
    expect(rows.map((r) => r.object)).toEqual([0, 0, 0, 1]);
  });

  it("reads each hit's edge, residual and amplitude, and a rejection's reason", () => {
    expect(rows[0]).toMatchObject({ status: "hit", edge: 0.4, residual: 0.02, amplitude: 64, reason: null });
    expect(rows[1]).toMatchObject({ status: "rejected", edge: null, residual: null, amplitude: null, reason: "no_edge" });
  });

  it("takes each caliper's box from the overlay by its id", () => {
    expect(rows[1]!.box).toMatchObject({ cx: 150, cy: 100, width: 20, height: 6 });
    // Turned a quarter, the 20 × 6 box stands 6 wide and 20 tall.
    expect(rows[1]!.bounds!.width).toBeCloseTo(6, 5);
    expect(rows[1]!.bounds!.height).toBeCloseTo(20, 5);
  });

  it("matches the backend's ids", () => {
    expect(caliperId(1, 0)).toBe("caliper-1-0");
  });
});

describe("filters", () => {
  const rows = describeCalipers(OBJECTS);

  it("splits on the verdict and counts each side", () => {
    expect(filterCalipers(rows, "hits").map((r) => r.id)).toEqual(["caliper-0-0", "caliper-0-2", "caliper-1-0"]);
    expect(filterCalipers(rows, "rejected").map((r) => r.id)).toEqual(["caliper-0-1"]);
    expect(filterCalipers(rows, "all")).toHaveLength(4);
    expect(filterCounts(rows)).toEqual({ all: 4, hits: 3, rejected: 1 });
  });
});

describe("pickCaliper", () => {
  const rows = describeCalipers(OBJECTS);

  it("finds the caliper whose box holds the point, in the box's own frame", () => {
    expect(pickCaliper(rows, { x: 108, y: 52 })).toBe("caliper-0-0");
    // The quarter-turned box reaches 10 px down and only 3 px across.
    expect(pickCaliper(rows, { x: 150, y: 108 })).toBe("caliper-0-1");
    expect(pickCaliper(rows, { x: 156, y: 100 })).toBeNull();
    expect(pickCaliper(rows, { x: 156, y: 100 }, 4)).toBe("caliper-0-1");
  });

  it("only picks among the rows it is given", () => {
    expect(pickCaliper(filterCalipers(rows, "hits"), { x: 150, y: 100 })).toBeNull();
  });
});

describe("withCaliperStates", () => {
  const overlay = toMeasurePrimitives(OBJECTS[0]!.overlay!);

  it("draws a caliper's box and edge mark together, in its state", () => {
    const out = withCaliperStates(overlay, "caliper-0-0", "caliper-0-2", new Set(["caliper-0-0", "caliper-0-2"]));
    const states = out.map((p) => [p.kind, p.id ?? null, p.state ?? "default"]);
    expect(states).toEqual([
      ["caliper", "caliper-0-0", "selected"],
      ["point", "caliper-0-0", "selected"],
      // Hidden by the filter: still there, faded.
      ["caliper", "caliper-0-1", "dimmed"],
      ["caliper", "caliper-0-2", "hover"],
      ["point", "caliper-0-2", "hover"],
      // The fit belongs to no caliper.
      ["circle", null, "default"],
    ]);
  });
});

describe("caliperProfile", () => {
  it("lays the samples out along the caliper, centred on the nominal edge", () => {
    const { series, edges, markers, xDomain } = caliperProfile(caliper(3, true));
    expect(series[0]!.points.map((p) => p.x)).toEqual([-10, -5, 0, 5, 10]);
    expect(xDomain).toEqual([-10, 10]);
    // So the edge rule lands on the step in the trace, between samples 2 and 3.
    expect(edges).toEqual([{ position: 0.4, label: "falling", tone: "signal" }]);
    expect(markers.map((m) => [m.position, m.tone])).toEqual([
      [0, "muted"],
      [0.4, "signal"],
    ]);
  });

  it("falls back to the step from zero without the span", () => {
    const old = caliper(0, false);
    old.profile = { ...old.profile, start_px: null, end_px: null };
    const { series, markers, xDomain } = caliperProfile(old);
    expect(series[0]!.points.map((p) => p.x)).toEqual([0, 5, 10, 15, 20]);
    expect(markers).toEqual([]);
    expect(xDomain).toBeUndefined();
  });

  it("says a reason as a reader would", () => {
    expect(reasonText("too_oblique")).toBe("too oblique");
    expect(reasonText(null)).toBe("");
  });
});
