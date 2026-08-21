import { describe, expect, it } from "vitest";

import {
  applyKeep,
  applySelect,
  boundsOf,
  boundsOfPoints,
  contoursInBox,
  describeContours,
  filterContours,
  invertKeep,
  pointsIn,
  sortContours,
  stepThrough,
} from "./contourSelection";
import type { ContourOut } from "../api/backend";

function contour(id: number, points: number[], extra: Partial<ContourOut> = {}): ContourOut {
  return {
    id,
    points,
    closed: false,
    length: points.length / 2,
    mean_strength: 0.5,
    ...extra,
  };
}

const CONTOURS: ContourOut[] = [
  contour(0, [10, 10, 20, 10, 20, 20], { length: 30, mean_strength: 0.9, closed: true }),
  contour(1, [100, 100, 400, 100], { length: 300, mean_strength: 0.2 }),
  contour(2, [50, 50, 55, 55, 60, 60, 65, 65], { length: 21, mean_strength: 0.6 }),
];

describe("describeContours", () => {
  it("derives the facts the list shows, once", () => {
    const stats = describeContours(CONTOURS);
    expect(stats.map((s) => s.points)).toEqual([3, 2, 4]);
    expect(stats.map((s) => s.id)).toEqual([0, 1, 2]);
    expect(stats[0]!.closed).toBe(true);
    expect(stats[0]!.bounds).toEqual({ x: 10, y: 10, width: 10, height: 10 });
  });

  it("gives an empty contour a degenerate box rather than NaN", () => {
    expect(boundsOfPoints([])).toEqual({ x: 0, y: 0, width: 0, height: 0 });
  });
});

describe("sortContours", () => {
  const stats = describeContours(CONTOURS);

  it("descends quantities and ascends the index", () => {
    expect(sortContours(stats, "length").map((s) => s.id)).toEqual([1, 0, 2]);
    expect(sortContours(stats, "strength").map((s) => s.id)).toEqual([0, 2, 1]);
    expect(sortContours(stats, "points").map((s) => s.id)).toEqual([2, 0, 1]);
    expect(sortContours(stats, "id").map((s) => s.id)).toEqual([0, 1, 2]);
  });

  it("breaks ties on id, so keyboard iteration cannot jump about", () => {
    const tied = describeContours([
      contour(7, [0, 0, 1, 1], { length: 5 }),
      contour(3, [0, 0, 1, 1], { length: 5 }),
    ]);
    expect(sortContours(tied, "length").map((s) => s.id)).toEqual([3, 7]);
  });

  it("does not mutate its input", () => {
    const before = stats.map((s) => s.id);
    sortContours(stats, "length");
    expect(stats.map((s) => s.id)).toEqual(before);
  });
});

describe("filterContours", () => {
  const stats = describeContours(CONTOURS);

  it("splits on keep state", () => {
    const kept = new Set([0, 2]);
    expect(filterContours(stats, "all", kept)).toHaveLength(3);
    expect(filterContours(stats, "kept", kept).map((s) => s.id)).toEqual([0, 2]);
    expect(filterContours(stats, "dropped", kept).map((s) => s.id)).toEqual([1]);
  });
});

describe("applySelect", () => {
  const order = [1, 0, 2];

  it("replaces, adds and toggles", () => {
    expect([...applySelect(new Set(), null, [0], "replace", order).selected]).toEqual([0]);
    expect([...applySelect(new Set([0]), 0, [2], "add", order).selected]).toEqual([0, 2]);
    expect([...applySelect(new Set([0, 2]), 2, [2], "toggle", order).selected]).toEqual([0]);
    expect([...applySelect(new Set([0]), 0, [2], "toggle", order).selected]).toEqual([0, 2]);
  });

  it("ranges over the list's own order, not over ids", () => {
    // Anchored on 1 (first in `order`), extending to 2 (last) takes everything between.
    const { selected } = applySelect(new Set([1]), 1, [2], "range", order);
    expect([...selected].sort()).toEqual([0, 1, 2]);
  });

  it("keeps the anchor still, so a second shift-click re-ranges from the same origin", () => {
    const first = applySelect(new Set([1]), 1, [0], "range", order);
    expect([...first.selected].sort()).toEqual([0, 1]);
    expect(first.anchor).toBe(1);

    const second = applySelect(first.selected, first.anchor, [2], "range", order);
    expect([...second.selected].sort()).toEqual([0, 1, 2]);
    expect(second.anchor).toBe(1);
  });

  it("ranges from the target itself when there is no anchor yet", () => {
    const { selected, anchor } = applySelect(new Set(), null, [0], "range", order);
    expect([...selected]).toEqual([0]);
    expect(anchor).toBe(0);
  });

  it("falls back to a plain selection when the order no longer holds the ids", () => {
    const { selected } = applySelect(new Set([9]), 9, [2], "range", order);
    expect([...selected]).toEqual([2]);
  });
});

describe("keep operations", () => {
  const stats = describeContours(CONTOURS);

  it("keeps and drops without touching the rest", () => {
    expect([...applyKeep(new Set([0, 1, 2]), [1], false)].sort()).toEqual([0, 2]);
    expect([...applyKeep(new Set([0]), [1, 2], true)].sort()).toEqual([0, 1, 2]);
  });

  it("inverts against the full inventory", () => {
    expect([...invertKeep(new Set([0, 2]), stats)]).toEqual([1]);
    expect([...invertKeep(new Set(), stats)].sort()).toEqual([0, 1, 2]);
  });

  it("counts the points a keep set carries", () => {
    expect(pointsIn(stats, new Set([0, 1, 2]))).toBe(9);
    expect(pointsIn(stats, new Set([1]))).toBe(2);
  });
});

describe("contoursInBox", () => {
  it("catches a contour by any vertex, not only its midpoint", () => {
    // Contour 1 runs from x=100 to x=400 at y=100; the band covers only its far end.
    const band = { x: 380, y: 90, width: 40, height: 20 };
    expect(contoursInBox(CONTOURS, band)).toEqual([1]);
  });

  it("returns nothing for a band over empty space", () => {
    expect(contoursInBox(CONTOURS, { x: 800, y: 800, width: 10, height: 10 })).toEqual([]);
  });
});

describe("boundsOf", () => {
  it("is the union, and null for nothing", () => {
    const stats = describeContours(CONTOURS);
    expect(boundsOf(stats)).toEqual({ x: 10, y: 10, width: 390, height: 90 });
    expect(boundsOf([])).toBeNull();
  });
});

describe("stepThrough", () => {
  const order = [1, 0, 2];

  it("walks the list order and wraps", () => {
    expect(stepThrough(order, null, 1)).toBe(1);
    expect(stepThrough(order, null, -1)).toBe(2);
    expect(stepThrough(order, 1, 1)).toBe(0);
    expect(stepThrough(order, 2, 1)).toBe(1);
    expect(stepThrough(order, 1, -1)).toBe(2);
  });

  it("restarts rather than dead-ending when the current id left the list", () => {
    expect(stepThrough(order, 99, 1)).toBe(1);
    expect(stepThrough([], 1, 1)).toBeNull();
  });
});
