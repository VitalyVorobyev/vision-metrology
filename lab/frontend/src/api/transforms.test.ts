import { describe, expect, it } from "vitest";

import { caliperToProfile, rectToRoi } from "./transforms";
import type { CaliperResultOut } from "./types";

describe("rectToRoi", () => {
  it("normalizes a rectangle dragged in the positive direction", () => {
    expect(rectToRoi(10, 20, 50, 80)).toEqual([10, 20, 40, 60]);
  });

  it("normalizes a rectangle dragged backwards (bottom-right to top-left)", () => {
    expect(rectToRoi(50, 80, 10, 20)).toEqual([10, 20, 40, 60]);
  });

  it("handles a drag with mixed direction per axis", () => {
    expect(rectToRoi(50, 20, 10, 80)).toEqual([10, 20, 40, 60]);
  });

  it("returns a zero-size roi for a degenerate drag", () => {
    expect(rectToRoi(5, 5, 5, 5)).toEqual([5, 5, 0, 0]);
  });
});

describe("caliperToProfile", () => {
  const base: CaliperResultOut = {
    index: 3,
    status: "hit",
    profile: {
      values: [10, 20, 200, 210],
      step_px: 0.5,
      edges: [{ pos_px: 1.0, polarity: "dark_to_bright" }],
    },
  };

  it("lays profile values out at arc-length = index * step_px", () => {
    const { series } = caliperToProfile(base);
    expect(series).toHaveLength(1);
    expect(series[0]?.name).toBe("caliper 3");
    expect(series[0]?.points).toEqual([
      { x: 0, y: 10 },
      { x: 0.5, y: 20 },
      { x: 1.0, y: 200 },
      { x: 1.5, y: 210 },
    ]);
  });

  it("marks a hit caliper's edges with the signal tone", () => {
    const { edges } = caliperToProfile(base);
    expect(edges).toEqual([{ position: 1.0, label: "dark_to_bright", tone: "signal" }]);
  });

  it("marks a rejected caliper's (empty) edges with the defect tone and empty edges list", () => {
    const rejected: CaliperResultOut = {
      ...base,
      status: "rejected",
      reason: "no_edge",
      profile: { ...base.profile, edges: [] },
    };
    const { edges } = caliperToProfile(rejected);
    expect(edges).toEqual([]);
  });
});
