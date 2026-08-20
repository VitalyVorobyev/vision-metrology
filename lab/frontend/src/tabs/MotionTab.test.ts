import { describe, expect, it } from "vitest";

import type { DisplacementResponse } from "../api/backend";
import { toggleSequence, trajectorySeries } from "./MotionTab";

describe("toggleSequence", () => {
  it("appends an id not already in the sequence", () => {
    expect(toggleSequence([], "img-1")).toEqual(["img-1"]);
    expect(toggleSequence(["img-1"], "img-2")).toEqual(["img-1", "img-2"]);
  });

  it("removes an id already in the sequence, preserving the rest's order", () => {
    expect(toggleSequence(["img-1", "img-2", "img-3"], "img-2")).toEqual(["img-1", "img-3"]);
  });

  it("clicking the same id twice is a no-op overall", () => {
    let seq = toggleSequence([], "img-1");
    seq = toggleSequence(seq, "img-1");
    expect(seq).toEqual([]);
  });
});

describe("trajectorySeries", () => {
  it("maps cumulative_x/y into two indexed series", () => {
    const resp: DisplacementResponse = {
      pairs: [
        { from_image_id: "img-1", to_image_id: "img-2", dx: 2.0, dy: -1.0, score: 0.9 },
        { from_image_id: "img-2", to_image_id: "img-3", dx: 3.0, dy: -0.5, score: 0.8 },
      ],
      cumulative_x: [0.0, 2.0, 5.0],
      cumulative_y: [0.0, -1.0, -1.5],
    };

    const series = trajectorySeries(resp);
    expect(series).toHaveLength(2);
    expect(series[0]?.name).toBe("cumulative dx");
    expect(series[0]?.points).toEqual([
      { x: 0, y: 0.0 },
      { x: 1, y: 2.0 },
      { x: 2, y: 5.0 },
    ]);
    expect(series[1]?.name).toBe("cumulative dy");
    expect(series[1]?.points).toEqual([
      { x: 0, y: 0.0 },
      { x: 1, y: -1.0 },
      { x: 2, y: -1.5 },
    ]);
  });

  it("an empty trajectory (no pairs) still yields the frame-0 zero point", () => {
    const resp: DisplacementResponse = { pairs: [], cumulative_x: [0.0], cumulative_y: [0.0] };
    const series = trajectorySeries(resp);
    expect(series[0]?.points).toEqual([{ x: 0, y: 0.0 }]);
  });
});
