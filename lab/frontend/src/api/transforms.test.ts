import { describe, expect, it } from "vitest";

import { formatMeasurement, rectToRoi } from "./transforms";

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

describe("formatMeasurement", () => {
  it("formats the px value in px mode", () => {
    expect(formatMeasurement("px", 39.988, 40.5, 2)).toBe("39.99 px");
  });

  it("formats the mm value in mm mode", () => {
    expect(formatMeasurement("mm", 39.988, 40.512, 2)).toBe("40.51 mm");
  });

  it("shows an em-dash when the requested unit's value is missing", () => {
    expect(formatMeasurement("mm", 39.988, null)).toBe("—");
    expect(formatMeasurement("mm", 39.988, undefined)).toBe("—");
    expect(formatMeasurement("px", null, 40.5)).toBe("—");
  });
});
