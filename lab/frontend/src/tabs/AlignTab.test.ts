import { describe, expect, it } from "vitest";

import type { Roi } from "../api/backend";
import { cropOutputSize, validityTone } from "./AlignTab";

describe("validityTone", () => {
  it("is normal above 90% valid", () => {
    expect(validityTone(0.95)).toBe("normal");
    expect(validityTone(1.0)).toBe("normal");
  });

  it("is a warning between 50% and 90%", () => {
    expect(validityTone(0.9)).toBe("warning");
    expect(validityTone(0.51)).toBe("warning");
  });

  it("is a defect at or below 50% valid", () => {
    expect(validityTone(0.5)).toBe("defect");
    expect(validityTone(0.0)).toBe("defect");
  });
});

describe("cropOutputSize", () => {
  it("scales the rect's width/height by px_per_unit, rounded", () => {
    const rect: Roi = [10, 20, 30, 15];
    expect(cropOutputSize(rect, 2.0)).toEqual([60, 30]);
  });

  it("depends only on the rect and px_per_unit, not on x/y", () => {
    const rect: Roi = [500, -300, 84, 84];
    expect(cropOutputSize(rect, 1.0)).toEqual([84, 84]);
  });

  it("floors at one pixel per side for a degenerate rect", () => {
    const rect: Roi = [0, 0, 0.1, 0.1];
    expect(cropOutputSize(rect, 0.1)).toEqual([1, 1]);
  });
});
