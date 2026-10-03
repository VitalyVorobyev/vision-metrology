import { describe, expect, it } from "vitest";

import {
  MIN_ROI,
  clampRoi,
  grabAt,
  handlePoint,
  moveRoi,
  resizeRoi,
  roiFromCorners,
  sameRoi,
} from "./roiEdit";
import type { Roi } from "../api/backend";

const IMAGE = { width: 1280, height: 1024 };
/** The ROI from the screenshot that started this. */
const ROI: Roi = [522.9, 357, 339.8, 274.5];

describe("handlePoint", () => {
  it("puts each handle on the edge it names", () => {
    expect(handlePoint(ROI, "nw")).toEqual({ x: 522.9, y: 357 });
    expect(handlePoint(ROI, "se")).toEqual({ x: 522.9 + 339.8, y: 357 + 274.5 });
    expect(handlePoint(ROI, "n")).toEqual({ x: 522.9 + 339.8 / 2, y: 357 });
    expect(handlePoint(ROI, "w")).toEqual({ x: 522.9, y: 357 + 274.5 / 2 });
  });
});

describe("resizeRoi", () => {
  it("moves the dragged edge and leaves the opposite one alone", () => {
    const next = resizeRoi(ROI, "e", { x: 900, y: 500 }, IMAGE);
    expect(next[0]).toBeCloseTo(ROI[0], 6);
    expect(next[1]).toBeCloseTo(ROI[1], 6);
    expect(next[0] + next[2]).toBeCloseTo(900, 6);
    expect(next[3]).toBeCloseTo(ROI[3], 6);
  });

  it("moves both edges of a corner", () => {
    const next = resizeRoi(ROI, "nw", { x: 400, y: 300 }, IMAGE);
    expect(next[0]).toBeCloseTo(400, 6);
    expect(next[1]).toBeCloseTo(300, 6);
    expect(next[0] + next[2]).toBeCloseTo(ROI[0] + ROI[2], 6);
    expect(next[1] + next[3]).toBeCloseTo(ROI[1] + ROI[3], 6);
  });

  it("flips through the opposite edge instead of collapsing", () => {
    const next = resizeRoi(ROI, "w", { x: 1000, y: 500 }, IMAGE);
    expect(next[0]).toBeCloseTo(ROI[0] + ROI[2], 6);
    expect(next[2]).toBeCloseTo(1000 - (ROI[0] + ROI[2]), 6);
  });

  it("keeps the box inside the image however far the pointer goes", () => {
    const next = resizeRoi(ROI, "se", { x: 99999, y: 99999 }, IMAGE);
    expect(next[0] + next[2]).toBeLessThanOrEqual(IMAGE.width);
    expect(next[1] + next[3]).toBeLessThanOrEqual(IMAGE.height);

    const back = resizeRoi(ROI, "nw", { x: -500, y: -500 }, IMAGE);
    expect(back[0]).toBeGreaterThanOrEqual(0);
    expect(back[1]).toBeGreaterThanOrEqual(0);
  });

  it("never produces a box too small to grab again", () => {
    const next = resizeRoi(ROI, "e", { x: ROI[0], y: 500 }, IMAGE);
    expect(next[2]).toBeGreaterThanOrEqual(MIN_ROI);
    expect(next[3]).toBeGreaterThanOrEqual(MIN_ROI);
  });
});

describe("moveRoi", () => {
  it("translates without resizing", () => {
    const next = moveRoi(ROI, 40, -25, IMAGE);
    expect(next[0]).toBeCloseTo(ROI[0] + 40, 6);
    expect(next[1]).toBeCloseTo(ROI[1] - 25, 6);
    expect(next[2]).toBe(ROI[2]);
    expect(next[3]).toBe(ROI[3]);
  });

  it("slides along the image edge rather than shrinking against it", () => {
    const next = moveRoi(ROI, 9999, 9999, IMAGE);
    expect(next[2]).toBe(ROI[2]);
    expect(next[3]).toBe(ROI[3]);
    expect(next[0] + next[2]).toBeCloseTo(IMAGE.width, 6);
    expect(next[1] + next[3]).toBeCloseTo(IMAGE.height, 6);
  });
});

describe("grabAt", () => {
  const radius = 8;

  it("prefers a handle to the interior, even where they overlap", () => {
    const tiny: Roi = [100, 100, 10, 10];
    expect(grabAt(tiny, { x: 100, y: 100 }, radius)).toBe("nw");
    expect(grabAt(tiny, { x: 110, y: 110 }, radius)).toBe("se");
  });

  it("reports the interior and misses honestly", () => {
    expect(grabAt(ROI, { x: 700, y: 500 }, radius)).toBe("move");
    expect(grabAt(ROI, { x: 100, y: 100 }, radius)).toBeNull();
  });

  it("grabs a handle from just outside the box, which is where a corner is aimed at", () => {
    expect(grabAt(ROI, { x: ROI[0] - 5, y: ROI[1] - 5 }, radius)).toBe("nw");
  });
});

describe("roiFromCorners", () => {
  it("normalises a drag in any direction", () => {
    expect(roiFromCorners({ x: 300, y: 200 }, { x: 100, y: 50 })).toEqual([100, 50, 200, 150]);
  });
});

describe("clampRoi", () => {
  it("holds a box inside the image and above the minimum", () => {
    expect(clampRoi([-50, -50, 2, 2], IMAGE)).toEqual([0, 0, MIN_ROI, MIN_ROI]);
    const huge = clampRoi([0, 0, 99999, 99999], IMAGE);
    expect(huge).toEqual([0, 0, IMAGE.width, IMAGE.height]);
  });
});

describe("sameRoi", () => {
  it("is what tells a preview its extraction is still current", () => {
    expect(sameRoi(ROI, [...ROI] as Roi)).toBe(true);
    expect(sameRoi(ROI, [ROI[0] + 1, ROI[1], ROI[2], ROI[3]])).toBe(false);
    expect(sameRoi(null, null)).toBe(true);
    expect(sameRoi(ROI, null)).toBe(false);
  });
});
