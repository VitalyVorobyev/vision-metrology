import { describe, expect, it } from "vitest";

import type { BatchFindItem, MatchOut, ModelGeometryOut, ModelOut } from "../api/backend";
import {
  describeMatches,
  frameVerdict,
  framesFound,
  matchBox,
  matchId,
  matchIndexOf,
  matchState,
  modelFrame,
  nextSort,
  pickMatch,
  sortMatches,
} from "./matchInventory";
import { boxBounds, boxContains, pickBox } from "./rotatedBox";

function match(x: number, y: number, extra: Partial<MatchOut> = {}): MatchOut {
  return { x, y, angle: 0, scale: 1, score: 0.9, support: 100, level: 0, ...extra };
}

/** A model taught from a 40 × 20 rectangle whose origin is its centre. */
const MODEL = { id: "model-1", roi: [80, 90, 40, 20], origin: [100, 100] } as unknown as ModelOut;
const FRAME = modelFrame(MODEL, null);

/** Three matches whose score, position and support orders all differ. */
const MATCHES: MatchOut[] = [
  match(300, 100, { score: 0.95, support: 80 }),
  match(100, 300, { score: 0.8, support: 120, angle: Math.PI / 2 }),
  match(200, 200, { score: 0.88, support: 95, scale: 1.5 }),
];

describe("rotatedBox", () => {
  const box = { cx: 50, cy: 50, width: 20, height: 10, angle: Math.PI / 2 };

  it("contains points in its own frame: a quarter turn swaps the extents", () => {
    // Turned a quarter, the box is 10 wide and 20 tall on the image.
    expect(boxContains(box, { x: 50, y: 59 })).toBe(true);
    expect(boxContains(box, { x: 59, y: 50 })).toBe(false);
    expect(boxContains(box, { x: 56, y: 50 }, 2)).toBe(true);
  });

  it("bounds the turned box, not the unturned one", () => {
    const bounds = boxBounds(box);
    expect(bounds.x).toBeCloseTo(45, 5);
    expect(bounds.y).toBeCloseTo(40, 5);
    expect(bounds.width).toBeCloseTo(10, 5);
    expect(bounds.height).toBeCloseTo(20, 5);
  });

  it("picks the box whose centre is nearest when boxes overlap", () => {
    const boxes = [
      { cx: 0, cy: 0, width: 30, height: 30, angle: 0 },
      { cx: 10, cy: 0, width: 30, height: 30, angle: 0 },
    ];
    expect(pickBox(boxes, (b) => b, { x: 8, y: 0 })).toBe(boxes[1]);
    expect(pickBox(boxes, (b) => b, { x: 2, y: 0 })).toBe(boxes[0]);
    expect(pickBox(boxes, (b) => b, { x: 100, y: 0 })).toBeNull();
  });
});

describe("modelFrame", () => {
  it("is the taught rectangle without the model's own points", () => {
    expect(FRAME).toEqual({ rect: { x: 80, y: 90, width: 40, height: 20 }, origin: [100, 100] });
  });

  it("is the points' extent when the shell reads them, whatever the rectangle was", () => {
    const geometry = {
      origin: [0, 0],
      points: [-5, -2, 1, 0, 15, 8, 1, 0],
    } as unknown as ModelGeometryOut;
    expect(modelFrame(MODEL, geometry)).toEqual({ rect: { x: -5, y: -2, width: 20, height: 10 }, origin: [0, 0] });
  });
});

describe("matchBox", () => {
  it("puts the model's rectangle where the pose puts its origin", () => {
    expect(matchBox(match(300, 100), FRAME)).toMatchObject({ cx: 300, cy: 100, width: 40, height: 20, angle: 0 });
  });

  it("turns and scales the rectangle about the origin, not about its own centre", () => {
    // A rectangle whose centre sits 10 px to the +x of the origin: at a quarter turn and
    // scale 2, its centre lands 20 px *below* the found position.
    const offset = { rect: { x: 0, y: -5, width: 20, height: 10 }, origin: [0, 0] as [number, number] };
    const box = matchBox(match(50, 50, { angle: Math.PI / 2, scale: 2 }), offset);
    expect(box.cx).toBeCloseTo(50, 5);
    expect(box.cy).toBeCloseTo(70, 5);
    expect(box.width).toBe(40);
    expect(box.height).toBe(20);
  });
});

describe("describeMatches", () => {
  it("keeps the search's order and reads the angle in degrees", () => {
    const stats = describeMatches(MATCHES, FRAME);
    expect(stats.map((s) => s.index)).toEqual([0, 1, 2]);
    expect(stats[1]!.angle).toBeCloseTo(90, 5);
    // A quarter turn stands the 40 × 20 box on end.
    expect(stats[1]!.bounds.width).toBeCloseTo(20, 4);
    expect(stats[1]!.bounds.height).toBeCloseTo(40, 4);
  });
});

describe("sortMatches", () => {
  const stats = describeMatches(MATCHES, FRAME);

  it("sorts by any column, both ways", () => {
    expect(sortMatches(stats, { key: "score", descending: true }).map((s) => s.index)).toEqual([0, 2, 1]);
    expect(sortMatches(stats, { key: "x", descending: false }).map((s) => s.index)).toEqual([1, 2, 0]);
    expect(sortMatches(stats, { key: "support", descending: true }).map((s) => s.index)).toEqual([1, 2, 0]);
    expect(sortMatches(stats, { key: "scale", descending: false }).map((s) => s.index)).toEqual([0, 1, 2]);
  });

  it("breaks ties on the index, so stepping cannot jump about", () => {
    const tied = describeMatches([match(0, 0), match(0, 0), match(0, 0)], FRAME);
    expect(sortMatches(tied, { key: "score", descending: true }).map((s) => s.index)).toEqual([0, 1, 2]);
  });

  it("starts a new column where reading it starts, and flips the same one", () => {
    expect(nextSort({ key: "score", descending: true }, "score")).toEqual({ key: "score", descending: false });
    expect(nextSort({ key: "score", descending: true }, "x")).toEqual({ key: "x", descending: false });
    expect(nextSort({ key: "x", descending: false }, "support")).toEqual({ key: "support", descending: true });
  });
});

describe("pickMatch", () => {
  const stats = describeMatches(MATCHES, FRAME);

  it("finds the match whose box holds the point", () => {
    expect(pickMatch(stats, { x: 315, y: 105 })).toBe(0);
    // The turned match is tall: 15 px below its origin is inside it, 15 px to the side is not.
    expect(pickMatch(stats, { x: 100, y: 315 })).toBe(1);
    expect(pickMatch(stats, { x: 115, y: 300 })).toBeNull();
  });

  it("widens by the tolerance", () => {
    expect(pickMatch(stats, { x: 323, y: 100 })).toBeNull();
    expect(pickMatch(stats, { x: 323, y: 100 }, 4)).toBe(0);
  });
});

describe("ids and states", () => {
  it("round-trips the overlay id and rejects anything else", () => {
    expect(matchIndexOf(matchId(7))).toBe(7);
    expect(matchIndexOf("caliper-0-1")).toBeNull();
    expect(matchIndexOf(null)).toBeNull();
  });

  it("lets the selection outrank the hover", () => {
    expect(matchState(2, 2, 2)).toBe("selected");
    expect(matchState(2, null, 2)).toBe("hover");
    expect(matchState(2, 1, null)).toBe("default");
  });
});

describe("batch verdicts", () => {
  const item = (matches: MatchOut[], error: string | null = null): BatchFindItem => ({
    image_id: "img",
    matches,
    elapsed_ms: 1,
    error,
  });

  it("tells a miss from a failure, and counts the hits", () => {
    expect(frameVerdict(undefined)).toBeNull();
    expect(frameVerdict(item([match(0, 0)]))).toBe("found");
    expect(frameVerdict(item([]))).toBe("not-found");
    expect(frameVerdict(item([], "decode failed"))).toBe("error");
    expect(framesFound([item([match(0, 0)]), item([]), item([], "x")])).toEqual({ found: 1, total: 3 });
  });
});
