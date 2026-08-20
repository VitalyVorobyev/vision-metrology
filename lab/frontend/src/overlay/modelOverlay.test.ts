import { describe, expect, it } from "vitest";

import { matchOverlay, modelOverlay } from "./modelOverlay";
import type { MatchOut, ModelGeometryOut } from "../api/backend";

/** Two points either side of the origin, gradients pointing along +x. */
const GEOMETRY: ModelGeometryOut = {
  model_id: "model-1",
  level: 0,
  origin: [100, 100],
  reference_angle: 0,
  points: [110, 100, 1, 0, 90, 100, 1, 0],
  frame: "model",
};

function matchAt(x: number, y: number, angle: number, scale: number): MatchOut {
  return { x, y, angle, scale, score: 0.9, support: 2, level: 0 };
}

describe("modelOverlay", () => {
  it("draws one orientation tick per model point, plus the origin and its datum arm", () => {
    const out = modelOverlay(GEOMETRY);
    const segments = out.filter((p) => p.kind === "segment");
    const points = out.filter((p) => p.kind === "point");
    // Two ticks + one datum arm.
    expect(segments).toHaveLength(3);
    expect(points).toHaveLength(1);
    expect(points[0]).toMatchObject({ x: 100, y: 100, cross: true });
  });
});

describe("matchOverlay", () => {
  it("places the model at the found pose, rotating about the model origin", () => {
    // A quarter turn, unit scale, origin landing at (300, 300). The point that
    // sat 10px to the +x of the origin must end up 10px *below* it.
    const out = matchOverlay(GEOMETRY, matchAt(300, 300, Math.PI / 2, 1));
    const first = out[0] as { x1: number; y1: number; x2: number; y2: number };
    // The tick is centred on the transformed point; its midpoint is what we check.
    expect((first.x1 + first.x2) / 2).toBeCloseTo(300, 4);
    expect((first.y1 + first.y2) / 2).toBeCloseTo(310, 4);
  });

  it("scales offsets from the origin but never the reported position", () => {
    const out = matchOverlay(GEOMETRY, matchAt(300, 300, 0, 2));
    const first = out[0] as { x1: number; y1: number; x2: number; y2: number };
    // +10 in the model frame becomes +20 at scale 2.
    expect((first.x1 + first.x2) / 2).toBeCloseTo(320, 4);
    expect((first.y1 + first.y2) / 2).toBeCloseTo(300, 4);

    const cross = out.find((p) => p.kind === "point") as { x: number; y: number };
    expect(cross.x).toBe(300);
    expect(cross.y).toBe(300);
  });

  it("rotates the gradient directions with the pose, not just the positions", () => {
    // At a quarter turn a gradient along +x must come out along +y, or the
    // ticks would all lie parallel to the taught orientation and say nothing.
    const out = matchOverlay(GEOMETRY, matchAt(0, 0, Math.PI / 2, 1));
    const first = out[0] as { x1: number; y1: number; x2: number; y2: number };
    expect(first.x2 - first.x1).toBeCloseTo(0, 4);
    expect(first.y2 - first.y1).toBeGreaterThan(0);
  });

  it("labels each instance with its score", () => {
    const out = matchOverlay(GEOMETRY, matchAt(10, 20, 0, 1));
    const cross = out.find((p) => p.kind === "point") as { label?: string };
    expect(cross.label).toBe("0.90");
  });
});
