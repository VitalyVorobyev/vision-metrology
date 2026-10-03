/**
 * Turning a model's points into something drawable.
 *
 * The one thing the lab never did: `ModelOut` carried point *counts*, so the
 * only evidence a model existed was a number. These helpers take the geometry
 * the backend now returns and produce `MeasurePrimitive`s — the same union
 * `MeasureOverlay` already draws for calipers and fits, in source-image
 * coordinates, so nothing new has to be rendered and the strokes stay one
 * screen pixel at any zoom.
 *
 * Model points are drawn as short **oriented ticks** rather than dots. A dot
 * says where an edge is; a tick says which way its gradient runs, which is the
 * thing the matcher actually scores and therefore the thing worth being able to
 * see. At a distance the ticks read as a contour anyway.
 *
 * They stay one `segment` per point rather than a `polyline` through the points,
 * because the points are not in contour order: the model stores them stratified
 * (a golden-ratio permutation, so every prefix the search scores early is spread
 * over the whole shape), and a line through them in that order is a scribble.
 *
 * Everything here is drawn in the overlay grammar's `model` role: it is what the
 * model predicts, not what was detected.
 */

import { caliperCorners, type MeasurePrimitive, type OverlayState } from "@vitavision/stage2d";

import type { MatchOut, ModelGeometryOut } from "../api/backend";
import type { RotatedBox } from "../state/rotatedBox";

/** Half-length of a point's orientation tick, in source-image pixels. */
const TICK = 2.5;
/** Length of the datum arm drawn from the origin. */
const ARM = 40;

/** A model drawn where it was taught. */
export function modelOverlay(geometry: ModelGeometryOut): MeasurePrimitive[] {
  const role = "model";
  const out: MeasurePrimitive[] = [];
  const { points, origin, reference_angle: refAngle } = geometry;

  for (let i = 0; i + 3 < points.length; i += 4) {
    const [x, y, dx, dy] = [points[i]!, points[i + 1]!, points[i + 2]!, points[i + 3]!];
    out.push({
      kind: "segment",
      role,
      x1: x - dx * TICK,
      y1: y - dy * TICK,
      x2: x + dx * TICK,
      y2: y + dy * TICK,
    });
  }

  out.push({ kind: "point", role, x: origin[0], y: origin[1], cross: true });
  out.push({
    kind: "segment",
    role,
    dashed: true,
    x1: origin[0],
    y1: origin[1],
    x2: origin[0] + ARM * Math.cos(refAngle),
    y2: origin[1] + ARM * Math.sin(refAngle),
  });
  return out;
}

/**
 * A model drawn at a found pose.
 *
 * `pose * p` for every model-frame point — the same transform
 * `ShapeMatch::pose` applies internally, reconstructed here from the reported
 * `(x, y, angle, scale)` and the model's origin. Drawing the *model* on the
 * instance is what makes a match verifiable at a glance: a cross with a score
 * next to it tells you the search returned something, not whether it was right.
 *
 * `state` is the overlay grammar's: `selected` is the match the table points at, `hover`
 * the one under the pointer. `id` (`match-<index>`) links every primitive of the match to
 * its row.
 */
export function matchOverlay(
  geometry: ModelGeometryOut,
  match: MatchOut,
  state: OverlayState = "default",
  id?: string,
): MeasurePrimitive[] {
  const role = "model";
  const { points, origin } = geometry;
  const cos = Math.cos(match.angle) * match.scale;
  const sin = Math.sin(match.angle) * match.scale;

  // `position` is where the model's origin landed, so the pose is "rotate and
  // scale about the origin, then put the origin there".
  const map = (x: number, y: number): [number, number] => {
    const [ux, uy] = [x - origin[0], y - origin[1]];
    return [match.x + cos * ux - sin * uy, match.y + sin * ux + cos * uy];
  };
  // Directions rotate but do not translate.
  const rot = (dx: number, dy: number): [number, number] => [
    Math.cos(match.angle) * dx - Math.sin(match.angle) * dy,
    Math.sin(match.angle) * dx + Math.cos(match.angle) * dy,
  ];

  const out: MeasurePrimitive[] = [];
  for (let i = 0; i + 3 < points.length; i += 4) {
    const [px, py] = map(points[i]!, points[i + 1]!);
    const [dx, dy] = rot(points[i + 2]!, points[i + 3]!);
    out.push({
      kind: "segment",
      role,
      state,
      id,
      x1: px - dx * TICK,
      y1: py - dy * TICK,
      x2: px + dx * TICK,
      y2: py + dy * TICK,
    });
  }

  out.push({
    kind: "point",
    role,
    state,
    id,
    x: match.x,
    y: match.y,
    cross: true,
    label: match.score.toFixed(2),
  });
  out.push({
    kind: "segment",
    role,
    state,
    id,
    x1: match.x,
    y1: match.y,
    x2: match.x + ARM * match.scale * Math.cos(match.angle),
    y2: match.y + ARM * match.scale * Math.sin(match.angle),
  });
  return out;
}

/**
 * A match without the model's geometry (the browser build reads no model points): a cross
 * at the found origin, labelled with the score.
 */
export function matchMarker(match: MatchOut, state: OverlayState = "default", id?: string): MeasurePrimitive {
  return { kind: "point", role: "model", state, id, x: match.x, y: match.y, cross: true, label: match.score.toFixed(2) };
}

/**
 * The extent a match occupies, as a dashed outline: drawn for the match being pointed at,
 * so the row under the pointer has a shape on the canvas and not only a heavier stroke.
 */
export function matchBoxOverlay(box: RotatedBox, state: OverlayState, id?: string): MeasurePrimitive {
  const corners = caliperCorners(box.cx, box.cy, box.width, box.height, box.angle);
  return {
    kind: "polyline",
    role: "model",
    state,
    id,
    dashed: true,
    closed: true,
    points: corners.flatMap((corner) => [corner.x, corner.y]),
  };
}
