/**
 * One place that decides what a press on the image means.
 *
 * The layers each want a full-frame surface — one to sweep-select contours, one to draw and
 * move the ROI — and only the topmost element under the pointer ever receives the event, so
 * two such surfaces is one surface that always wins and one that never fires. Hit-testing
 * therefore lives here, once, and the layers are left as drawing plus their own small
 * targets (a contour's fat stroke, an ROI handle's square), stacked in the order a hand
 * expects to reach them:
 *
 *     background surface  →  contour strokes  →  ROI handles  →  datum handles
 *
 * The rule for the background is a priority list rather than a mode: grab the ROI's inside
 * if the press landed there, draw a box if that is what the tool is for or there is no box
 * yet, sweep if the gesture says sweep, and otherwise decline — a declined press bubbles to
 * the stage and pans, which is what makes "drag the picture" work without a tool for it.
 *
 * Once a drag starts, its `pointermove`/`pointerup` are taken from **window**, not from the
 * element that was pressed. `setPointerCapture` routes every later event to that one
 * element, so a drag begun on an ROI handle delivered its moves to the handle — a sibling of
 * the surface that carries the move handler, never an ancestor of it — and the box simply did
 * not move. Listening at the window also means a drag survives the pointer leaving the
 * canvas, which is exactly what dragging a corner outward does.
 */

import { useStage } from "@vitavision/lab-ui";
import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { ContourOut, Roi } from "../api/backend";
import type { CanvasTool } from "../state/LabContext";
import { contoursInBox, type Bounds, type SelectMode } from "./contourSelection";
import {
  HANDLE_CURSOR,
  MIN_ROI,
  clampRoi,
  grabAt,
  moveRoi,
  resizeRoi,
  roiFromCorners,
  type RoiGrab,
} from "./roiEdit";

/** How far from a handle's centre a press still grabs it, in screen pixels. */
export const GRAB_PX = 11;

type Drag =
  | { kind: "roi"; grab: RoiGrab; from: { x: number; y: number }; start: Roi }
  | { kind: "draw"; from: { x: number; y: number } }
  | { kind: "band"; from: { x: number; y: number }; additive: boolean };

export interface CanvasInteraction {
  /** The ROI as it is being dragged, or `null` when nothing is in flight. */
  draftRoi: Roi | null;
  /** The rubber band, in image coordinates. */
  band: Bounds | null;
  cursor: string | undefined;
  surface: {
    onPointerDown: (event: ReactPointerEvent<SVGRectElement>) => void;
    onPointerMove: (event: ReactPointerEvent<SVGRectElement>) => void;
  };
  /** Wire onto an ROI handle so it can start a resize from on top of everything else. */
  grabHandle: (grab: RoiGrab, event: ReactPointerEvent<SVGElement>) => void;
  /**
   * Start a rubber band from a layer that sits above the surface.
   *
   * Only the topmost element under the pointer is a press's target, so a contour stroke that
   * merely declines a shift-press does not hand it down to the surface — it hands it *up* to
   * the stage, which pans. A sweep that cannot start on a contour is a sweep that cannot
   * start anywhere useful on a frame that is mostly contours, so the stroke calls this
   * instead of declining.
   */
  startBand: (event: ReactPointerEvent<SVGElement>) => void;
}

export function useCanvasInteraction({
  roi,
  onRoi,
  roiEditable,
  tool,
  contours,
  onSelect,
}: {
  roi: Roi | null;
  onRoi: (roi: Roi) => void;
  roiEditable: boolean;
  tool: CanvasTool;
  contours: ContourOut[] | null;
  onSelect: ((ids: number[], mode: SelectMode) => void) | null;
}): CanvasInteraction {
  const stage = useStage();
  const [draftRoi, setDraftRoi] = useState<Roi | null>(null);
  const [band, setBand] = useState<Bounds | null>(null);
  const [cursor, setCursor] = useState<string | undefined>(undefined);
  const drag = useRef<Drag | null>(null);
  const draft = useRef<Roi | null>(null);

  const at = useCallback(
    (event: { clientX: number; clientY: number }) =>
      stage.toImage({ x: event.clientX, y: event.clientY }),
    [stage],
  );

  // Mirrored into state only so the window listeners can be attached and removed; the ref is
  // what the handlers read, because a re-render per pointermove would be a cost for nothing.
  const [dragging, setDragging] = useState(false);

  const begin = (event: ReactPointerEvent<SVGElement>, next: Drag) => {
    event.stopPropagation();
    event.preventDefault();
    drag.current = next;
    setDragging(true);
  };

  const wantsBand = (event: { shiftKey: boolean }) =>
    contours !== null && onSelect !== null && (event.shiftKey || tool === "marquee");

  const startBandAt = (event: ReactPointerEvent<SVGElement>, p: { x: number; y: number }) => {
    if (!wantsBand(event)) return false;
    begin(event, { kind: "band", from: p, additive: event.metaKey || event.ctrlKey });
    setBand({ x: p.x, y: p.y, width: 0, height: 0 });
    return true;
  };

  const startBand = (event: ReactPointerEvent<SVGElement>) => {
    if (event.button !== 0 || stage.panMode) return;
    startBandAt(event, at(event));
  };

  const grabHandle = useCallback(
    (grab: RoiGrab, event: ReactPointerEvent<SVGElement>) => {
      if (!roiEditable || event.button !== 0 || stage.panMode || roi === null) return;
      begin(event, { kind: "roi", grab, from: at(event), start: roi });
      draft.current = roi;
      setDraftRoi(roi);
    },
    [roiEditable, stage.panMode, roi, at],
  );

  const onPointerDown = (event: ReactPointerEvent<SVGRectElement>) => {
    if (event.button !== 0 || stage.panMode) return;
    const p = at(event);

    const grab = roiEditable && roi !== null ? grabAt(roi, p, stage.imageLength(GRAB_PX)) : null;

    // An edge or a corner first: those are unambiguous, and a handle is a small target the
    // user has deliberately aimed at.
    if (grab !== null && grab !== "move" && roi !== null) {
      begin(event, { kind: "roi", grab, from: p, start: roi });
      draft.current = roi;
      setDraftRoi(roi);
      return;
    }

    // Then a sweep — *before* the region's inside. Contours are usually inside the region
    // (that is what a region is for), so checking the interior first meant shift-dragging
    // over them moved the box instead of selecting them.
    if (wantsBand(event) && startBandAt(event, p)) return;

    if (grab === "move" && roi !== null) {
      begin(event, { kind: "roi", grab, from: p, start: roi });
      draft.current = roi;
      setDraftRoi(roi);
      return;
    }

    if (roiEditable && (tool === "box" || roi === null)) {
      begin(event, { kind: "draw", from: p });
      const seed = clampRoi([p.x, p.y, MIN_ROI, MIN_ROI], stage.image);
      draft.current = seed;
      setDraftRoi(seed);
      return;
    }

    // Declined: the press bubbles to the stage, which pans.
  };

  const onPointerMove = (event: ReactPointerEvent<SVGRectElement>) => {
    if (drag.current !== null) return;
    setCursor(hoverCursor(roi, roiEditable, tool, stage, at(event)));
  };

  /* The live drag, at the window. Re-bound whenever the geometry it reads changes, which is
   * cheap: `dragging` only flips twice per gesture. */
  useEffect(() => {
    if (!dragging) return;

    const move = (event: PointerEvent) => {
      const state = drag.current;
      if (state === null) return;
      const p = at(event);
      if (state.kind === "band") {
        setBand(boxBetween(state.from, p));
        return;
      }
      const next =
        state.kind === "draw"
          ? clampRoi(roiFromCorners(state.from, p), stage.image)
          : state.grab === "move"
            ? moveRoi(state.start, p.x - state.from.x, p.y - state.from.y, stage.image)
            : resizeRoi(state.start, state.grab, p, stage.image);
      draft.current = next;
      setDraftRoi(next);
    };

    const up = (event: PointerEvent) => {
      const state = drag.current;
      drag.current = null;
      setDragging(false);
      if (state === null) return;

      if (state.kind === "band") {
        const box = boxBetween(state.from, at(event));
        setBand(null);
        // An empty sweep clears the selection, which is how you let go of one with a tool in
        // hand rather than having to find empty background to click.
        onSelect?.(contours ? contoursInBox(contours, box) : [], state.additive ? "add" : "replace");
        return;
      }

      const next = draft.current;
      draft.current = null;
      setDraftRoi(null);
      // A draw that never became a box is a mis-click, not an ROI of four pixels.
      if (next && (state.kind !== "draw" || (next[2] > MIN_ROI && next[3] > MIN_ROI))) onRoi(next);
    };

    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, [dragging, at, stage.image, contours, onSelect, onRoi]);

  return {
    draftRoi,
    band,
    cursor,
    grabHandle,
    startBand,
    surface: { onPointerDown, onPointerMove },
  };
}

function hoverCursor(
  roi: Roi | null,
  roiEditable: boolean,
  tool: CanvasTool,
  stage: { panMode: boolean; imageLength: (css: number) => number },
  p: { x: number; y: number },
): string | undefined {
  if (stage.panMode) return undefined;
  if (tool === "marquee") return "crosshair";
  if (roiEditable && roi !== null) {
    const grab = grabAt(roi, p, stage.imageLength(GRAB_PX));
    if (grab !== null) return HANDLE_CURSOR[grab];
  }
  if (roiEditable && (tool === "box" || roi === null)) return "crosshair";
  return undefined;
}

function boxBetween(a: { x: number; y: number }, b: { x: number; y: number }): Bounds {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(b.x - a.x),
    height: Math.abs(b.y - a.y),
  };
}
