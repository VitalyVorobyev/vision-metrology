/**
 * The sweep: a rubber band that selects every contour it touches.
 *
 * Only the topmost element under the pointer receives a press, so the band has to be
 * startable from three places: the bare-image surface underneath everything, a contour's
 * hit stroke (a frame with a hundred and sixty-six contours is mostly strokes), and the
 * region editor's interior, which would otherwise take the press as a move. The surface is
 * wired here; the other two call `startBand`.
 *
 * Shift-drag sweeps in any tool, and any drag sweeps with the marquee tool. A press the band
 * has no use for is declined, and bubbles to the stage, which pans.
 *
 * Once a band starts, its `pointermove`/`pointerup` are taken from **window**, not from the
 * element that was pressed. `setPointerCapture` would route every later event to that one
 * element, which is the stroke or the region, not the surface that holds the band. Listening
 * at the window also lets a band survive the pointer leaving the canvas.
 */

import { useStage } from "@vitavision/stage2d";
import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { ContourOut } from "../api/backend";
import type { CanvasTool } from "../state/LabContext";
import { contoursInBox, type Bounds, type SelectMode } from "./contourSelection";

interface Band {
  from: { x: number; y: number };
  additive: boolean;
}

export interface CanvasInteraction {
  /** The rubber band, in image coordinates, while one is being drawn. */
  band: Bounds | null;
  cursor: string | undefined;
  surface: {
    onPointerDown: (event: ReactPointerEvent<SVGRectElement>) => void;
  };
  /**
   * Start a band from a layer above the surface, if the press asks for one.
   *
   * Claims the press (stops propagation) only when it starts a band, so a caller can offer
   * every press and let the rest through.
   */
  startBand: (event: ReactPointerEvent<Element>) => void;
}

export function useCanvasInteraction({
  tool,
  contours,
  onSelect,
}: {
  tool: CanvasTool;
  contours: ContourOut[] | null;
  onSelect: ((ids: number[], mode: SelectMode) => void) | null;
}): CanvasInteraction {
  const stage = useStage();
  const [band, setBand] = useState<Bounds | null>(null);
  const dragRef = useRef<Band | null>(null);
  // Mirrored into state only so the window listeners can be attached and removed; the ref is
  // what the handlers read, because a re-render per pointermove would be a cost for nothing.
  const [dragging, setDragging] = useState(false);

  const at = useCallback(
    (event: { clientX: number; clientY: number }) =>
      stage.toImage({ x: event.clientX, y: event.clientY }),
    [stage],
  );

  const startBand = (event: ReactPointerEvent<Element>) => {
    if (event.button !== 0 || stage.panMode) return;
    if (contours === null || onSelect === null) return;
    if (!event.shiftKey && tool !== "marquee") return;
    event.stopPropagation();
    event.preventDefault();
    const from = at(event);
    dragRef.current = { from, additive: event.metaKey || event.ctrlKey };
    setDragging(true);
    setBand({ x: from.x, y: from.y, width: 0, height: 0 });
  };

  useEffect(() => {
    if (!dragging) return;

    const move = (event: PointerEvent) => {
      const state = dragRef.current;
      if (state !== null) setBand(boxBetween(state.from, at(event)));
    };

    const up = (event: PointerEvent) => {
      const state = dragRef.current;
      dragRef.current = null;
      setDragging(false);
      setBand(null);
      if (state === null) return;
      // An empty sweep clears the selection, which is how you let go of one with a tool in
      // hand rather than having to find empty background to click.
      const box = boxBetween(state.from, at(event));
      onSelect?.(contours ? contoursInBox(contours, box) : [], state.additive ? "add" : "replace");
    };

    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, [dragging, at, contours, onSelect]);

  return {
    band,
    cursor: !stage.panMode && tool === "marquee" ? "crosshair" : undefined,
    startBand,
    surface: { onPointerDown: startBand },
  };
}

function boxBetween(a: { x: number; y: number }, b: { x: number; y: number }): Bounds {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(b.x - a.x),
    height: Math.abs(b.y - a.y),
  };
}
