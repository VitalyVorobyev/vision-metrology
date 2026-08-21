/**
 * The ROI, as an object rather than a gesture.
 *
 * Drawing a box was all the lab could do: every press started a new one from a corner, so a
 * box that was nearly right had to be redrawn, and one drawn at fit could not be refined by
 * zooming in first. Here it has eight handles and an interior, and the numbers in the panel
 * are the same box from the other side.
 *
 * Handle squares are sized in image pixels through `stage.imageLength`, which makes them a
 * constant number of *screen* pixels at any zoom. Its predecessor sized the datum's handles
 * as `strokeWidth × 4` inside a stretched viewBox — around three screen pixels, which is a
 * control that reads as decoration.
 *
 * Handles sit above the contours so a corner is always grabbable; the box's *interior* is
 * handled by the background surface underneath them (see `useCanvasInteraction`).
 */

import { toneColor, useStage } from "@vitavision/lab-ui";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { Roi } from "../api/backend";
import { ROI_HANDLES, handlePoint, type RoiGrab, type RoiHandle } from "./roiEdit";

/** Handle square side, in screen pixels. */
const HANDLE_PX = 9;

const TONE = toneColor("signal");

export function RoiLayer({
  roi,
  editable,
  onGrab,
}: {
  roi: Roi;
  editable: boolean;
  onGrab: (grab: RoiGrab, event: ReactPointerEvent<SVGElement>) => void;
}) {
  const stage = useStage();
  const size = stage.imageLength(HANDLE_PX);

  return (
    <svg
      viewBox={`0 0 ${stage.image.width} ${stage.image.height}`}
      className="absolute inset-0 h-full w-full"
      style={{ pointerEvents: "none" }}
    >
      <rect
        x={roi[0]}
        y={roi[1]}
        width={roi[2]}
        height={roi[3]}
        fill="none"
        stroke={TONE}
        strokeWidth={1.5}
        strokeDasharray="6 4"
        vectorEffect="non-scaling-stroke"
      />
      {editable &&
        ROI_HANDLES.map((handle) => (
          <Handle key={handle} handle={handle} roi={roi} size={size} onGrab={onGrab} />
        ))}
    </svg>
  );
}

function Handle({
  handle,
  roi,
  size,
  onGrab,
}: {
  handle: RoiHandle;
  roi: Roi;
  size: number;
  onGrab: (grab: RoiGrab, event: ReactPointerEvent<SVGElement>) => void;
}) {
  const centre = handlePoint(roi, handle);
  return (
    <rect
      x={centre.x - size / 2}
      y={centre.y - size / 2}
      width={size}
      height={size}
      fill={TONE}
      stroke="#0b0d0f"
      strokeWidth={1}
      vectorEffect="non-scaling-stroke"
      style={{ pointerEvents: "all", cursor: CURSORS[handle] }}
      onPointerDown={(event) => onGrab(handle, event)}
    />
  );
}

const CURSORS: Record<RoiHandle, string> = {
  nw: "nwse-resize",
  n: "ns-resize",
  ne: "nesw-resize",
  e: "ew-resize",
  se: "nwse-resize",
  s: "ns-resize",
  sw: "nesw-resize",
  w: "ew-resize",
};
