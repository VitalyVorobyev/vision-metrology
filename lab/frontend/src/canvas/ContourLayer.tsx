/**
 * The contours a model is about to be built from, drawn so they can be argued with.
 *
 * This is the step teaching never had. A rectangle ROI on a round or L-shaped part always
 * drags in some background, and until now the only evidence of that was a point count — so
 * the fix was to nudge `min_contrast` and hope. Here the candidates are on the image, each
 * one selectable, and the inventory beside them says which is which.
 *
 * Drawn in **source-image coordinates inside the stage's transform**, so it moves with the
 * photograph. Its predecessor was mounted outside the transform (the old canvas captured
 * every `pointerdown`, leaving no way for an interactive layer to live inside it), which
 * meant contours stayed pinned at fit scale while the image zoomed away underneath them.
 *
 * Hit-testing is the browser's: each polyline gets a fat transparent stroke under the
 * visible one, so a one-pixel contour is a comfortable click target. Both strokes use
 * `vectorEffect="non-scaling-stroke"`, so widths are screen pixels at any zoom with no
 * compensation arithmetic to get wrong. Sweep-selection is *not* here — a full-frame
 * capture surface in this layer would be the topmost element over the whole image and would
 * swallow every press meant for the ROI beneath it, so it lives once in
 * `useCanvasInteraction`.
 */

import { toneColor, useStage } from "@vitavision/lab-ui";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { ContourSelection, LayerVisibility } from "../state/LabContext";
import type { SelectMode } from "./contourSelection";

/** Click target width, in screen pixels. */
const HIT_WIDTH = 14;
/** Vertices become worth drawing once one image pixel is this many screen pixels. */
const VERTEX_SCALE = 3;

const KEPT = toneColor("signal");
const DROPPED = toneColor("muted");
const SELECTED = toneColor("warn");

export function ContourLayer({
  selection,
  layers,
  sweeping,
  onSweep,
}: {
  selection: ContourSelection;
  layers: LayerVisibility;
  /** True when the active tool means "sweep", so a press on a stroke starts one. */
  sweeping: boolean;
  /** Hands a sweep press to the shared interaction state — see `useCanvasInteraction`. */
  onSweep: (event: ReactPointerEvent<SVGElement>) => void;
}) {
  const stage = useStage();
  const { contours, kept, selected, hovered } = selection;

  /**
   * Shift is not handled here: it means "sweep" on the canvas (see `onPointerDown` below),
   * and range-select lives in the inspector's list, where a range over rows is something a
   * reader can see. So a press on a stroke is either a plain selection or a meta-toggle.
   */
  const pick = (id: number, event: ReactPointerEvent<SVGPathElement>) => {
    const mode: SelectMode = event.metaKey || event.ctrlKey ? "toggle" : "replace";
    selection.onSelect([id], mode);
  };

  const showVertices = layers.vertices && stage.view.scale >= VERTEX_SCALE;

  return (
    <svg
      viewBox={`0 0 ${stage.image.width} ${stage.image.height}`}
      className="absolute inset-0 h-full w-full"
      // Only the fat hit strokes below take events; everything else here is drawing, and a
      // press on bare image must reach the surface underneath so it can pan or sweep.
      style={{ pointerEvents: "none" }}
    >
      {contours.map((contour) => {
        const isKept = kept.has(contour.id);
        if (isKept ? !layers.kept : !layers.dropped) return null;

        const isSelected = selected.has(contour.id);
        const isHovered = hovered === contour.id;
        const d = pathOf(contour.points, contour.closed);

        return (
          <g key={contour.id}>
            <path
              d={d}
              fill="none"
              stroke="transparent"
              strokeWidth={HIT_WIDTH}
              vectorEffect="non-scaling-stroke"
              style={{ cursor: "pointer", pointerEvents: "stroke" }}
              onPointerDown={(event) => {
                if (stage.panMode || event.button !== 0) return;
                // A sweep has to be able to start anywhere, and on a frame with a hundred
                // and sixty-six contours "anywhere" is usually *on* one. Declining would
                // send the press *up* to the stage (which pans), not down to the surface —
                // only the topmost element is a target — so start the band from here.
                if (event.shiftKey || sweeping) {
                  event.stopPropagation();
                  onSweep(event);
                  return;
                }
                // Claim the press so it never reads as the start of a pan, and select on
                // the press rather than the click: a selection that waits for pointerup
                // feels like a lag on a canvas where every other gesture is immediate.
                event.stopPropagation();
                pick(contour.id, event);
              }}
              onPointerEnter={() => selection.onHover(contour.id)}
              onPointerLeave={() => selection.onHover(null)}
            />
            <path
              d={d}
              fill="none"
              stroke={isSelected ? SELECTED : isKept ? KEPT : DROPPED}
              strokeWidth={isSelected ? 2.5 : isKept ? 1.5 : 1}
              strokeDasharray={isKept ? undefined : "4 4"}
              strokeOpacity={isKept || isSelected ? 1 : 0.45}
              vectorEffect="non-scaling-stroke"
              style={{ pointerEvents: "none" }}
            />
            {isHovered && !isSelected && (
              <path
                d={d}
                fill="none"
                stroke="#ffffff"
                strokeWidth={2.5}
                strokeOpacity={0.9}
                vectorEffect="non-scaling-stroke"
                style={{ pointerEvents: "none" }}
              />
            )}
            {showVertices && (isSelected || isHovered) && (
              <Vertices points={contour.points} radius={stage.imageLength(1.6)} />
            )}
          </g>
        );
      })}
    </svg>
  );
}

/**
 * The contour's own samples, drawn only for the one being looked at and only once a pixel
 * is big enough to hold a dot. "7365 points" is otherwise a number with nothing behind it;
 * drawing all of them at once is both unreadable and slow.
 */
function Vertices({ points, radius }: { points: number[]; radius: number }) {
  const dots = [];
  for (let i = 0; i + 1 < points.length; i += 2) {
    dots.push(
      <circle key={i} cx={points[i]} cy={points[i + 1]} r={radius} fill="#ffffff" fillOpacity={0.85} />,
    );
  }
  return <g style={{ pointerEvents: "none" }}>{dots}</g>;
}

/** `[x0, y0, x1, y1, …]` as an SVG path. */
export function pathOf(points: number[], closed: boolean): string {
  if (points.length < 4) return "";
  let d = `M ${points[0]} ${points[1]}`;
  for (let i = 2; i < points.length; i += 2) d += ` L ${points[i]} ${points[i + 1]}`;
  return closed ? `${d} Z` : d;
}
