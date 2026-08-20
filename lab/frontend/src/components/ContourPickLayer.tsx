/**
 * The contours a model is about to be built from, drawn so they can be argued
 * with.
 *
 * This is the step teaching never had. A rectangle ROI on a round or L-shaped
 * part always drags in some background, and until now the only evidence of that
 * was a point count — so the fix was to nudge `min_contrast` and hope. Here the
 * candidates are on the image, each one clickable: keep the part, drop the
 * bench it is sitting on.
 *
 * Drawn as SVG in source-image coordinates, like `MeasureOverlay`, but with
 * `pointer-events` on — which is why it is a sibling of `ZoomPanCanvas` rather
 * than a child (the canvas captures every `pointerdown` for panning), and why
 * it draws its own geometry instead of pushing primitives into the overlay.
 *
 * Hit-testing is the browser's: each polyline gets a fat transparent stroke
 * under the visible one, so a 1-pixel contour is still a comfortable click
 * target at any zoom.
 */

import { cn, contentUnder, strokeWidthFor, type View } from "@vitavision/lab-ui";
import { useRef, useState } from "react";
import type { RefObject } from "react";

import type { ContourSelection } from "../state/LabContext";

/** Click target width in *screen* pixels, independent of zoom. */
const HIT_WIDTH = 12;

export function ContourPickLayer({
  frameRef,
  view,
  nativeWidth,
  nativeHeight,
  selection,
}: {
  frameRef: RefObject<HTMLDivElement | null>;
  view: View;
  nativeWidth: number;
  nativeHeight: number;
  selection: ContourSelection;
}) {
  const [hovered, setHovered] = useState<number | null>(null);
  const drag = useRef<{ x0: number; y0: number; rect: DOMRect } | null>(null);
  const [box, setBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  const stroke = strokeWidthFor(view.zoom);
  const hit = (HIT_WIDTH * stroke) / 1;

  /** Rubber-band select: everything whose midpoint falls inside the box flips. */
  const applyBox = (rect: DOMRect, x0: number, y0: number, x1: number, y1: number) => {
    const a = contentUnder(view, { clientX: x0, clientY: y0 }, rect);
    const b = contentUnder(view, { clientX: x1, clientY: y1 }, rect);
    if (!a || !b) return;
    const lo = { x: Math.min(a.u, b.u) * nativeWidth, y: Math.min(a.v, b.v) * nativeHeight };
    const hi = { x: Math.max(a.u, b.u) * nativeWidth, y: Math.max(a.v, b.v) * nativeHeight };
    for (const c of selection.contours) {
      const n = c.points.length / 2;
      if (n === 0) continue;
      const mid = (n >> 1) * 2;
      const [mx, my] = [c.points[mid]!, c.points[mid + 1]!];
      if (mx >= lo.x && mx <= hi.x && my >= lo.y && my <= hi.y) selection.onToggle(c.id);
    }
  };

  return (
    <div
      className="absolute inset-0"
      onPointerDown={(event) => {
        // A drag on empty space is a rubber band; a click on a contour is
        // handled by the polyline's own `onClick` below and never reaches here
        // with a meaningful box.
        const rect = frameRef.current?.getBoundingClientRect();
        if (!rect || !event.shiftKey) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = { x0: event.clientX, y0: event.clientY, rect };
        setBox({ x: event.clientX - rect.left, y: event.clientY - rect.top, w: 0, h: 0 });
      }}
      onPointerMove={(event) => {
        const d = drag.current;
        if (!d) return;
        setBox({
          x: Math.min(d.x0, event.clientX) - d.rect.left,
          y: Math.min(d.y0, event.clientY) - d.rect.top,
          w: Math.abs(event.clientX - d.x0),
          h: Math.abs(event.clientY - d.y0),
        });
      }}
      onPointerUp={(event) => {
        const d = drag.current;
        drag.current = null;
        setBox(null);
        if (!d) return;
        applyBox(d.rect, d.x0, d.y0, event.clientX, event.clientY);
      }}
    >
      <svg
        viewBox={`0 0 ${nativeWidth} ${nativeHeight}`}
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full"
      >
        {selection.contours.map((c) => {
          const kept = selection.kept.has(c.id);
          const d = pathOf(c.points, c.closed);
          return (
            <g key={c.id}>
              <path
                d={d}
                fill="none"
                stroke="transparent"
                strokeWidth={hit}
                style={{ cursor: "pointer", pointerEvents: "stroke" }}
                onClick={() => selection.onToggle(c.id)}
                onPointerEnter={() => setHovered(c.id)}
                onPointerLeave={() => setHovered((h) => (h === c.id ? null : h))}
              />
              <path
                d={d}
                fill="none"
                className={cn(
                  kept ? "stroke-signal" : "stroke-fg-subtle",
                  hovered === c.id && "stroke-fg",
                )}
                strokeWidth={stroke * (kept ? 1.6 : 1)}
                strokeDasharray={kept ? undefined : `${stroke * 3} ${stroke * 3}`}
                style={{ pointerEvents: "none" }}
                opacity={kept ? 1 : 0.55}
              />
            </g>
          );
        })}
      </svg>

      {box && (
        <div
          className="pointer-events-none absolute border border-signal bg-signal/10"
          style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
        />
      )}
    </div>
  );
}

/** `[x0, y0, x1, y1, …]` as an SVG path. */
export function pathOf(points: number[], closed: boolean): string {
  if (points.length < 4) return "";
  let d = `M ${points[0]} ${points[1]}`;
  for (let i = 2; i < points.length; i += 2) d += ` L ${points[i]} ${points[i + 1]}`;
  return closed ? `${d} Z` : d;
}
