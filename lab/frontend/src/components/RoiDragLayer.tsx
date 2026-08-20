/**
 * A lightweight click-drag-release ROI picker, layered over `ZoomPanCanvas` rather than
 * inside its transformed stack: `ZoomPanCanvas`'s own box captures every `pointerdown` for
 * panning, so this renders as a sibling covering the same box and only turns its own
 * `pointer-events` on while `active`, letting normal pan/zoom through otherwise.
 *
 * Built on `contentUnder`, the same pure fraction-of-frame math `ZoomPanCanvas` uses for
 * hover reporting — the one thing worth getting from the package rather than
 * reimplementing, since a hand-rolled version could silently disagree with panning about
 * where zero is.
 */

import { contentUnder, type View } from "@vitavision/lab-ui";
import { useRef, useState } from "react";
import type { RefObject } from "react";

import type { Roi } from "../api/backend";
import { rectToRoi } from "../api/transforms";

export function RoiDragLayer({
  frameRef,
  view,
  nativeWidth,
  nativeHeight,
  active,
  onRoiChange,
}: {
  frameRef: RefObject<HTMLDivElement | null>;
  view: View;
  nativeWidth: number;
  nativeHeight: number;
  active: boolean;
  onRoiChange: (roi: Roi) => void;
}) {
  const start = useRef<{ clientX: number; clientY: number; rect: DOMRect } | null>(null);
  const [preview, setPreview] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  if (!active) return null;

  const previewFromClient = (rect: DOMRect, x0: number, y0: number, x1: number, y1: number) => ({
    x: Math.min(x0, x1) - rect.left,
    y: Math.min(y0, y1) - rect.top,
    w: Math.abs(x1 - x0),
    h: Math.abs(y1 - y0),
  });

  return (
    <div
      className="absolute inset-0 cursor-crosshair"
      onPointerDown={(event) => {
        const rect = frameRef.current?.getBoundingClientRect();
        if (!rect) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        start.current = { clientX: event.clientX, clientY: event.clientY, rect };
        setPreview(previewFromClient(rect, event.clientX, event.clientY, event.clientX, event.clientY));
      }}
      onPointerMove={(event) => {
        const origin = start.current;
        if (!origin) return;
        setPreview(previewFromClient(origin.rect, origin.clientX, origin.clientY, event.clientX, event.clientY));
      }}
      onPointerUp={(event) => {
        const origin = start.current;
        start.current = null;
        setPreview(null);
        if (!origin) return;
        const p0 = contentUnder(view, { clientX: origin.clientX, clientY: origin.clientY }, origin.rect);
        const p1 = contentUnder(view, { clientX: event.clientX, clientY: event.clientY }, origin.rect);
        if (!p0 || !p1) return;
        const roi = rectToRoi(p0.u * nativeWidth, p0.v * nativeHeight, p1.u * nativeWidth, p1.v * nativeHeight);
        if (roi[2] > 2 && roi[3] > 2) onRoiChange(roi);
      }}
    >
      {preview && (
        <div
          className="absolute border-2 border-signal bg-signal/10"
          style={{ left: preview.x, top: preview.y, width: preview.w, height: preview.h }}
        />
      )}
    </div>
  );
}
