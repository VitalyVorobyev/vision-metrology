/**
 * The model's own frame, as two things you can drag: where its origin sits, and
 * which way is 0°.
 *
 * Both are real model parameters (`ShapeModelConfig::origin` and
 * `reference_angle`) that were previously unreachable, so every model got the
 * centroid of its own points as an origin and the teach image's axes as its
 * zero. For a part with a natural orientation — a can end's tab, a connector's
 * key — that means a reported angle nobody can interpret and a rectified crop
 * that comes out at whatever angle the part happened to be lying at. Neither is
 * something an algorithm can infer; both are something a person knows at a
 * glance.
 */

import { contentUnder, strokeWidthFor, type View } from "@vitavision/lab-ui";
import { useRef } from "react";
import type { RefObject } from "react";

import type { FrameHandles } from "../state/LabContext";

/** Arm length of the orientation handle, in source-image pixels. */
const ARM = 60;

export function FrameHandleLayer({
  frameRef,
  view,
  nativeWidth,
  nativeHeight,
  handles,
}: {
  frameRef: RefObject<HTMLDivElement | null>;
  view: View;
  nativeWidth: number;
  nativeHeight: number;
  handles: FrameHandles;
}) {
  const dragging = useRef<"origin" | "angle" | null>(null);
  const stroke = strokeWidthFor(view.zoom);

  const [ox, oy] = handles.origin;
  const tip: [number, number] = [
    ox + ARM * Math.cos(handles.angle),
    oy + ARM * Math.sin(handles.angle),
  ];

  const imagePointAt = (clientX: number, clientY: number): [number, number] | null => {
    const rect = frameRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const p = contentUnder(view, { clientX, clientY }, rect);
    if (!p) return null;
    return [p.u * nativeWidth, p.v * nativeHeight];
  };

  return (
    <div
      // Transparent to the mouse except for the two handles, which opt back in
      // below: everything else here is decoration, and panning must still work
      // through it. Once a handle captures the pointer its move events are
      // delivered to it and bubble up here regardless of this setting.
      className="absolute inset-0"
      style={{ pointerEvents: "none" }}
      onPointerMove={(event) => {
        const mode = dragging.current;
        if (mode === null) return;
        const p = imagePointAt(event.clientX, event.clientY);
        if (!p) return;
        if (mode === "origin") handles.onOrigin(p);
        else handles.onAngle(Math.atan2(p[1] - oy, p[0] - ox));
      }}
      onPointerUp={() => {
        dragging.current = null;
      }}
    >
      <svg
        viewBox={`0 0 ${nativeWidth} ${nativeHeight}`}
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full"
        style={{ pointerEvents: "none" }}
      >
        {/* The 0° arm. */}
        <line
          x1={ox}
          y1={oy}
          x2={tip[0]}
          y2={tip[1]}
          className="stroke-signal"
          strokeWidth={stroke * 1.5}
        />
        <circle
          cx={tip[0]}
          cy={tip[1]}
          r={stroke * 4}
          className="fill-signal"
          style={{ pointerEvents: "all", cursor: "grab" }}
          onPointerDown={(event) => {
            (event.target as Element).setPointerCapture?.(event.pointerId);
            dragging.current = "angle";
          }}
        />
        {/* The origin cross. */}
        <line x1={ox - 10} y1={oy} x2={ox + 10} y2={oy} className="stroke-signal" strokeWidth={stroke} />
        <line x1={ox} y1={oy - 10} x2={ox} y2={oy + 10} className="stroke-signal" strokeWidth={stroke} />
        <circle
          cx={ox}
          cy={oy}
          r={stroke * 4}
          className="fill-signal/30 stroke-signal"
          strokeWidth={stroke}
          style={{ pointerEvents: "all", cursor: "move" }}
          onPointerDown={(event) => {
            (event.target as Element).setPointerCapture?.(event.pointerId);
            dragging.current = "origin";
          }}
        />
      </svg>
    </div>
  );
}
