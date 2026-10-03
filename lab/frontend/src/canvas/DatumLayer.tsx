/**
 * The model's own frame, as two things you can drag: where its origin sits, and which way
 * is 0°.
 *
 * Both are real model parameters (`ShapeModelConfig::origin` and `reference_angle`) that
 * were previously unreachable, so every model got the centroid of its own points as an
 * origin and the teach image's axes as its zero. For a part with a natural orientation — a
 * can end's tab, a connector's key — that means a reported angle nobody can interpret and a
 * rectified crop that comes out at whatever angle the part happened to be lying at.
 *
 * Two things about how it is drawn are corrections rather than taste. The handles are sized
 * in **screen** pixels (they used to come out around three, which is a control that reads as
 * decoration), and the datum has its own colour: drawn in the contour cyan, on a canvas
 * covered in cyan contours, it was one more inert line among a hundred and sixty-six.
 */

import { imageViewBox, toneColor, useStage } from "@vitavision/lab-ui";
import { useRef } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { FrameHandles } from "../state/LabContext";

/** Arm length and handle sizes, in screen pixels — the arm is a control, not a measurement. */
const ARM_PX = 78;
const ORIGIN_PX = 11;
const TIP_PX = 9;
const CROSS_PX = 13;
/** Held modifiers snap the arm to this many degrees. */
const SNAP_DEGREES = 15;

const TONE = toneColor("normal");

export function DatumLayer({ handles }: { handles: FrameHandles }) {
  const stage = useStage();
  const dragging = useRef<"origin" | "angle" | null>(null);

  const [ox, oy] = handles.origin;
  const arm = stage.imageLength(ARM_PX);
  const cross = stage.imageLength(CROSS_PX);
  const tip: [number, number] = [ox + arm * Math.cos(handles.angle), oy + arm * Math.sin(handles.angle)];

  const start = (mode: "origin" | "angle") => (event: ReactPointerEvent<SVGElement>) => {
    if (event.button !== 0 || stage.panMode) return;
    event.stopPropagation();
    (event.target as Element).setPointerCapture?.(event.pointerId);
    dragging.current = mode;
  };

  const onMove = (event: ReactPointerEvent<SVGGElement>) => {
    const mode = dragging.current;
    if (mode === null) return;
    event.stopPropagation();
    const p = stage.toImage({ x: event.clientX, y: event.clientY });
    if (mode === "origin") {
      handles.onOrigin([p.x, p.y]);
      return;
    }
    const raw = Math.atan2(p.y - oy, p.x - ox);
    handles.onAngle(event.shiftKey ? snap(raw, SNAP_DEGREES) : raw);
  };

  const end = () => {
    dragging.current = null;
  };

  return (
    <svg
      viewBox={imageViewBox(stage.image)}
      className="absolute inset-0 h-full w-full"
      style={{ pointerEvents: "none" }}
    >
      <g onPointerMove={onMove} onPointerUp={end} onPointerCancel={end} onLostPointerCapture={end}>
        <line
          x1={ox}
          y1={oy}
          x2={tip[0]}
          y2={tip[1]}
          stroke={TONE}
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
        />
        <line x1={ox - cross} y1={oy} x2={ox + cross} y2={oy} stroke={TONE} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        <line x1={ox} y1={oy - cross} x2={ox} y2={oy + cross} stroke={TONE} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />

        <circle
          cx={tip[0]}
          cy={tip[1]}
          r={stage.imageLength(TIP_PX / 2)}
          fill={TONE}
          stroke="#0b0d0f"
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
          style={{ pointerEvents: "all", cursor: "grab" }}
          onPointerDown={start("angle")}
        >
          <title>Drag to set the model&apos;s 0° direction (hold shift to snap to 15°)</title>
        </circle>

        <circle
          cx={ox}
          cy={oy}
          r={stage.imageLength(ORIGIN_PX / 2)}
          fill={TONE}
          fillOpacity={0.35}
          stroke={TONE}
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
          style={{ pointerEvents: "all", cursor: "move" }}
          onPointerDown={start("origin")}
        >
          <title>Drag to set the model origin</title>
        </circle>
      </g>
    </svg>
  );
}

function snap(radians: number, degrees: number): number {
  const step = (degrees * Math.PI) / 180;
  return Math.round(radians / step) * step;
}
