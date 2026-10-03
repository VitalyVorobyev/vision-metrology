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
 * It is drawn as the overlay grammar's origin glyph: a ring with its i and j axes, in the
 * `model` role colour, with a halo under every stroke so it holds on a white or a black
 * part. The handles are sized in **screen** pixels (`useScreenPx`): at around three image
 * pixels a control reads as decoration. And it is not the contours' colour: in that colour,
 * on a canvas covered in contours, it was one more inert line among a hundred and
 * sixty-six.
 */

import { imageViewBox, overlayRole, useScreenPx, useStage, useStageDrag } from "@vitavision/stage2d";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { FrameHandles } from "../state/LabContext";

/** The i arm, to the 0° handle, in screen pixels: the arm is a control, not a measurement. */
const ARM_PX = 78;
/** The j arm. Shorter, so the two axes are told apart without labels. */
const J_ARM_PX = 30;
/** The origin ring's radius, the overlay grammar's origin glyph. */
const ORIGIN_PX = 7;
const TIP_PX = 9;
/** Held modifiers snap the arm to this many degrees. */
const SNAP_DEGREES = 15;

/** The datum is the model's own frame, so it takes the `model` role, as the model's points do. */
const COLOUR = overlayRole("model");
const HALO = overlayRole("halo");

export function DatumLayer({ handles }: { handles: FrameHandles }) {
  const stage = useStage();
  const px = useScreenPx();
  const startDrag = useStageDrag();

  const [ox, oy] = handles.origin;
  const cos = Math.cos(handles.angle);
  const sin = Math.sin(handles.angle);
  const ring = px(ORIGIN_PX);
  const tip: [number, number] = [ox + px(ARM_PX) * cos, oy + px(ARM_PX) * sin];
  // +90° from i in image coordinates, where y points down. Both axes start at the ring, so
  // its centre stays clear.
  const jTip: [number, number] = [ox - px(J_ARM_PX) * sin, oy + px(J_ARM_PX) * cos];
  const axes =
    `M${ox + ring * cos} ${oy + ring * sin}L${tip[0]} ${tip[1]}` +
    `M${ox - ring * sin} ${oy + ring * cos}L${jTip[0]} ${jTip[1]}`;

  /* A window-level drag (stage2d's `useStageDrag`), so the handle keeps following a pointer
   * that outruns it. */
  const grab = (mode: "origin" | "angle") => (event: ReactPointerEvent<SVGElement>) => {
    if (event.button !== 0 || stage.panMode) return;
    const { onOrigin, onAngle } = handles;
    startDrag(event, {
      onMove: (p, move) => {
        if (mode === "origin") {
          onOrigin([p.x, p.y]);
          return;
        }
        const raw = Math.atan2(p.y - oy, p.x - ox);
        onAngle(move.shiftKey ? snap(raw, SNAP_DEGREES) : raw);
      },
    });
  };

  return (
    <svg
      viewBox={imageViewBox(stage.image)}
      className="pointer-events-none absolute inset-0 h-full w-full overflow-visible"
    >
      <g fill="none" strokeLinecap="round">
        <path d={axes} stroke={HALO} strokeWidth={px(1.5 + 2)} />
        <path d={axes} stroke={COLOUR} strokeWidth={px(1.5)} />
      </g>

      <circle
        data-datum="angle"
        cx={tip[0]}
        cy={tip[1]}
        r={px(TIP_PX / 2)}
        fill={COLOUR}
        stroke={HALO}
        strokeWidth={px(1)}
        style={{ pointerEvents: "all", cursor: "grab" }}
        onPointerDown={grab("angle")}
      >
        <title>Drag to set the model&apos;s 0° direction (hold shift to snap to 15°)</title>
      </circle>

      <circle
        cx={ox}
        cy={oy}
        r={ring}
        fill="none"
        stroke={HALO}
        strokeWidth={px(1.5 + 2)}
      />
      <circle
        data-datum="origin"
        cx={ox}
        cy={oy}
        r={ring}
        fill={COLOUR}
        fillOpacity={0.25}
        stroke={COLOUR}
        strokeWidth={px(1.5)}
        style={{ pointerEvents: "all", cursor: "move" }}
        onPointerDown={grab("origin")}
      >
        <title>Drag to set the model origin</title>
      </circle>
    </svg>
  );
}

function snap(radians: number, degrees: number): number {
  const step = (degrees * Math.PI) / 180;
  return Math.round(radians / step) * step;
}
