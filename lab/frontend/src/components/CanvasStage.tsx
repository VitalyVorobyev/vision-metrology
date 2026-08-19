import {
  FULL_TIER_ZOOM,
  MeasureOverlay,
  RESET_VIEW,
  ZoomPanCanvas,
  type MeasurePrimitive,
  type View,
} from "@vitavision/lab-ui";
import { useRef, useState } from "react";

import { imageTierUrl } from "../api/client";
import type { ImageOut, OverlayPrimitiveOut, Roi } from "../api/types";
import { RoiDragLayer } from "./RoiDragLayer";

/** Backend overlay primitives are already `MeasurePrimitive`-shaped (same fields, same
 * units) — this only narrows `null` to `undefined`, which JSON round-tripping introduces
 * and the lab-ui type does not carry. */
function toMeasurePrimitive(p: OverlayPrimitiveOut): MeasurePrimitive {
  return Object.fromEntries(
    Object.entries(p).filter(([, v]) => v !== null).map(([k, v]) => [k, v ?? undefined]),
  ) as unknown as MeasurePrimitive;
}

export function CanvasStage({
  image,
  overlay,
  roiMode,
  onRoiChange,
  roiPreview,
}: {
  image: ImageOut;
  overlay: OverlayPrimitiveOut[];
  roiMode: boolean;
  onRoiChange?: (roi: Roi) => void;
  /** The last-drawn/loaded ROI, shown as a dashed reference rectangle even outside
   * `roiMode` (e.g. once a model has been taught). */
  roiPreview?: Roi | null;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<View>(RESET_VIEW);
  const tier = view.zoom > FULL_TIER_ZOOM ? "full" : "preview";

  const primitives: MeasurePrimitive[] = overlay.map(toMeasurePrimitive);
  if (roiPreview) {
    const [x, y, w, h] = roiPreview;
    primitives.push(
      { kind: "segment", tone: "muted", dashed: true, x1: x, y1: y, x2: x + w, y2: y },
      { kind: "segment", tone: "muted", dashed: true, x1: x + w, y1: y, x2: x + w, y2: y + h },
      { kind: "segment", tone: "muted", dashed: true, x1: x + w, y1: y + h, x2: x, y2: y + h },
      { kind: "segment", tone: "muted", dashed: true, x1: x, y1: y + h, x2: x, y2: y },
    );
  }

  return (
    <div ref={frameRef} className="relative h-full w-full">
      <ZoomPanCanvas view={view} onView={setView} nativeWidth={image.width} className="h-full w-full">
        {/* eslint-disable-next-line jsx-a11y/img-redundant-alt -- key is content identity, not decoration */}
        <img
          key={`${image.id}-${tier}`}
          src={imageTierUrl(image.id, tier)}
          alt={image.filename}
          className="h-full w-full object-contain"
          draggable={false}
        />
        <MeasureOverlay
          nativeWidth={image.width}
          nativeHeight={image.height}
          primitives={primitives}
          strokeScale={view.zoom}
          className="pointer-events-none absolute inset-0 h-full w-full"
        />
      </ZoomPanCanvas>
      {roiMode && onRoiChange && (
        <RoiDragLayer
          frameRef={frameRef}
          view={view}
          nativeWidth={image.width}
          nativeHeight={image.height}
          active={roiMode}
          onRoiChange={onRoiChange}
        />
      )}
    </div>
  );
}
