/**
 * The image and everything drawn over it — one component, mounted once by the
 * shell and shared by every workspace.
 *
 * Three layers, stacked in the order a user needs to reach them:
 *   1. the image itself, inside `ZoomPanCanvas`;
 *   2. `MeasureOverlay` — vector primitives in source-image coordinates,
 *      `pointer-events: none`, so panning still works through it;
 *   3. the interactive layers (ROI drag, contour picking, frame handles),
 *      siblings of the canvas rather than children of it, because
 *      `ZoomPanCanvas` captures every `pointerdown` for panning.
 */

import {
  FULL_TIER_ZOOM,
  MeasureOverlay,
  RESET_VIEW,
  Skeleton,
  ZoomPanCanvas,
  type MeasurePrimitive,
  type View,
} from "@vitavision/lab-ui";
import { useRef, useState } from "react";

import type { ContourSelection, FrameHandles } from "../state/LabContext";
import type { ImageOut, Roi } from "../api/backend";
import { useImageUrl } from "../hooks/useImageUrl";
import { ContourPickLayer } from "./ContourPickLayer";
import { FrameHandleLayer } from "./FrameHandleLayer";
import { RoiDragLayer } from "./RoiDragLayer";

export function CanvasStage({
  image,
  overlay,
  roiMode,
  onRoiChange,
  roiPreview,
  contourSelection,
  frameHandles,
}: {
  image: ImageOut;
  overlay: MeasurePrimitive[];
  roiMode: boolean;
  onRoiChange?: (roi: Roi) => void;
  /** The last-drawn/loaded ROI, shown as a dashed reference rectangle even outside
   * `roiMode` (e.g. once a model has been taught). */
  roiPreview?: Roi | null;
  contourSelection?: ContourSelection | null;
  frameHandles?: FrameHandles | null;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<View>(RESET_VIEW);
  const tier = view.zoom > FULL_TIER_ZOOM ? "full" : "preview";
  const { url, loading, error } = useImageUrl(image.id, tier);

  const primitives: MeasurePrimitive[] = [...overlay];
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
        {url !== null && (
          /* eslint-disable-next-line jsx-a11y/img-redundant-alt -- key is content identity, not decoration */
          <img
            key={`${image.id}-${tier}`}
            src={url}
            alt={image.filename}
            className="h-full w-full object-contain"
            draggable={false}
          />
        )}
        <MeasureOverlay
          nativeWidth={image.width}
          nativeHeight={image.height}
          primitives={primitives}
          strokeScale={view.zoom}
          className="pointer-events-none absolute inset-0 h-full w-full"
        />
      </ZoomPanCanvas>

      {contourSelection && (
        <ContourPickLayer
          frameRef={frameRef}
          view={view}
          nativeWidth={image.width}
          nativeHeight={image.height}
          selection={contourSelection}
        />
      )}
      {frameHandles && (
        <FrameHandleLayer
          frameRef={frameRef}
          view={view}
          nativeWidth={image.width}
          nativeHeight={image.height}
          handles={frameHandles}
        />
      )}
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

      {/* The image is a file the webview loads; while that is in flight the
          frame should say so rather than sit empty and look broken. */}
      {loading && url === null && (
        <Skeleton className="pointer-events-none absolute inset-0 h-full w-full" />
      )}
      {error !== null && (
        <div className="absolute inset-x-0 bottom-0 bg-defect/10 px-3 py-2 text-xs text-defect">
          {error}
        </div>
      )}
    </div>
  );
}
