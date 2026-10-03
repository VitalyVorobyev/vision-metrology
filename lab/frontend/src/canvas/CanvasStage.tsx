/**
 * The image and everything drawn over it — one component, mounted once by the shell and
 * shared by every workspace.
 *
 * Every layer is a child of the **same** transform (`ImageStage`), which is the whole point:
 * the previous arrangement had the photograph inside the transform and the interactive
 * layers outside it, because the old canvas captured every `pointerdown` for panning. That
 * bought interactivity at the price of registration — contours, ROI and datum stayed pinned
 * at fit scale while the image zoomed and panned away underneath them, and a window resize
 * moved the two apart even at rest.
 *
 * Stacking order is the order a hand expects to reach things, and it is load-bearing:
 *
 *     photograph → results overlay → sweep surface → region → contours → datum
 *
 * The sweep surface is the bare-image target (see `useCanvasInteraction`). The region sits
 * under the contours so a contour inside it stays clickable and hoverable; its interior moves
 * it, and with the box tool its own surface draws a new one. The layers above carry only
 * their own small targets, so a contour stroke and a datum handle each win over the
 * background without competing with each other.
 */

import {
  ImageLayer,
  ImageStage,
  MeasureOverlay,
  RectRoiEditor,
  StageReadout,
  StageToolbar,
  imageViewBox,
  useStage,
} from "@vitavision/stage2d";
import type { MeasurePrimitive, Rect } from "@vitavision/stage2d";
import { Skeleton, toneColor } from "@vitavision/ui";
import { useCallback, useMemo, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { ImageOut } from "../api/backend";
import { useImageUrl, useLazyImageUrl } from "../hooks/useImageUrl";
import { useLab } from "../state/LabContext";
import { ContourLayer } from "./ContourLayer";
import { DatumLayer } from "./DatumLayer";
import { LayersMenu, ToolGroup } from "./CanvasControls";
import { MIN_ROI, rectToRoi, roiToRect } from "./roi";
import { useCanvasInteraction } from "./useCanvasInteraction";

const BAND = toneColor("warn");

/** The `preview` tier's long edge (see the backend's media tiers). */
const PREVIEW_LONG_EDGE = 1024;
/**
 * Above about 4× a bilinear filter is drawing something the sensor never recorded, and on a
 * metrology bench the sensor's own samples are the thing worth looking at.
 */
const PIXELATED_ABOVE = 4;

export function CanvasStage({ image }: { image: ImageOut }) {
  const { view, setView, tool, setTool, layers, setLayer, contourSelection, canvas } = useLab();
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);

  const size = useMemo(
    () => ({ width: image.width, height: image.height }),
    [image.width, image.height],
  );

  const clearSelection = useCallback(() => {
    contourSelection?.onSelect([], "replace");
  }, [contourSelection]);

  return (
    <ImageStage
      // The panels command the canvas ("frame this contour") through the stage's own handle.
      ref={canvas}
      image={size}
      view={view}
      onView={setView}
      // A frame opens whole, however small: the first question about a frame is what is on
      // it, not what its pixels look like at 1:1.
      initialView="fit"
      onHover={setCursor}
      onBackgroundClick={clearSelection}
      // The arrows step the contour inventory in this app (see `routes/teach`), which is a
      // better use of them than a pan that dragging already does.
      panKeys={false}
      label={`${image.filename} — image canvas`}
      toolbar={
        <StageToolbar>
          <ToolGroup tool={tool} onTool={setTool} />
          <LayersMenu layers={layers} onLayer={setLayer} />
        </StageToolbar>
      }
      readout={<StageReadout cursor={cursor} />}
    >
      <Layers image={image} />
    </ImageStage>
  );
}

/**
 * Inside the stage, so it can read the transform.
 *
 * `ImageStage` provides its context to its children, and this is where the app's own layers
 * and the one interaction surface they share are assembled.
 */
function Layers({ image }: { image: ImageOut }) {
  const stage = useStage();
  const { overlay, roi, setRoi, roiMode, contourSelection, frameHandles, layers, tool } = useLab();

  const interaction = useCanvasInteraction({
    tool,
    contours: contourSelection?.contours ?? null,
    onSelect: contourSelection?.onSelect ?? null,
  });

  /* The region as it is being dragged. The shared `roi` changes once, on release: the Teach
   * panel re-extracts when it changes, and a change per pointer move would also re-render
   * every consumer of the lab state for a box nobody has let go of yet. */
  const [draft, setDraft] = useState<Rect | null>(null);
  const region = draft ?? (roi === null ? null : roiToRect(roi));
  const commitRegion = useCallback(
    (next: Rect) => {
      setDraft(null);
      setRoi(rectToRoi(next));
    },
    [setRoi],
  );
  const drawRegion = roiMode && (tool === "box" || roi === null);

  /* A sweep outranks the region's interior, which would otherwise take the press as a move:
   * contours are usually inside the region (that is what a region is for), so shift-dragging
   * over them has to select them. Offered in the capture phase, before the editor hears the
   * press; a handle is the one target that outranks a sweep. */
  const sweepOverRegion = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.target instanceof Element && event.target.closest("[data-handle]")) return;
    interaction.startBand(event);
  };

  const primitives: MeasurePrimitive[] = layers.model ? overlay : [];

  return (
    <>
      <Photograph image={image} />

      <MeasureOverlay
        nativeWidth={image.width}
        nativeHeight={image.height}
        primitives={primitives}
        strokeScale={stage.view.scale}
        className="pointer-events-none absolute inset-0 h-full w-full"
      />

      {/* The bare-image target. It declines any press it has no use for, and a declined
          press bubbles to the stage and pans. */}
      <svg
        viewBox={imageViewBox(image)}
        className="absolute inset-0 h-full w-full"
        style={{ pointerEvents: "none" }}
      >
        <rect
          x={-0.5}
          y={-0.5}
          width={image.width}
          height={image.height}
          fill="transparent"
          style={{ pointerEvents: "all", cursor: interaction.cursor }}
          {...interaction.surface}
        />
        {interaction.band && (
          <rect
            x={interaction.band.x}
            y={interaction.band.y}
            width={interaction.band.width}
            height={interaction.band.height}
            fill={BAND}
            fillOpacity={0.12}
            stroke={BAND}
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
            style={{ pointerEvents: "none" }}
          />
        )}
      </svg>

      {/* Hidden with its layer, except while a box is being drawn: "Redraw" with the layer
          off would otherwise do nothing visible at all. */}
      {(layers.roi || drawRegion) && (
        <div className="pointer-events-none absolute inset-0" onPointerDownCapture={sweepOverRegion}>
          <RectRoiEditor
            value={region}
            onValueChange={setDraft}
            onCommit={commitRegion}
            editable={roiMode}
            draw={drawRegion}
            minSize={MIN_ROI}
          />
        </div>
      )}

      {contourSelection && (
        <ContourLayer
          selection={contourSelection}
          layers={layers}
          sweeping={tool === "marquee"}
          onSweep={interaction.startBand}
        />
      )}

      {frameHandles && layers.datum && <DatumLayer handles={frameHandles} />}
    </>
  );
}

/**
 * The photograph: the preview tier until the stage would magnify it, then full resolution.
 *
 * stage2d's `ImageLayer` makes that swap, but it needs the full tier's URL up front, and on
 * the desktop asking for a tier's URL is what renders it. So the full URL is resolved only
 * once the preview would be magnified (`ImageLayer`'s own rule), and kept for this frame
 * after that, so zooming back out does not drop the pixels already fetched.
 */
function Photograph({ image }: { image: ImageOut }) {
  const stage = useStage();
  // Tiers scale the long edge down to `PREVIEW_LONG_EDGE`, and never up.
  const previewWidth =
    image.width * Math.min(1, PREVIEW_LONG_EDGE / Math.max(image.width, image.height));
  const magnified = stage.box.width > 0 && stage.view.scale * image.width > previewWidth;
  const [fullFor, setFullFor] = useState<string | null>(null);
  if (magnified && fullFor !== image.id) setFullFor(image.id);

  const preview = useImageUrl(image.id, "preview");
  const full = useLazyImageUrl(image.id, "full", fullFor === image.id);
  const error = preview.error ?? full.error;

  return (
    <>
      {preview.url !== null && (
        <ImageLayer
          key={image.id}
          // Until the full tier's URL exists, the preview stands in for it.
          src={full.url ?? preview.url}
          preview={{ src: preview.url, width: previewWidth }}
          alt={image.filename}
          pixelatedAbove={PIXELATED_ABOVE}
        />
      )}

      {/* The image is a file the webview loads; while that is in flight the frame should
          say so rather than sit empty and look broken. */}
      {preview.loading && preview.url === null && (
        <Skeleton className="pointer-events-none absolute inset-0 h-full w-full" />
      )}
      {error !== null && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-defect/10 px-3 py-2 text-xs text-defect">
          {error}
        </div>
      )}
    </>
  );
}
