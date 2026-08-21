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
 *     photograph → results overlay → interaction surface → contours → ROI handles → datum
 *
 * The interaction surface is the only full-frame target (see `useCanvasInteraction`); the
 * layers above it carry their own small ones, so a contour stroke, an ROI corner and a datum
 * handle each win over the background without competing with each other.
 */

import {
  ImageStage,
  MeasureOverlay,
  StageReadout,
  StageToolbar,
  imageViewBox,
  toneColor,
  useStage,
} from "@vitavision/lab-ui";
import type { MeasurePrimitive } from "@vitavision/lab-ui";
import { useCallback, useEffect, useMemo, useState } from "react";

import type { ImageOut } from "../api/backend";
import { useLab } from "../state/LabContext";
import { ContourLayer } from "./ContourLayer";
import { DatumLayer } from "./DatumLayer";
import { ImageLayer } from "./ImageLayer";
import { LayersMenu, ToolGroup } from "./CanvasControls";
import { RoiLayer } from "./RoiLayer";
import { useCanvasInteraction } from "./useCanvasInteraction";

const BAND = toneColor("warn");

export function CanvasStage({ image }: { image: ImageOut }) {
  const { view, setView, tool, setTool, layers, setLayer, contourSelection } = useLab();
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
      image={size}
      view={view}
      onView={setView}
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
  const { overlay, roi, setRoi, roiMode, contourSelection, frameHandles, layers, tool, canvas } =
    useLab();

  /* The panel commands the canvas through this — "frame this contour" is an inspector
   * action with a canvas effect, and the stage is the only thing that knows the viewport's
   * size, so the panel borrows its `frame` rather than reconstructing the box. */
  useEffect(() => {
    canvas.current = { frame: stage.frame, fit: stage.fit };
    return () => {
      canvas.current = null;
    };
  }, [canvas, stage.frame, stage.fit]);

  const interaction = useCanvasInteraction({
    roi,
    onRoi: setRoi,
    roiEditable: roiMode,
    tool,
    contours: contourSelection?.contours ?? null,
    onSelect: contourSelection?.onSelect ?? null,
  });

  const shownRoi = interaction.draftRoi ?? roi;
  const primitives: MeasurePrimitive[] = layers.model ? overlay : [];

  return (
    <>
      <ImageLayer image={image} />

      <MeasureOverlay
        nativeWidth={image.width}
        nativeHeight={image.height}
        primitives={primitives}
        strokeScale={stage.view.scale}
        className="pointer-events-none absolute inset-0 h-full w-full"
      />

      {/* The one full-frame target. It declines any press it has no use for, and a declined
          press bubbles to the stage and pans. */}
      <svg
        viewBox={imageViewBox(image)}
        className="absolute inset-0 h-full w-full"
        style={{ pointerEvents: "none" }}
      >
        <rect
          x={0}
          y={0}
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

      {contourSelection && (
        <ContourLayer
          selection={contourSelection}
          layers={layers}
          sweeping={tool === "marquee"}
          onSweep={interaction.startBand}
        />
      )}

      {layers.roi && shownRoi && (
        <RoiLayer roi={shownRoi} editable={roiMode} onGrab={interaction.grabHandle} />
      )}

      {frameHandles && layers.datum && <DatumLayer handles={frameHandles} />}
    </>
  );
}
