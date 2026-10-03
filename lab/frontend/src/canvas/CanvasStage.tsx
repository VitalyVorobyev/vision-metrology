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
 *     photograph → results overlay → surface → region → contours → sweep band → datum
 *
 * The surface (stage2d's `StageSurface`) is the bare-image target: it starts a sweep and
 * declines everything else, so the stage pans. The region sits under the contours so a
 * contour inside it stays clickable and hoverable; its interior moves it, and with the box
 * tool its own surface draws a new one. The contours (stage2d's `PolylineSet`) and the datum
 * carry only their own small targets, so a contour stroke and a datum handle each win over
 * the background without competing with each other.
 */

import {
  ImageLayer,
  ImageStage,
  MeasureOverlay,
  PolylineSet,
  RectRoiEditor,
  StageReadout,
  StageSurface,
  StageToolbar,
  buildPolylineIndex,
  imageViewBox,
  overlayRole,
  polylinesInRect,
  useScreenPx,
  useStage,
  useStageDrag,
} from "@vitavision/stage2d";
import type {
  MeasurePrimitive,
  Point,
  PolylineId,
  PolylineSetItem,
  Rect,
  StageDrag,
  StagePress,
} from "@vitavision/stage2d";
import { Skeleton } from "@vitavision/ui";
import { useCallback, useMemo, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { ImageOut } from "../api/backend";
import { useImageUrl, useLazyImageUrl } from "../hooks/useImageUrl";
import type { SelectMode } from "../state/contourInventory";
import { useLab } from "../state/LabContext";
import type { ContourSelection, LayerVisibility } from "../state/LabContext";
import { DatumLayer } from "./DatumLayer";
import { LayersMenu, ToolGroup } from "./CanvasControls";
import { MIN_ROI, rectToRoi, roiToRect } from "./roi";

/** A dropped contour: present, but not in play. */
const DROPPED_STROKE = overlayRole("structure");
const SELECTION = overlayRole("selection");
const VERTEX = overlayRole("label");
const HALO = overlayRole("halo");
/** Vertices become worth drawing once one image pixel is this many screen pixels. */
const VERTEX_SCALE = 3;
/** The most vertices drawn at once; past this the dots are noise and a cost. */
const MAX_VERTICES = 5000;

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

  const contours = useContourItems(contourSelection, layers);
  const selected = contourSelection?.selected;
  const hovered = contourSelection?.hovered ?? null;
  /** The contours being looked at: their vertices are drawn when the zoom allows. */
  const lookedAt = useMemo(
    () => new Set(hovered === null ? (selected ?? []) : [...(selected ?? []), hovered]),
    [selected, hovered],
  );
  const sweep = useSweep({
    items: contours,
    onSelect: contourSelection?.onSelect ?? null,
    marquee: tool === "marquee",
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

  /* A sweep outranks the region's interior, which would otherwise take the press as a move,
   * and a contour's own click: contours are usually inside the region (that is what a region
   * is for), so shift-dragging over them has to select them. Offered in the capture phase,
   * before either layer hears the press; a region handle is the one target that outranks a
   * sweep. */
  const offerSweep = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.target instanceof Element && event.target.closest("[data-handle]")) return;
    sweep.offer(event);
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
          press reaches the stage, which pans. */}
      <StageSurface onPress={sweep.onPress} cursor={sweep.cursor} />

      <div className="pointer-events-none absolute inset-0" onPointerDownCapture={offerSweep}>
        {/* Hidden with its layer, except while a box is being drawn: "Redraw" with the layer
            off would otherwise do nothing visible at all. */}
        {(layers.roi || drawRegion) && (
          <RectRoiEditor
            value={region}
            onValueChange={setDraft}
            onCommit={commitRegion}
            editable={roiMode}
            draw={drawRegion}
            minSize={MIN_ROI}
          />
        )}

        {contourSelection && contours && (
          <PolylineSet
            label="Contours"
            items={contours}
            selected={contourSelection.selected}
            hovered={contourSelection.hovered}
            onHover={(id) => contourSelection.onHover(contourId(id))}
            onSelect={(ids, mode) => contourSelection.onSelect(ids.map(Number), mode)}
            // The vertices are drawn below, where they show on a selected line too.
            vertexScale={Infinity}
          />
        )}
      </div>

      {contourSelection && contours && layers.vertices && stage.view.scale >= VERTEX_SCALE && (
        <ContourVertices items={contours} ids={lookedAt} />
      )}

      {sweep.band && <SweepBand rect={sweep.band} />}

      {frameHandles && layers.datum && <DatumLayer handles={frameHandles} />}
    </>
  );
}

/**
 * The contours as `PolylineSet` items: kept ones in the set's own `feature` colour, dropped
 * ones dashed in `structure`, each kind hidden with its layer.
 *
 * Keyed on the contours and the keep set, not on the selection object, which is rebuilt on
 * every hover: a new item list rebuilds the set's spatial index.
 */
function useContourItems(
  selection: ContourSelection | null,
  layers: LayerVisibility,
): PolylineSetItem[] | null {
  const contours = selection?.contours ?? null;
  const kept = selection?.kept ?? null;
  const showKept = layers.kept;
  const showDropped = layers.dropped;
  return useMemo(() => {
    if (contours === null || kept === null) return null;
    const items: PolylineSetItem[] = [];
    for (const contour of contours) {
      const isKept = kept.has(contour.id);
      if (isKept ? !showKept : !showDropped) continue;
      const line = { id: contour.id, points: contour.points, closed: contour.closed };
      items.push(isKept ? line : { ...line, stroke: DROPPED_STROKE, dashed: true });
    }
    return items;
  }, [contours, kept, showKept, showDropped]);
}

/**
 * The samples of the contours being looked at (hovered or selected), once a pixel is big
 * enough to hold a dot. "7365 points" is otherwise a number with nothing behind it; drawing
 * all of them at once is both unreadable and slow.
 *
 * Drawn here rather than by `PolylineSet`, whose dots are the selection colour 3 px wide: on a
 * selected line, which is the selection colour 2.5 px wide, they disappear exactly where they
 * are wanted. These are the `label` role on a halo.
 */
function ContourVertices({ items, ids }: { items: PolylineSetItem[]; ids: ReadonlySet<number> }) {
  const stage = useStage();
  const px = useScreenPx();
  const d = useMemo(() => {
    let path = "";
    let count = 0;
    for (const item of items) {
      if (!ids.has(Number(item.id))) continue;
      const p = item.points;
      for (let i = 0; i + 1 < p.length && count < MAX_VERTICES; i += 2, count++) {
        path += `M${p[i]} ${p[i + 1]}h0`;
      }
    }
    return path;
  }, [items, ids]);
  if (d === "") return null;
  return (
    <svg
      viewBox={imageViewBox(stage.image)}
      className="pointer-events-none absolute inset-0 h-full w-full overflow-visible"
      aria-hidden
    >
      <g fill="none" strokeLinecap="round">
        <path d={d} stroke={HALO} strokeWidth={px(5)} />
        <path data-vertices="" d={d} stroke={VERTEX} strokeWidth={px(3)} />
      </g>
    </svg>
  );
}

/** The ids are the backend's contour ids, which are numbers. */
function contourId(id: PolylineId | null): number | null {
  return id === null ? null : Number(id);
}

/**
 * The sweep: a rubber band that selects every drawn contour it touches.
 *
 * Shift-drag sweeps in any tool, and any drag sweeps with the marquee tool. Only the topmost
 * element under the pointer receives a press, so a band has to be startable from the bare
 * image (the surface's `onPress`) and from the layers above it (`offer`, called in the
 * capture phase): a frame with a hundred and sixty-six contours is mostly strokes, and the
 * region's interior covers most of the rest. `PolylineSet` has a band of its own, but only
 * for presses on its lines, so the lab draws one band for all three.
 *
 * The drag itself is stage2d's `useStageDrag`: it claims the press and listens on `window`,
 * so the band survives the pointer leaving the canvas.
 */
function useSweep({
  items,
  onSelect,
  marquee,
}: {
  items: PolylineSetItem[] | null;
  onSelect: ((ids: number[], mode: SelectMode) => void) | null;
  marquee: boolean;
}) {
  const stage = useStage();
  const startDrag = useStageDrag();
  const [band, setBand] = useState<Rect | null>(null);

  const wants = (shift: boolean) => items !== null && onSelect !== null && (shift || marquee);

  const drag = (from: Point, additive: boolean): StageDrag => {
    setBand({ x: from.x, y: from.y, width: 0, height: 0 });
    return {
      onMove: (point) => setBand(boxBetween(from, point)),
      onEnd: (point) => {
        setBand(null);
        // Indexed on release rather than kept: one sweep is the only reader, and the set
        // changes with every keep and drop.
        const caught = items === null ? [] : polylinesInRect(buildPolylineIndex(items), boxBetween(from, point));
        // An empty sweep clears the selection, which is how you let go of one with a tool in
        // hand rather than having to find empty background to click.
        onSelect?.(caught.map(Number), additive ? "add" : "replace");
      },
      onCancel: () => setBand(null),
    };
  };

  return {
    band,
    cursor: items !== null && marquee ? "crosshair" : undefined,
    onPress: (press: StagePress) => (wants(press.shiftKey) ? drag(press.point, press.metaKey) : null),
    offer: (event: ReactPointerEvent<Element>) => {
      if (event.button !== 0 || stage.panMode || !wants(event.shiftKey)) return;
      const from = stage.toImage({ x: event.clientX, y: event.clientY });
      startDrag(event, drag(from, event.metaKey || event.ctrlKey));
    },
  };
}

function boxBetween(a: Point, b: Point): Rect {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(b.x - a.x),
    height: Math.abs(b.y - a.y),
  };
}

/** The band, drawn as `PolylineSet` draws its own: the selection colour at 12 %. */
function SweepBand({ rect }: { rect: Rect }) {
  const stage = useStage();
  const px = useScreenPx();
  return (
    <svg
      viewBox={imageViewBox(stage.image)}
      className="pointer-events-none absolute inset-0 h-full w-full overflow-visible"
      aria-hidden
      data-sweep=""
    >
      <rect
        x={rect.x}
        y={rect.y}
        width={rect.width}
        height={rect.height}
        fill={SELECTION}
        fillOpacity={0.12}
        stroke={SELECTION}
        strokeWidth={px(1)}
      />
    </svg>
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
