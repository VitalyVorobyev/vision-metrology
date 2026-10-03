/**
 * What every workspace shares: which frame is on the canvas, which model is in
 * hand, and what is drawn over the image.
 *
 * `App.tsx` used to hold all of this in `useState` and thread it through six
 * sibling tabs as props. That works while the tabs are siblings; it stops
 * working the moment they become routes, because the canvas has to outlive the
 * panel beside it — you do not want the image to unmount and re-fetch when you
 * step from Teach to Find.
 *
 * The overlay is a *slot*, not a value the shell computes: a route pushes what
 * it wants drawn and clears it on the way out. That keeps the drawing decision
 * next to the data that justifies it, and keeps the shell from knowing what a
 * caliper or a model point is.
 */

import { useQuery } from "@tanstack/react-query";
import type { MeasurePrimitive, StageView } from "@vitavision/lab-ui";
import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import type { ReactNode, RefObject } from "react";

import type { ContourStat, SelectMode } from "../canvas/contourSelection";
import { getBackend } from "../api/backend";
import type {
  CalibrationOut,
  ContourOut,
  FindRequestFull,
  ImageOut,
  MatchOut,
  ModelOut,
  Roi,
} from "../api/backend";

/**
 * The interactive contour layer the Teach step puts on the canvas.
 *
 * `kept` and `selected` are deliberately separate. `kept` is what the model will be built
 * from; `selected` is the working set the keep actions operate on. Conflating them — which
 * is what the first version did, where a click *was* a drop — leaves no way to look at a
 * contour without changing the model, and no way to act on several at once.
 */
export interface ContourSelection {
  contours: ContourOut[];
  /** Per-contour facts, derived once by the panel and shared with the canvas. */
  stats: ContourStat[];
  /** Ids currently kept. Everything else is drawn as excluded. */
  kept: ReadonlySet<number>;
  /** Ids in the working set. */
  selected: ReadonlySet<number>;
  /** The one under the pointer, wherever the pointer is — canvas or list. */
  hovered: number | null;
  /**
   * Ids in the inspector's current sort order. The canvas needs it so a shift-click range
   * means the same rows the list would have ranged over; ordering by id instead would
   * select something else entirely whenever the list is sorted by length or strength.
   */
  order: number[];
  onHover: (id: number | null) => void;
  onSelect: (ids: number[], mode: SelectMode) => void;
  onKeep: (ids: number[], keep: boolean) => void;
}

/** What a drag on the canvas means. Handles and contours stay live in every mode. */
export type CanvasTool = "pan" | "box" | "marquee";

/** What is drawn over the image. Every layer is answerable for its own pixels. */
export interface LayerVisibility {
  roi: boolean;
  kept: boolean;
  dropped: boolean;
  vertices: boolean;
  datum: boolean;
  model: boolean;
}

export const DEFAULT_LAYERS: LayerVisibility = {
  roi: true,
  kept: true,
  dropped: true,
  vertices: true,
  datum: true,
  model: true,
};

/** A draggable handle on the canvas — the model origin and its 0° direction. */
export interface FrameHandles {
  origin: [number, number];
  angle: number;
  onOrigin: (p: [number, number]) => void;
  onAngle: (a: number) => void;
}

interface LabState {
  images: ImageOut[];
  models: ModelOut[];
  calibrations: CalibrationOut[];
  imagesLoading: boolean;

  selectedImage: ImageOut | null;
  selectImage: (id: string | null) => void;
  selectedModel: ModelOut | null;
  selectModel: (id: string | null) => void;

  roi: Roi | null;
  setRoi: (roi: Roi | null) => void;

  overlay: MeasurePrimitive[];
  setOverlay: (o: MeasurePrimitive[]) => void;

  matches: MatchOut[];
  /**
   * The request that produced `matches`, kept so a later step can reproduce
   * the *same* result set.
   *
   * Verify rectifies a found instance by index, and `rectify` re-runs the
   * search internally. Re-running it with different settings would renumber
   * the matches, so index `2` in the table and index `2` in the crop cache
   * would be different instances — a verification panel showing the wrong
   * part, which is worse than none.
   */
  lastFind: FindRequestFull | null;
  setMatches: (m: MatchOut[], req: FindRequestFull) => void;
  /** Which match the user is pointing at, so table and canvas agree. */
  highlightedMatch: number | null;
  setHighlightedMatch: (i: number | null) => void;

  contourSelection: ContourSelection | null;
  setContourSelection: (s: ContourSelection | null) => void;

  frameHandles: FrameHandles | null;
  setFrameHandles: (h: FrameHandles | null) => void;

  /** Draw an ROI rectangle and report drags into it. */
  roiMode: boolean;
  setRoiMode: (on: boolean) => void;

  /**
   * The canvas transform, held here rather than inside the canvas so a panel can command
   * it — "frame this contour" is an inspector action with a canvas effect. `null` until the
   * viewport has been measured, which is what lets the stage choose the opening view.
   */
  view: StageView | null;
  setView: (view: StageView | null) => void;

  layers: LayerVisibility;
  setLayer: (key: keyof LayerVisibility, on: boolean) => void;

  tool: CanvasTool;
  setTool: (tool: CanvasTool) => void;

  /**
   * What a panel may ask the canvas to do, published by the canvas while it is mounted.
   *
   * A ref rather than state because it is a set of imperative handles, not a value to
   * render: putting them in the context value would re-render every consumer whenever the
   * canvas remounted, to hand back functions nothing displays. `null` when no canvas is up
   * (the Library workspace replaces it), which callers must treat as "not now" rather than
   * as an error.
   */
  canvas: RefObject<CanvasCommands | null>;
}

export interface CanvasCommands {
  /** Put a rect (image coordinates) on screen with a margin. */
  frame: (rect: { x: number; y: number; width: number; height: number }, pad?: number) => void;
  fit: () => void;
}

const Ctx = createContext<LabState | null>(null);

export function LabProvider({ children }: { children: ReactNode }) {
  const backend = getBackend();
  const imagesQuery = useQuery({ queryKey: ["images"], queryFn: () => backend.listImages() });
  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: () => backend.listModels() });
  const calibrationsQuery = useQuery({
    queryKey: ["calibrations"],
    queryFn: () => backend.listCalibrations(),
  });

  const [selectedImageId, setSelectedImageId] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [roi, setRoi] = useState<Roi | null>(null);
  const [overlay, setOverlay] = useState<MeasurePrimitive[]>([]);
  const [matches, setMatchList] = useState<MatchOut[]>([]);
  const [lastFind, setLastFind] = useState<FindRequestFull | null>(null);
  const setMatches = useCallback((m: MatchOut[], req: FindRequestFull) => {
    setMatchList(m);
    setLastFind(req);
  }, []);
  const [highlightedMatch, setHighlightedMatch] = useState<number | null>(null);
  const [contourSelection, setContourSelection] = useState<ContourSelection | null>(null);
  const [frameHandles, setFrameHandles] = useState<FrameHandles | null>(null);
  const [roiMode, setRoiMode] = useState(false);
  const [view, setView] = useState<StageView | null>(null);
  const [layers, setLayers] = useState<LayerVisibility>(DEFAULT_LAYERS);
  const [tool, setTool] = useState<CanvasTool>("pan");

  const canvas = useRef<CanvasCommands | null>(null);

  const setLayer = useCallback((key: keyof LayerVisibility, on: boolean) => {
    setLayers((current) => ({ ...current, [key]: on }));
  }, []);

  const images = useMemo(() => imagesQuery.data ?? [], [imagesQuery.data]);
  const models = useMemo(() => modelsQuery.data ?? [], [modelsQuery.data]);
  const calibrations = useMemo(() => calibrationsQuery.data ?? [], [calibrationsQuery.data]);

  const selectImage = useCallback((id: string | null) => {
    setSelectedImageId(id);
    // A different frame invalidates everything drawn over the old one. Leaving
    // it up is worse than clearing it: an overlay from another image looks like
    // a result about this one.
    setRoi(null);
    setOverlay([]);
    setMatchList([]);
    setLastFind(null);
    setHighlightedMatch(null);
    setContourSelection(null);
    setFrameHandles(null);
    // A different frame is a different size and a different subject; keeping a pan offset
    // across it would open the new one scrolled to a corner of the old one.
    setView(null);
  }, []);

  const value = useMemo<LabState>(
    () => ({
      images,
      models,
      calibrations,
      imagesLoading: imagesQuery.isLoading,
      selectedImage: images.find((i) => i.id === selectedImageId) ?? null,
      selectImage,
      selectedModel: models.find((m) => m.id === selectedModelId) ?? null,
      selectModel: setSelectedModelId,
      roi,
      setRoi,
      overlay,
      setOverlay,
      matches,
      lastFind,
      setMatches,
      highlightedMatch,
      setHighlightedMatch,
      contourSelection,
      setContourSelection,
      frameHandles,
      setFrameHandles,
      roiMode,
      setRoiMode,
      view,
      setView,
      layers,
      setLayer,
      tool,
      setTool,
      canvas,
    }),
    [
      images,
      models,
      calibrations,
      imagesQuery.isLoading,
      selectedImageId,
      selectedModelId,
      selectImage,
      roi,
      overlay,
      matches,
      lastFind,
      setMatches,
      highlightedMatch,
      contourSelection,
      frameHandles,
      roiMode,
      view,
      layers,
      setLayer,
      tool,
    ],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useLab(): LabState {
  const ctx = useContext(Ctx);
  if (ctx === null) throw new Error("useLab must be used inside <LabProvider>.");
  return ctx;
}
