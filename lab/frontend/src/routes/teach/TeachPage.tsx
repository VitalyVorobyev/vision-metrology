/**
 * Teaching, as a thing you can look at while you do it.
 *
 * The screen this replaces was three numbered instructions whose bodies were read-only
 * sentences: it told you what to do next and nothing about what you had. A hundred and
 * sixty-six contours were on the image in one colour, seven thousand points were a number,
 * and the only gesture was "click a contour to drop it" — so there was no way to look at one
 * without changing the model, no way to act on several, and no way to step through them.
 *
 * What is here instead is the work: the region as four editable numbers, the extraction and
 * whether it is still current, the contour inventory with the two facts that separate the
 * part from its background, the datum, and — once it exists — what the model actually came
 * out as, still drawn over the contours it was built from.
 */

import { Button, Callout, ErrorBox } from "@vitavision/lab-ui";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";

import { getBackend } from "../../api/backend";
import type { ContourOut, ModelOut, Roi } from "../../api/backend";
import {
  applyKeep,
  applySelect,
  boundsOf,
  describeContours,
  filterContours,
  invertKeep,
  pointsIn,
  sortContours,
  stepThrough,
  type KeepFilter,
  type SelectMode,
  type SortKey,
} from "../../canvas/contourSelection";
import { sameRoi } from "../../canvas/roiEdit";
import { modelOverlay } from "../../overlay/modelOverlay";
import { RecognizeShell } from "../RecognizeShell";
import { useLab } from "../../state/LabContext";
import { ContourSection } from "./ContourSection";
import { DatumSection } from "./DatumSection";
import { ExtractSection } from "./ExtractSection";
import { ModelSection } from "./ModelSection";
import { RoiSection } from "./RoiSection";

/** An extraction, with the inputs that give its contour ids meaning. */
interface Preview {
  contours: ContourOut[];
  roi: Roi;
  minContrast: number;
}

/** How long to wait after a drag before re-extracting on its own. */
const AUTO_EXTRACT_MS = 350;

export function TeachPage() {
  const backend = getBackend();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const {
    selectedImage,
    roi,
    setRoi,
    setRoiMode,
    setContourSelection,
    setFrameHandles,
    setOverlay,
    selectModel,
    setTool,
    tool,
    canvas,
  } = useLab();

  const [minContrast, setMinContrast] = useState(0.1);
  const [numLevels, setNumLevels] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [kept, setKept] = useState<Set<number>>(new Set());
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [anchor, setAnchor] = useState<number | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [origin, setOrigin] = useState<[number, number] | null>(null);
  const [angle, setAngle] = useState(0);
  const [sort, setSort] = useState<SortKey>("length");
  const [filter, setFilter] = useState<KeepFilter>("all");
  const [follow, setFollow] = useState(false);
  const [model, setModel] = useState<ModelOut | null>(null);

  // Drawing a box is what this step is for, so the ROI layer is editable the whole time —
  // including after a model is built, so the next one can be framed without a mode switch.
  useEffect(() => {
    setRoiMode(true);
    return () => setRoiMode(false);
  }, [setRoiMode]);

  const stats = useMemo(
    () => (preview === null ? [] : describeContours(preview.contours)),
    [preview],
  );
  const ordered = useMemo(() => sortContours(stats, sort), [stats, sort]);
  const visible = useMemo(() => filterContours(ordered, filter, kept), [ordered, filter, kept]);
  /* The canvas ranges over the *visible* order, so a shift-click means the rows the list
   * would have ranged over — an id-ordered range would select something else entirely
   * whenever the list is sorted by length or strength, which is most of the time. */
  const order = useMemo(() => visible.map((stat) => stat.id), [visible]);

  const totalPoints = useMemo(() => stats.reduce((sum, stat) => sum + stat.points, 0), [stats]);
  const keptPoints = useMemo(() => pointsIn(stats, kept), [stats, kept]);
  const selectedPoints = useMemo(() => pointsIn(stats, selected), [stats, selected]);
  const keptBounds = useMemo(
    () => boundsOf(stats.filter((stat) => kept.has(stat.id))),
    [stats, kept],
  );

  /** Whether the preview still describes the current inputs — see `ExtractSection`. */
  const stale =
    preview !== null && (!sameRoi(preview.roi, roi) || preview.minContrast !== minContrast);
  const curated = preview !== null && kept.size !== preview.contours.length;

  const extract = useMutation({
    mutationFn: async () => {
      const request = { image_id: selectedImage!.id, roi: roi as Roi, min_contrast: minContrast };
      const started = performance.now();
      const response = await backend.teachPreview(request);
      return { response, elapsed: performance.now() - started, request };
    },
    onSuccess: ({ response, elapsed, request }) => {
      setPreview({ contours: response.contours, roi: request.roi, minContrast: request.min_contrast });
      setElapsedMs(elapsed);
      // Everything is kept by default: the rectangle's own answer, which the user then
      // subtracts from. Starting empty would make the common case — the box is already
      // right — into work.
      setKept(new Set(response.contours.map((contour) => contour.id)));
      setSelected(new Set());
      setAnchor(null);
      setOrigin((current) =>
        current ?? [request.roi[0] + request.roi[2] / 2, request.roi[1] + request.roi[3] / 2],
      );
    },
  });

  /*
   * Re-extract on its own only while there is nothing to lose. Contour ids are positions in
   * the extraction, so a re-run renumbers them and any curation has to be discarded with
   * it; doing that silently to someone who has just spent a minute picking edges is worse
   * than making them press a button. Until they have picked anything, though, a stale
   * preview is only a stale picture, and asking them to refresh it by hand is friction.
   */
  const extractMutate = extract.mutate;
  useEffect(() => {
    if (!stale || curated || extract.isPending || roi === null) return;
    const timer = window.setTimeout(() => extractMutate(), AUTO_EXTRACT_MS);
    return () => window.clearTimeout(timer);
  }, [stale, curated, extract.isPending, roi, extractMutate]);

  const onSelect = useCallback(
    (ids: number[], mode: SelectMode) => {
      setSelected((current) => {
        const next = applySelect(current, anchor, ids, mode, order);
        setAnchor(next.anchor);
        return next.selected;
      });
    },
    [anchor, order],
  );

  const onKeep = useCallback((ids: number[], keep: boolean) => {
    setKept((current) => applyKeep(current, ids, keep));
  }, []);

  // Push the interactive layers into the shared canvas while this step is up.
  useEffect(() => {
    if (preview === null) {
      setContourSelection(null);
      return;
    }
    setContourSelection({
      contours: preview.contours,
      stats,
      kept,
      selected,
      hovered,
      order,
      onHover: setHovered,
      onSelect,
      onKeep,
    });
    return () => setContourSelection(null);
  }, [preview, stats, kept, selected, hovered, order, onSelect, onKeep, setContourSelection]);

  useEffect(() => {
    if (origin === null) {
      setFrameHandles(null);
      return;
    }
    setFrameHandles({ origin, angle, onOrigin: setOrigin, onAngle: setAngle });
    return () => setFrameHandles(null);
  }, [origin, angle, setFrameHandles]);

  /** Put a set of contours on screen — the panel commanding the canvas. */
  const frameIds = useCallback(
    (ids: ReadonlySet<number>) => {
      const bounds = boundsOf(stats.filter((stat) => ids.has(stat.id)));
      // The stage owns the viewport measurement, so the framing is its arithmetic, not a
      // reconstruction from the transform.
      if (bounds) canvas.current?.frame(bounds, 0.35);
    },
    [stats, canvas],
  );

  const frameSelection = useCallback(() => frameIds(selected), [frameIds, selected]);

  useEffect(() => {
    if (!follow || selected.size === 0) return;
    frameIds(selected);
    // Deliberately keyed on the selection alone: re-framing whenever the view changes would
    // fight the user's own pan.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [follow, selected]);

  /*
   * Keyboard, at the window: the inventory is worked from both the list and the image, and
   * a binding that only fires when a particular element has focus is a binding that does
   * nothing right after you click a contour.
   */
  const step = useCallback(
    (delta: 1 | -1) => {
      const current = selected.size === 1 ? [...selected][0]! : null;
      const next = stepThrough(order, current, delta);
      if (next !== null) onSelect([next], "replace");
    },
    [order, selected, onSelect],
  );

  const keyState = useRef({ step, selected, onKeep, frameSelection, hasPreview: preview !== null });
  keyState.current = { step, selected, onKeep, frameSelection, hasPreview: preview !== null };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const { step, selected, onKeep, frameSelection, hasPreview } = keyState.current;
      if (!hasPreview || isTypingTarget(event.target) || event.metaKey || event.ctrlKey) return;
      switch (event.key) {
        case "ArrowDown":
          step(1);
          break;
        case "ArrowUp":
          step(-1);
          break;
        case "Delete":
        case "Backspace":
          if (selected.size === 0) return;
          onKeep([...selected], false);
          break;
        case "Enter":
          if (selected.size === 0) return;
          setKept(new Set(selected));
          break;
        case " ":
          if (selected.size === 0) return;
          setKept((current) => {
            const allKept = [...selected].every((id) => current.has(id));
            return applyKeep(current, selected, !allKept);
          });
          break;
        case "f":
        case "F":
          frameSelection();
          break;
        case "Escape":
          setSelected(new Set());
          setAnchor(null);
          break;
        default:
          return;
      }
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const teach = useMutation({
    mutationFn: async () => {
      const built = await backend.teachModel({
        image_id: selectedImage!.id,
        roi: roi as Roi,
        min_contrast: minContrast,
        num_levels: numLevels === "" ? null : Number(numLevels),
        // Curation fields only when there is a curation to send: without a preview the
        // caller has not chosen anything, and an empty selection would be a model with no
        // points rather than the rectangle's own answer.
        ...(preview !== null ? { keep_contours: [...kept], origin, reference_angle: angle } : {}),
      });
      // Draw what was learned, in the frame of the image it was learned from.
      const geometry = await backend.modelGeometry(built.id, 0, "reference");
      return { built, geometry };
    },
    onSuccess: ({ built, geometry }) => {
      void queryClient.invalidateQueries({ queryKey: ["models"] });
      selectModel(built.id);
      setModel(built);
      // The contours stay: "is the model what I picked?" is the question a build raises,
      // and clearing the evidence was what made it unanswerable.
      setOverlay(modelOverlay(geometry));
    },
  });

  const canPreview = backend.canOpenFiles();
  const buildBlocked = roi === null || (preview !== null && (kept.size === 0 || stale));

  return (
    <RecognizeShell>
      <div className="flex flex-col gap-2">
        <RoiSection
          roi={roi}
          onRoi={setRoi}
          image={{ width: selectedImage?.width ?? 0, height: selectedImage?.height ?? 0 }}
          drawing={tool === "box"}
          onRedraw={() => setTool(tool === "box" ? "pan" : "box")}
        />

        <ExtractSection
          minContrast={minContrast}
          onMinContrast={setMinContrast}
          numLevels={numLevels}
          onNumLevels={setNumLevels}
          onExtract={() => extract.mutate()}
          extracting={extract.isPending}
          error={extract.isError ? (extract.error as Error).message : null}
          stale={stale && curated}
          elapsedMs={elapsedMs}
          disabled={roi === null || !selectedImage || !canPreview}
          disabledReason={
            canPreview
              ? null
              : "Curating contours needs the desktop app; the browser build teaches from the rectangle alone."
          }
          hasPreview={preview !== null}
        />

        {preview !== null && (
          <ContourSection
            stats={stats}
            visible={visible}
            kept={kept}
            selected={selected}
            hovered={hovered}
            sort={sort}
            onSort={setSort}
            filter={filter}
            onFilter={setFilter}
            keptPoints={keptPoints}
            totalPoints={totalPoints}
            selectedPoints={selectedPoints}
            onSelect={onSelect}
            onKeep={onKeep}
            onHover={setHovered}
            onKeepAll={() => setKept(new Set(stats.map((stat) => stat.id)))}
            onDropAll={() => setKept(new Set())}
            onInvert={() => setKept(invertKeep(kept, stats))}
            onFrame={frameSelection}
            follow={follow}
            onFollow={setFollow}
          />
        )}

        <DatumSection
          origin={origin}
          angle={angle}
          onOrigin={setOrigin}
          onAngle={setAngle}
          roi={roi}
          keptBounds={keptBounds}
        />

        {model !== null && (
          <ModelSection
            model={model}
            rebuilding={teach.isPending}
            onRebuild={() => teach.mutate()}
            onFind={() => void navigate("/recognize/find")}
          />
        )}

        {teach.isError && <ErrorBox>{(teach.error as Error).message}</ErrorBox>}
        {preview !== null && kept.size === 0 && (
          <Callout tone="warning">
            Nothing is kept, so there is no model to build. Keep at least one contour.
          </Callout>
        )}

        {/* Pinned, because a build button that scrolls away below a hundred and sixty-six
            rows is a build button you cannot reach. */}
        <div className="sticky bottom-0 -mx-2 -mb-2 border-t border-line bg-surface px-2 py-2">
          <Button
            variant="primary"
            size="md"
            className="w-full"
            disabled={buildBlocked}
            loading={teach.isPending}
            onClick={() => teach.mutate()}
          >
            {model === null ? "Build model" : "Build another"}
          </Button>
        </div>
      </div>
    </RecognizeShell>
  );
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}
