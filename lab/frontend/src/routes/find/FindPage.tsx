/**
 * Search, with the result drawn on the part and listed beside it.
 *
 * Each match is drawn as the model at its found pose, so a wrong pose is visible at a
 * glance, and listed in the match inventory (`MatchSection`), linked to the canvas: hover
 * either and both light up, select either and the selection is what Verify compares.
 *
 * The request exposes the library's speed knobs (angle range, match cap, greediness): the
 * only search a full 360° sweep with no cap allows is the slowest one available. On the
 * desktop the same search runs over every open frame, and stepping through the frames then
 * shows each frame's own matches.
 */

import {
  Button,
  Disclosure,
  ErrorBox,
  Field,
  NumberInput,
  Panel,
  ProgressBar,
  Select,
  Slider,
} from "@vitavision/ui";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useEffectEvent, useMemo, useState } from "react";
import { useNavigate } from "react-router";

import { getBackend } from "../../api/backend";
import type { BatchProgress, FindRequestFull } from "../../api/backend";
import { matchBoxOverlay, matchMarker, matchOverlay } from "../../overlay/modelOverlay";
import { stepThrough } from "../../state/contourInventory";
import { isTypingTarget } from "../../state/keyboard";
import { useLab } from "../../state/LabContext";
import {
  DEFAULT_MATCH_SORT,
  describeMatches,
  frameVerdict,
  framesFound,
  matchId,
  matchIndexOf,
  matchState,
  modelFrame,
  nextSort,
  pickMatch,
  sortMatches,
  type MatchSort,
} from "../../state/matchInventory";
import { RecognizeShell } from "../RecognizeShell";
import { MatchSection } from "./MatchSection";

/** Margin around a framed match, as a fraction of its size. */
const FRAME_PAD = 0.25;
/** The smallest box a match is framed with: a match with no known extent is still a place. */
const MIN_FRAME = 48;

export function FindPage() {
  const backend = getBackend();
  const desktop = backend.canOpenFiles();
  const navigate = useNavigate();
  const {
    images,
    selectedImage,
    models,
    selectedModel,
    selectModel,
    setOverlay,
    matches,
    lastFind,
    setMatches,
    highlightedMatch,
    setHighlightedMatch,
    batch,
    setBatch,
    setOverlayPicker,
    canvas,
  } = useLab();

  const [minScore, setMinScore] = useState(0.7);
  const [angleLo, setAngleLo] = useState<string>("");
  const [angleHi, setAngleHi] = useState<string>("");
  const [maxMatches, setMaxMatches] = useState<string>("1");
  const [greediness, setGreediness] = useState(0.9);
  const [sort, setSort] = useState<MatchSort>(DEFAULT_MATCH_SORT);
  const [hovered, setHovered] = useState<number | null>(null);
  const [progress, setProgress] = useState<BatchProgress | null>(null);

  const modelId = selectedModel?.id ?? models[0]?.id ?? "";
  useEffect(() => {
    if (selectedModel === null && models.length > 0) selectModel(models[0]!.id);
  }, [models, selectedModel, selectModel]);

  /** The search the panel describes, less its frame. */
  const searchRequest = (): Omit<FindRequestFull, "image_id"> => ({
    model_id: modelId,
    min_score: minScore,
    max_matches: maxMatches === "" ? null : Number(maxMatches),
    roi: null,
    angle_range:
      angleLo !== "" && angleHi !== ""
        ? [(Number(angleLo) * Math.PI) / 180, (Number(angleHi) * Math.PI) / 180]
        : null,
    tuning: { greediness },
  });

  const search = useMutation({
    mutationFn: async () => {
      const req: FindRequestFull = { ...searchRequest(), image_id: selectedImage!.id };
      return { res: await backend.find(req), req };
    },
    onSuccess: ({ res, req }) => {
      setMatches(res.matches, req);
      setHighlightedMatch(null);
    },
  });

  useEffect(() => backend.onBatchProgress(setProgress), [backend]);
  const searchAll = useMutation({
    mutationFn: async () => {
      const request = searchRequest();
      const res = await backend.batchFind({ ...request, image_ids: images.map((image) => image.id) });
      return { request, res };
    },
    onSuccess: ({ request, res }) => {
      setProgress(null);
      setBatch({ request, items: new Map(res.items.map((item) => [item.image_id, item])) });
    },
  });

  /* The model the listed matches came from, which is not necessarily the one picked now. Its
   * own points (desktop) draw each match as the model; its extent boxes it. */
  const matchModelId = lastFind?.model_id ?? null;
  const geometry = useQuery({
    queryKey: ["model-geometry", matchModelId, 0, "model"],
    queryFn: () => backend.modelGeometry(matchModelId!, 0, "model"),
    enabled: desktop && matchModelId !== null,
    staleTime: Infinity,
  });
  const geometryData = geometry.data ?? null;
  const matchModel = useMemo(
    () => models.find((model) => model.id === matchModelId) ?? null,
    [models, matchModelId],
  );
  const frame = useMemo(() => modelFrame(matchModel, geometryData), [matchModel, geometryData]);

  const stats = useMemo(() => describeMatches(matches, frame), [matches, frame]);
  const ordered = useMemo(() => sortMatches(stats, sort), [stats, sort]);
  const order = useMemo(() => ordered.map((stat) => stat.index), [ordered]);

  /* The matches on the canvas, each primitive carrying its match's id and state. The one
   * being pointed at also gets its extent outlined, so the row under the pointer has a shape
   * on the image, not only a heavier stroke. */
  useEffect(() => {
    setOverlay(
      stats.flatMap((stat) => {
        const match = matches[stat.index]!;
        const state = matchState(stat.index, highlightedMatch, hovered);
        const id = matchId(stat.index);
        const marks =
          geometryData === null ? [matchMarker(match, state, id)] : matchOverlay(geometryData, match, state, id);
        return state === "default" || stat.box.width === 0 ? marks : [matchBoxOverlay(stat.box, state, id), ...marks];
      }),
    );
  }, [stats, matches, geometryData, highlightedMatch, hovered, setOverlay]);

  // The canvas side of the list: what is under the pointer, and what a click selects.
  useEffect(() => {
    if (stats.length === 0) {
      setOverlayPicker(null);
      return;
    }
    setOverlayPicker({
      pick: (point, tolerance) => {
        const index = pickMatch(stats, point, tolerance);
        return index === null ? null : matchId(index);
      },
      hovered: hovered === null ? null : matchId(hovered),
      onHover: (id) => setHovered(matchIndexOf(id)),
      onSelect: (id) => setHighlightedMatch(matchIndexOf(id)),
    });
    return () => setOverlayPicker(null);
  }, [stats, hovered, setHighlightedMatch, setOverlayPicker]);

  /** Put one match on screen: the panel commanding the canvas through the stage's handle. */
  const frameMatch = useCallback(
    (index: number) => {
      const stat = stats[index];
      if (stat === undefined) return;
      const { x, y, width, height } = stat.bounds;
      const w = Math.max(width, MIN_FRAME);
      const h = Math.max(height, MIN_FRAME);
      canvas.current?.frame({ x: x + (width - w) / 2, y: y + (height - h) / 2, width: w, height: h }, FRAME_PAD);
    },
    [stats, canvas],
  );

  const onKey = useEffectEvent((event: KeyboardEvent) => {
    if (order.length === 0 || isTypingTarget(event.target) || event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }
    switch (event.key) {
      case "ArrowDown":
      case "ArrowUp":
        setHighlightedMatch(stepThrough(order, highlightedMatch, event.key === "ArrowDown" ? 1 : -1));
        break;
      case "f":
      case "F":
        if (highlightedMatch === null) return;
        frameMatch(highlightedMatch);
        break;
      case "Escape":
        if (highlightedMatch === null) return;
        setHighlightedMatch(null);
        break;
      default:
        return;
    }
    event.preventDefault();
  });
  useEffect(() => {
    const listener = (event: KeyboardEvent) => onKey(event);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  const angleHint = angleLo !== "" && angleHi !== "" ? "narrowed" : "full 360°";
  const verdict = selectedImage === null ? null : frameVerdict(batch?.items.get(selectedImage.id));
  const batchError = selectedImage === null ? null : (batch?.items.get(selectedImage.id)?.error ?? null);
  const found = batch === null ? null : framesFound(batch.items.values());
  const searched = lastFind !== null || verdict !== null;

  return (
    <RecognizeShell>
      <div className="flex flex-col gap-3">
        <Panel title="Find">
          <div className="flex flex-col gap-3">
            <Field label="Model">
              <Select
                value={modelId}
                onValueChange={selectModel}
                options={models.map((m) => ({ value: m.id, label: `${m.id} (${m.image_id})` }))}
                placeholder="Choose a model…"
              />
            </Field>
            <Field label="Min score" annotation="0–1">
              <Slider min={0} max={1} step={0.05} value={minScore} onValueChange={setMinScore} />
            </Field>
            <Field label="Max matches" annotation="blank = every instance">
              <NumberInput
                min={1}
                value={maxMatches}
                onChange={(e) => setMaxMatches(e.target.value)}
                placeholder="all"
              />
            </Field>

            <Disclosure summary={`Search effort — angle sweep ${angleHint}`}>
              <div className="flex flex-col gap-3">
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Angle min" annotation="degrees">
                    <NumberInput value={angleLo} onChange={(e) => setAngleLo(e.target.value)} placeholder="any" />
                  </Field>
                  <Field label="Angle max" annotation="degrees">
                    <NumberInput value={angleHi} onChange={(e) => setAngleHi(e.target.value)} placeholder="any" />
                  </Field>
                </div>
                <Field label="Greediness" annotation="0 never misses a match; 1 is fastest and may">
                  <Slider min={0} max={1} step={0.05} value={greediness} onValueChange={setGreediness} />
                </Field>
              </div>
            </Disclosure>

            <div className="flex gap-2">
              <Button
                variant="primary"
                className="flex-1"
                disabled={modelId === "" || selectedImage === null}
                loading={search.isPending}
                onClick={() => search.mutate()}
              >
                Find
              </Button>
              {desktop && images.length > 1 && (
                <Button
                  disabled={modelId === "" || searchAll.isPending}
                  loading={searchAll.isPending}
                  onClick={() => searchAll.mutate()}
                  title="Run this search on every open frame"
                >
                  In all {images.length} frames
                </Button>
              )}
            </div>
            {searchAll.isPending && progress && (
              <ProgressBar
                fraction={progress.done / Math.max(progress.total, 1)}
                label={`${progress.done}/${progress.total} frames`}
              />
            )}
            {found !== null && !searchAll.isPending && (
              <div className="flex items-center justify-between gap-2 text-[11px] text-fg-muted">
                <span>
                  Batch: found in <span className="font-mono text-fg tabular-nums">{found.found}</span> of{" "}
                  <span className="font-mono tabular-nums">{found.total}</span> frames. Misses are marked in the
                  frame strip.
                </span>
                <Button size="sm" variant="ghost" onClick={() => setBatch(null)}>
                  Clear
                </Button>
              </div>
            )}
            {search.isError && <ErrorBox>{search.error.message}</ErrorBox>}
            {searchAll.isError && <ErrorBox>{searchAll.error.message}</ErrorBox>}
            {batchError !== null && <ErrorBox>{batchError}</ErrorBox>}
          </div>
        </Panel>

        {searched && (
          <MatchSection
            rows={ordered}
            total={matches.length}
            sort={sort}
            onSort={(key) => setSort((current) => nextSort(current, key))}
            selected={highlightedMatch}
            hovered={hovered}
            onSelect={setHighlightedMatch}
            onHover={setHovered}
            onFrame={() => {
              if (highlightedMatch !== null) frameMatch(highlightedMatch);
            }}
            onVerify={() => void navigate("/recognize/verify")}
            empty={
              verdict === "not-found" && lastFind === null
                ? "The batch run found no match on this frame."
                : "No matches at this score threshold."
            }
          />
        )}
      </div>
    </RecognizeShell>
  );
}
