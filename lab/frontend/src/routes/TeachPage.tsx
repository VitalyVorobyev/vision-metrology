/**
 * Teaching, as three steps you can see and argue with.
 *
 * The old Teach tab was a rectangle, two numbers, and a button that printed
 * "Taught model-3: 4 levels, 812/210/… points" — and then *cleared the ROI*, so
 * the canvas went back to a bare image. Nothing showed what had been learned,
 * so nothing could be corrected, and the two parameters that decide whether a
 * model is usable on a non-rectangular part (which edges are in it, and which
 * way is 0°) had no controls at all.
 *
 *   1. Draw the box.
 *   2. Curate: the candidate contours appear on the image; click to drop the
 *      ones that belong to the background rather than the part.
 *   3. Frame: drag the origin and the 0° arm to the part's natural datum.
 *
 * Then build — and the model stays drawn over the image, which is how you tell
 * that step 2 worked.
 */

import {
  Badge,
  Button,
  Callout,
  Disclosure,
  ErrorBox,
  Field,
  NumberInput,
  Panel,
  Section,
  Slider,
} from "@vitavision/lab-ui";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";

import { getBackend } from "../api/backend";
import type { ContourOut, Roi } from "../api/backend";
import { modelOverlay } from "../overlay/modelOverlay";
import { RecognizeShell } from "./RecognizeShell";
import { useLab } from "../state/LabContext";

export function TeachPage() {
  const backend = getBackend();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const {
    selectedImage,
    roi,
    setRoiMode,
    setContourSelection,
    setFrameHandles,
    setOverlay,
    selectModel,
  } = useLab();

  const [minContrast, setMinContrast] = useState(0.1);
  const [numLevels, setNumLevels] = useState<string>("");
  const [contours, setContours] = useState<ContourOut[] | null>(null);
  const [kept, setKept] = useState<Set<number>>(new Set());
  const [origin, setOrigin] = useState<[number, number] | null>(null);
  const [angle, setAngle] = useState(0);

  // Drawing a box is what this step is for, so the drag layer is on the whole
  // time — including after a model is built, so the next one can be framed
  // without a mode switch.
  useEffect(() => {
    setRoiMode(true);
    return () => setRoiMode(false);
  }, [setRoiMode]);

  const preview = useMutation({
    mutationFn: () =>
      backend.teachPreview({
        image_id: selectedImage!.id,
        roi: roi as Roi,
        min_contrast: minContrast,
      }),
    onSuccess: (res) => {
      setContours(res.contours);
      // Everything is kept by default: the rectangle's own answer, which the
      // user then subtracts from. Starting empty would make the common case
      // (the box is already right) into work.
      setKept(new Set(res.contours.map((c) => c.id)));
      if (roi) setOrigin([roi[0] + roi[2] / 2, roi[1] + roi[3] / 2]);
    },
  });

  const toggle = useCallback((id: number) => {
    setKept((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // Push the interactive layers into the shared canvas while this step is up.
  useEffect(() => {
    if (contours === null) {
      setContourSelection(null);
      return;
    }
    setContourSelection({ contours, kept, onToggle: toggle });
    return () => setContourSelection(null);
  }, [contours, kept, toggle, setContourSelection]);

  useEffect(() => {
    if (origin === null) {
      setFrameHandles(null);
      return;
    }
    setFrameHandles({ origin, angle, onOrigin: setOrigin, onAngle: setAngle });
    return () => setFrameHandles(null);
  }, [origin, angle, setFrameHandles]);

  const teach = useMutation({
    mutationFn: async () => {
      const model = await backend.teachModel({
        image_id: selectedImage!.id,
        roi: roi as Roi,
        min_contrast: minContrast,
        num_levels: numLevels === "" ? null : Number(numLevels),
        // Curation fields only when there is a curation to send: without a
        // preview the caller has not chosen anything, and an empty selection
        // would be a model with no points rather than the rectangle's own
        // answer.
        ...(contours !== null
          ? { keep_contours: [...kept], origin, reference_angle: angle }
          : {}),
      });
      // Draw what was learned, in the frame of the image it was learned from.
      const geometry = await backend.modelGeometry(model.id, 0, "reference");
      return { model, geometry };
    },
    onSuccess: ({ model, geometry }) => {
      void queryClient.invalidateQueries({ queryKey: ["models"] });
      selectModel(model.id);
      setContours(null);
      setOverlay(modelOverlay(geometry));
    },
  });

  const keptPoints = useMemo(() => {
    if (contours === null) return null;
    return contours
      .filter((c) => kept.has(c.id))
      .reduce((n, c) => n + c.points.length / 2, 0);
  }, [contours, kept]);

  const totalPoints = useMemo(
    () => contours?.reduce((n, c) => n + c.points.length / 2, 0) ?? 0,
    [contours],
  );

  return (
    <RecognizeShell>
      <div className="flex flex-col gap-3">
        <Panel title="Teach">
          <div className="flex flex-col gap-4">
            <Section step={1} title="Drag a box on the image" hint="Frame the feature to recognise.">
              <p className="text-xs text-fg-muted">
                {roi
                  ? `ROI: x=${roi[0].toFixed(1)} y=${roi[1].toFixed(1)} w=${roi[2].toFixed(1)} h=${roi[3].toFixed(1)}`
                  : "No ROI drawn yet."}
              </p>
            </Section>

            <Section step={2} title="Choose the edges" hint="Click a contour to drop it; shift-drag to sweep.">
              <div className="flex flex-col gap-3">
                <Field label="Min contrast" annotation="fraction of the ROI's range, 0–1">
                  <Slider
                    min={0.01}
                    max={0.6}
                    step={0.01}
                    value={minContrast}
                    onValueChange={setMinContrast}
                  />
                </Field>
                <Button
                  size="sm"
                  disabled={!roi || !selectedImage || !backend.canOpenFiles()}
                  loading={preview.isPending}
                  onClick={() => preview.mutate()}
                >
                  {contours === null ? "Show candidate edges" : "Re-extract"}
                </Button>
                {!backend.canOpenFiles() && (
                  <p className="text-xs text-fg-subtle">
                    Curating contours needs the desktop app; the browser build teaches from the
                    rectangle alone.
                  </p>
                )}
                {preview.isError && <ErrorBox>{(preview.error as Error).message}</ErrorBox>}
                {contours !== null && (
                  <p className="text-xs text-fg-muted">
                    <Badge tone="neutral">{kept.size}</Badge> of {contours.length} contours ·{" "}
                    {keptPoints} of {totalPoints} points
                  </p>
                )}
              </div>
            </Section>

            <Section step={3} title="Set the datum" hint="Drag the cross and the arm on the image.">
              {origin === null ? (
                <p className="text-xs text-fg-subtle">
                  Extract the edges first — the datum handles appear with them.
                </p>
              ) : (
                <p className="font-mono text-xs text-fg-muted">
                  origin ({origin[0].toFixed(1)}, {origin[1].toFixed(1)}) · 0° at{" "}
                  {((angle * 180) / Math.PI).toFixed(1)}°
                </p>
              )}
            </Section>

            <Disclosure summary="Advanced">
              <Field label="Pyramid levels" annotation="blank = automatic">
                <NumberInput
                  min={1}
                  max={8}
                  value={numLevels}
                  onChange={(e) => setNumLevels(e.target.value)}
                  placeholder="auto"
                />
              </Field>
            </Disclosure>

            <Button
              variant="primary"
              disabled={!roi || (contours !== null && kept.size === 0)}
              loading={teach.isPending}
              onClick={() => teach.mutate()}
            >
              Build model
            </Button>

            {teach.isError && <ErrorBox>{(teach.error as Error).message}</ErrorBox>}
            {teach.isSuccess && (
              <Callout tone="info">
                <div className="flex flex-col gap-2">
                  <span>
                    {teach.data.model.id}: {teach.data.model.num_levels_built} levels,{" "}
                    {teach.data.model.point_counts.join("/")} points. It is drawn over the image.
                  </span>
                  <Button size="sm" onClick={() => void navigate("/recognize/find")}>
                    Find it elsewhere →
                  </Button>
                </div>
              </Callout>
            )}
          </div>
        </Panel>
      </div>
    </RecognizeShell>
  );
}
