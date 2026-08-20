/**
 * Was it really found there?
 *
 * `score` answers that with one number, and one number cannot distinguish a
 * correct pose from a plausible wrong one — which is exactly the failure that
 * matters, because a wrong pose feeds a fixture, and a wrong fixture measures
 * the wrong part of the part. So this shows the model and the instance
 * rectified into the same frame and interleaved: registration you can see.
 */

import { Callout, Empty, ErrorBox, Field, NumberInput, Panel, Table } from "@vitavision/lab-ui";
import { useMemo, useState } from "react";

import { getBackend } from "../api/backend";
import type { MatchOut, Roi } from "../api/backend";
import { Triptych } from "../components/Triptych";
import { useAsyncUrl } from "../hooks/useImageUrl";
import { RecognizeShell } from "./RecognizeShell";
import { useLab } from "../state/LabContext";

export function VerifyPage() {
  const backend = getBackend();
  const { selectedImage, selectedModel, matches, lastFind, highlightedMatch, setHighlightedMatch } =
    useLab();
  const [pxPerUnit, setPxPerUnit] = useState(1);

  const index = highlightedMatch ?? 0;
  const match: MatchOut | undefined = matches[index];

  // The model's own ROI is the natural crop: it is the region the model was
  // taught from, so both halves show exactly the taught feature.
  const rect = useMemo<Roi | null>(() => (selectedModel ? selectedModel.roi : null), [selectedModel]);

  const modelCrop = useAsyncUrl(
    selectedModel !== null && rect !== null && backend.canOpenFiles()
      ? () => backend.modelCropUrl(selectedModel.id, rect, pxPerUnit)
      : null,
    [selectedModel?.id, rect?.join(","), pxPerUnit],
  );

  const sampleCrop = useAsyncUrl(
    selectedImage !== null &&
    selectedModel !== null &&
    rect !== null &&
    match !== undefined &&
    lastFind !== null
      ? async () => {
          // The *same* thresholds the visible match list came from. `rectify`
          // re-runs the search to place its crops, so anything else here would
          // renumber the instances and this panel would show a different part
          // than the row that opened it.
          await backend.rectify({
            image_id: selectedImage.id,
            model_id: selectedModel.id,
            min_score: lastFind.min_score,
            max_matches: lastFind.max_matches ?? null,
            angle_range: lastFind.angle_range ?? null,
            scale_range: lastFind.scale_range ?? null,
            refinement: lastFind.refinement ?? null,
            min_contrast: lastFind.min_contrast ?? null,
            tuning: lastFind.tuning ?? null,
            crop: { rect, px_per_unit: pxPerUnit, normalize_scale: true },
          });
          const key = backend.rectifyCropUrl(selectedImage.id, selectedModel.id, index);
          return backend.resolveCropUrl(key);
        }
      : null,
    [
      selectedImage?.id,
      selectedModel?.id,
      rect?.join(","),
      pxPerUnit,
      index,
      lastFind?.min_score,
      lastFind?.max_matches,
      lastFind?.angle_range?.join(","),
    ],
  );

  return (
    <RecognizeShell>
      <div className="flex flex-col gap-3">
        <Panel title="Verify">
          {match === undefined || selectedModel === null ? (
            <Empty>Run a search first — this compares a found instance against the model.</Empty>
          ) : (
            <div className="flex flex-col gap-3">
              <Triptych
                modelUrl={modelCrop.url}
                sampleUrl={sampleCrop.url}
                loading={modelCrop.loading || sampleCrop.loading}
              />
              {(modelCrop.error ?? sampleCrop.error) !== null && (
                <ErrorBox>{modelCrop.error ?? sampleCrop.error}</ErrorBox>
              )}
              <Callout tone="info">
                A correct pose makes the checkerboard seamless: edges run straight across every
                tile boundary. A step at the boundaries is the registration error, made visible.
              </Callout>
              <Field label="Crop resolution" annotation="destination pixels per model pixel">
                <NumberInput
                  min={0.25}
                  max={4}
                  step={0.25}
                  value={pxPerUnit}
                  onChange={(e) => setPxPerUnit(Number(e.target.value))}
                />
              </Field>
            </div>
          )}
        </Panel>

        {matches.length > 1 && (
          <Panel title="Instances">
            <Table
              columns={[
                { key: "i", header: "#", numeric: true, cell: (_m: MatchOut, ) => "" },
                { key: "score", header: "score", numeric: true, cell: (m: MatchOut) => m.score.toFixed(3) },
                {
                  key: "angle",
                  header: "angle°",
                  numeric: true,
                  cell: (m: MatchOut) => ((m.angle * 180) / Math.PI).toFixed(2),
                },
              ]}
              rows={matches}
              rowKey={(_m, i) => i}
              isRowActive={(_m, i) => i === index}
              onRowClick={(_m, i) => setHighlightedMatch(i)}
              empty="No matches."
            />
          </Panel>
        )}
      </div>
    </RecognizeShell>
  );
}
