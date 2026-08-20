/**
 * Search, with the result drawn on the part rather than described in a table.
 *
 * Two things were wrong before. The overlay was a cross and a score — enough to
 * know the search returned *something*, not enough to know it was right. And
 * the request exposed four fields out of the library's dozen, so the only
 * search anyone could run was a full 360° sweep with no match cap: the slowest
 * one available, which then read as the library being slow.
 */

import {
  Button,
  Disclosure,
  ErrorBox,
  Field,
  NumberInput,
  Panel,
  Select,
  Slider,
  Table,
} from "@vitavision/lab-ui";
import { useMutation } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";

import { getBackend } from "../api/backend";
import type { MatchOut, ModelGeometryOut } from "../api/backend";
import { matchOverlay } from "../overlay/modelOverlay";
import { RecognizeShell } from "./RecognizeShell";
import { useLab } from "../state/LabContext";

export function FindPage() {
  const backend = getBackend();
  const navigate = useNavigate();
  const {
    selectedImage,
    models,
    selectedModel,
    selectModel,
    setOverlay,
    matches,
    setMatches,
    highlightedMatch,
    setHighlightedMatch,
  } = useLab();

  const [minScore, setMinScore] = useState(0.7);
  const [angleLo, setAngleLo] = useState<string>("");
  const [angleHi, setAngleHi] = useState<string>("");
  const [maxMatches, setMaxMatches] = useState<string>("1");
  const [greediness, setGreediness] = useState(0.9);
  const [geometry, setGeometry] = useState<ModelGeometryOut | null>(null);

  const modelId = selectedModel?.id ?? models[0]?.id ?? "";
  useEffect(() => {
    if (selectedModel === null && models.length > 0) selectModel(models[0]!.id);
  }, [models, selectedModel, selectModel]);

  const search = useMutation({
    mutationFn: async () => {
      const req = {
        image_id: selectedImage!.id,
        model_id: modelId,
        min_score: minScore,
        max_matches: maxMatches === "" ? null : Number(maxMatches),
        roi: null,
        angle_range: (angleLo !== "" && angleHi !== ""
          ? [(Number(angleLo) * Math.PI) / 180, (Number(angleHi) * Math.PI) / 180]
          : null) as [number, number] | null,
        tuning: { greediness },
      };
      const res = await backend.find(req);
      // The model's own geometry, in the frame a pose consumes, so each match
      // can be drawn as the model rather than as a marker.
      const geom = backend.canOpenFiles()
        ? await backend.modelGeometry(modelId, 0, "model")
        : null;
      return { res, geom, req };
    },
    onSuccess: ({ res, geom, req }) => {
      setMatches(res.matches, req);
      setGeometry(geom);
      setOverlay(geom === null ? crossesOnly(res.matches) : res.matches.flatMap((m) => matchOverlay(geom, m)));
    },
  });

  // Hovering a row re-draws that one match in the accent tone, so the table and
  // the picture are talking about the same thing.
  useEffect(() => {
    if (geometry === null || matches.length === 0) return;
    setOverlay(
      matches.flatMap((m, i) =>
        matchOverlay(geometry, m, i === highlightedMatch ? "signal" : "normal"),
      ),
    );
  }, [highlightedMatch, matches, geometry, setOverlay]);

  const angleHint = useMemo(
    () => (angleLo !== "" && angleHi !== "" ? "narrowed" : "full 360°"),
    [angleLo, angleHi],
  );

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
              <Slider
                min={0}
                max={1}
                step={0.05}
                value={minScore}
                onValueChange={setMinScore}
              />
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
                    <NumberInput
                      value={angleLo}
                      onChange={(e) => setAngleLo(e.target.value)}
                      placeholder="any"
                    />
                  </Field>
                  <Field label="Angle max" annotation="degrees">
                    <NumberInput
                      value={angleHi}
                      onChange={(e) => setAngleHi(e.target.value)}
                      placeholder="any"
                    />
                  </Field>
                </div>
                <Field
                  label="Greediness"
                  annotation="0 never misses a match; 1 is fastest and may"
                >
                  <Slider
                    min={0}
                    max={1}
                    step={0.05}
                    value={greediness}
                    onValueChange={setGreediness}
                  />
                </Field>
              </div>
            </Disclosure>

            <Button
              variant="primary"
              disabled={modelId === "" || selectedImage === null}
              loading={search.isPending}
              onClick={() => search.mutate()}
            >
              Find
            </Button>
            {search.isError && <ErrorBox>{(search.error as Error).message}</ErrorBox>}
          </div>
        </Panel>

        {matches.length > 0 && (
          <Panel title={`Matches (${matches.length})`}>
            <Table
              columns={[
                { key: "x", header: "x", numeric: true, cell: (m: MatchOut) => m.x.toFixed(2) },
                { key: "y", header: "y", numeric: true, cell: (m: MatchOut) => m.y.toFixed(2) },
                {
                  key: "angle",
                  header: "angle°",
                  numeric: true,
                  cell: (m: MatchOut) => ((m.angle * 180) / Math.PI).toFixed(2),
                },
                { key: "score", header: "score", numeric: true, cell: (m: MatchOut) => m.score.toFixed(3) },
              ]}
              rows={matches}
              rowKey={(m, i) => `${m.x}-${m.y}-${i}`}
              isRowActive={(_, i) => i === highlightedMatch}
              onRowHover={(_, i) => setHighlightedMatch(i)}
              onRowClick={(_, i) => {
                setHighlightedMatch(i);
                void navigate("/recognize/verify");
              }}
              empty="No matches at this score threshold."
            />
          </Panel>
        )}
      </div>
    </RecognizeShell>
  );
}

/** Fallback for the browser build, which has no `model_geometry` route. */
function crossesOnly(matches: MatchOut[]) {
  return matches.map((m) => ({
    kind: "point" as const,
    tone: "signal" as const,
    x: m.x,
    y: m.y,
    cross: true,
    label: m.score.toFixed(2),
  }));
}
