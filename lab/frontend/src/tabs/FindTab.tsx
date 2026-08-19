import { Button, ErrorBox, Field, NumberInput, Panel, Select, Table } from "@vitavision/lab-ui";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { runFind } from "../api/client";
import type { ImageOut, MatchOut, ModelOut, OverlayPrimitiveOut } from "../api/types";

export function FindTab({
  image,
  models,
  onMatches,
}: {
  image: ImageOut;
  models: ModelOut[];
  onMatches: (matches: MatchOut[], overlay: OverlayPrimitiveOut[]) => void;
}) {
  const [modelId, setModelId] = useState(models[0]?.id ?? "");
  const [minScore, setMinScore] = useState(0.7);
  const [angleLo, setAngleLo] = useState<string>("");
  const [angleHi, setAngleHi] = useState<string>("");

  const mutation = useMutation({
    mutationFn: () =>
      runFind({
        image_id: image.id,
        model_id: modelId,
        min_score: minScore,
        max_matches: null,
        roi: null,
        angle_range: angleLo !== "" && angleHi !== "" ? [Number(angleLo), Number(angleHi)] : null,
      }),
    onSuccess: (res) => {
      const overlay: OverlayPrimitiveOut[] = res.matches.map((m) => ({
        kind: "point",
        tone: "signal",
        x: m.x,
        y: m.y,
        cross: true,
        label: m.score.toFixed(2),
      }));
      onMatches(res.matches, overlay);
    },
  });

  return (
    <div className="flex flex-col gap-4">
      <Panel title="Find">
        <div className="flex flex-col gap-3">
          <Field label="Model">
            <Select
              value={modelId}
              onValueChange={setModelId}
              options={models.map((m) => ({ value: m.id, label: `${m.id} (${m.image_id})` }))}
              placeholder="Choose a model…"
            />
          </Field>
          <Field label="Min score" annotation="0–1">
            <NumberInput min={0} max={1} step={0.05} value={minScore} onChange={(e) => setMinScore(Number(e.target.value))} />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Angle min" annotation="degrees, optional">
              <NumberInput value={angleLo} onChange={(e) => setAngleLo(e.target.value)} placeholder="any" />
            </Field>
            <Field label="Angle max" annotation="degrees, optional">
              <NumberInput value={angleHi} onChange={(e) => setAngleHi(e.target.value)} placeholder="any" />
            </Field>
          </div>
          <Button variant="primary" disabled={!modelId} loading={mutation.isPending} onClick={() => mutation.mutate()}>
            Find
          </Button>
          {mutation.isError && <ErrorBox>{(mutation.error as Error).message}</ErrorBox>}
        </div>
      </Panel>

      {mutation.data && (
        <Panel title={`Matches (${mutation.data.matches.length})`}>
          <Table
            columns={[
              { key: "x", header: "x", numeric: true, cell: (m: MatchOut) => m.x.toFixed(2) },
              { key: "y", header: "y", numeric: true, cell: (m: MatchOut) => m.y.toFixed(2) },
              { key: "angle", header: "angle°", numeric: true, cell: (m: MatchOut) => ((m.angle * 180) / Math.PI).toFixed(2) },
              { key: "scale", header: "scale", numeric: true, cell: (m: MatchOut) => m.scale.toFixed(3) },
              { key: "score", header: "score", numeric: true, cell: (m: MatchOut) => m.score.toFixed(3) },
              { key: "level", header: "level", numeric: true, cell: (m: MatchOut) => String(m.level) },
            ]}
            rows={mutation.data.matches}
            rowKey={(m, i) => `${m.x}-${m.y}-${i}`}
            empty="No matches at this score threshold."
          />
        </Panel>
      )}
    </div>
  );
}
