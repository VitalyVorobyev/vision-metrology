import {
  Badge,
  Button,
  ErrorBox,
  Field,
  LineProfile,
  NumberInput,
  Panel,
  Section,
  Select,
  Table,
} from "@vitavision/lab-ui";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { runMeasure } from "../api/client";
import { caliperToProfile } from "../api/transforms";
import type {
  CaliperResultOut,
  ImageOut,
  MeasureObjectIn,
  MeasureObjectResultOut,
  ModelOut,
  OverlayPrimitiveOut,
} from "../api/types";

type Kind = "circle" | "line";

function emptyObject(kind: Kind): MeasureObjectIn {
  const base = { n_calipers: 16, caliper_len: 15, caliper_width: 6, label: kind };
  return kind === "circle" ? { kind, ...base, cx: 0, cy: 0, r: 30 } : { kind, ...base, ax: 0, ay: 0, bx: 50, by: 0 };
}

export function MeasureTab({
  image,
  models,
  onResult,
}: {
  image: ImageOut;
  models: ModelOut[];
  onResult: (overlay: OverlayPrimitiveOut[]) => void;
}) {
  const [modelId, setModelId] = useState(models[0]?.id ?? "");
  const [minScore, setMinScore] = useState(0.7);
  const [objects, setObjects] = useState<MeasureObjectIn[]>([]);
  const [draftKind, setDraftKind] = useState<Kind>("circle");
  const [selected, setSelected] = useState<{ object: number; caliper: number } | null>(null);

  const mutation = useMutation({
    mutationFn: () =>
      runMeasure({ image_id: image.id, model_id: modelId, min_score: minScore, objects }),
    onSuccess: (res) => {
      onResult(res.objects.flatMap((o) => o.overlay));
      setSelected(null);
    },
  });

  const addObject = () => setObjects((prev) => [...prev, emptyObject(draftKind)]);
  const removeObject = (i: number) => setObjects((prev) => prev.filter((_, idx) => idx !== i));
  const patchObject = (i: number, patch: Partial<MeasureObjectIn>) =>
    setObjects((prev) => prev.map((o, idx) => (idx === i ? { ...o, ...patch } : o)));

  const results = mutation.data?.objects ?? [];
  const activeCaliper: CaliperResultOut | null =
    selected && results[selected.object] ? (results[selected.object]?.calipers[selected.caliper] ?? null) : null;

  return (
    <div className="flex flex-col gap-4">
      <Panel title="Measure">
        <div className="flex flex-col gap-3">
          <Field label="Model (defines model frame)">
            <Select
              value={modelId}
              onValueChange={setModelId}
              options={models.map((m) => ({ value: m.id, label: `${m.id} (${m.image_id})` }))}
              placeholder="Choose a model…"
            />
          </Field>
          <Field label="Auto-find min score" annotation="fixture comes from the top find match">
            <NumberInput min={0} max={1} step={0.05} value={minScore} onChange={(e) => setMinScore(Number(e.target.value))} />
          </Field>

          <Section step={1} title="Objects" hint="Nominal geometry, in the model's own frame">
            <div className="flex items-center gap-2">
              <Select
                value={draftKind}
                onValueChange={(v) => setDraftKind(v as Kind)}
                options={[
                  { value: "circle", label: "circle" },
                  { value: "line", label: "line" },
                ]}
              />
              <Button size="sm" onClick={addObject}>
                Add object
              </Button>
            </div>

            <ul className="mt-3 flex flex-col gap-3">
              {objects.map((obj, i) => (
                <li key={i} className="rounded-control border border-line p-2.5">
                  <div className="mb-2 flex items-center justify-between">
                    <Badge tone="neutral">
                      {i}: {obj.kind}
                    </Badge>
                    <Button size="sm" variant="ghost" onClick={() => removeObject(i)}>
                      Remove
                    </Button>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    {obj.kind === "circle" ? (
                      <>
                        <NumberField label="cx" value={obj.cx} onChange={(v) => patchObject(i, { cx: v })} />
                        <NumberField label="cy" value={obj.cy} onChange={(v) => patchObject(i, { cy: v })} />
                        <NumberField label="r" value={obj.r} onChange={(v) => patchObject(i, { r: v })} />
                      </>
                    ) : (
                      <>
                        <NumberField label="ax" value={obj.ax} onChange={(v) => patchObject(i, { ax: v })} />
                        <NumberField label="ay" value={obj.ay} onChange={(v) => patchObject(i, { ay: v })} />
                        <NumberField label="bx" value={obj.bx} onChange={(v) => patchObject(i, { bx: v })} />
                        <NumberField label="by" value={obj.by} onChange={(v) => patchObject(i, { by: v })} />
                      </>
                    )}
                    <NumberField
                      label="n_calipers"
                      value={obj.n_calipers}
                      onChange={(v) => patchObject(i, { n_calipers: v })}
                    />
                    <NumberField
                      label="caliper_len"
                      value={obj.caliper_len}
                      onChange={(v) => patchObject(i, { caliper_len: v })}
                    />
                    <NumberField
                      label="caliper_width"
                      value={obj.caliper_width}
                      onChange={(v) => patchObject(i, { caliper_width: v })}
                    />
                  </div>
                </li>
              ))}
            </ul>
          </Section>

          <Button
            variant="primary"
            disabled={!modelId || objects.length === 0}
            loading={mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            Run measure
          </Button>
          {mutation.isError && <ErrorBox>{(mutation.error as Error).message}</ErrorBox>}
        </div>
      </Panel>

      {mutation.data && (
        <Panel title="Results">
          <div className="flex flex-col gap-1 pb-3 text-xs text-fg-muted">
            fixture ({mutation.data.fixture_source}): x={mutation.data.fixture.x.toFixed(2)} y=
            {mutation.data.fixture.y.toFixed(2)} angle={((mutation.data.fixture.angle * 180) / Math.PI).toFixed(2)}°
            scale={mutation.data.fixture.scale.toFixed(3)}
          </div>
          <Table
            columns={[
              { key: "i", header: "#", cell: (r: MeasureObjectResultOut) => results.indexOf(r) },
              { key: "kind", header: "kind", cell: (r: MeasureObjectResultOut) => r.label ?? r.kind },
              { key: "rms", header: "rms", numeric: true, cell: (r) => (r.rms !== null && r.rms !== undefined ? r.rms.toFixed(3) : "—") },
              {
                key: "max_dev",
                header: "max_dev",
                numeric: true,
                cell: (r) => (r.max_dev !== null && r.max_dev !== undefined ? r.max_dev.toFixed(3) : "—"),
              },
              { key: "n_used", header: "n_used", numeric: true, cell: (r) => r.n_used ?? "—" },
              {
                key: "rejects",
                header: "rejected",
                numeric: true,
                cell: (r) => r.calipers.filter((c) => c.status === "rejected").length,
              },
            ]}
            rows={results}
            rowKey={(_r, i) => i}
            empty="No results."
          />
          {results.some((r) => r.message) && (
            <ul className="mt-2 flex flex-col gap-1">
              {results
                .filter((r) => r.message)
                .map((r, i) => (
                  <li key={i}>
                    <ErrorBox>{r.message}</ErrorBox>
                  </li>
                ))}
            </ul>
          )}

          <div className="mt-4 flex flex-col gap-2">
            <h3 className="text-xs font-semibold text-fg-muted">Calipers</h3>
            {results.map((r, oi) => (
              <div key={oi} className="flex flex-wrap gap-1.5">
                {r.calipers.map((c) => (
                  <button
                    key={c.index}
                    type="button"
                    onClick={() => setSelected({ object: oi, caliper: c.index })}
                    title={c.reason ?? undefined}
                    className={
                      "rounded px-1.5 py-0.5 font-mono text-[10px] " +
                      (c.status === "hit" ? "bg-signal/15 text-signal" : "bg-defect/15 text-defect") +
                      (selected?.object === oi && selected.caliper === c.index ? " ring-1 ring-fg" : "")
                    }
                  >
                    {oi}.{c.index}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </Panel>
      )}

      {activeCaliper && (
        <Panel title={`Caliper profile — ${activeCaliper.status}${activeCaliper.reason ? ` (${activeCaliper.reason})` : ""}`}>
          <LineProfile {...caliperToProfile(activeCaliper)} label="intensity along caliper axis" />
        </Panel>
      )}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number | null | undefined;
  onChange: (v: number) => void;
}) {
  return (
    <Field label={label} className="gap-1">
      <NumberInput value={value ?? 0} onChange={(e) => onChange(Number(e.target.value))} />
    </Field>
  );
}
