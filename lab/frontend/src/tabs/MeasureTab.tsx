/**
 * Calipers and fits at the found pose, and the evidence behind each number.
 *
 * The objects are nominal geometry in the model's frame; the backend places them at the
 * auto-found fixture and measures every caliper once. The results are per object (the fit,
 * its residual statistics) and per caliper: the caliper inventory (`CaliperSection`) lists
 * each one's verdict, edge, residual and amplitude, linked to its box and edge mark on the
 * canvas, and draws the selected one's profile.
 */

import {
  Badge,
  Button,
  ErrorBox,
  Field,
  NumberInput,
  Panel,
  Section,
  SegmentedControl,
  Select,
  Table,
} from "@vitavision/ui";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from "react";

import { getBackend } from "../api/backend";
import type {
  CalibrationOut,
  ImageOut,
  MeasureObjectIn,
  MeasureObjectResultOut,
  ModelOut,
} from "../api/backend";
import { formatMeasurement, type MeasureUnit } from "../api/transforms";
import { toMeasurePrimitives } from "../overlay/toMeasurePrimitive";
import {
  describeCalipers,
  filterCalipers,
  filterCounts,
  pickCaliper,
  withCaliperStates,
  type CaliperFilter,
} from "../state/caliperInventory";
import { stepThrough } from "../state/contourInventory";
import { isTypingTarget } from "../state/keyboard";
import { useLab } from "../state/LabContext";
import { atLeast } from "../state/rotatedBox";
import { CaliperProfilePanel, CaliperSection } from "./CaliperSection";

/** Framing a caliper shows at least this much image around it, in image pixels. */
const CALIPER_CONTEXT = 64;
/** Margin around a framed caliper, as a fraction of the framed box. */
const FRAME_PAD = 0.15;

type Kind = "circle" | "line";

function emptyObject(kind: Kind): MeasureObjectIn {
  const base = { n_calipers: 16, caliper_len: 15, caliper_width: 6, label: kind };
  return kind === "circle" ? { kind, ...base, cx: 0, cy: 0, r: 30 } : { kind, ...base, ax: 0, ay: 0, bx: 50, by: 0 };
}

export function MeasureTab({
  image,
  models,
  calibrations,
}: {
  image: ImageOut;
  models: ModelOut[];
  calibrations: CalibrationOut[];
}) {
  const { setOverlay, setOverlayPicker, canvas } = useLab();
  const [modelId, setModelId] = useState(models[0]?.id ?? "");
  const [minScore, setMinScore] = useState(0.7);
  const [objects, setObjects] = useState<MeasureObjectIn[]>([]);
  const [draftKind, setDraftKind] = useState<Kind>("circle");
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [filter, setFilter] = useState<CaliperFilter>("all");
  const [calibrationId, setCalibrationId] = useState("");
  const [cameraIndex, setCameraIndex] = useState(0);
  const [unit, setUnit] = useState<MeasureUnit>("px");
  const queryClient = useQueryClient();
  const calibrationFileInputRef = useRef<HTMLInputElement>(null);
  const uploadCalibrationMutation = useMutation({
    mutationFn: (file: File) => getBackend().uploadCalibration(file),
    onSuccess: (calibration) => {
      void queryClient.invalidateQueries({ queryKey: ["calibrations"] });
      setCalibrationId(calibration.id);
    },
  });

  const mutation = useMutation({
    mutationFn: async () => {
      const response = await getBackend().measure({
        image_id: image.id,
        model_id: modelId,
        min_score: minScore,
        objects,
        camera_index: cameraIndex,
        ...(calibrationId ? { calibration_id: calibrationId } : {}),
      });
      return { response, imageId: image.id };
    },
    onSuccess: () => {
      setSelected(null);
      setHovered(null);
    },
  });

  const addObject = () => setObjects((prev) => [...prev, emptyObject(draftKind)]);
  const removeObject = (i: number) => setObjects((prev) => prev.filter((_, idx) => idx !== i));
  const patchObject = (i: number, patch: Partial<MeasureObjectIn>) =>
    setObjects((prev) => prev.map((o, idx) => (idx === i ? { ...o, ...patch } : o)));

  /* A measurement is about the frame it ran on. Stepping to another frame leaves the objects
   * set up, and the result behind: drawing it over the new frame would look like a result
   * about that one. */
  const data = mutation.data?.imageId === image.id ? mutation.data.response : undefined;
  const results = useMemo(() => data?.objects ?? [], [data]);
  const rows = useMemo(() => describeCalipers(results), [results]);
  const counts = useMemo(() => filterCounts(rows), [rows]);
  const visible = useMemo(() => filterCalipers(rows, filter), [rows, filter]);
  const order = useMemo(() => visible.map((row) => row.id), [visible]);
  const shown = useMemo(() => new Set(order), [order]);
  const overlay = useMemo(() => toMeasurePrimitives(results.flatMap((o) => o.overlay ?? [])), [results]);
  const active = rows.find((row) => row.id === selected) ?? null;

  /* The measurement on the canvas, each caliper's box and edge mark in its state. Pushed only
   * once there is a result: until then the canvas keeps what the previous step drew. */
  useEffect(() => {
    if (data === undefined) return;
    setOverlay(withCaliperStates(overlay, selected, hovered, shown));
  }, [data, overlay, selected, hovered, shown, setOverlay]);

  // The canvas side of the list: the caliper under the pointer, and what a click selects.
  useEffect(() => {
    if (visible.length === 0) {
      setOverlayPicker(null);
      return;
    }
    setOverlayPicker({
      pick: (point, tolerance) => pickCaliper(visible, point, tolerance),
      hovered,
      onHover: setHovered,
      onSelect: setSelected,
    });
    return () => setOverlayPicker(null);
  }, [visible, hovered, setOverlayPicker]);

  const frameCaliper = useCallback(
    (id: string) => {
      const bounds = rows.find((row) => row.id === id)?.bounds;
      if (bounds) canvas.current?.frame(atLeast(bounds, CALIPER_CONTEXT), FRAME_PAD);
    },
    [rows, canvas],
  );

  const onKey = useEffectEvent((event: KeyboardEvent) => {
    if (order.length === 0 || isTypingTarget(event.target) || event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }
    switch (event.key) {
      case "ArrowDown":
      case "ArrowUp":
        setSelected(stepThrough(order, selected, event.key === "ArrowDown" ? 1 : -1));
        break;
      case "f":
      case "F":
        if (selected === null) return;
        frameCaliper(selected);
        break;
      case "Escape":
        if (selected === null) return;
        setSelected(null);
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

  const activeEdge = active?.caliper.profile.edges[0];
  const activeMm =
    calibrationId && unit === "mm" && activeEdge?.x_mm != null && activeEdge.y_mm != null
      ? `${activeEdge.x_mm.toFixed(3)}, ${activeEdge.y_mm.toFixed(3)}`
      : null;

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
            <NumberInput min={0} max={1} step={0.05} value={minScore} onValueChange={setMinScore} />
          </Field>

          <Field
            label="Calibration"
            annotation="optional — augments results with millimetre values via the z=0 plane"
          >
            <div className="flex items-center gap-2">
              <Select
                value={calibrationId}
                onValueChange={setCalibrationId}
                options={[
                  { value: "", label: "None (pixels only)" },
                  ...calibrations.map((c) => ({ value: c.id, label: `${c.id} (${c.format}, ${c.n_cameras} cam)` })),
                ]}
                className="flex-1"
              />
              <Button
                size="sm"
                variant="ghost"
                loading={uploadCalibrationMutation.isPending}
                onClick={() => calibrationFileInputRef.current?.click()}
              >
                Upload…
              </Button>
              <input
                ref={calibrationFileInputRef}
                type="file"
                accept="application/json,.json"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) uploadCalibrationMutation.mutate(file);
                  e.target.value = "";
                }}
              />
            </div>
            {uploadCalibrationMutation.isError && (
              <ErrorBox>{uploadCalibrationMutation.error.message}</ErrorBox>
            )}
          </Field>
          {calibrationId && (
            <Field label="Camera index">
              <NumberInput min={0} step={1} value={cameraIndex} onValueChange={setCameraIndex} />
            </Field>
          )}

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
          {mutation.isError && <ErrorBox>{mutation.error.message}</ErrorBox>}
        </div>
      </Panel>

      {data && (
        <Panel title="Results">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex flex-col gap-1 text-xs text-fg-muted">
              fixture ({data.fixture_source}): x={data.fixture.x.toFixed(2)} y=
              {data.fixture.y.toFixed(2)} angle={((data.fixture.angle * 180) / Math.PI).toFixed(2)}°
              scale={data.fixture.scale.toFixed(3)}
            </div>
            {calibrationId && (
              <SegmentedControl
                value={unit}
                onValueChange={(v) => setUnit((v || "px") as MeasureUnit)}
                options={[
                  { value: "px", label: "px" },
                  { value: "mm", label: "mm" },
                ]}
                aria-label="Measurement unit"
              />
            )}
          </div>
          <Table
            columns={[
              { key: "i", header: "#", cell: (r: MeasureObjectResultOut) => results.indexOf(r) },
              { key: "kind", header: "kind", cell: (r: MeasureObjectResultOut) => r.label ?? r.kind },
              {
                key: "r",
                header: "radius",
                numeric: true,
                cell: (r) => formatMeasurement(unit, r.circle_r, r.circle_r_mm),
              },
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
                cell: (r) => (r.calipers ?? []).filter((c) => c.status === "rejected").length,
              },
            ]}
            rows={results}
            rowKey={(_r, i) => i}
            empty="No results."
          />
          {results.some((r) => r.message) && (
            <ul className="mt-2 flex flex-col gap-1">
              {results.flatMap((r, i) =>
                r.message ? [<li key={`${i}-${r.message}`}><ErrorBox>{r.message}</ErrorBox></li>] : [],
              )}
            </ul>
          )}
        </Panel>
      )}

      {data && rows.length > 0 && (
        <CaliperSection
          rows={visible}
          counts={counts}
          filter={filter}
          onFilter={setFilter}
          selected={selected}
          hovered={hovered}
          onSelect={setSelected}
          onHover={setHovered}
        />
      )}

      {active && <CaliperProfilePanel row={active} mm={activeMm} />}
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
      <NumberInput value={value ?? 0} onValueChange={onChange} />
    </Field>
  );
}
