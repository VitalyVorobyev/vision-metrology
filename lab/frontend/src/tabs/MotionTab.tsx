import {
  Badge,
  Button,
  ErrorBox,
  Field,
  LineChart,
  NumberInput,
  Panel,
  Section,
  Select,
  cn,
} from "@vitavision/lab-ui";
import type { Series } from "@vitavision/lab-ui";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { getBackend } from "../api/backend";
import { Thumb } from "../components/Thumb";
import type { DisplacementResponse, ImageOut, Roi } from "../api/backend";

/** Cumulative trajectory as two `LineChart` series (frame index on x, px on y) —
 * pure so the mapping is covered without rendering the chart. */
export function trajectorySeries(resp: DisplacementResponse): Series[] {
  return [
    {
      name: "cumulative dx",
      points: resp.cumulative_x.map((v, i) => ({ x: i, y: v })),
    },
    {
      name: "cumulative dy",
      points: resp.cumulative_y.map((v, i) => ({ x: i, y: v })),
    },
  ];
}

/** Toggle `id` in an ordered sequence: append if absent, drop if present. This
 * is the whole multi-select — click images in the rail below in the order you
 * want them tracked; click again to remove one. */
export function toggleSequence(sequence: string[], id: string): string[] {
  return sequence.includes(id) ? sequence.filter((existing) => existing !== id) : [...sequence, id];
}

const REFINE_OPTIONS = [
  { value: "lucas_kanade", label: "Lucas-Kanade" },
  { value: "none", label: "quadratic only" },
];

/** Inter-frame displacement over an ordered sequence of frames
 * (`POST /api/displacement`, `corr::displacement`): drag a tracking window on
 * the currently shown image, pick which frames to track it through and in
 * what order, run, and read the cumulative trajectory + per-pair scores. */
export function MotionTab({
  images,
  windowRoi,
}: {
  images: ImageOut[];
  windowRoi: Roi | null;
}) {
  const [sequence, setSequence] = useState<string[]>([]);
  const [searchX, setSearchX] = useState(12);
  const [searchY, setSearchY] = useState(12);
  const [refine, setRefine] = useState<"none" | "lucas_kanade">("lucas_kanade");
  const [lkIters, setLkIters] = useState(3);
  const [minScore, setMinScore] = useState(0.5);

  const mutation = useMutation({
    mutationFn: () =>
      getBackend().displacement({
        image_ids: sequence,
        window: windowRoi as Roi,
        search_x: searchX,
        search_y: searchY,
        refine,
        lk_iters: lkIters,
        min_score: minScore,
      }),
  });

  const canRun = sequence.length >= 2 && windowRoi !== null && windowRoi[2] > 0 && windowRoi[3] > 0;

  return (
    <div className="flex flex-col gap-4">
      <Panel title="Motion">
        <div className="flex flex-col gap-4">
          <Section step={1} title="Drag a box on the image" hint="Window to track, in the currently shown frame.">
            <p className="text-xs text-fg-muted">
              {windowRoi
                ? `window: x=${windowRoi[0].toFixed(1)} y=${windowRoi[1].toFixed(1)} w=${windowRoi[2].toFixed(1)} h=${windowRoi[3].toFixed(1)}`
                : "No window drawn yet."}
            </p>
          </Section>

          <Section step={2} title="Pick an ordered frame sequence" hint="Click images below, in the order to track.">
            <div className="flex flex-wrap gap-2">
              {images.map((img) => {
                const order = sequence.indexOf(img.id);
                const selected = order >= 0;
                return (
                  <button
                    key={img.id}
                    type="button"
                    onClick={() => setSequence((prev) => toggleSequence(prev, img.id))}
                    className={cn(
                      "relative flex h-12 w-12 items-center justify-center rounded border",
                      selected ? "border-signal bg-signal/10" : "border-line hover:border-line-strong",
                    )}
                  >
                    <Thumb
                      imageId={img.id}
                      alt={img.filename}
                      className="h-full w-full rounded object-contain"
                    />
                    {selected && (
                      <Badge tone="neutral" className="absolute -right-1.5 -top-1.5">
                        {order + 1}
                      </Badge>
                    )}
                  </button>
                );
              })}
            </div>
            {sequence.length > 0 && (
              <Button size="sm" variant="ghost" onClick={() => setSequence([])}>
                Clear sequence ({sequence.length})
              </Button>
            )}
          </Section>

          <Section step={3} title="Config">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Search x" annotation="px">
                <NumberInput min={1} value={searchX} onChange={(e) => setSearchX(Number(e.target.value))} />
              </Field>
              <Field label="Search y" annotation="px">
                <NumberInput min={1} value={searchY} onChange={(e) => setSearchY(Number(e.target.value))} />
              </Field>
              <Field label="Refine">
                <Select
                  value={refine}
                  onValueChange={(v) => setRefine(v as "none" | "lucas_kanade")}
                  options={REFINE_OPTIONS}
                />
              </Field>
              <Field label="LK iterations">
                <NumberInput
                  min={1}
                  max={20}
                  value={lkIters}
                  onChange={(e) => setLkIters(Number(e.target.value))}
                  disabled={refine === "none"}
                />
              </Field>
              <Field label="Min score" annotation="-1..1">
                <NumberInput
                  min={-1}
                  max={1}
                  step={0.05}
                  value={minScore}
                  onChange={(e) => setMinScore(Number(e.target.value))}
                />
              </Field>
            </div>
          </Section>

          <Button variant="primary" disabled={!canRun} loading={mutation.isPending} onClick={() => mutation.mutate()}>
            Run
          </Button>
          {mutation.isError && <ErrorBox>{(mutation.error as Error).message}</ErrorBox>}
        </div>
      </Panel>

      {mutation.data && (
        <Panel title="Trajectory">
          <LineChart
            series={trajectorySeries(mutation.data)}
            label="Cumulative displacement"
            xLabel="frame index"
            yLabel="px"
          />
          <ul className="mt-3 flex flex-col gap-1">
            {mutation.data.pairs.map((p, i) => (
              <li
                key={`${p.from_image_id}-${p.to_image_id}`}
                className="flex items-center justify-between gap-2 font-mono text-[11px] text-fg-subtle"
              >
                <span>
                  {i}: {p.from_image_id} to {p.to_image_id}
                </span>
                <span>
                  dx={p.dx.toFixed(3)} dy={p.dy.toFixed(3)} score={p.score.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}
