import { Badge, Button, Callout, ErrorBox, Field, NumberInput, Panel, Section } from "@vitavision/lab-ui";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { createModel } from "../api/client";
import type { ImageOut, ModelOut, Roi } from "../api/types";

export function TeachTab({
  image,
  roi,
  onRoiCommitted,
  models,
}: {
  image: ImageOut;
  roi: Roi | null;
  onRoiCommitted: () => void;
  models: ModelOut[];
}) {
  const [minContrast, setMinContrast] = useState(0.1);
  const [numLevels, setNumLevels] = useState<string>("");
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: () =>
      createModel({
        image_id: image.id,
        roi: roi as Roi,
        min_contrast: minContrast,
        num_levels: numLevels === "" ? null : Number(numLevels),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["models"] });
      onRoiCommitted();
    },
  });

  const imageModels = models.filter((m) => m.image_id === image.id);

  return (
    <div className="flex flex-col gap-4">
      <Panel title="Teach">
        <div className="flex flex-col gap-4">
          <Section step={1} title="Drag a box on the image" hint="Frame the feature to recognize later.">
            <p className="text-xs text-fg-muted">
              {roi
                ? `ROI: x=${roi[0].toFixed(1)} y=${roi[1].toFixed(1)} w=${roi[2].toFixed(1)} h=${roi[3].toFixed(1)}`
                : "No ROI drawn yet."}
            </p>
          </Section>

          <Section step={2} title="Model parameters">
            <div className="flex flex-col gap-3">
              <Field label="Min contrast" annotation="fraction of dtype range, 0–1">
                <NumberInput
                  min={0.01}
                  max={1}
                  step={0.01}
                  value={minContrast}
                  onChange={(e) => setMinContrast(Number(e.target.value))}
                />
              </Field>
              <Field label="Pyramid levels" annotation="blank = automatic">
                <NumberInput
                  min={1}
                  max={8}
                  value={numLevels}
                  onChange={(e) => setNumLevels(e.target.value)}
                  placeholder="auto"
                />
              </Field>
            </div>
          </Section>

          <Button
            variant="primary"
            disabled={!roi}
            loading={mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            Teach model
          </Button>

          {mutation.isError && <ErrorBox>{(mutation.error as Error).message}</ErrorBox>}
          {mutation.isSuccess && (
            <Callout tone="info">
              Taught {mutation.data.id}: {mutation.data.num_levels_built} levels,{" "}
              {mutation.data.point_counts.join("/")} points, origin (
              {mutation.data.origin[0].toFixed(1)}, {mutation.data.origin[1].toFixed(1)}).
            </Callout>
          )}
        </div>
      </Panel>

      <Panel title="Models taught on this image">
        {imageModels.length === 0 ? (
          <p className="text-xs text-fg-muted">None yet.</p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {imageModels.map((m) => (
              <li key={m.id} className="flex items-center gap-2 text-xs">
                <Badge tone="neutral">{m.id}</Badge>
                <span className="font-mono text-fg-subtle">
                  roi=[{m.roi.map((v) => v.toFixed(0)).join(",")}] levels={m.num_levels_built}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
