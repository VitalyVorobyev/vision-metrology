import { ImageStage, StageToolbar, type StageView } from "@vitavision/stage2d";
import {
  Badge,
  Button,
  Dialog,
  ErrorBox,
  Field,
  NumberInput,
  Panel,
  Select,
  Switch,
  VectorInput,
} from "@vitavision/ui";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { getBackend } from "../api/backend";
import { CropImage } from "../components/CropImage";
import type { ImageOut, ModelOut, Roi } from "../api/backend";
import { QuantityInput } from "../components/QuantityInput";

const RECT_LABELS = ["x", "y", "w", "h"] as const;

/** Badge tone for a crop's validity fraction — normal above 90%, a caveat down to 50%,
 * a defect below that (mostly border-fill, not real image content). Exported and pure so
 * the thresholds are covered without rendering the component. */
export function validityTone(validity: number): "normal" | "warning" | "defect" {
  if (validity > 0.9) return "normal";
  if (validity > 0.5) return "warning";
  return "defect";
}

/** `CropSpec::output_size` mirrored client-side, for the "output crop is this big"
 * annotation under the form — fixed by the rect and `px_per_unit` alone, matching the
 * Rust/Python `output_size()` (round, floor at one pixel per side). */
export function cropOutputSize(rect: Roi, pxPerUnit: number): [number, number] {
  const [, , w, h] = rect;
  return [Math.max(1, Math.round(w * pxPerUnit)), Math.max(1, Math.round(h * pxPerUnit))];
}

/** Crop-spec form + rectified-crop grid over `POST /api/rectify`: teach a model
 * elsewhere (Teach tab), then here, find every instance of it in the selected image and
 * rectify each into a canonical, model-frame crop — the seam an anomaly pipeline needs
 * (see `crates/vision-metrology/src/matching/crop.rs`). */
export function AlignTab({ image, models }: { image: ImageOut; models: ModelOut[] }) {
  const firstModel = models[0] ?? null;
  const [modelId, setModelId] = useState(firstModel?.id ?? "");
  const [rect, setRect] = useState<Roi>(firstModel?.roi ?? [0, 0, 0, 0]);
  const [pxPerUnit, setPxPerUnit] = useState(1);
  const [normalizeScale, setNormalizeScale] = useState(true);
  const [minScore, setMinScore] = useState(0.7);
  const [maxMatches, setMaxMatches] = useState<string>("");
  const [zoomIndex, setZoomIndex] = useState<number | null>(null);
  // `null` opens the stage at its own sensible view (1:1 if the crop fits, else fit).
  const [zoomView, setZoomView] = useState<StageView | null>(null);

  // Picking a model reseeds the crop rect from its own ROI — a sensible default the
  // reader can then adjust, rather than a stale rect left over from a different model.
  const selectModel = (id: string) => {
    setModelId(id);
    const m = models.find((mm) => mm.id === id);
    if (m) setRect(m.roi);
  };

  const mutation = useMutation({
    mutationFn: () =>
      getBackend().rectify({
        image_id: image.id,
        model_id: modelId,
        min_score: minScore,
        max_matches: maxMatches === "" ? null : Number(maxMatches),
        crop: { rect, px_per_unit: pxPerUnit, normalize_scale: normalizeScale },
      }),
    onSuccess: () => setZoomIndex(null),
  });

  const openZoom = (index: number) => {
    setZoomIndex(index);
    setZoomView(null);
  };

  const [, , rw, rh] = rect;
  const canRun = modelId !== "" && rw > 0 && rh > 0;

  return (
    <div className="flex flex-col gap-4">
      <Panel title="Align">
        <div className="flex flex-col gap-3">
          <Field label="Model">
            <Select
              value={modelId}
              onValueChange={selectModel}
              options={models.map((m) => ({ value: m.id, label: `${m.id} (${m.image_id})` }))}
              placeholder="Choose a model…"
            />
          </Field>

          <Field label="Crop rect" annotation="model-frame coordinates">
            <VectorInput
              value={rect}
              onValueChange={(v) => setRect([v[0] ?? 0, v[1] ?? 0, v[2] ?? 0, v[3] ?? 0])}
              labels={RECT_LABELS}
              precision={1}
              aria-label="Crop rect"
            />
          </Field>

          <Field label="Pixels per model unit">
            <QuantityInput min={0.1} step={0.1} value={pxPerUnit} onValueChange={setPxPerUnit} />
          </Field>

          <Switch
            checked={normalizeScale}
            onCheckedChange={setNormalizeScale}
            label="Normalize scale"
            description="Render at model scale (canonical), not the found scale."
          />

          <div className="grid grid-cols-2 gap-3">
            <Field label="Min score" annotation="0–1">
              <QuantityInput min={0} max={1} step={0.05} value={minScore} onValueChange={setMinScore} />
            </Field>
            <Field label="Max matches" annotation="optional">
              <NumberInput
                value={maxMatches}
                onChange={(e) => setMaxMatches(e.target.value)}
                placeholder="all"
              />
            </Field>
          </div>

          <p className="text-xs text-fg-subtle">
            Output crop: {cropOutputSize(rect, pxPerUnit).join(" x ")} px, fixed by the spec
            above regardless of the found pose.
          </p>

          <Button variant="primary" disabled={!canRun} loading={mutation.isPending} onClick={() => mutation.mutate()}>
            Rectify
          </Button>
          {mutation.isError && <ErrorBox>{mutation.error.message}</ErrorBox>}
        </div>
      </Panel>

      {mutation.data && (
        <Panel title={`Crops (${mutation.data.matches.length})`}>
          {mutation.data.matches.length === 0 ? (
            <p className="text-xs text-fg-subtle">No matches at this score threshold.</p>
          ) : (
            <div className="grid grid-cols-3 gap-3">
              {mutation.data.matches.map((m) => (
                <button
                  key={m.index}
                  type="button"
                  className="flex flex-col items-start gap-1 rounded border border-line p-1.5 text-left hover:border-line-strong"
                  onClick={() => openZoom(m.index)}
                >
                  <CropImage
                    imageId={image.id}
                    modelId={modelId}
                    index={m.index}
                    alt={`rectified crop, match ${m.index}`}
                    className="w-full rounded bg-raised"
                  />
                  <div className="flex w-full items-center justify-between gap-1 text-xs">
                    <Badge tone={validityTone(m.validity)}>{(m.validity * 100).toFixed(0)}% valid</Badge>
                    <span className="text-fg-muted">score {m.score.toFixed(2)}</span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </Panel>
      )}

      <Dialog
        open={zoomIndex !== null}
        onOpenChange={(open) => !open && setZoomIndex(null)}
        title={zoomIndex !== null ? `Match ${zoomIndex}` : "Crop"}
      >
        {zoomIndex !== null && mutation.data && (
          <ImageStage
            image={{ width: mutation.data.width, height: mutation.data.height }}
            view={zoomView}
            onView={setZoomView}
            toolbar={<StageToolbar />}
            className="h-80 w-full"
            label={`Rectified crop, match ${zoomIndex}`}
          >
            <CropImage
              imageId={image.id}
              modelId={modelId}
              index={zoomIndex}
              alt={`rectified crop, match ${zoomIndex}, zoomed`}
              className="h-full w-full"
            />
          </ImageStage>
        )}
      </Dialog>
    </div>
  );
}
