/**
 * What the edge detector is asked for, and whether what is on screen still answers it.
 *
 * The staleness line is the point of this panel. Contour ids are **positional** — the
 * builder re-runs the same extraction and trusts it to be deterministic, so a `keep_contours`
 * list only means anything for the exact `(image, roi, min_contrast)` it came from
 * (`src-tauri/src/commands/teach.rs`). Nothing in the request records which preview a
 * selection came from, so moving the box after curating used to build a model out of
 * whichever contours happened to land on those indices — silently, and with no error.
 */

import { Button, Callout, ErrorBox, Field, NumberInput, Panel, Slider } from "@vitavision/lab-ui";

export function ExtractSection({
  minContrast,
  onMinContrast,
  numLevels,
  onNumLevels,
  onExtract,
  extracting,
  error,
  stale,
  elapsedMs,
  disabled,
  disabledReason,
  hasPreview,
}: {
  minContrast: number;
  onMinContrast: (value: number) => void;
  numLevels: string;
  onNumLevels: (value: string) => void;
  onExtract: () => void;
  extracting: boolean;
  error: string | null;
  stale: boolean;
  elapsedMs: number | null;
  disabled: boolean;
  disabledReason: string | null;
  hasPreview: boolean;
}) {
  return (
    <Panel
      title="Extraction"
      actions={
        elapsedMs !== null && (
          <span className="font-mono text-[10px] text-fg-subtle tabular-nums">
            {elapsedMs.toFixed(0)} ms
          </span>
        )
      }
    >
      <div className="flex flex-col gap-2">
        <Field label="Min contrast" annotation="fraction of the ROI's range">
          <div className="flex items-center gap-2">
            <Slider
              min={0.01}
              max={0.6}
              step={0.01}
              value={minContrast}
              onValueChange={onMinContrast}
              className="flex-1"
            />
            <NumberInput
              min={0.01}
              max={0.6}
              step={0.01}
              value={minContrast}
              onChange={(event) => {
                // See `RoiSection`: an empty field is mid-edit, not a zero.
                if (event.target.value.trim() === "") return;
                const value = Number(event.target.value);
                if (Number.isFinite(value)) onMinContrast(clamp(value, 0.01, 0.6));
              }}
              className="w-16 px-1.5 text-[11px]"
              aria-label="Min contrast"
            />
          </div>
        </Field>

        <Field label="Pyramid levels" annotation="blank = automatic">
          <NumberInput
            min={1}
            max={8}
            value={numLevels}
            onChange={(event) => onNumLevels(event.target.value)}
            placeholder="auto"
            className="w-20 px-1.5 text-[11px]"
          />
        </Field>

        <Button variant={stale ? "primary" : "secondary"} loading={extracting} disabled={disabled} onClick={onExtract}>
          {hasPreview ? "Re-extract" : "Show candidate edges"}
        </Button>

        {disabledReason && <p className="text-[11px] text-fg-subtle">{disabledReason}</p>}
        {stale && (
          <Callout tone="warning">
            The region or the contrast changed since this preview. Contour numbers are
            positions in the extraction, so the selection below no longer names the same
            edges — re-extract before building.
          </Callout>
        )}
        {error && <ErrorBox>{error}</ErrorBox>}
      </div>
    </Panel>
  );
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
