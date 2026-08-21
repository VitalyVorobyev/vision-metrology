/**
 * The ROI as four numbers you can type, beside the box you can drag.
 *
 * It used to be a read-only sentence. A region is the one input to teaching that a person
 * often knows exactly — "the same crop as last time", "square, centred on the tab" — and a
 * box that can only be dragged cannot express that.
 */

import { Button, NumberInput, Panel } from "@vitavision/lab-ui";

import type { Roi } from "../../api/backend";
import { clampRoi } from "../../canvas/roiEdit";

export function RoiSection({
  roi,
  onRoi,
  image,
  onRedraw,
  drawing,
}: {
  roi: Roi | null;
  onRoi: (roi: Roi) => void;
  image: { width: number; height: number };
  onRedraw: () => void;
  drawing: boolean;
}) {
  // An empty field is a field being retyped, not a zero: `Number("")` is `0`, which would
  // snap the box to the image's corner between two keystrokes.
  const set = (index: 0 | 1 | 2 | 3, raw: string) => {
    const value = Number(raw);
    if (!roi || raw.trim() === "" || !Number.isFinite(value)) return;
    const next: Roi = [...roi];
    next[index] = value;
    onRoi(clampRoi(next, image));
  };

  return (
    <Panel
      title="Region"
      actions={
        <Button
          variant={drawing ? "primary" : "ghost"}
          onClick={onRedraw}
          title="Drag a fresh box on the image"
        >
          {drawing ? "Drawing…" : "Redraw"}
        </Button>
      }
    >
      {roi === null ? (
        <p className="text-xs text-fg-muted">
          Drag a box on the image to frame the feature to recognise.
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          <div className="grid grid-cols-4 gap-1.5">
            {(["x", "y", "w", "h"] as const).map((name, index) => (
              <label key={name} className="flex flex-col gap-0.5">
                <span className="font-mono text-[10px] text-fg-subtle">{name}</span>
                <NumberInput
                  value={round(roi[index as 0 | 1 | 2 | 3])}
                  step={1}
                  onChange={(event) => set(index as 0 | 1 | 2 | 3, event.target.value)}
                  className="px-1.5 text-[11px]"
                  aria-label={`ROI ${name}`}
                />
              </label>
            ))}
          </div>
          <div className="flex items-center gap-2 font-mono text-[10px] text-fg-subtle tabular-nums">
            <span>
              {Math.round(roi[2] * roi[3]).toLocaleString()} px² ·{" "}
              {((100 * roi[2] * roi[3]) / (image.width * image.height)).toFixed(1)}% of frame
            </span>
            <Button
              variant="ghost"
              className="ml-auto"
              onClick={() => onRoi([0, 0, image.width, image.height])}
            >
              Whole frame
            </Button>
          </div>
        </div>
      )}
    </Panel>
  );
}

function round(value: number): string {
  return Number.isFinite(value) ? String(Math.round(value * 10) / 10) : "";
}
