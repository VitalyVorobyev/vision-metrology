/**
 * The ROI as four numbers you can type, beside the box you can drag.
 *
 * A region is the one input to teaching that a person often knows exactly — "the same crop
 * as last time", "square, centred on the tab" — and a box that can only be dragged cannot
 * express that.
 */

import { Button, Panel, VectorInput } from "@vitavision/ui";

import type { Roi } from "../../api/backend";
import { clampRoi } from "../../canvas/roi";

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
          <VectorInput
            value={roi}
            onValueChange={(v) => onRoi(clampRoi([v[0] ?? 0, v[1] ?? 0, v[2] ?? 0, v[3] ?? 0], image))}
            labels={["x", "y", "w", "h"]}
            unit="px"
            step={1}
            precision={1}
            aria-label="ROI"
          />
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
