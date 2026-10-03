/**
 * The model's datum, as numbers as well as handles.
 *
 * `origin` and `reference_angle` decide what a reported pose *means* — where the part's
 * zero is and which way its 0° points — and neither is something an algorithm can infer.
 * The drag handles are the fast way to set them; these fields are the exact way, which
 * matters when the answer is "the tab's centre" or "45°, exactly".
 */

import type { Rect } from "@vitavision/stage2d";
import { Button, Field, NumberInput, Panel, VectorInput } from "@vitavision/ui";

import type { Roi } from "../../api/backend";

export function DatumSection({
  origin,
  angle,
  onOrigin,
  onAngle,
  roi,
  keptBounds,
}: {
  origin: [number, number] | null;
  angle: number;
  onOrigin: (p: [number, number]) => void;
  onAngle: (radians: number) => void;
  roi: Roi | null;
  keptBounds: Rect | null;
}) {
  if (origin === null) {
    return (
      <Panel title="Datum">
        <p className="text-xs text-fg-muted">
          Extract the edges first — the datum handles appear with them.
        </p>
      </Panel>
    );
  }

  const degrees = (angle * 180) / Math.PI;

  return (
    <Panel title="Datum">
      <div className="flex flex-col gap-2">
        <VectorInput
          value={origin}
          onValueChange={(v) => onOrigin([v[0] ?? origin[0], v[1] ?? origin[1]])}
          labels={["x", "y"]}
          unit="px"
          step={1}
          precision={1}
          aria-label="Origin"
        />
        <Field label="0° at">
          <NumberInput
            unit="°"
            step={1}
            value={Math.round(degrees * 10) / 10}
            onValueChange={(value) => onAngle((value * Math.PI) / 180)}
            className="px-1.5 text-[11px]"
          />
        </Field>

        <div className="flex flex-wrap items-center gap-1">
          <Button
            variant="ghost"
            disabled={roi === null}
            onClick={() => roi && onOrigin([roi[0] + roi[2] / 2, roi[1] + roi[3] / 2])}
          >
            Centre on region
          </Button>
          <Button
            variant="ghost"
            disabled={keptBounds === null}
            onClick={() =>
              keptBounds &&
              onOrigin([
                keptBounds.x + keptBounds.width / 2,
                keptBounds.y + keptBounds.height / 2,
              ])
            }
          >
            Centre on kept
          </Button>
          <Button variant="ghost" onClick={() => onAngle(0)}>
            0°
          </Button>
        </div>
        <p className="text-[10px] text-fg-subtle">
          Drag the green cross and arm on the image; hold shift while dragging the arm to snap
          to 15°.
        </p>
      </div>
    </Panel>
  );
}
