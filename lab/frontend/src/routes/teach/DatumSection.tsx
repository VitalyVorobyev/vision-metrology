/**
 * The model's datum, as numbers as well as handles.
 *
 * `origin` and `reference_angle` decide what a reported pose *means* — where the part's
 * zero is and which way its 0° points — and neither is something an algorithm can infer.
 * The drag handles are the fast way to set them; these fields are the exact way, which
 * matters when the answer is "the tab's centre" or "45°, exactly".
 */

import { Button, Field, NumberInput, Panel } from "@vitavision/lab-ui";

import type { Roi } from "../../api/backend";
import type { Bounds } from "../../canvas/contourSelection";

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
  keptBounds: Bounds | null;
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
        <div className="grid grid-cols-3 gap-1.5">
          <Field label="origin x">
            <NumberInput
              value={round(origin[0])}
              step={1}
              onChange={(event) => onOrigin([Number(event.target.value), origin[1]])}
              className="px-1.5 text-[11px]"
            />
          </Field>
          <Field label="origin y">
            <NumberInput
              value={round(origin[1])}
              step={1}
              onChange={(event) => onOrigin([origin[0], Number(event.target.value)])}
              className="px-1.5 text-[11px]"
            />
          </Field>
          <Field label="0° at" annotation="°">
            <NumberInput
              value={round(degrees)}
              step={1}
              onChange={(event) => {
                const value = Number(event.target.value);
                if (Number.isFinite(value)) onAngle((value * Math.PI) / 180);
              }}
              className="px-1.5 text-[11px]"
            />
          </Field>
        </div>

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

function round(value: number): string {
  return Number.isFinite(value) ? String(Math.round(value * 10) / 10) : "";
}
