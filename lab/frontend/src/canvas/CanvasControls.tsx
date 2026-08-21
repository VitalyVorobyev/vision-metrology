/**
 * The lab's own additions to the viewer's toolbar: what a drag means, and what is drawn.
 *
 * Both belong over the image rather than in the inspector. A tool is a property of the next
 * gesture, and a layer toggle is a question about what is currently on screen — putting
 * either in a side panel spends inspector width on it permanently and puts the answer a
 * column away from the thing it describes.
 */

import { Hand, Layers, SquareDashed, SquareMousePointer } from "lucide-react";
import { StageButton, StageToolbarDivider, cn, focusRing } from "@vitavision/lab-ui";
import { useEffect, useRef, useState } from "react";

import type { CanvasTool, LayerVisibility } from "../state/LabContext";

/**
 * `pan` is the neutral tool rather than a hand: dragging bare image pans in every mode, and
 * contours, ROI handles and the datum stay live in every mode too. What a tool changes is
 * only what a drag on *bare image* means.
 */
const TOOLS: { value: CanvasTool; label: string; icon: typeof Hand }[] = [
  { value: "pan", label: "Pan and select — drag the image, click a contour", icon: Hand },
  { value: "box", label: "Draw a new region box", icon: SquareDashed },
  { value: "marquee", label: "Sweep-select contours (or hold shift in any tool)", icon: SquareMousePointer },
];

export function ToolGroup({
  tool,
  onTool,
}: {
  tool: CanvasTool;
  onTool: (tool: CanvasTool) => void;
}) {
  return (
    <>
      {TOOLS.map(({ value, label, icon: Icon }) => (
        <StageButton key={value} label={label} pressed={tool === value} onClick={() => onTool(value)}>
          <Icon className="size-4" aria-hidden />
        </StageButton>
      ))}
    </>
  );
}

const LAYER_LABELS: { key: keyof LayerVisibility; label: string; swatch?: string }[] = [
  { key: "roi", label: "ROI box", swatch: "var(--signal)" },
  { key: "kept", label: "Kept contours", swatch: "var(--signal)" },
  { key: "dropped", label: "Dropped contours", swatch: "var(--fg-subtle)" },
  { key: "vertices", label: "Edge points (at 3× and above)" },
  { key: "datum", label: "Datum", swatch: "var(--normal)" },
  { key: "model", label: "Model points", swatch: "var(--signal-strong)" },
];

export function LayersMenu({
  layers,
  onLayer,
}: {
  layers: LayerVisibility;
  onLayer: (key: keyof LayerVisibility, on: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => {
      if (!box.current?.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [open]);

  const hidden = LAYER_LABELS.filter(({ key }) => !layers[key]).length;

  return (
    <>
      <StageToolbarDivider />
      <div className="relative" ref={box}>
        <StageButton
          label={hidden > 0 ? `Layers (${hidden} hidden)` : "Layers"}
          pressed={hidden > 0}
          onClick={() => setOpen((value) => !value)}
        >
          <Layers className="size-4" aria-hidden />
        </StageButton>
        {open && (
          <div className="absolute bottom-9 left-0 z-20 min-w-56 rounded-panel border border-line bg-overlay p-1 shadow-lg">
            {LAYER_LABELS.map(({ key, label, swatch }) => (
              <label
                key={key}
                className={cn(
                  "flex cursor-pointer items-center gap-2 rounded-control px-2 py-1 text-xs text-fg hover:bg-raised",
                  focusRing,
                )}
              >
                <input
                  type="checkbox"
                  checked={layers[key]}
                  onChange={(event) => onLayer(key, event.target.checked)}
                  className="size-3.5 accent-[var(--signal)]"
                />
                {swatch && (
                  <span
                    aria-hidden
                    className="size-2 shrink-0 rounded-full"
                    style={{ background: swatch }}
                  />
                )}
                {label}
              </label>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
