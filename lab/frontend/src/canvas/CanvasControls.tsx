/**
 * The lab's own additions to the viewer's toolbar: what a drag means, and what is drawn.
 *
 * Both belong over the image rather than in the inspector. A tool is a property of the next
 * gesture, and a layer toggle is a question about what is currently on screen — putting
 * either in a side panel spends inspector width on it permanently and puts the answer a
 * column away from the thing it describes.
 */

import { Hand, Layers, SquareDashed, SquareMousePointer } from "lucide-react";
import { StageButton, StageToolbarDivider, overlayRole } from "@vitavision/stage2d";
import { DropdownMenu, MenuCheckboxItem } from "@vitavision/ui";

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
  { key: "roi", label: "ROI box", swatch: overlayRole("selection") },
  { key: "kept", label: "Kept contours", swatch: "var(--signal)" },
  { key: "dropped", label: "Dropped contours", swatch: "var(--fg-subtle)" },
  { key: "vertices", label: "Edge points (at 3× and above)" },
  { key: "datum", label: "Datum", swatch: "var(--normal)" },
  { key: "model", label: "Model points", swatch: "var(--signal-strong)" },
];

/**
 * The layer toggles, with each layer's colour beside its name and the hidden count in the
 * button's name.
 *
 * Not stage2d's `StageLayersMenu`: its layers are named by a plain string, so a menu item
 * cannot carry a swatch, and its one `label` is both the button's name and the menu's
 * heading, so "Layers (2 hidden)" would head the menu too.
 */
export function LayersMenu({
  layers,
  onLayer,
}: {
  layers: LayerVisibility;
  onLayer: (key: keyof LayerVisibility, on: boolean) => void;
}) {
  const hidden = LAYER_LABELS.filter(({ key }) => !layers[key]).length;

  return (
    <>
      <StageToolbarDivider />
      <DropdownMenu
        side="top"
        trigger={
          <StageButton label={hidden > 0 ? `Layers (${hidden} hidden)` : "Layers"} pressed={hidden > 0}>
            <Layers className="size-4" aria-hidden />
          </StageButton>
        }
      >
        {LAYER_LABELS.map(({ key, label, swatch }) => (
          <MenuCheckboxItem key={key} checked={layers[key]} onCheckedChange={(on) => onLayer(key, on)}>
            <span className="flex items-center gap-2">
              {swatch && (
                <span aria-hidden className="size-2 shrink-0 rounded-full" style={{ background: swatch }} />
              )}
              {label}
            </span>
          </MenuCheckboxItem>
        ))}
      </DropdownMenu>
    </>
  );
}
