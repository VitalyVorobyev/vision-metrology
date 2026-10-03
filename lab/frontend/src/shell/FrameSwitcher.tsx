/**
 * Which frame is on the canvas — as a control, in the header, on every screen.
 *
 * Changing frame must not require a round trip through Library: on Find the whole task is
 * "run this model against a different frame".
 *
 * Two controls, for the two ways a frame is chosen. Stepping is workbench's
 * `SequenceNavigator`: the neighbours as a thumbnail strip, previous and next, the position,
 * and `[` / `]` from anywhere, because a capture is an ordered set and stepping through it one
 * frame at a time is what the Find and Verify steps are for. Jumping is the frame menu: every
 * frame by name and size.
 */

import { Button, cn, focusRing, Listbox, Popover } from "@vitavision/ui";
import { SequenceNavigator } from "@vitavision/workbench";
import { ChevronDown, Images } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router";

import type { ImageOut } from "../api/backend";
import { Thumb } from "../components/Thumb";
import { useLab } from "../state/LabContext";

export function FrameSwitcher() {
  const { images, selectedImage, selectImage, selectedModel, models, selectModel } = useLab();
  const navigate = useNavigate();
  const [open, setOpen] = useState<"frames" | "models" | null>(null);
  const imageById = useMemo(() => new Map(images.map((image) => [image.id, image])), [images]);
  const sequence = useMemo(
    () => images.map((image) => ({ id: image.id, label: image.filename })),
    [images],
  );

  if (images.length === 0) {
    return (
      <Button size="sm" variant="ghost" icon={<Images />} onClick={() => void navigate("/library")}>
        Open frames…
      </Button>
    );
  }

  return (
    <div className="flex min-w-0 flex-1 items-center gap-2">
      <SequenceNavigator
        aria-label="Frame sequence"
        items={sequence}
        value={selectedImage?.id ?? null}
        onValueChange={selectImage}
        // A capture is browsed in both directions; the last frame steps on to the first.
        wrap
        renderThumbnail={(item) => (
          <Thumb imageId={item.id} alt="" className="h-full w-full object-cover" />
        )}
        // Sized to its content: a short capture keeps Next beside its frames, and a long one
        // scrolls inside the strip rather than pushing the frame menu off the bar.
        className="max-w-xl [&>ol]:flex-initial"
      />

      <Popover
        open={open === "frames"}
        onOpenChange={(isOpen) => setOpen(isOpen ? "frames" : null)}
        align="start"
        aria-label="Frames"
        trigger={
          <button
            type="button"
            className={cn(
              "flex h-7 shrink-0 items-center gap-1.5 rounded-control px-2 font-mono text-xs text-fg hover:bg-raised",
              focusRing,
            )}
          >
            <span className="max-w-52 truncate">{selectedImage?.filename ?? "no frame"}</span>
            <ChevronDown className="size-3.5 text-fg-subtle" aria-hidden />
          </button>
        }
      >
        <Listbox
          aria-label="Frames"
          autoFocus
          className="max-h-96 min-w-72 overflow-y-auto"
          options={images.map((image) => ({ value: image.id, label: image.filename }))}
          value={selectedImage?.id ?? null}
          onValueChange={(id) => {
            selectImage(id);
            setOpen(null);
          }}
          renderOption={(option) => {
            const image = imageById.get(option.value);
            return image ? <FrameRow image={image} /> : option.label;
          }}
        />
      </Popover>

      {selectedModel && (
        <Popover
          open={open === "models"}
          onOpenChange={(isOpen) => setOpen(isOpen ? "models" : null)}
          align="start"
          aria-label="Models"
          trigger={
            <button
              type="button"
              className={cn(
                "ml-1 flex h-7 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-control bg-signal/10 px-2 font-mono text-xs text-signal hover:bg-signal/20",
                focusRing,
              )}
            >
              {selectedModel.id}
              <ChevronDown className="size-3.5" aria-hidden />
            </button>
          }
        >
          <Listbox
            aria-label="Models"
            autoFocus
            className="max-h-96 min-w-56 overflow-y-auto"
            options={models.map((model) => ({
              value: model.id,
              label: model.id,
              description: model.point_counts.join("/"),
            }))}
            value={selectedModel.id}
            onValueChange={(id) => {
              selectModel(id);
              setOpen(null);
            }}
            renderOption={(option) => (
              <span className="flex items-baseline gap-2 font-mono text-xs">
                {option.label}
                <span className="text-fg-subtle">{option.description}</span>
              </span>
            )}
          />
        </Popover>
      )}
    </div>
  );
}

function FrameRow({ image }: { image: ImageOut }) {
  return (
    <span className="flex w-full items-center gap-2">
      <Thumb imageId={image.id} alt="" className="size-8 shrink-0 rounded object-cover" />
      <span className="min-w-0 flex-1 truncate font-mono text-xs text-fg">{image.filename}</span>
      <span className="shrink-0 font-mono text-[10px] text-fg-subtle tabular-nums">
        {image.width}×{image.height}
      </span>
    </span>
  );
}
