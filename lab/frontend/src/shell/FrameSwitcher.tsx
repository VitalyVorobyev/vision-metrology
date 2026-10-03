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
 *
 * After a batch find, both mark the frames where the model was not found (or the search
 * failed): the tail of a capture is the part worth stepping to.
 */

import { Button, cn, focusRing, Listbox, Popover } from "@vitavision/ui";
import { SequenceNavigator } from "@vitavision/workbench";
import { ChevronDown, Images, SearchX } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router";

import type { ImageOut } from "../api/backend";
import { Thumb } from "../components/Thumb";
import { useLab } from "../state/LabContext";
import { frameVerdict, type FrameVerdict } from "../state/matchInventory";

/** What a frame's mark says, for its tooltip and accessible name. */
const MISS_LABEL: Record<Exclude<FrameVerdict, "found">, string> = {
  "not-found": "not found",
  error: "search failed",
};

export function FrameSwitcher() {
  const { images, selectedImage, selectImage, selectedModel, models, selectModel, batch } = useLab();
  const navigate = useNavigate();
  const [open, setOpen] = useState<"frames" | "models" | null>(null);
  const imageById = useMemo(() => new Map(images.map((image) => [image.id, image])), [images]);
  /** The frames a batch run found nothing on, with why. */
  const misses = useMemo(() => {
    const out = new Map<string, Exclude<FrameVerdict, "found">>();
    if (batch === null) return out;
    for (const [id, item] of batch.items) {
      const verdict = frameVerdict(item);
      if (verdict === "not-found" || verdict === "error") out.set(id, verdict);
    }
    return out;
  }, [batch]);
  const sequence = useMemo(
    () =>
      images.map((image) => {
        const miss = misses.get(image.id);
        return { id: image.id, label: miss ? `${image.filename} (${MISS_LABEL[miss]})` : image.filename };
      }),
    [images, misses],
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
          <span className="relative block h-full w-full">
            <Thumb imageId={item.id} alt="" className="h-full w-full object-cover" />
            {misses.has(item.id) && <MissMark />}
          </span>
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
            return image ? <FrameRow image={image} miss={misses.get(image.id)} /> : option.label;
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

/** A batch miss on a thumbnail: a defect-coloured rule along its foot and a corner glyph. */
function MissMark() {
  return (
    <span data-miss="" aria-hidden className="pointer-events-none absolute inset-0">
      <span className="absolute inset-x-0 bottom-0 h-1 bg-defect" />
      <span className="absolute top-0.5 right-0.5 grid size-3.5 place-items-center rounded-full bg-defect text-surface">
        <SearchX className="size-2.5" />
      </span>
    </span>
  );
}

function FrameRow({ image, miss }: { image: ImageOut; miss?: Exclude<FrameVerdict, "found"> | undefined }) {
  return (
    <span className="flex w-full items-center gap-2">
      <Thumb imageId={image.id} alt="" className="size-8 shrink-0 rounded object-cover" />
      <span className="min-w-0 flex-1 truncate font-mono text-xs text-fg">{image.filename}</span>
      {miss && <span className="shrink-0 text-[10px] text-defect">{MISS_LABEL[miss]}</span>}
      <span className="shrink-0 font-mono text-[10px] text-fg-subtle tabular-nums">
        {image.width}×{image.height}
      </span>
    </span>
  );
}
