/**
 * Which frame is on the canvas — as a control, in the header, on every screen.
 *
 * Changing frame must not require a round trip through Library: on Find the whole task is
 * "run this model against a different frame".
 *
 * `[` and `]` step the sequence, because a capture is an ordered set and stepping through it
 * one frame at a time is what the Find and Verify steps are for.
 */

import { Button, cn, focusRing, Listbox, Popover } from "@vitavision/ui";
import { ChevronDown, ChevronLeft, ChevronRight, Images } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";

import type { ImageOut } from "../api/backend";
import { Thumb } from "../components/Thumb";
import { useLab } from "../state/LabContext";

export function FrameSwitcher() {
  const { images, selectedImage, selectImage, selectedModel, models, selectModel } = useLab();
  const navigate = useNavigate();
  const [open, setOpen] = useState<"frames" | "models" | null>(null);
  const imageById = useMemo(() => new Map(images.map((image) => [image.id, image])), [images]);

  const index = selectedImage ? images.findIndex((image) => image.id === selectedImage.id) : -1;
  const step = (delta: 1 | -1) => {
    if (images.length === 0) return;
    const next = index < 0 ? 0 : (index + delta + images.length) % images.length;
    selectImage(images[next]!.id);
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (isTypingTarget(event.target)) return;
      if (event.key === "[") step(-1);
      else if (event.key === "]") step(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (images.length === 0) {
    return (
      <Button size="sm" variant="ghost" icon={<Images />} onClick={() => void navigate("/library")}>
        Open frames…
      </Button>
    );
  }

  return (
    <div className="flex items-center gap-1">
      <IconStep label="Previous frame ([)" onClick={() => step(-1)} disabled={images.length < 2}>
        <ChevronLeft className="size-4" aria-hidden />
      </IconStep>

      <Popover
        open={open === "frames"}
        onOpenChange={(isOpen) => setOpen(isOpen ? "frames" : null)}
        align="start"
        aria-label="Frames"
        trigger={
          <button
            type="button"
            className={cn(
              "flex h-7 items-center gap-1.5 rounded-control px-2 font-mono text-xs text-fg hover:bg-raised",
              focusRing,
            )}
          >
            <span className="max-w-52 truncate">{selectedImage?.filename ?? "no frame"}</span>
            <span className="text-fg-subtle tabular-nums">
              {index >= 0 ? `${index + 1}/${images.length}` : `–/${images.length}`}
            </span>
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

      <IconStep label="Next frame (])" onClick={() => step(1)} disabled={images.length < 2}>
        <ChevronRight className="size-4" aria-hidden />
      </IconStep>

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
                "ml-1 flex h-7 items-center gap-1.5 rounded-control bg-signal/10 px-2 font-mono text-xs text-signal hover:bg-signal/20",
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

function IconStep({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "grid size-7 place-items-center rounded-control text-fg-muted hover:bg-raised hover:text-fg",
        "disabled:pointer-events-none disabled:opacity-40",
        focusRing,
      )}
    >
      {children}
    </button>
  );
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}
