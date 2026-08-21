/**
 * Which frame is on the canvas — as a control, in the header, on every screen.
 *
 * It used to be dead text. Changing frame meant navigating to Library, clicking a card and
 * navigating back, which is a long way round on Teach and impossible to discover on Find,
 * where the whole task is "run this model against a different frame" and there was no way
 * to pick one.
 *
 * `[` and `]` step the sequence, because a capture is an ordered set and stepping through it
 * one frame at a time is what the Find and Verify steps are for.
 */

import { Button, cn, focusRing } from "@vitavision/lab-ui";
import { ChevronDown, ChevronLeft, ChevronRight, Images } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";

import type { ImageOut } from "../api/backend";
import { Thumb } from "../components/Thumb";
import { useLab } from "../state/LabContext";

export function FrameSwitcher() {
  const { images, selectedImage, selectImage, selectedModel, models, selectModel } = useLab();
  const navigate = useNavigate();
  const [open, setOpen] = useState<"frames" | "models" | null>(null);
  const box = useRef<HTMLDivElement>(null);

  const index = selectedImage ? images.findIndex((image) => image.id === selectedImage.id) : -1;
  const step = (delta: 1 | -1) => {
    if (images.length === 0) return;
    const next = index < 0 ? 0 : (index + delta + images.length) % images.length;
    selectImage(images[next]!.id);
  };

  useEffect(() => {
    if (open === null) return;
    const close = (event: MouseEvent) => {
      if (!box.current?.contains(event.target as Node)) setOpen(null);
    };
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [open]);

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
    <div ref={box} className="flex items-center gap-1">
      <IconStep label="Previous frame ([)" onClick={() => step(-1)} disabled={images.length < 2}>
        <ChevronLeft className="size-4" aria-hidden />
      </IconStep>

      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open === "frames"}
        onClick={() => setOpen((value) => (value === "frames" ? null : "frames"))}
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

      <IconStep label="Next frame (])" onClick={() => step(1)} disabled={images.length < 2}>
        <ChevronRight className="size-4" aria-hidden />
      </IconStep>

      {selectedModel && (
        <button
          type="button"
          aria-haspopup="listbox"
          aria-expanded={open === "models"}
          onClick={() => setOpen((value) => (value === "models" ? null : "models"))}
          className={cn(
            "ml-1 flex h-7 items-center gap-1.5 rounded-control bg-signal/10 px-2 font-mono text-xs text-signal hover:bg-signal/20",
            focusRing,
          )}
        >
          {selectedModel.id}
          <ChevronDown className="size-3.5" aria-hidden />
        </button>
      )}

      {open === "frames" && (
        <Dropdown>
          {images.map((image) => (
            <FrameRow
              key={image.id}
              image={image}
              active={image.id === selectedImage?.id}
              onPick={() => {
                selectImage(image.id);
                setOpen(null);
              }}
            />
          ))}
        </Dropdown>
      )}

      {open === "models" && (
        <Dropdown>
          {models.map((model) => (
            <button
              key={model.id}
              type="button"
              onClick={() => {
                selectModel(model.id);
                setOpen(null);
              }}
              className={cn(
                "flex w-full items-baseline gap-2 rounded-control px-2 py-1 text-left font-mono text-xs hover:bg-raised",
                model.id === selectedModel?.id ? "text-signal" : "text-fg",
                focusRing,
              )}
            >
              {model.id}
              <span className="text-fg-subtle">{model.point_counts.join("/")}</span>
            </button>
          ))}
        </Dropdown>
      )}
    </div>
  );
}

function Dropdown({ children }: { children: React.ReactNode }) {
  return (
    <div
      role="listbox"
      className="absolute top-11 left-0 z-30 max-h-96 min-w-72 overflow-y-auto rounded-panel border border-line bg-overlay p-1 shadow-lg"
    >
      {children}
    </div>
  );
}

function FrameRow({
  image,
  active,
  onPick,
}: {
  image: ImageOut;
  active: boolean;
  onPick: () => void;
}) {
  return (
    <button
      type="button"
      role="option"
      aria-selected={active}
      onClick={onPick}
      className={cn(
        "flex w-full items-center gap-2 rounded-control p-1 text-left hover:bg-raised",
        active && "bg-signal/10",
        focusRing,
      )}
    >
      <Thumb imageId={image.id} alt="" className="size-8 shrink-0 rounded object-cover" />
      <span className="min-w-0 flex-1 truncate font-mono text-xs text-fg">{image.filename}</span>
      <span className="shrink-0 font-mono text-[10px] text-fg-subtle tabular-nums">
        {image.width}×{image.height}
      </span>
    </button>
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
