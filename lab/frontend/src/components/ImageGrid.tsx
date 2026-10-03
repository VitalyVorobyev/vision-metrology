/**
 * The open frames, as a grid of thumbnails.
 *
 * A grid rather than the old 224 px column: a capture is browsed by comparing
 * frames, and a single-file-wide list makes that a scrolling exercise.
 *
 * Each card fetches its thumbnail **only once it is near the viewport** (`Thumb`),
 * which is what keeps a folder of thousands of frames cheap to open.
 *
 * Not workbench's `SequenceNavigator`: that is one row of small thumbnails for
 * stepping, which the header uses. Browsing a capture wants a grid, each frame's
 * name and size, and a double-click to open it.
 */

import { cn, focusRing } from "@vitavision/ui";

import type { ImageOut } from "../api/backend";
import { Thumb } from "./Thumb";

export function ImageGrid({
  images,
  selectedId,
  onSelect,
  onOpen,
}: {
  images: ImageOut[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onOpen: (id: string) => void;
}) {
  return (
    <ul className="grid h-full grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] content-start gap-3 overflow-y-auto p-1">
      {images.map((img) => (
        <li key={img.id}>
          <Card
            image={img}
            selected={img.id === selectedId}
            onSelect={() => onSelect(img.id)}
            onOpen={() => onOpen(img.id)}
          />
        </li>
      ))}
    </ul>
  );
}

function Card({
  image,
  selected,
  onSelect,
  onOpen,
}: {
  image: ImageOut;
  selected: boolean;
  onSelect: () => void;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      onDoubleClick={onOpen}
      title={image.path ?? image.filename}
      className={cn(
        "flex w-full flex-col gap-1 rounded-control border p-1.5 text-left transition-colors",
        focusRing,
        selected ? "border-signal bg-signal/10" : "border-line hover:border-line-strong",
      )}
    >
      <div className="aspect-square w-full overflow-hidden rounded bg-canvas">
        <Thumb imageId={image.id} alt={image.filename} className="h-full w-full object-contain" />
      </div>
      <span className="truncate text-xs text-fg-muted">{image.filename}</span>
      <span className="font-mono text-[10px] text-fg-subtle">
        {image.width}×{image.height}
      </span>
    </button>
  );
}
