/**
 * The open frames, as a grid of thumbnails.
 *
 * A grid rather than the old 224 px column: a capture is browsed by comparing
 * frames, and a single-file-wide list makes that a scrolling exercise.
 *
 * Each card fetches its thumbnail **only once it is near the viewport**. That
 * is not a nicety: scanning a folder deliberately decodes nothing, and a grid
 * that then asks for every thumbnail at once would spend exactly the work the
 * scan avoided — one decode, resize and PNG encode per frame, three thousand
 * times, before the user has looked at any of them. Tiers are cached on disk,
 * so scrolling back is free after the first pass.
 */

import { cn, focusRing } from "@vitavision/lab-ui";
import { useEffect, useRef, useState } from "react";

import type { ImageOut } from "../api/backend";
import { useLazyImageUrl } from "../hooks/useImageUrl";

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
  const ref = useRef<HTMLButtonElement>(null);
  const [near, setNear] = useState(false);
  const { url } = useLazyImageUrl(image.id, "thumb", near);

  useEffect(() => {
    const el = ref.current;
    // No IntersectionObserver (jsdom, an old webview): fetch rather than show
    // an empty grid forever. Degrading to the eager behaviour is the right
    // failure here — it is slower, not wrong.
    if (el === null || typeof IntersectionObserver === "undefined") {
      setNear(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setNear(true);
          observer.disconnect();
        }
      },
      // A screen of margin, so a scroll finds thumbnails already arriving.
      { rootMargin: "300px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <button
      ref={ref}
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
        {url !== null && (
          <img src={url} alt={image.filename} className="h-full w-full object-contain" draggable={false} />
        )}
      </div>
      <span className="truncate text-xs text-fg-muted">{image.filename}</span>
      <span className="font-mono text-[10px] text-fg-subtle">
        {image.width}×{image.height}
      </span>
    </button>
  );
}
