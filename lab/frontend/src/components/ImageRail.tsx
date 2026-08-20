import { Badge, Button, Empty, cn } from "@vitavision/lab-ui";
import { useRef } from "react";
import type { ChangeEvent } from "react";

import { getBackend } from "../api/backend";
import type { ImageOut } from "../api/backend";

export function ImageRail({
  images,
  selectedId,
  onSelect,
  onUpload,
  uploading,
}: {
  images: ImageOut[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onUpload: (file: File) => void;
  uploading: boolean;
}) {
  const fileInput = useRef<HTMLInputElement>(null);

  const handleFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) onUpload(file);
    event.target.value = "";
  };

  return (
    <div className="flex h-full w-56 shrink-0 flex-col gap-3 border-r border-line bg-surface p-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight text-fg">Images</h2>
        <Badge tone="neutral">{images.length}</Badge>
      </div>
      <Button
        size="sm"
        variant="primary"
        loading={uploading}
        onClick={() => fileInput.current?.click()}
      >
        Upload PNG/BMP
      </Button>
      <input ref={fileInput} type="file" accept="image/png,image/bmp" className="hidden" onChange={handleFile} />

      <div className="flex-1 overflow-y-auto">
        {images.length === 0 ? (
          <Empty>No images yet. Upload one to begin.</Empty>
        ) : (
          <ul className="flex flex-col gap-2">
            {images.map((img) => (
              <li key={img.id}>
                <button
                  type="button"
                  onClick={() => onSelect(img.id)}
                  className={cn(
                    "flex w-full flex-col gap-1 rounded-control border p-1.5 text-left transition-colors",
                    img.id === selectedId
                      ? "border-signal bg-signal/10"
                      : "border-line hover:border-line-strong",
                  )}
                >
                  <img
                    src={getBackend().imageUrl(img.id, "thumb")}
                    alt={img.filename}
                    className="aspect-square w-full rounded bg-canvas object-contain"
                  />
                  <span className="truncate text-xs text-fg-muted" title={img.filename}>
                    {img.filename}
                  </span>
                  <span className="font-mono text-[10px] text-fg-subtle">
                    {img.width}×{img.height}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
