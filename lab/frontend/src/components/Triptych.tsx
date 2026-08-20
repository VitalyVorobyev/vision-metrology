/**
 * Model, blend, instance — side by side, same size, same orientation.
 *
 * Two crops of the same rectangle in the model's own frame: one from the
 * reference image the model was taught on, one from the frame the match was
 * found in. If the pose is right they are the same picture, and the only honest
 * way to see that is to interleave them — a checkerboard makes a sub-pixel
 * misregistration show up as a visible step at every tile boundary, which no
 * pair of side-by-side images and no single number does.
 *
 * The compositing is client-side on a `<canvas>`: both crops are already
 * fetched for the outer panels, so the middle one costs nothing but a draw.
 */

import { SegmentedControl, Skeleton } from "@vitavision/lab-ui";
import { useEffect, useRef, useState } from "react";

export type BlendMode = "checker" | "wipe" | "difference";

export function Triptych({
  modelUrl,
  sampleUrl,
  loading,
}: {
  modelUrl: string | null;
  sampleUrl: string | null;
  loading: boolean;
}) {
  const [mode, setMode] = useState<BlendMode>("checker");
  const [wipe, setWipe] = useState(0.5);
  const [tile, setTile] = useState(16);

  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-3 gap-2">
        <Pane label="Model" url={modelUrl} loading={loading} />
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wide text-fg-subtle">Blend</span>
          <div className="aspect-square w-full overflow-hidden rounded border border-line bg-canvas">
            <BlendCanvas
              aUrl={modelUrl}
              bUrl={sampleUrl}
              mode={mode}
              wipe={wipe}
              tile={tile}
            />
          </div>
        </div>
        <Pane label="Found" url={sampleUrl} loading={loading} />
      </div>

      <SegmentedControl
        aria-label="Blend mode"
        value={mode}
        onValueChange={(v) => setMode(v as BlendMode)}
        options={[
          { value: "checker", label: "Checker" },
          { value: "wipe", label: "Wipe" },
          { value: "difference", label: "Difference" },
        ]}
      />
      {mode === "checker" && (
        <label className="flex items-center gap-2 text-xs text-fg-muted">
          Tile
          <input
            type="range"
            min={4}
            max={64}
            step={4}
            value={tile}
            onChange={(e) => setTile(Number(e.target.value))}
            className="flex-1"
          />
          <span className="font-mono">{tile}px</span>
        </label>
      )}
      {mode === "wipe" && (
        <label className="flex items-center gap-2 text-xs text-fg-muted">
          Split
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={wipe}
            onChange={(e) => setWipe(Number(e.target.value))}
            className="flex-1"
          />
        </label>
      )}
    </div>
  );
}

function Pane({ label, url, loading }: { label: string; url: string | null; loading: boolean }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-wide text-fg-subtle">{label}</span>
      <div className="aspect-square w-full overflow-hidden rounded border border-line bg-canvas">
        {url !== null ? (
          <img src={url} alt={label} className="h-full w-full object-contain" />
        ) : loading ? (
          <Skeleton className="h-full w-full" />
        ) : null}
      </div>
    </div>
  );
}

function BlendCanvas({
  aUrl,
  bUrl,
  mode,
  wipe,
  tile,
}: {
  aUrl: string | null;
  bUrl: string | null;
  mode: BlendMode;
  wipe: number;
  tile: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (aUrl === null || bUrl === null) return;
    const canvas = ref.current;
    if (canvas === null) return;
    let cancelled = false;

    void Promise.all([loadImage(aUrl), loadImage(bUrl)]).then(([a, b]) => {
      if (cancelled) return;
      const w = Math.max(a.width, b.width);
      const h = Math.max(a.height, b.height);
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (ctx === null) return;
      ctx.clearRect(0, 0, w, h);

      if (mode === "difference") {
        ctx.drawImage(a, 0, 0, w, h);
        ctx.globalCompositeOperation = "difference";
        ctx.drawImage(b, 0, 0, w, h);
        ctx.globalCompositeOperation = "source-over";
        return;
      }

      if (mode === "wipe") {
        const split = Math.round(w * wipe);
        ctx.drawImage(a, 0, 0, w, h);
        ctx.save();
        ctx.beginPath();
        ctx.rect(split, 0, w - split, h);
        ctx.clip();
        ctx.drawImage(b, 0, 0, w, h);
        ctx.restore();
        // The seam itself, so the split is readable on a uniform region.
        ctx.fillStyle = "rgba(59, 201, 219, 0.9)";
        ctx.fillRect(split - 1, 0, 2, h);
        return;
      }

      // Checker: A everywhere, B through a chequered clip.
      ctx.drawImage(a, 0, 0, w, h);
      ctx.save();
      ctx.beginPath();
      for (let y = 0; y < h; y += tile) {
        for (let x = 0; x < w; x += tile) {
          if (((x / tile) | 0) % 2 === ((y / tile) | 0) % 2) continue;
          ctx.rect(x, y, tile, tile);
        }
      }
      ctx.clip();
      ctx.drawImage(b, 0, 0, w, h);
      ctx.restore();
    });

    return () => {
      cancelled = true;
    };
  }, [aUrl, bUrl, mode, wipe, tile]);

  return <canvas ref={ref} className="h-full w-full object-contain" />;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`could not load ${src}`));
    img.src = src;
  });
}
