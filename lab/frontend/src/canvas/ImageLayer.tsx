/**
 * The photograph.
 *
 * Sized `h-full w-full` of a stage that is laid out at the image's own pixel size, so this
 * is the natural size with no `object-contain` letterbox for an overlay to disagree with —
 * the disagreement being exactly what made image and overlay move differently on resize.
 *
 * Past `PIXELATED_SCALE` the browser's smoothing is turned off. Above about 4× a bilinear
 * filter is drawing something the sensor never recorded, and on a metrology bench the
 * sensor's own samples are the thing worth looking at.
 */

import { Skeleton, useStage } from "@vitavision/lab-ui";

import type { ImageOut } from "../api/backend";
import { useImageUrl } from "../hooks/useImageUrl";

/** The `preview` tier's long edge (see the backend's media tiers). */
const PREVIEW_TIER_PX = 1024;
const PIXELATED_SCALE = 4;

export function ImageLayer({ image }: { image: ImageOut }) {
  const stage = useStage();
  // Ask for real pixels once the preview tier would be upsampled — a threshold in the
  // tier's own units rather than a zoom number that means nothing on a different image.
  const tier = stage.view.scale * image.width > PREVIEW_TIER_PX ? "full" : "preview";
  const { url, loading, error } = useImageUrl(image.id, tier);

  return (
    <>
      {url !== null && (
        /* eslint-disable-next-line jsx-a11y/img-redundant-alt -- key is content identity, not decoration */
        <img
          key={`${image.id}-${tier}`}
          src={url}
          alt={image.filename}
          className="absolute inset-0 h-full w-full"
          draggable={false}
          style={{
            imageRendering: stage.view.scale >= PIXELATED_SCALE ? "pixelated" : undefined,
            pointerEvents: "none",
          }}
        />
      )}

      {/* The image is a file the webview loads; while that is in flight the frame should
          say so rather than sit empty and look broken. */}
      {loading && url === null && (
        <Skeleton className="pointer-events-none absolute inset-0 h-full w-full" />
      )}
      {error !== null && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-defect/10 px-3 py-2 text-xs text-defect">
          {error}
        </div>
      )}
    </>
  );
}
