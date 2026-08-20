/**
 * A rectified crop, resolved through the backend.
 *
 * `rectifyCropUrl` returns a *name* for a crop; on the desktop the pixels then
 * have to be fetched, because a crop is computed per request and never written
 * to disk (unlike an image tier, which is cached as a file and served
 * directly). This component owns that asynchrony so its call sites can go on
 * treating a crop as a picture.
 */

import { cn } from "@vitavision/lab-ui";

import { getBackend } from "../api/backend";
import { useAsyncUrl } from "../hooks/useImageUrl";

export function CropImage({
  imageId,
  modelId,
  index,
  alt,
  className,
}: {
  imageId: string;
  modelId: string;
  index: number;
  alt: string;
  className?: string;
}) {
  const backend = getBackend();
  const { url } = useAsyncUrl(
    () => backend.resolveCropUrl(backend.rectifyCropUrl(imageId, modelId, index)),
    [imageId, modelId, index],
  );
  if (url === null) return <div className={cn("bg-canvas", className)} aria-label={alt} />;
  return <img src={url} alt={alt} className={className} draggable={false} />;
}
