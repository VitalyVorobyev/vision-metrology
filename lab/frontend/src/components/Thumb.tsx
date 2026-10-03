/**
 * One image's thumbnail, fetched once it comes near the viewport.
 *
 * Small on purpose: several places want a thumbnail inside a control of their own (the frame
 * strip, the frame menu, a grid card), and the thing they must not do is call `imageUrl`
 * inline: it is asynchronous, and a component that does not own the loading state ends up
 * rendering nothing forever. Nor should they fetch eagerly; see `useNearViewport`.
 */

import { cn } from "@vitavision/ui";

import { useLazyImageUrl } from "../hooks/useImageUrl";
import { useNearViewport } from "../hooks/useNearViewport";

export function Thumb({
  imageId,
  alt,
  className,
}: {
  imageId: string;
  alt: string;
  className?: string;
}) {
  const [ref, near] = useNearViewport();
  const { url } = useLazyImageUrl(imageId, "thumb", near);
  if (url === null) return <div ref={ref} className={cn("bg-canvas", className)} aria-label={alt} />;
  return <img ref={ref} src={url} alt={alt} className={className} draggable={false} />;
}
