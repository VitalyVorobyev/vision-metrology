/**
 * One image's thumbnail, resolved through `useImageUrl`.
 *
 * Small on purpose: several places want a thumbnail inside a control of their
 * own (a sequence picker, a grid card), and the thing they must not do is call
 * `imageUrl` inline — it is asynchronous, and a component that does not own the
 * loading state ends up rendering nothing forever.
 */

import { cn } from "@vitavision/lab-ui";

import { useImageUrl } from "../hooks/useImageUrl";

export function Thumb({
  imageId,
  alt,
  className,
}: {
  imageId: string;
  alt: string;
  className?: string;
}) {
  const { url } = useImageUrl(imageId, "thumb");
  if (url === null) return <div className={cn("bg-canvas", className)} aria-label={alt} />;
  return <img src={url} alt={alt} className={className} draggable={false} />;
}
