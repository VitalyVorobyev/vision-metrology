/**
 * Whether an element has come near the viewport — once true, it stays true.
 *
 * Thumbnails are fetched only once they are about to be seen. Scanning a folder decodes
 * nothing, and a list that then asked for every thumbnail at once would spend exactly the
 * work the scan avoided: on the desktop each one is a decode, a resize and a PNG encode, three
 * thousand times, before anyone has looked at any of them. Tiers are cached on disk, so
 * scrolling back is free after the first pass.
 *
 * Without `IntersectionObserver` (an old webview, a test DOM) everything counts as near:
 * slower, not wrong.
 */

import { useEffect, useState } from "react";

/** A screen of margin, so a scroll finds thumbnails already arriving. */
const MARGIN = "300px";

export function useNearViewport(): [ref: (element: Element | null) => void, near: boolean] {
  const [element, setElement] = useState<Element | null>(null);
  const [seen, setSeen] = useState(false);
  const observable = typeof IntersectionObserver !== "undefined";

  useEffect(() => {
    if (seen || element === null || !observable) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setSeen(true);
          observer.disconnect();
        }
      },
      { rootMargin: MARGIN },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [element, seen, observable]);

  return [setElement, seen || !observable];
}
