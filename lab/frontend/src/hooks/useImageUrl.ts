/**
 * Resolve a backend image URL, with loading state.
 *
 * This exists because `LabBackend.imageUrl` is asynchronous, and it is
 * asynchronous because on the desktop the tier has to be rendered and cached
 * before there is a URL to give. The previous design faked a synchronous answer
 * with a 1×1 placeholder and relied on *something else* re-rendering the
 * component once the real bytes arrived — so the main canvas showed nothing at
 * all until the user happened to click a thumbnail, and crossing the zoom
 * threshold re-armed the same trap.
 *
 * Owning the state in a hook is what makes the arrival of the URL a render.
 */

import { useEffect, useState } from "react";

import { getBackend } from "../api/backend";
import type { ImageTier } from "../api/backend";

export interface ImageUrlState {
  url: string | null;
  loading: boolean;
  error: string | null;
}

export function useImageUrl(imageId: string | null, tier: ImageTier): ImageUrlState {
  return useResolvedUrl(imageId, tier, true);
}

/**
 * `useImageUrl`, but only once `enabled` turns true.
 *
 * A folder of three thousand frames is listed without decoding any of them —
 * and that laziness is thrown away if the grid then asks for three thousand
 * thumbnails at once, since each one is a decode plus a resize plus a PNG
 * encode. Gating on visibility (see `ImageGrid`'s IntersectionObserver) keeps
 * the cost proportional to what is actually on screen.
 */
export function useLazyImageUrl(
  imageId: string | null,
  tier: ImageTier,
  enabled: boolean,
): ImageUrlState {
  return useResolvedUrl(imageId, tier, enabled);
}

function useResolvedUrl(
  imageId: string | null,
  tier: ImageTier,
  enabled: boolean,
): ImageUrlState {
  const [state, setState] = useState<ImageUrlState>({
    url: null,
    loading: imageId !== null && enabled,
    error: null,
  });

  useEffect(() => {
    if (imageId === null || !enabled) {
      setState({ url: null, loading: false, error: null });
      return;
    }
    // `cancelled` rather than an AbortController: the work is a command that
    // will finish either way, and all we need is to not write the result of a
    // stale request over a newer one (fast tier switching while zooming).
    let cancelled = false;
    setState((prev) => ({ ...prev, loading: true, error: null }));
    getBackend()
      .imageUrl(imageId, tier)
      .then((url) => {
        if (!cancelled) setState({ url, loading: false, error: null });
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setState({
            url: null,
            loading: false,
            error: e instanceof Error ? e.message : String(e),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [imageId, tier, enabled]);

  return state;
}

/**
 * The same, for a URL that is produced by an arbitrary async thunk — model
 * crops, rectified crops. `deps` decides when it re-runs.
 */
export function useAsyncUrl(
  make: (() => Promise<string>) | null,
  deps: readonly unknown[],
): ImageUrlState {
  const [state, setState] = useState<ImageUrlState>({
    url: null,
    loading: make !== null,
    error: null,
  });

  useEffect(() => {
    if (make === null) {
      setState({ url: null, loading: false, error: null });
      return;
    }
    let cancelled = false;
    setState((prev) => ({ ...prev, loading: true, error: null }));
    make()
      .then((url) => {
        if (!cancelled) setState({ url, loading: false, error: null });
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setState({
            url: null,
            loading: false,
            error: e instanceof Error ? e.message : String(e),
          });
        }
      });
    return () => {
      cancelled = true;
    };
    // The thunk is rebuilt on every render by design; `deps` is what the caller
    // says actually identifies the request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
