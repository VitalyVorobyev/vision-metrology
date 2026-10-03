/**
 * The inspector column: resizable, remembered, and dense.
 *
 * Two complaints, one component. It was a fixed 22rem, which is the wrong width for both
 * jobs it has to do — too wide for a page of readouts, too narrow for a list of a hundred
 * and sixty-six contours you are meant to sort and pick through. And it was laid out at
 * page density: a panel body's `p-4` inside a section's `gap-3` inside the column's own
 * padding is three margins deep before a control appears, which on a permanent surface that
 * keeps accumulating tools is simply less instrument on screen.
 *
 * The width is stored per install rather than per route: it is a property of the display
 * and the person, not of the task.
 */

import { DensityProvider } from "@vitavision/lab-ui";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

const STORAGE_KEY = "metrology-lab-inspector-width";
const DEFAULT_WIDTH = 384;
const MIN_WIDTH = 288;
const MAX_WIDTH = 640;

export function InspectorColumn({ children }: { children: ReactNode }) {
  const [width, setWidth] = useState(readWidth);
  const drag = useRef<{ x: number; width: number } | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, String(width));
    } catch {
      // A private window, or storage turned off. The column still works; it just forgets.
    }
  }, [width]);

  const onMove = useCallback((event: PointerEvent) => {
    const state = drag.current;
    if (!state) return;
    // The column is on the right, so dragging left widens it.
    setWidth(clamp(state.width + (state.x - event.clientX), MIN_WIDTH, MAX_WIDTH));
  }, []);

  const onUp = useCallback(() => {
    drag.current = null;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  useEffect(() => {
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [onMove, onUp]);

  return (
    <div className="flex min-h-0 shrink-0" style={{ width }}>
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize the inspector"
        tabIndex={0}
        onPointerDown={(event) => {
          drag.current = { x: event.clientX, width };
          document.body.style.cursor = "col-resize";
          document.body.style.userSelect = "none";
        }}
        onDoubleClick={() => setWidth(DEFAULT_WIDTH)}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") setWidth((w) => clamp(w + 24, MIN_WIDTH, MAX_WIDTH));
          else if (event.key === "ArrowRight") setWidth((w) => clamp(w - 24, MIN_WIDTH, MAX_WIDTH));
          else return;
          event.preventDefault();
        }}
        className="w-1 shrink-0 cursor-col-resize bg-line transition-colors hover:bg-signal focus-visible:bg-signal focus-visible:outline-none"
      />
      <aside className="min-w-0 flex-1 overflow-y-auto bg-surface p-2">
        <DensityProvider value="compact">{children}</DensityProvider>
      </aside>
    </div>
  );
}

function readWidth(): number {
  try {
    const stored = Number(window.localStorage.getItem(STORAGE_KEY));
    if (Number.isFinite(stored) && stored > 0) return clamp(stored, MIN_WIDTH, MAX_WIDTH);
  } catch {
    // See above.
  }
  return DEFAULT_WIDTH;
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
