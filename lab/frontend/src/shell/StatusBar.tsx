/**
 * How long the last operation took.
 *
 * The desktop shell has always emitted `lab://progress` with a measured
 * duration, and nothing ever listened — so "find feels slow" had no number
 * attached to it and no way to tell a slow search from a frozen window. One
 * line at the bottom of the screen turns that into an observation: the library
 * publishes ~3.5 ms for a full 360° search of a 1280×1024 frame, so anything
 * far from that is the shell's doing, not the matcher's.
 */

import { cn } from "@vitavision/lab-ui";
import { useEffect, useState } from "react";

import { getBackend } from "../api/backend";
import type { OpProgress } from "../api/backend";

interface Entry {
  op: string;
  elapsedMs: number | null;
  running: boolean;
}

export function StatusBar({ note }: { note?: string }) {
  const [entry, setEntry] = useState<Entry | null>(null);

  useEffect(
    () =>
      getBackend().onProgress((p: OpProgress) => {
        setEntry({
          op: p.op,
          elapsedMs: p.elapsed_ms,
          running: p.stage === "started",
        });
      }),
    [],
  );

  return (
    <footer className="flex h-7 shrink-0 items-center gap-3 border-t border-line bg-surface px-4 text-[11px] text-fg-subtle">
      {entry === null ? (
        <span>Ready</span>
      ) : entry.running ? (
        <span className="flex items-center gap-1.5">
          <span className="size-1.5 animate-pulse rounded-full bg-signal" aria-hidden />
          {entry.op}…
        </span>
      ) : (
        <span className={cn("font-mono", entry.elapsedMs !== null && entry.elapsedMs > 1000 && "text-warn")}>
          {entry.op} {entry.elapsedMs?.toFixed(1)} ms
        </span>
      )}
      {note && <span className="ml-auto truncate">{note}</span>}
    </footer>
  );
}
