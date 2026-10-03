/**
 * The frame every workspace lives in: header, rail, canvas, inspector, status.
 *
 * Built on `@vitavision/workbench`'s `AppShell`, which owns the viewport-filling layout, the
 * landmarks and the resizable, remembered inspector. This component decides what goes in
 * each slot.
 *
 * The canvas is mounted **here**, once, and every route draws into it through `LabContext`.
 * That is what lets stepping from Teach to Find keep the image on screen at the same zoom
 * instead of unmounting and re-fetching it.
 *
 * The workspace rail goes in workbench's `rail` slot, a fixed-width `<nav>` at the far left
 * outside the resizable panels, rather than in `left`, which is a resizable panel.
 */

import { DensityProvider, Empty, Skeleton, ThemeToggle } from "@vitavision/ui";
import { AppShell as WorkbenchShell } from "@vitavision/workbench";
import type { ReactNode } from "react";

import { CanvasStage } from "../canvas/CanvasStage";
import { useLab } from "../state/LabContext";
import { FrameSwitcher } from "./FrameSwitcher";
import { StatusBar } from "./StatusBar";
import { LAB_THEME_STORAGE_KEY } from "./theme";
import { WorkspaceRail } from "./WorkspaceRail";

/** Where the inspector's width is remembered (`<key>:right`). */
const SHELL_STORAGE_KEY = "metrology-lab-shell";

export function AppShell({
  steps,
  inspector,
  /** Set by a workspace that owns the whole area (the Library grid), which
   * replaces the canvas rather than drawing over it. */
  fullBleed,
}: {
  steps?: ReactNode;
  inspector: ReactNode;
  fullBleed?: ReactNode;
}) {
  const { selectedImage, imagesLoading } = useLab();

  return (
    <WorkbenchShell
      className="h-full"
      storageKey={SHELL_STORAGE_KEY}
      header={
        // `relative` because the frame switcher's dropdown is positioned against this bar.
        <div className="relative flex items-center gap-3 border-b border-line bg-surface px-3 py-1.5">
          <h1 className="shrink-0 text-sm font-semibold tracking-tight text-fg">Visual Metrology Lab</h1>
          <FrameSwitcher />
          <div className="ml-auto">
            <ThemeToggle storageKey={LAB_THEME_STORAGE_KEY} />
          </div>
        </div>
      }
      rail={<WorkspaceRail />}
      main={
        <div className="flex h-full min-h-0 flex-col">
          {steps && <div className="border-b border-line bg-surface">{steps}</div>}
          <div className="min-h-0 flex-1 p-2">
            {fullBleed ??
              (imagesLoading ? (
                <Skeleton className="h-full w-full" />
              ) : selectedImage === null ? (
                <Empty>Open a frame to begin — the Library workspace is where they come from.</Empty>
              ) : (
                <CanvasStage image={selectedImage} />
              ))}
          </div>
        </div>
      }
      right={
        <div className="p-2">
          <DensityProvider value="compact">{inspector}</DensityProvider>
        </div>
      }
      rightSize={{ defaultSize: 384, minSize: 288, maxSize: 640 }}
      bottom={<StatusBar />}
    />
  );
}
