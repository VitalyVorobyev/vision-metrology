/**
 * The frame every workspace lives in: rail, header, canvas, inspector, status.
 *
 * The canvas is mounted **here**, once, and every route draws into it through
 * `LabContext`. That is what lets stepping from Teach to Find keep the image on
 * screen at the same zoom instead of unmounting and re-fetching it.
 *
 * Layout is deliberately three fixed columns and one focal area: navigation is
 * as narrow as it can be and still be readable, the inspector is a fixed
 * column so controls do not move when their content changes, and everything
 * left over is image. On a workbench the picture is the subject; the chrome is
 * there to be aimed, not admired.
 */

import { Empty, Skeleton, ThemeToggle } from "@vitavision/lab-ui";
import type { ReactNode } from "react";

import { CanvasStage } from "../canvas/CanvasStage";
import { LAB_THEME_STORAGE_KEY } from "./theme";
import { FrameSwitcher } from "./FrameSwitcher";
import { InspectorColumn } from "./InspectorColumn";
import { useLab } from "../state/LabContext";
import { StatusBar } from "./StatusBar";
import { WorkspaceRail } from "./WorkspaceRail";

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
    <div className="flex h-full flex-col">
      {/* `relative` because the frame switcher's dropdown is positioned against this bar. */}
      <div className="relative flex items-center gap-3 border-b border-line bg-surface px-3 py-1.5">
        <h1 className="shrink-0 text-sm font-semibold tracking-tight text-fg">Visual Metrology Lab</h1>
        <FrameSwitcher />
        <div className="ml-auto">
          <ThemeToggle storageKey={LAB_THEME_STORAGE_KEY} />
        </div>
      </div>

      <div className="flex min-h-0 flex-1">
        <WorkspaceRail />

        <div className="flex min-w-0 flex-1 flex-col">
          {steps && <div className="border-b border-line bg-surface">{steps}</div>}

          <div className="flex min-h-0 flex-1">
            <main className="min-w-0 flex-1 p-2">
              {fullBleed ??
                (imagesLoading ? (
                  <Skeleton className="h-full w-full" />
                ) : selectedImage === null ? (
                  <Empty>Open a frame to begin — the Library workspace is where they come from.</Empty>
                ) : (
                  <CanvasStage image={selectedImage} />
                ))}
            </main>

            <InspectorColumn>{inspector}</InspectorColumn>
          </div>
        </div>
      </div>

      <StatusBar />
    </div>
  );
}
