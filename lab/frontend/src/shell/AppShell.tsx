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

import { Empty, PageHeader, Skeleton, ThemeToggle } from "@vitavision/lab-ui";
import type { ReactNode } from "react";

import { CanvasStage } from "../components/CanvasStage";
import { useLab } from "../state/LabContext";
import { StatusBar } from "./StatusBar";
import { WorkspaceRail } from "./WorkspaceRail";

/** This app's own `localStorage` key, so it does not share one with a sibling lab. */
const THEME_STORAGE_KEY = "metrology-lab-theme";

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
  const {
    selectedImage,
    selectedModel,
    imagesLoading,
    overlay,
    roi,
    setRoi,
    roiMode,
    contourSelection,
    frameHandles,
  } = useLab();

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-4 border-b border-line bg-surface px-4 py-2">
        <PageHeader
          title="Visual Metrology Lab"
          meta={
            <span className="font-mono text-xs">
              {selectedImage ? selectedImage.filename : "no frame"}
              {selectedModel && ` · ${selectedModel.id}`}
            </span>
          }
        />
        <div className="ml-auto">
          <ThemeToggle storageKey={THEME_STORAGE_KEY} />
        </div>
      </div>

      <div className="flex min-h-0 flex-1">
        <WorkspaceRail />

        <div className="flex min-w-0 flex-1 flex-col">
          {steps && <div className="border-b border-line bg-surface">{steps}</div>}

          <div className="flex min-h-0 flex-1">
            <main className="min-w-0 flex-1 p-3">
              {fullBleed ??
                (imagesLoading ? (
                  <Skeleton className="h-full w-full" />
                ) : selectedImage === null ? (
                  <Empty>Open a frame to begin — the Library workspace is where they come from.</Empty>
                ) : (
                  <CanvasStage
                    image={selectedImage}
                    overlay={overlay}
                    roiMode={roiMode}
                    onRoiChange={setRoi}
                    roiPreview={roi}
                    contourSelection={contourSelection}
                    frameHandles={frameHandles}
                  />
                ))}
            </main>

            <aside className="w-[22rem] shrink-0 overflow-y-auto border-l border-line bg-surface p-3">
              {inspector}
            </aside>
          </div>
        </div>
      </div>

      <StatusBar />
    </div>
  );
}
