/**
 * The match inventory, end to end through the real page and the shared lab state.
 *
 * The arithmetic (order, boxes, picking) is tested in `state/matchInventory.test.ts`; this
 * covers the wiring between the list, the overlay the canvas draws, and the picker the
 * canvas asks, which is where a row and a match drift apart if they are going to. The canvas
 * itself needs layout, so the harness stands in for it: it reads the overlay's ids and
 * states, and hovers and clicks through `overlayPicker` as the canvas does.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { MeasurePrimitive, StageHandle } from "@vitavision/stage2d";
import { TooltipProvider } from "@vitavision/ui";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useEffect, useImperativeHandle } from "react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type * as BackendModule from "../../api/backend";
import type { BatchFindResponse, ImageOut, LabBackend, MatchOut, ModelOut } from "../../api/backend";
import { FrameSwitcher } from "../../shell/FrameSwitcher";
import { LabProvider, useLab } from "../../state/LabContext";
import { FindPage } from "./FindPage";

const FRAMES: ImageOut[] = [
  { id: "img-1", filename: "a.png", width: 640, height: 480 } as ImageOut,
  { id: "img-2", filename: "b.png", width: 640, height: 480 } as ImageOut,
];

/** Taught from a 40 × 20 rectangle around its origin. */
const MODEL = {
  id: "model-1",
  image_id: "img-1",
  roi: [80, 90, 40, 20],
  origin: [100, 100],
  min_contrast: 0.1,
  num_levels: null,
  num_levels_built: 3,
  point_counts: [40, 20, 10],
} as ModelOut;

function match(x: number, y: number, score: number, support: number): MatchOut {
  return { x, y, angle: 0, scale: 1, score, support, level: 0 };
}

/** Search order is by score; x order and support order are different again. */
const MATCHES: MatchOut[] = [match(300, 100, 0.95, 80), match(100, 300, 0.9, 120), match(200, 200, 0.85, 95)];

const find = vi.fn<LabBackend["find"]>();
const batchFind = vi.fn<LabBackend["batchFind"]>();

function fakeBackend(): LabBackend {
  return {
    canOpenFiles: () => true,
    listImages: async () => FRAMES,
    listModels: async () => [MODEL],
    listCalibrations: async () => [],
    imageUrl: async () => "data:image/png;base64,",
    find,
    batchFind,
    // No points: the matches are drawn as markers and boxed by the taught rectangle.
    modelGeometry: async () => ({ model_id: "model-1", level: 0, origin: [100, 100], reference_angle: 0, points: [], frame: "model" }),
    onProgress: () => () => {},
    onBatchProgress: () => () => {},
    onThumbReady: () => () => {},
    prewarmThumbnails: async () => {},
  } as unknown as LabBackend;
}

vi.mock("../../api/backend", async (importOriginal) => {
  const actual = await importOriginal<typeof BackendModule>();
  return { ...actual, getBackend: () => fakeBackend() };
});

/** The page without the shell: `RecognizeShell` mounts the canvas, which needs layout. */
vi.mock("../RecognizeShell", () => ({
  RecognizeShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

/** What the harness exposes of the shared state, refreshed on every render. */
const lab: {
  overlay: MeasurePrimitive[];
  highlightedMatch: number | null;
  hover: (point: { x: number; y: number }) => void;
  click: (point: { x: number; y: number }) => void;
} = { overlay: [], highlightedMatch: null, hover: () => {}, click: () => {} };

const frame = vi.fn<StageHandle["frame"]>();

/** Selects the first frame, stands in for the canvas, and exposes the state the test reads. */
function Harness() {
  const state = useLab();
  const { images, selectedImage, selectImage, canvas, overlayPicker } = state;
  useEffect(() => {
    if (images.length > 0 && selectedImage === null) selectImage(images[0]!.id);
  }, [images, selectedImage, selectImage]);
  useImperativeHandle(canvas, () => ({ frame, fit: () => {}, zoomTo: () => {} }), []);
  useEffect(() => {
    lab.overlay = state.overlay;
    lab.highlightedMatch = state.highlightedMatch;
    lab.hover = (point) => overlayPicker?.onHover(overlayPicker.pick(point, 0));
    lab.click = (point) => overlayPicker?.onSelect(overlayPicker.pick(point, 0));
  });
  return selectedImage === null ? <p>no-frame</p> : <FindPage />;
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <TooltipProvider>
          <LabProvider>
            <FrameSwitcher />
            <Harness />
          </LabProvider>
        </TooltipProvider>
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

async function search() {
  const button = await screen.findByRole("button", { name: /^find$/i });
  await waitFor(() => expect(button.hasAttribute("disabled")).toBe(false));
  fireEvent.click(button);
  await screen.findByRole("table", { name: "Matches on this frame" });
}

function table() {
  return screen.getByRole("table", { name: "Matches on this frame" });
}

/** The list's rows, by the index each names, in the order they are rendered. */
function rowIndices(): string[] {
  return within(table())
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell")[0]!.textContent.trim());
}

function row(index: number): HTMLElement {
  return within(table())
    .getAllByRole("row")
    .slice(1)
    .find((r) => within(r).getAllByRole("cell")[0]!.textContent.trim() === String(index))!;
}

/** The overlay state each match is drawn in, by index. */
function drawnStates(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const p of lab.overlay) if (p.id && p.kind === "point") out[p.id] = p.state ?? "default";
  return out;
}

describe("FindPage", () => {
  beforeEach(() => {
    frame.mockReset();
    find.mockReset().mockResolvedValue({ matches: MATCHES });
    batchFind.mockReset();
  });

  it("lists every match, best first, and sorts by a column", async () => {
    renderPage();
    await search();
    expect(rowIndices()).toEqual(["0", "1", "2"]);

    fireEvent.click(screen.getByRole("button", { name: /sort by x/i }));
    expect(rowIndices()).toEqual(["1", "2", "0"]);

    fireEvent.click(screen.getByRole("button", { name: /sort by supp/i }));
    expect(rowIndices()).toEqual(["1", "2", "0"]);
    fireEvent.click(screen.getByRole("button", { name: /sort by supp/i }));
    expect(rowIndices()).toEqual(["0", "2", "1"]);
  });

  it("gives every match's primitives its id, drawn in its state", async () => {
    renderPage();
    await search();
    expect(drawnStates()).toEqual({ "match-0": "default", "match-1": "default", "match-2": "default" });

    fireEvent.pointerEnter(row(2));
    expect(drawnStates()["match-2"]).toBe("hover");
    // The hovered match is also outlined by its extent.
    expect(lab.overlay.some((p) => p.kind === "polyline" && p.id === "match-2")).toBe(true);

    fireEvent.pointerLeave(row(2));
    expect(drawnStates()["match-2"]).toBe("default");
  });

  it("lights the row of the match under the pointer on the canvas, and selects it on a click", async () => {
    renderPage();
    await search();

    act(() => lab.hover({ x: 205, y: 195 }));
    expect(row(2).getAttribute("data-state")).toBe("active");
    expect(row(0).getAttribute("data-state")).toBeNull();

    act(() => lab.click({ x: 205, y: 195 }));
    expect(lab.highlightedMatch).toBe(2);
    expect(drawnStates()["match-2"]).toBe("selected");

    // A click on bare image clears the selection.
    act(() => lab.click({ x: 600, y: 20 }));
    expect(lab.highlightedMatch).toBeNull();
  });

  it("selects from the list, steps in list order, frames, and clears", async () => {
    renderPage();
    await search();
    fireEvent.click(screen.getByRole("button", { name: /sort by x/i }));

    fireEvent.click(row(2));
    expect(lab.highlightedMatch).toBe(2);
    expect(screen.getByText(/score 0\.850/)).toBeTruthy();

    // The list reads 1, 2, 0: down from 2 is 0, and down again wraps to 1.
    fireEvent.keyDown(window, { key: "ArrowDown" });
    expect(lab.highlightedMatch).toBe(0);
    fireEvent.keyDown(window, { key: "ArrowDown" });
    expect(lab.highlightedMatch).toBe(1);

    fireEvent.keyDown(window, { key: "f" });
    expect(frame).toHaveBeenCalledTimes(1);
    const [rect] = frame.mock.calls[0]!;
    // Match 1 is the taught 40 × 20 rectangle centred on (100, 300).
    expect(rect.x + rect.width / 2).toBeCloseTo(100, 4);
    expect(rect.y + rect.height / 2).toBeCloseTo(300, 4);

    fireEvent.keyDown(window, { key: "Escape" });
    expect(lab.highlightedMatch).toBeNull();
  });

  it("shows each frame's own batch result and marks the frames it missed", async () => {
    const response: BatchFindResponse = {
      items: [
        { image_id: "img-1", matches: [MATCHES[0]!], elapsed_ms: 3, error: null },
        { image_id: "img-2", matches: [], elapsed_ms: 3, error: null },
      ],
    };
    batchFind.mockResolvedValue(response);
    renderPage();

    const all = await screen.findByRole("button", { name: /in all 2 frames/i });
    await waitFor(() => expect(all.hasAttribute("disabled")).toBe(false));
    fireEvent.click(all);

    expect(await screen.findByText(/frames\. Misses are marked/)).toBeTruthy();
    expect(batchFind.mock.calls[0]![0].image_ids).toEqual(["img-1", "img-2"]);
    expect(rowIndices()).toEqual(["0"]);

    // The miss is marked in the strip, by name as well as by colour.
    const strip = screen.getByRole("list", { name: "Frame sequence" });
    expect(within(strip).getByRole("button", { name: "b.png (not found)" })).toBeTruthy();
    expect(within(strip).getByRole("button", { name: "a.png" })).toBeTruthy();

    fireEvent.keyDown(window, { key: "]" });
    expect(await screen.findByText("The batch run found no match on this frame.")).toBeTruthy();
    expect(lab.overlay).toEqual([]);
  });
});
