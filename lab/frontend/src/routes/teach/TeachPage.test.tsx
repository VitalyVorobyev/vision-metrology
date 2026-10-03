/**
 * The teaching workflow, end to end through the real components.
 *
 * The pure modules (`canvas/contourSelection`, `canvas/roiEdit`) are tested on their own;
 * what this covers is the wiring between them, the panel and the shared canvas state —
 * which is where the behaviour the old screen lacked actually lives. In particular the
 * stale-preview rule, which is a correctness property rather than a nicety: contour ids are
 * positions in an extraction, so a curated selection sent with a changed region names
 * different edges, silently.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@vitavision/lab-ui";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useEffect } from "react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ContourOut, ImageOut, LabBackend } from "../../api/backend";
import { LabProvider, useLab } from "../../state/LabContext";
import { TeachPage } from "./TeachPage";

const IMAGE: ImageOut = {
  id: "img-1",
  filename: "8.bmp",
  width: 1280,
  height: 1024,
} as ImageOut;

/** Three contours whose length order differs from their id order, so sorting is visible. */
const CONTOURS: ContourOut[] = [
  { id: 0, points: [10, 10, 20, 10, 20, 20], closed: true, length: 30, mean_strength: 0.9 },
  { id: 1, points: [100, 100, 400, 100], closed: false, length: 300, mean_strength: 0.2 },
  { id: 2, points: [50, 50, 55, 55, 60, 60, 65, 65], closed: false, length: 21, mean_strength: 0.6 },
];

const teachPreview = vi.fn();
const teachModel = vi.fn();

function fakeBackend(): LabBackend {
  return {
    canOpenFiles: () => true,
    listImages: async () => [IMAGE],
    listModels: async () => [],
    listCalibrations: async () => [],
    imageUrl: async () => "data:image/png;base64,",
    teachPreview,
    teachModel,
    modelGeometry: async () => ({ points: [], origin: [0, 0], reference_angle: 0 }),
    onProgress: () => () => {},
    onBatchProgress: () => () => {},
    onThumbReady: () => () => {},
    prewarmThumbnails: async () => {},
  } as unknown as LabBackend;
}

vi.mock("../../api/backend", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/backend")>();
  return { ...actual, getBackend: () => fakeBackend() };
});

/** The page without the shell: `RecognizeShell` mounts the canvas, which needs layout. */
vi.mock("../RecognizeShell", () => ({
  RecognizeShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <TooltipProvider>
          <LabProvider>
            <Harness />
          </LabProvider>
        </TooltipProvider>
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

/** Selects the frame and seeds a region, which the canvas would otherwise do by drag. */
function Harness() {
  return (
    <>
      <Seed />
      <TeachPage />
    </>
  );
}

function Seed() {
  const { images, selectedImage, selectImage, roi, setRoi } = useLab();
  // In an effect, not during render: `selectImage` clears the region, so calling it from a
  // render pass wipes whatever the previous step just set.
  useEffect(() => {
    if (images.length > 0 && selectedImage === null) selectImage(images[0]!.id);
  }, [images, selectedImage, selectImage]);
  return (
    <button type="button" data-testid="seed-roi" onClick={() => setRoi([500, 350, 340, 275])}>
      {selectedImage === null ? "seed:no-frame" : roi ? "seed:has-roi" : "seed:ready"}
    </button>
  );
}

async function extractContours() {
  await screen.findByText("seed:ready");
  fireEvent.click(screen.getByTestId("seed-roi"));
  await screen.findByText("seed:has-roi");
  fireEvent.click(screen.getByRole("button", { name: /show candidate edges/i }));
  await screen.findByText("9 / 9 pts");
}

/** The inventory's rows, in the order they are rendered. */
function rowIds(): string[] {
  const rows = screen.getAllByRole("row").slice(1); // drop the header
  return rows.map((row) => within(row).getAllByRole("cell")[0]!.textContent!.trim());
}

describe("TeachPage", () => {
  beforeEach(() => {
    teachPreview.mockReset().mockResolvedValue({ contours: CONTOURS, total_points: 9 });
    teachModel.mockReset().mockResolvedValue({
      id: "model-1",
      image_id: IMAGE.id,
      roi: [500, 350, 340, 275],
      min_contrast: 0.1,
      num_levels: null,
      num_levels_built: 3,
      origin: [670, 487],
      point_counts: [9, 5, 2],
    });
  });

  it("lists every contour with its own facts, longest first", async () => {
    renderPage();
    await extractContours();

    // Longest first is the default, and it is not the extraction order.
    expect(rowIds()).toEqual(["1", "0", "2"]);
    const first = screen.getAllByRole("row")[1]!;
    expect(within(first).getByText("300")).toBeTruthy(); // length
    expect(within(first).getByText("2")).toBeTruthy(); // points
    expect(within(first).getByText("0.20")).toBeTruthy(); // mean strength
  });

  it("keeps everything by default and counts the points behind it", async () => {
    renderPage();
    await extractContours();
    expect(screen.getByText(/kept of 3/)).toBeTruthy();
    expect(screen.getByText("9 / 9 pts")).toBeTruthy();
  });

  it("drops a contour from the model without deleting it from the list", async () => {
    renderPage();
    await extractContours();

    fireEvent.click(screen.getByLabelText("Keep contour 1"));

    expect(screen.getByText("7 / 9 pts")).toBeTruthy();
    expect(rowIds()).toEqual(["1", "0", "2"]);
  });

  it("filters to what is kept, and to what is not", async () => {
    renderPage();
    await extractContours();
    fireEvent.click(screen.getByLabelText("Keep contour 1"));

    fireEvent.click(screen.getByRole("radio", { name: "Dropped" }));
    expect(rowIds()).toEqual(["1"]);

    fireEvent.click(screen.getByRole("radio", { name: "Kept" }));
    expect(rowIds()).toEqual(["0", "2"]);
  });

  it("selects from the list and offers the actions for the selection", async () => {
    renderPage();
    await extractContours();

    fireEvent.click(screen.getAllByRole("row")[1]!);
    expect(screen.getByText(/1 selected · 2 pts/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Drop" }));
    expect(screen.getByText("7 / 9 pts")).toBeTruthy();
  });

  it("steps through the inventory with the arrow keys, in list order", async () => {
    renderPage();
    await extractContours();

    fireEvent.keyDown(window, { key: "ArrowDown" });
    expect(screen.getByText(/1 selected · 2 pts/)).toBeTruthy(); // id 1, two points

    fireEvent.keyDown(window, { key: "ArrowDown" });
    expect(screen.getByText(/1 selected · 3 pts/)).toBeTruthy(); // id 0, three points

    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByText(/selected ·/)).toBeNull();
  });

  it("drops the selection on Delete", async () => {
    renderPage();
    await extractContours();
    fireEvent.keyDown(window, { key: "ArrowDown" });
    fireEvent.keyDown(window, { key: "Delete" });
    expect(screen.getByText("7 / 9 pts")).toBeTruthy();
  });

  it("blocks the build once the region no longer matches the extraction", async () => {
    renderPage();
    await extractContours();
    // Curate, so the auto re-extract deliberately does not fire.
    fireEvent.click(screen.getByLabelText("Keep contour 1"));

    fireEvent.change(screen.getByLabelText("ROI x"), { target: { value: "600" } });

    expect(await screen.findByText(/no longer names the same edges/i)).toBeTruthy();
    expect(screen.getByRole("button", { name: /build model/i }).hasAttribute("disabled")).toBe(true);
  });

  it("re-extracts on its own while there is no curation to lose", async () => {
    renderPage();
    await extractContours();
    expect(teachPreview).toHaveBeenCalledTimes(1);

    fireEvent.change(screen.getByLabelText("ROI x"), { target: { value: "600" } });

    await waitFor(() => expect(teachPreview).toHaveBeenCalledTimes(2), { timeout: 2000 });
    expect(screen.queryByText(/no longer names the same edges/i)).toBeNull();
  });

  it("sends the kept ids, the datum and the region it previewed", async () => {
    renderPage();
    await extractContours();
    fireEvent.click(screen.getByLabelText("Keep contour 1"));

    fireEvent.click(screen.getByRole("button", { name: /build model/i }));

    await waitFor(() => expect(teachModel).toHaveBeenCalledTimes(1));
    const request = teachModel.mock.calls[0]![0];
    expect(request.keep_contours.sort()).toEqual([0, 2]);
    expect(request.roi).toEqual([500, 350, 340, 275]);
    expect(request.min_contrast).toBe(0.1);
    expect(request.origin).toEqual([670, 487.5]);
  });

  it("keeps the contours on screen after a build, so the model can be compared with them", async () => {
    renderPage();
    await extractContours();
    fireEvent.click(screen.getByRole("button", { name: /build model/i }));

    expect(await screen.findByText("model-1")).toBeTruthy();
    expect(rowIds()).toEqual(["1", "0", "2"]);
    expect(screen.getByRole("button", { name: /build another/i })).toBeTruthy();
  });
});
