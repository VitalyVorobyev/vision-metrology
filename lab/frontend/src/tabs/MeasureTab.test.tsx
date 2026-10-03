/**
 * The caliper inventory, end to end through the real Measure tab and the shared lab state.
 *
 * The joins (rows, boxes, states, profile axis) are tested in
 * `state/caliperInventory.test.ts`; this covers the wiring between the list, the overlay the
 * canvas draws and the picker the canvas asks. The canvas needs layout, so the harness
 * stands in for it, as in `FindPage.test.tsx`.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { MeasurePrimitive, StageHandle } from "@vitavision/stage2d";
import { TooltipProvider } from "@vitavision/ui";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useEffect, useImperativeHandle } from "react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type * as BackendModule from "../api/backend";
import type { ImageOut, LabBackend, MeasureResponse, ModelOut } from "../api/backend";
import { LabProvider, useLab } from "../state/LabContext";
import { MeasureTab } from "./MeasureTab";

const IMAGE = { id: "img-1", filename: "disc.png", width: 128, height: 128 } as ImageOut;
const MODEL = { id: "model-1", image_id: "img-1", roi: [24, 24, 80, 80], origin: [64, 64] } as unknown as ModelOut;

/** A rim of three calipers, the middle one rejected; boxes 16 × 8, sampled ±8 about the rim. */
const RESPONSE: MeasureResponse = {
  fixture: { x: 64, y: 64, angle: 0, scale: 1 },
  fixture_source: "auto_find",
  objects: [
    {
      kind: "circle",
      label: "rim",
      circle_cx: 64,
      circle_cy: 64,
      circle_r: 24,
      rms: 0.01,
      max_dev: 0.02,
      n_used: 2,
      calipers: [0, 1, 2].map((index) => ({
        index,
        status: index === 1 ? ("rejected" as const) : ("hit" as const),
        reason: index === 1 ? "too_oblique" : null,
        residual: index === 1 ? null : index === 0 ? -0.018 : 0.011,
        profile: {
          values: [200, 200, 200, 110, 20, 20, 20],
          step_px: 1,
          start_px: -8,
          end_px: 8,
          edges:
            index === 1 ? [] : [{ pos_px: index === 0 ? -0.05 : 0.12, polarity: "falling", amplitude: 64.2 }],
        },
      })),
      overlay: [
        { kind: "caliper", id: "caliper-0-0", tone: "signal", cx: 88, cy: 64, width: 16, height: 8, angle: 0 },
        { kind: "point", id: "caliper-0-0", tone: "signal", x: 87.95, y: 64, cross: true },
        { kind: "caliper", id: "caliper-0-1", tone: "defect", cx: 64, cy: 88, width: 16, height: 8, angle: Math.PI / 2 },
        { kind: "caliper", id: "caliper-0-2", tone: "signal", cx: 40, cy: 64, width: 16, height: 8, angle: Math.PI },
        { kind: "point", id: "caliper-0-2", tone: "signal", x: 39.88, y: 64, cross: true },
        { kind: "circle", tone: "normal", cx: 64, cy: 64, r: 24 },
      ],
    },
  ],
};

const measure = vi.fn<LabBackend["measure"]>();

function fakeBackend(): LabBackend {
  return {
    canOpenFiles: () => false,
    listImages: async () => [IMAGE],
    listModels: async () => [MODEL],
    listCalibrations: async () => [],
    imageUrl: async () => "data:image/png;base64,",
    measure,
    onProgress: () => () => {},
    onBatchProgress: () => () => {},
    onThumbReady: () => () => {},
    prewarmThumbnails: async () => {},
  } as unknown as LabBackend;
}

vi.mock("../api/backend", async (importOriginal) => {
  const actual = await importOriginal<typeof BackendModule>();
  return { ...actual, getBackend: () => fakeBackend() };
});

const lab: {
  overlay: MeasurePrimitive[];
  hover: (point: { x: number; y: number }) => void;
  click: (point: { x: number; y: number }) => void;
} = { overlay: [], hover: () => {}, click: () => {} };

const frame = vi.fn<StageHandle["frame"]>();

/** Selects the frame, stands in for the canvas, and exposes the state the test reads. */
function Harness() {
  const state = useLab();
  const { images, models, calibrations, selectedImage, selectImage, canvas, overlayPicker } = state;
  useEffect(() => {
    if (images.length > 0 && selectedImage === null) selectImage(images[0]!.id);
  }, [images, selectedImage, selectImage]);
  useImperativeHandle(canvas, () => ({ frame, fit: () => {}, zoomTo: () => {} }), []);
  useEffect(() => {
    lab.overlay = state.overlay;
    lab.hover = (point) => overlayPicker?.onHover(overlayPicker.pick(point, 0));
    lab.click = (point) => overlayPicker?.onSelect(overlayPicker.pick(point, 0));
  });
  if (selectedImage === null || models.length === 0) return <p>loading</p>;
  return <MeasureTab image={selectedImage} models={models} calibrations={calibrations} />;
}

function renderTab() {
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

async function runMeasure() {
  fireEvent.click(await screen.findByRole("button", { name: "Add object" }));
  const run = screen.getByRole("button", { name: "Run measure" });
  await waitFor(() => expect(run.hasAttribute("disabled")).toBe(false));
  fireEvent.click(run);
  await screen.findByRole("table", { name: "Calipers" });
}

function rows(): HTMLElement[] {
  return within(screen.getByRole("table", { name: "Calipers" })).getAllByRole("row").slice(1);
}

function rowIds(): string[] {
  return rows().map((row) => within(row).getAllByRole("cell")[0]!.textContent.trim());
}

/** The state each caliper's box is drawn in, by id. */
function boxStates(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const p of lab.overlay) if (p.kind === "caliper" && p.id) out[p.id] = p.state ?? "default";
  return out;
}

describe("MeasureTab's caliper inventory", () => {
  beforeEach(() => {
    frame.mockReset();
    measure.mockReset().mockResolvedValue(RESPONSE);
  });

  it("lists every caliper with its verdict, edge, residual and amplitude", async () => {
    renderTab();
    await runMeasure();

    expect(rowIds()).toEqual(["0.0", "0.1", "0.2"]);
    const [hit, rejected] = rows();
    expect(within(hit!).getByText("hit")).toBeTruthy();
    expect(within(hit!).getByText("-0.05")).toBeTruthy();
    expect(within(hit!).getByText("-0.018")).toBeTruthy();
    expect(within(hit!).getByText("64.2")).toBeTruthy();
    expect(within(rejected!).getByText("too oblique")).toBeTruthy();
  });

  it("filters to the hits or the rejections, and fades the rest on the canvas", async () => {
    renderTab();
    await runMeasure();

    fireEvent.click(screen.getByRole("radio", { name: /Rejected 1/ }));
    expect(rowIds()).toEqual(["0.1"]);
    expect(boxStates()).toEqual({ "caliper-0-0": "dimmed", "caliper-0-1": "default", "caliper-0-2": "dimmed" });

    fireEvent.click(screen.getByRole("radio", { name: /Hits 2/ }));
    expect(rowIds()).toEqual(["0.0", "0.2"]);
  });

  it("links hover both ways with the canvas", async () => {
    renderTab();
    await runMeasure();

    fireEvent.pointerEnter(rows()[2]!);
    expect(boxStates()["caliper-0-2"]).toBe("hover");
    // The edge mark goes with its box.
    expect(lab.overlay.find((p) => p.kind === "point" && p.id === "caliper-0-2")?.state).toBe("hover");
    fireEvent.pointerLeave(rows()[2]!);

    act(() => lab.hover({ x: 66, y: 93 }));
    expect(rows()[1]!.getAttribute("data-state")).toBe("active");
    expect(rows()[0]!.getAttribute("data-state")).toBeNull();
  });

  it("selects from either side and shows the selected caliper's profile", async () => {
    renderTab();
    await runMeasure();

    act(() => lab.click({ x: 90, y: 65 }));
    expect(boxStates()["caliper-0-0"]).toBe("selected");
    expect(await screen.findByRole("img", { name: "Intensity along caliper 0.0" })).toBeTruthy();
    expect(screen.getByText(/-0\.050 px from nominal, falling/)).toBeTruthy();

    fireEvent.click(rows()[1]!);
    expect(screen.getByRole("img", { name: "Intensity along caliper 0.1" })).toBeTruthy();
    expect(screen.getByText(/rejected \(too oblique\), so nothing from it went into the fit/)).toBeTruthy();
  });

  it("steps with the arrows, frames with F, and clears with Esc", async () => {
    renderTab();
    await runMeasure();

    fireEvent.keyDown(window, { key: "ArrowDown" });
    expect(boxStates()["caliper-0-0"]).toBe("selected");
    fireEvent.keyDown(window, { key: "ArrowUp" });
    expect(boxStates()["caliper-0-2"]).toBe("selected");

    fireEvent.keyDown(window, { key: "f" });
    expect(frame).toHaveBeenCalledTimes(1);
    const [rect] = frame.mock.calls[0]!;
    // Caliper 0.2 is centred on (40, 64), framed with its surroundings rather than alone.
    expect(rect.x + rect.width / 2).toBeCloseTo(40, 4);
    expect(rect.y + rect.height / 2).toBeCloseTo(64, 4);
    expect(rect.width).toBeGreaterThanOrEqual(64);

    fireEvent.keyDown(window, { key: "Escape" });
    expect(Object.values(boxStates()).includes("selected")).toBe(false);
    expect(screen.queryByRole("img", { name: /Intensity along caliper/ })).toBeNull();
  });
});
