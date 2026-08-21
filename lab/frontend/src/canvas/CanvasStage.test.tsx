/**
 * The property the old canvas could not hold: every layer registered with the photograph,
 * at any viewport shape, and interactive layers that live inside the transform.
 *
 * The bug this replaces was invisible in a screenshot at one window size and obvious at the
 * next, so it is worth asserting structurally — the stage is laid out at the image's own
 * pixel size and every overlay's `viewBox` covers exactly that image, which is what makes the
 * mapping the identity no matter what shape the panel is — offset by the half pixel between
 * "the centre of pixel i" (what a detector reports) and "the leading edge of pixel i" (what
 * CSS and SVG mean by it).
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@vitavision/lab-ui";
import { fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ContourOut, ImageOut, LabBackend } from "../api/backend";
import { LabProvider, useLab } from "../state/LabContext";
import { CanvasStage } from "./CanvasStage";
import { describeContours } from "./contourSelection";

const IMAGE: ImageOut = { id: "img-1", filename: "8.bmp", width: 1280, height: 1024 } as ImageOut;

const CONTOURS: ContourOut[] = [
  { id: 0, points: [600, 400, 700, 400, 700, 500], closed: false, length: 200, mean_strength: 0.7 },
  { id: 1, points: [900, 800, 950, 850], closed: false, length: 70, mean_strength: 0.3 },
];

vi.mock("../api/backend", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/backend")>();
  return {
    ...actual,
    getBackend: () =>
      ({
        canOpenFiles: () => true,
        listImages: async () => [IMAGE],
        listModels: async () => [],
        listCalibrations: async () => [],
        imageUrl: async () => "data:image/png;base64,",
        onProgress: () => () => {},
        onThumbReady: () => () => {},
        prewarmThumbnails: async () => {},
      }) as unknown as LabBackend,
  };
});

/** happy-dom lays nothing out, so the viewport has to be told how big it is. */
function withViewport(box: { width: number; height: number }) {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(() => ({
    left: 0,
    top: 0,
    right: box.width,
    bottom: box.height,
    width: box.width,
    height: box.height,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  }));
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(private readonly callback: ResizeObserverCallback) {}
      observe(element: Element) {
        this.callback(
          [{ target: element, contentRect: box } as unknown as ResizeObserverEntry],
          this as unknown as ResizeObserver,
        );
      }
      unobserve() {}
      disconnect() {}
    },
  );
}

const selectSpy = vi.fn();
const roiSpy = vi.fn();

function Seed({ withContours = false }: { withContours?: boolean }) {
  const { images, selectedImage, selectImage, roi, setRoi, setRoiMode, setContourSelection } =
    useLab();
  useEffect(() => {
    if (roi) roiSpy(roi);
  }, [roi]);
  useEffect(() => {
    if (images.length > 0 && selectedImage === null) selectImage(images[0]!.id);
  }, [images, selectedImage, selectImage]);
  // Only once a frame is selected: `selectImage` clears the region and the contour layer,
  // so seeding them before it lands would be undone by it.
  useEffect(() => {
    if (selectedImage === null) return;
    setRoi([500, 350, 340, 275]);
    setRoiMode(true);
    if (withContours) {
      setContourSelection({
        contours: CONTOURS,
        stats: describeContours(CONTOURS),
        kept: new Set([0, 1]),
        selected: new Set(),
        hovered: null,
        order: [0, 1],
        onHover: () => {},
        onSelect: selectSpy,
        onKeep: () => {},
      });
    }
  }, [selectedImage, setRoi, setRoiMode, setContourSelection, withContours]);
  return null;
}

function Canvas() {
  const { selectedImage } = useLab();
  return selectedImage ? <CanvasStage image={selectedImage} /> : null;
}

function renderCanvas(withContours = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <TooltipProvider>
          <LabProvider>
            <Seed withContours={withContours} />
            <Canvas />
          </LabProvider>
        </TooltipProvider>
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

describe("CanvasStage", () => {
  beforeEach(() => {
    selectSpy.mockReset();
    roiSpy.mockReset();
  });

  /** The last region the provider saw. */
  const lastRoi = () => roiSpy.mock.calls[roiSpy.mock.calls.length - 1]?.[0] as number[] | undefined;

  /**
   * Image pixels → client pixels for a 1200×500 viewport at fit, computed here rather than
   * read off the DOM so a drag can be aimed at a handle by the coordinate it is drawn at.
   */
  const SCALE = 500 / 1024;
  const TX = (1200 - 1280 * SCALE) / 2;
  const client = (x: number, y: number) => ({ clientX: x * SCALE + TX, clientY: y * SCALE });
  const drag = (from: SVGElement, to: { clientX: number; clientY: number }, init = {}) => {
    fireEvent.pointerDown(from, { button: 0, pointerId: 1, ...init });
    fireEvent(window, new window.PointerEvent("pointermove", to));
    fireEvent(window, new window.PointerEvent("pointerup", to));
  };

  it.each([
    ["a wide panel", { width: 1200, height: 500 }],
    ["a tall panel", { width: 400, height: 900 }],
    ["a square panel", { width: 700, height: 700 }],
  ])("registers every layer with the photograph in %s", async (_name, box) => {
    withViewport(box);
    const { container } = renderCanvas(true);
    await screen.findByRole("application");

    const stage = container.querySelector("[data-stage]") as HTMLElement;
    // The layers are sized by this box, so it is the whole of the registration contract.
    expect(stage.style.width).toBe("1280px");
    expect(stage.style.height).toBe("1024px");

    const overlays = stage.querySelectorAll("svg");
    expect(overlays.length).toBeGreaterThanOrEqual(3); // results, surface, contours, roi
    for (const svg of overlays) {
      // `imageViewBox`, not `0 0 W H`: a contour vertex at `i` must land on the centre of
      // pixel `i`, not on its boundary. Every layer has to agree, or they disagree with
      // each other as well as with the photograph.
      expect(svg.getAttribute("viewBox")).toBe("-0.5 -0.5 1280 1024");
    }

    // And the photograph is laid out at that same size rather than letterboxed inside it —
    // the `object-contain` that used to disagree with the overlays is gone.
    const image = stage.querySelector("img") as HTMLImageElement | null;
    if (image) expect(image.className).not.toContain("object-contain");
  });

  it("draws the region's eight handles inside the transform", async () => {
    withViewport({ width: 1200, height: 500 });
    const { container } = renderCanvas();
    await screen.findByRole("application");

    const stage = container.querySelector("[data-stage]") as HTMLElement;
    const handles = stage.querySelectorAll('rect[style*="-resize"]');
    expect(handles.length).toBe(8);
    // Sized in image units so they come out a constant number of *screen* pixels — the
    // predecessor's datum handles were about three, which reads as decoration.
    const side = Number((handles[0] as SVGRectElement).getAttribute("width"));
    expect(side).toBeGreaterThan(9); // fit here is well under 1:1, so the square is larger
  });

  it("selects a contour from the canvas, and claims the press so it is not a pan", async () => {
    withViewport({ width: 1200, height: 500 });
    const { container } = renderCanvas(true);
    await screen.findByRole("application");

    const stage = container.querySelector("[data-stage]") as HTMLElement;
    const hitStroke = stage.querySelector('path[stroke="transparent"]') as SVGPathElement;
    expect(hitStroke).toBeTruthy();

    fireEvent.pointerDown(hitStroke, { button: 0, clientX: 100, clientY: 100, pointerId: 1 });
    expect(selectSpy).toHaveBeenCalledWith([0], "replace");

    fireEvent.pointerDown(hitStroke, { button: 0, clientX: 100, clientY: 100, pointerId: 2, metaKey: true });
    expect(selectSpy).toHaveBeenLastCalledWith([0], "toggle");
  });

  /*
   * Both of these are regressions rather than features. A drag begun on a handle used to do
   * nothing, because `setPointerCapture` routed its moves to the handle — a sibling of the
   * surface that carried the move handler, never an ancestor. And a sweep begun on a contour
   * used to pan, because declining a press hands it *up* to the stage, not down to the layer
   * beneath.
   */
  it("resizes the region by its corner handle, wherever the drag ends up", async () => {
    withViewport({ width: 1200, height: 500 });
    const { container } = renderCanvas();
    await screen.findByRole("application");
    const before = lastRoi()!;

    const stage = container.querySelector("[data-stage]") as HTMLElement;
    const handles = stage.querySelectorAll('rect[style*="nwse-resize"]');
    const se = handles[handles.length - 1] as SVGRectElement; // south-east

    // Grab the south-east corner where it is drawn and pull it out and down.
    drag(se, client(1000, 800), client(before[0]! + before[2]!, before[1]! + before[3]!));

    const after = lastRoi()!;
    expect(after[0]).toBeCloseTo(before[0]!, 3);
    expect(after[1]).toBeCloseTo(before[1]!, 3);
    expect(after[2]).toBeGreaterThan(before[2]!);
    expect(after[3]).toBeGreaterThan(before[3]!);
  });

  it("sweeps a selection even when the press lands on a contour", async () => {
    withViewport({ width: 1200, height: 500 });
    const { container } = renderCanvas(true);
    await screen.findByRole("application");

    const stage = container.querySelector("[data-stage]") as HTMLElement;
    const hitStroke = stage.querySelector('path[stroke="transparent"]') as SVGPathElement;

    drag(hitStroke, client(1000, 900), { ...client(100, 100), shiftKey: true });

    // A sweep, not the single-contour selection a plain press would have made.
    const swept = selectSpy.mock.calls[selectSpy.mock.calls.length - 1]!;
    expect(swept[1]).toBe("replace");
    expect(swept[0].length).toBeGreaterThan(1);
  });

  it("puts the zoom controls over the image, not in a panel", async () => {
    withViewport({ width: 1200, height: 500 });
    renderCanvas();
    const canvas = await screen.findByRole("application");

    for (const label of ["Zoom out", "Zoom in", "Fit to window", "Actual size (100%)"]) {
      expect(canvas.contains(screen.getByLabelText(label))).toBe(true);
    }
  });
});
