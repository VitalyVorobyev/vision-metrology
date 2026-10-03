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
import { TooltipProvider } from "@vitavision/ui";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { StageHandle } from "@vitavision/stage2d";
import type { RefObject } from "react";

import type * as BackendModule from "../api/backend";
import type { ContourOut, ImageOut, LabBackend } from "../api/backend";
import type { SelectMode } from "./contourSelection";
import { LabProvider, useLab } from "../state/LabContext";
import { CanvasStage } from "./CanvasStage";
import { describeContours } from "./contourSelection";

const IMAGE: ImageOut = { id: "img-1", filename: "8.bmp", width: 1280, height: 1024 } as ImageOut;
/** What the fake backend lists; a test may swap in a different frame. */
let frames: ImageOut[] = [IMAGE];

const CONTOURS: ContourOut[] = [
  { id: 0, points: [600, 400, 700, 400, 700, 500], closed: false, length: 200, mean_strength: 0.7 },
  { id: 1, points: [900, 800, 950, 850], closed: false, length: 70, mean_strength: 0.3 },
];

vi.mock("../api/backend", async (importOriginal) => {
  const actual = await importOriginal<typeof BackendModule>();
  return {
    ...actual,
    getBackend: () =>
      ({
        canOpenFiles: () => true,
        listImages: async () => frames,
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
          this,
        );
      }
      unobserve() {}
      disconnect() {}
    },
  );
}

const selectSpy = vi.fn<(ids: number[], mode: SelectMode) => void>();
const roiSpy = vi.fn();
/** Hears the lab's handle on the stage, as a panel holds it. */
const handleSpy = vi.fn<(handle: RefObject<StageHandle | null>) => void>();

function Seed({ withContours = false }: { withContours?: boolean }) {
  const { images, selectedImage, selectImage, roi, setRoi, setRoiMode, setContourSelection, canvas } =
    useLab();
  useEffect(() => handleSpy(canvas), [canvas]);
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
    frames = [IMAGE];
  });

  /** The last region the provider saw. */
  const lastRoi = () => roiSpy.mock.calls[roiSpy.mock.calls.length - 1]?.[0] as number[] | undefined;

  /**
   * Image pixels → client pixels for a 1200×500 viewport at fit, computed here rather than
   * read off the DOM so a drag can be aimed at a handle by the coordinate it is drawn at.
   */
  const SCALE = 500 / 1024;
  const TX = (1200 - 1280 * SCALE) / 2;
  // `+ 0.5`: image coordinates name pixel centres, and client coordinates name edges.
  const client = (x: number, y: number) => ({
    clientX: (x + 0.5) * SCALE + TX,
    clientY: (y + 0.5) * SCALE,
  });
  /** A drag whose moves arrive at the window — how the sweep listens. */
  const drag = (from: Element, to: { clientX: number; clientY: number }, init = {}) => {
    fireEvent.pointerDown(from, { button: 0, pointerId: 1, ...init });
    fireEvent(window, new window.PointerEvent("pointermove", to));
    fireEvent(window, new window.PointerEvent("pointerup", to));
  };
  /** A drag whose moves arrive at the pressed element — what pointer capture does, and how
   * the region editor listens. */
  const captured = (
    target: Element,
    from: { clientX: number; clientY: number },
    to: { clientX: number; clientY: number },
  ) => {
    fireEvent.pointerDown(target, { button: 0, pointerId: 1, ...from });
    fireEvent.pointerMove(target, { pointerId: 1, ...to });
    fireEvent.pointerUp(target, { button: 0, pointerId: 1, ...to });
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
    const image = stage.querySelector<HTMLImageElement>("img");
    if (image) expect(image.className).not.toContain("object-contain");
  });

  it("draws the region's eight handles inside the transform", async () => {
    withViewport({ width: 1200, height: 500 });
    const { container } = renderCanvas();
    await screen.findByRole("application");

    const stage = container.querySelector("[data-stage]") as HTMLElement;
    const handles = stage.querySelectorAll("[data-handle]");
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

  it("resizes the region by its corner handle, and commits once, on release", async () => {
    withViewport({ width: 1200, height: 500 });
    const { container } = renderCanvas();
    await screen.findByRole("application");
    const before = lastRoi()!;
    const commits = roiSpy.mock.calls.length;

    const se = container.querySelector("[data-stage] [data-handle=se]")!;
    // Grab the south-east corner where it is drawn and pull it out and down.
    captured(se, client(before[0]! + before[2]!, before[1]! + before[3]!), client(1000, 800));

    const after = lastRoi()!;
    expect(after[0]).toBeCloseTo(before[0]!, 3);
    expect(after[1]).toBeCloseTo(before[1]!, 3);
    // The corner lands on the pixel the pointer let go over.
    expect(after[0]! + after[2]!).toBeCloseTo(1000, 3);
    expect(after[1]! + after[3]!).toBeCloseTo(800, 3);
    // The shared region changes on release only: the Teach panel re-extracts on a change.
    expect(roiSpy.mock.calls.length).toBe(commits + 1);
  });

  it("moves the region by its interior, held inside the image", async () => {
    withViewport({ width: 1200, height: 500 });
    renderCanvas();
    await screen.findByRole("application");
    const [x, y, w, h] = lastRoi()!;

    // From the centre, far past the bottom-right corner: it slides to the edge, unshrunk.
    const region = screen.getByRole("button", { name: /^Region:/ });
    captured(region, client(x! + w! / 2, y! + h! / 2), client(5000, 5000));

    expect(lastRoi()).toEqual([1280 - w!, 1024 - h!, w, h]);
  });

  /*
   * Both of these are regressions rather than features. A sweep begun on a contour used to
   * pan, because declining a press hands it *up* to the stage, not down to the layer beneath.
   * And a sweep begun inside the region used to move the region, because its interior took
   * the press first — and contours are usually inside the region.
   */

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

  it("sweeps from inside the region instead of moving it", async () => {
    withViewport({ width: 1200, height: 500 });
    renderCanvas(true);
    await screen.findByRole("application");
    const before = lastRoi()!;
    const commits = roiSpy.mock.calls.length;

    const region = screen.getByRole("button", { name: /^Region:/ });
    drag(region, client(1000, 900), { ...client(550, 380), shiftKey: true });

    const swept = selectSpy.mock.calls[selectSpy.mock.calls.length - 1]!;
    expect(swept).toEqual([[0, 1], "replace"]);
    expect(roiSpy.mock.calls.length).toBe(commits);
    expect(lastRoi()).toEqual(before);
  });

  it("opens a small frame at fit, not at 1:1", async () => {
    frames = [{ ...IMAGE, width: 320, height: 240 }];
    withViewport({ width: 1200, height: 500 });
    const { container } = renderCanvas();
    await screen.findByRole("application");

    const stage = container.querySelector("[data-stage]") as HTMLElement;
    // 500 / 240: the frame fills the viewport's height.
    expect(stage.style.transform).toContain(`scale(${500 / 240})`);
  });

  it("hands the panels the stage's own handle", async () => {
    withViewport({ width: 1200, height: 500 });
    const { container } = renderCanvas();
    await screen.findByRole("application");

    const handle = handleSpy.mock.calls[handleSpy.mock.calls.length - 1]![0];
    act(() => handle.current!.zoomTo(2));
    const stage = container.querySelector("[data-stage]") as HTMLElement;
    expect(stage.style.transform).toContain("scale(2)");
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
