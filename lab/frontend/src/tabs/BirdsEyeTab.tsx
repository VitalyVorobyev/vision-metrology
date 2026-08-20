import {
  Button,
  Dialog,
  ErrorBox,
  Field,
  NumberInput,
  Panel,
  ReadoutStrip,
  RESET_VIEW,
  Section,
  SegmentedControl,
  Select,
  Switch,
  Table,
  ZoomPanCanvas,
  type View,
} from "@vitavision/lab-ui";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { getBackend } from "../api/backend";
import type { CalibrationOut, ImageOut, MosaicCameraCoverageOut, MosaicRequest } from "../api/backend";

type OverlayMode = "none" | "source_id" | "feather";

const OVERLAY_OPTIONS: { value: OverlayMode; label: string }[] = [
  { value: "none", label: "None" },
  { value: "source_id", label: "Source tint" },
  { value: "feather", label: "Feather" },
];

/** Camera picker state: one image id per calibration camera index, `""` until chosen.
 * Exported and pure so "run enabled once every slot is filled" is covered without
 * rendering the component. */
export function allCamerasChosen(picks: string[]): boolean {
  return picks.length > 0 && picks.every((id) => id !== "");
}

/** Builds the `MosaicRequest.cameras` array from the per-index picker state, skipping
 * unset slots (defensive — `allCamerasChosen` is what gates the Run button). */
export function cameraPicksToRequest(picks: string[]): MosaicRequest["cameras"] {
  return picks
    .map((image_id, camera_index) => ({ camera_index, image_id }))
    .filter((c) => c.image_id !== "");
}

/** Bird's-eye mosaic: N calibrated cameras' rectified views of the calibration's shared
 * `z = 0` plane, composited with no-blending nearest-camera-centre priority
 * (`POST /api/mosaic`, `crates/vision-metrology/tests/mosaic.rs` / `examples/birdseye_mosaic.rs`
 * mirror the exact same rule). The overlay toggle switches which of the three server-rendered
 * views is shown — plain composite, `source_id` tint, or the opt-in display-only feather —
 * never changes what was computed. */
export function BirdsEyeTab({
  images,
  calibrations,
}: {
  images: ImageOut[];
  calibrations: CalibrationOut[];
}) {
  const [calibrationId, setCalibrationId] = useState("");
  const [cameraPicks, setCameraPicks] = useState<string[]>([]);
  const [autoGrid, setAutoGrid] = useState(true);
  const [originX, setOriginX] = useState(0);
  const [originY, setOriginY] = useState(0);
  const [mmPerPx, setMmPerPx] = useState(0.05);
  const [gridWidth, setGridWidth] = useState(800);
  const [gridHeight, setGridHeight] = useState(600);
  const [overlay, setOverlay] = useState<OverlayMode>("none");
  const [zoomOpen, setZoomOpen] = useState(false);
  const [zoomView, setZoomView] = useState<View>(RESET_VIEW);

  const calibration = calibrations.find((c) => c.id === calibrationId) ?? null;

  const selectCalibration = (id: string) => {
    setCalibrationId(id);
    const cal = calibrations.find((c) => c.id === id);
    setCameraPicks(cal ? Array(cal.n_cameras).fill("") : []);
  };

  const setCameraPick = (index: number, imageId: string) => {
    setCameraPicks((prev) => {
      const next = [...prev];
      next[index] = imageId;
      return next;
    });
  };

  const mutation = useMutation({
    mutationFn: () =>
      getBackend().mosaic({
        calibration_id: calibrationId,
        cameras: cameraPicksToRequest(cameraPicks),
        grid: autoGrid
          ? {}
          : {
              origin_mm: [originX, originY],
              mm_per_px: mmPerPx,
              width: gridWidth,
              height: gridHeight,
            },
      }),
    onSuccess: () => setOverlay("none"),
  });

  const canRun = calibrationId !== "" && allCamerasChosen(cameraPicks);

  const overlayUrl = (id: string) => {
    if (overlay === "source_id") return getBackend().mosaicSourceIdUrl(id);
    return getBackend().mosaicImageUrl(id, overlay === "feather");
  };

  const openZoom = () => {
    setZoomView(RESET_VIEW);
    setZoomOpen(true);
  };

  return (
    <div className="flex flex-col gap-4">
      <Panel title="Bird's-eye">
        <div className="flex flex-col gap-4">
          <Section step={1} title="Calibration">
            <Field label="Calibration">
              <Select
                value={calibrationId}
                onValueChange={selectCalibration}
                options={calibrations.map((c) => ({
                  value: c.id,
                  label: `${c.id} (${c.n_cameras} cameras, ${c.format})`,
                }))}
                placeholder="Choose a calibration…"
              />
            </Field>
          </Section>

          {calibration && (
            <Section
              step={2}
              title="Pick one image per camera"
              hint={`Calibration has ${calibration.n_cameras} camera(s).`}
            >
              <div className="flex flex-col gap-2">
                {cameraPicks.map((picked, i) => (
                  <Field key={i} label={`Camera ${i}`}>
                    <Select
                      value={picked}
                      onValueChange={(id) => setCameraPick(i, id)}
                      options={images.map((img) => ({ value: img.id, label: `${img.id} (${img.filename})` }))}
                      placeholder="Choose an image…"
                    />
                  </Field>
                ))}
              </div>
            </Section>
          )}

          <Section step={3} title="Grid" hint="Auto-fits from the cameras' own footprints on the plane by default.">
            <Switch checked={autoGrid} onCheckedChange={setAutoGrid} label="Auto-fit" />
            {!autoGrid && (
              <div className="mt-3 grid grid-cols-2 gap-3">
                <Field label="Origin x" annotation="mm">
                  <NumberInput value={originX} onChange={(e) => setOriginX(Number(e.target.value))} />
                </Field>
                <Field label="Origin y" annotation="mm">
                  <NumberInput value={originY} onChange={(e) => setOriginY(Number(e.target.value))} />
                </Field>
                <Field label="mm / px">
                  <NumberInput
                    min={0.001}
                    step={0.01}
                    value={mmPerPx}
                    onChange={(e) => setMmPerPx(Number(e.target.value))}
                  />
                </Field>
                <Field label="Width" annotation="px">
                  <NumberInput min={1} value={gridWidth} onChange={(e) => setGridWidth(Number(e.target.value))} />
                </Field>
                <Field label="Height" annotation="px">
                  <NumberInput min={1} value={gridHeight} onChange={(e) => setGridHeight(Number(e.target.value))} />
                </Field>
              </div>
            )}
          </Section>

          <Button variant="primary" disabled={!canRun} loading={mutation.isPending} onClick={() => mutation.mutate()}>
            Build mosaic
          </Button>
          {mutation.isError && <ErrorBox>{(mutation.error as Error).message}</ErrorBox>}
        </div>
      </Panel>

      {mutation.data && (
        <Panel
          title="Mosaic"
          actions={
            <SegmentedControl
              value={overlay === "none" ? "" : overlay}
              defaultValue="none"
              options={OVERLAY_OPTIONS.map((o) => ({ value: o.value === "none" ? "" : o.value, label: o.label }))}
              onValueChange={(v) => setOverlay((v === "" ? "none" : v) as OverlayMode)}
              aria-label="Overlay mode"
            />
          }
        >
          <div className="flex flex-col gap-3">
            <ReadoutStrip
              items={[
                { label: "size", value: `${mutation.data.width} x ${mutation.data.height} px` },
                { label: "mm/px", value: mutation.data.mm_per_px.toFixed(4) },
                { label: "union coverage", value: `${(mutation.data.union_coverage_fraction * 100).toFixed(1)}%` },
                { label: "overlap", value: `${(mutation.data.overlap_fraction * 100).toFixed(1)}%` },
                {
                  label: "seam disparity p95",
                  value: mutation.data.seam_disparity_p95 == null ? "—" : mutation.data.seam_disparity_p95.toFixed(1),
                },
              ]}
            />

            <button
              type="button"
              className="overflow-hidden rounded border border-line hover:border-line-strong"
              onClick={openZoom}
            >
              {/* eslint-disable-next-line jsx-a11y/img-redundant-alt -- key is content identity, not decoration */}
              <img
                src={overlayUrl(mutation.data.id)}
                alt="bird's-eye mosaic"
                className="w-full bg-raised"
                draggable={false}
              />
            </button>

            <Table<MosaicCameraCoverageOut>
              columns={[
                { key: "camera", header: "camera", cell: (r) => r.camera_index },
                { key: "image", header: "image", cell: (r) => r.image_id },
                {
                  key: "coverage",
                  header: "coverage",
                  numeric: true,
                  cell: (r) => `${(r.coverage_fraction * 100).toFixed(1)}%`,
                },
              ]}
              rows={mutation.data.cameras}
              rowKey={(r) => r.camera_index}
            />
          </div>
        </Panel>
      )}

      <Dialog open={zoomOpen} onOpenChange={setZoomOpen} title="Bird's-eye mosaic">
        {zoomOpen && mutation.data && (
          <ZoomPanCanvas
            view={zoomView}
            onView={setZoomView}
            nativeWidth={mutation.data.width}
            className="h-96 w-full"
            fitLabel={null}
          >
            {/* eslint-disable-next-line jsx-a11y/img-redundant-alt -- key is content identity, not decoration */}
            <img
              src={overlayUrl(mutation.data.id)}
              alt="bird's-eye mosaic, zoomed"
              className="h-full w-full object-contain"
              draggable={false}
            />
          </ZoomPanCanvas>
        )}
      </Dialog>
    </div>
  );
}
