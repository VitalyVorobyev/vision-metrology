/**
 * Measuring against a found pose, and the canonical crops that come out of it.
 *
 * Two steps rather than two of six flat tabs: aligning is what you do *with* a
 * measurement's fixture, not a separate activity.
 */

import { useEffect } from "react";

import { toMeasurePrimitives } from "../overlay/toMeasurePrimitive";
import { AlignTab } from "../tabs/AlignTab";
import { MeasureTab } from "../tabs/MeasureTab";
import { AppShell } from "../shell/AppShell";
import { Stepper } from "../shell/Stepper";
import { useLab } from "../state/LabContext";
import { Empty } from "@vitavision/lab-ui";

function GaugeShell({ children }: { children: React.ReactNode }) {
  const { selectedImage, models } = useLab();
  const steps = [
    {
      to: "/gauge/measure",
      label: "Measure",
      blockedBy: models.length === 0 ? "Teach a model first." : undefined,
    },
    {
      to: "/gauge/align",
      label: "Align",
      blockedBy: models.length === 0 ? "Teach a model first." : undefined,
    },
  ];
  return (
    <AppShell
      steps={<Stepper steps={steps} label="Gauge steps" />}
      inspector={selectedImage === null ? <Empty>Open a frame first.</Empty> : children}
    />
  );
}

export function MeasurePage() {
  const { selectedImage, models, calibrations, setOverlay, setRoiMode } = useLab();
  useEffect(() => setRoiMode(false), [setRoiMode]);
  return (
    <GaugeShell>
      {selectedImage && (
        <MeasureTab
          image={selectedImage}
          models={models}
          calibrations={calibrations}
          onResult={(o) => setOverlay(toMeasurePrimitives(o))}
        />
      )}
    </GaugeShell>
  );
}

export function AlignPage() {
  const { selectedImage, models, setRoiMode } = useLab();
  useEffect(() => setRoiMode(false), [setRoiMode]);
  return (
    <GaugeShell>{selectedImage && <AlignTab image={selectedImage} models={models} />}</GaugeShell>
  );
}
