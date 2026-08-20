/**
 * The two camera-level jobs: what moved between frames, and what several
 * calibrated cameras see of one plane.
 *
 * Neither has anything to do with teaching a model, which is why neither is on
 * screen while you are.
 */

import { Empty } from "@vitavision/lab-ui";
import { useEffect } from "react";

import { BirdsEyeTab } from "../tabs/BirdsEyeTab";
import { MotionTab } from "../tabs/MotionTab";
import { AppShell } from "../shell/AppShell";
import { Stepper } from "../shell/Stepper";
import { useLab } from "../state/LabContext";

function CameraShell({ children }: { children: React.ReactNode }) {
  const { images } = useLab();
  const steps = [
    {
      to: "/camera/motion",
      label: "Motion",
      blockedBy: images.length < 2 ? "Open at least two frames." : undefined,
    },
    { to: "/camera/mosaic", label: "Bird's-eye" },
  ];
  return <AppShell steps={<Stepper steps={steps} label="Camera steps" />} inspector={children} />;
}

export function MotionPage() {
  const { images, roi, setRoiMode } = useLab();
  // Motion measures displacement inside a window the user draws, so this step
  // owns the drag layer the same way Teach does.
  useEffect(() => {
    setRoiMode(true);
    return () => setRoiMode(false);
  }, [setRoiMode]);
  return (
    <CameraShell>
      <MotionTab images={images} windowRoi={roi} />
    </CameraShell>
  );
}

export function MosaicPage() {
  const { images, calibrations, setRoiMode } = useLab();
  useEffect(() => setRoiMode(false), [setRoiMode]);
  return (
    <CameraShell>
      {images.length === 0 ? (
        <Empty>Open some frames first.</Empty>
      ) : (
        <BirdsEyeTab images={images} calibrations={calibrations} />
      )}
    </CameraShell>
  );
}
