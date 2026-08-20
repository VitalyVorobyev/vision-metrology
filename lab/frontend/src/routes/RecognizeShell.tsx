/**
 * The Recognize workspace's frame: the three steps, and the gate on each.
 *
 * A step is blocked rather than empty when its input is missing — see
 * `Stepper` for why that distinction is the whole point of showing a sequence.
 */

import type { ReactNode } from "react";

import { AppShell } from "../shell/AppShell";
import { Stepper } from "../shell/Stepper";
import { useLab } from "../state/LabContext";

export function RecognizeShell({ children }: { children: ReactNode }) {
  const { selectedImage, models, matches } = useLab();

  const steps = [
    {
      to: "/recognize/teach",
      label: "Teach",
      blockedBy: selectedImage ? undefined : "Open a frame first.",
    },
    {
      to: "/recognize/find",
      label: "Find",
      blockedBy:
        models.length === 0 ? "Teach a model first." : selectedImage ? undefined : "Open a frame first.",
    },
    {
      to: "/recognize/verify",
      label: "Verify",
      blockedBy: matches.length === 0 ? "Run a search first." : undefined,
    },
  ];

  return <AppShell steps={<Stepper steps={steps} label="Recognize steps" />} inspector={children} />;
}
