/**
 * The steps inside a workspace, in the order the data flows.
 *
 * A step whose precondition is not met is disabled and says why. That is the
 * whole point of showing them as a sequence: "Find" with no model is not an
 * empty results table, it is a step that cannot start yet, and the difference
 * is the difference between a user who knows what to do next and one who
 * concludes the search is broken.
 */

import { cn, focusRing } from "@vitavision/lab-ui";
import { NavLink } from "react-router";

export interface Step {
  to: string;
  label: string;
  /** When set, the step is unavailable and this says what is missing. */
  blockedBy?: string;
}

export function Stepper({ steps, label }: { steps: Step[]; label: string }) {
  return (
    <nav aria-label={label} className="flex items-center gap-1 px-4 py-2">
      {steps.map((step, i) => (
        <div key={step.to} className="flex items-center gap-1">
          {i > 0 && <span className="mx-1 h-px w-6 bg-line" aria-hidden />}
          {step.blockedBy ? (
            <span
              title={step.blockedBy}
              aria-disabled="true"
              className="flex cursor-not-allowed items-center gap-1.5 rounded-control px-2.5 py-1 text-xs text-fg-subtle"
            >
              <StepNumber index={i} muted />
              {step.label}
            </span>
          ) : (
            <NavLink
              to={step.to}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-1.5 rounded-control px-2.5 py-1 text-xs transition-colors",
                  focusRing,
                  isActive ? "bg-raised font-medium text-fg" : "text-fg-muted hover:text-fg",
                )
              }
            >
              <StepNumber index={i} />
              {step.label}
            </NavLink>
          )}
        </div>
      ))}
    </nav>
  );
}

function StepNumber({ index, muted = false }: { index: number; muted?: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        "grid size-4 place-items-center rounded-full text-[10px] font-semibold",
        muted ? "bg-line text-fg-subtle" : "bg-signal/15 text-signal",
      )}
    >
      {index + 1}
    </span>
  );
}
