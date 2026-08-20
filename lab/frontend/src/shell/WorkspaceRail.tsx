/**
 * The four things this workbench is for, as places rather than as tabs.
 *
 * The lab used to present six flat tabs — Teach, Find, Measure, Align, Motion,
 * Bird's-eye — side by side in one strip. That reads as six equal, unrelated
 * options, when in fact the first four are one task done in order and the last
 * two are separate jobs entirely. Grouping them into workspaces means the
 * screen only ever offers what the current job needs: nothing about camera
 * mosaics is on screen while a model is being taught.
 */

import { cn, focusRing } from "@vitavision/lab-ui";
import { Boxes, Crosshair, Images, Ruler } from "lucide-react";
import { NavLink } from "react-router";

const WORKSPACES = [
  { to: "/library", icon: Images, label: "Library", hint: "Frames and models" },
  { to: "/recognize", icon: Crosshair, label: "Recognize", hint: "Teach, find, verify" },
  { to: "/gauge", icon: Ruler, label: "Gauge", hint: "Measure and align" },
  { to: "/camera", icon: Boxes, label: "Camera", hint: "Motion and mosaic" },
] as const;

export function WorkspaceRail() {
  return (
    <nav
      aria-label="Workspaces"
      className="flex w-[4.5rem] shrink-0 flex-col gap-1 border-r border-line bg-surface px-2 py-3"
    >
      {WORKSPACES.map(({ to, icon: Icon, label, hint }) => (
        <NavLink
          key={to}
          to={to}
          title={hint}
          className={({ isActive }) =>
            cn(
              "flex flex-col items-center gap-1 rounded-control px-1 py-2 text-[10px] font-medium transition-colors",
              focusRing,
              isActive
                ? "bg-signal/10 text-signal"
                : "text-fg-subtle hover:bg-raised hover:text-fg-muted",
            )
          }
        >
          {({ isActive }) => (
            <>
              <Icon className="size-5" aria-hidden />
              <span aria-current={isActive ? "page" : undefined}>{label}</span>
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}
