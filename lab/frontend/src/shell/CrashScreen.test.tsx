import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { CrashBoundary, installCrashHandlers } from "./CrashScreen";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function Throws(): never {
  throw new Error("`Tooltip` must be used within `TooltipProvider`");
}

describe("CrashBoundary", () => {
  it("shows what was thrown, where React 19 would otherwise unmount the root", () => {
    // React logs the caught error itself; the point of the test is the screen, not the log.
    vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <CrashBoundary>
        <Throws />
      </CrashBoundary>,
    );

    expect(screen.getByRole("alert").textContent).toContain(
      "`Tooltip` must be used within `TooltipProvider`",
    );
  });

  it("stays out of the way while nothing is wrong", () => {
    render(
      <CrashBoundary>
        <p>the workbench</p>
      </CrashBoundary>,
    );

    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByText("the workbench")).toBeTruthy();
  });
});

describe("installCrashHandlers", () => {
  it("paints an uncaught error into an empty root — the black-window case", () => {
    const container = document.createElement("div");
    document.body.append(container);
    installCrashHandlers(container);

    window.dispatchEvent(
      new ErrorEvent("error", {
        error: new Error("styles.css failed to load"),
        message: "styles.css failed to load",
      }),
    );

    expect(container.textContent).toContain("styles.css failed to load");
    container.remove();
  });

  it("leaves a working screen alone: a live UI reports its own failures", () => {
    const container = document.createElement("div");
    container.append(document.createElement("main"));
    document.body.append(container);
    installCrashHandlers(container);

    window.dispatchEvent(
      new ErrorEvent("error", { error: new Error("a find that failed"), message: "boom" }),
    );

    expect(container.textContent).not.toContain("a find that failed");
    container.remove();
  });
});
