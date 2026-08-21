/**
 * The one screen this app must always be able to draw: the one that says it broke.
 *
 * A desktop build has no console. When the React tree throws during render, React 19
 * unmounts the whole root — and `index.html`'s `color-scheme: dark` then paints the empty
 * document black. That is indistinguishable from a hung window, a failed dev server or a
 * crashed backend, and it is what a `<ThemeToggle>` rendered outside `TooltipProvider`
 * looked like from the outside: a black window, no message, nowhere to start.
 *
 * So this file deliberately depends on **nothing**: no `@vitavision/lab-ui` import, no
 * Tailwind class, no design token, no router. Inline styles only, with its own colours. A
 * crash screen that needs the stylesheet is another black window on the day the stylesheet
 * is what failed.
 */

import { Component } from "react";
import type { CSSProperties, ErrorInfo, ReactNode } from "react";

const PANEL = {
  position: "fixed",
  inset: "0",
  zIndex: "2147483647",
  overflow: "auto",
  padding: "2rem",
  background: "#17181a",
  color: "#e8e8ea",
  font: "13px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace",
} satisfies CSSProperties;

const TITLE = {
  margin: "0 0 0.75rem",
  font: "600 15px/1.4 ui-sans-serif, system-ui, sans-serif",
  color: "#ff6b6b",
} satisfies CSSProperties;

const MESSAGE = {
  margin: "0 0 1rem",
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
} satisfies CSSProperties;

const DETAIL = {
  margin: "0",
  padding: "0.75rem",
  borderRadius: "6px",
  background: "#101113",
  color: "#9aa0a6",
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
} satisfies CSSProperties;

const HEADLINE = "The lab could not draw its window.";

/** Whatever was thrown, as something with a message and maybe a stack. */
function asError(thrown: unknown): Error {
  return thrown instanceof Error ? thrown : new Error(String(thrown));
}

export function CrashScreen({
  message,
  detail,
}: {
  message: string;
  detail?: string | null;
}) {
  return (
    <div style={PANEL} role="alert">
      <h1 style={TITLE}>{HEADLINE}</h1>
      <p style={MESSAGE}>{message}</p>
      {detail != null && detail !== "" && <pre style={DETAIL}>{detail}</pre>}
    </div>
  );
}

interface CrashBoundaryState {
  error: Error | null;
  componentStack: string | null;
}

/**
 * Catches what React can catch: a throw during render, in a lifecycle, or in a
 * constructor, anywhere below it. Everything else — a module that throws while it is being
 * evaluated, a rejected promise nobody awaited — reaches [`installCrashHandlers`] instead.
 */
export class CrashBoundary extends Component<{ children: ReactNode }, CrashBoundaryState> {
  override state: CrashBoundaryState = { error: null, componentStack: null };

  static getDerivedStateFromError(thrown: unknown): Pick<CrashBoundaryState, "error"> {
    return { error: asError(thrown) };
  }

  override componentDidCatch(thrown: unknown, info: ErrorInfo): void {
    // Still log: the console is worth having when there *is* one (browser build, or the
    // desktop build's inspector), and the screen below is worth having when there is not.
    console.error(thrown);
    this.setState({ componentStack: info.componentStack ?? null });
  }

  override render(): ReactNode {
    const { error, componentStack } = this.state;
    if (error === null) return this.props.children;
    return <CrashScreen message={error.message} detail={componentStack ?? error.stack ?? null} />;
  }
}

/**
 * The same screen, painted straight into the DOM.
 *
 * Not React, on purpose: by the time a global `error` fires, the root may already have
 * unmounted itself, and asking React to render is asking the thing that just died.
 *
 * Paints **only when the container is empty** — that is the exact condition this exists
 * for (a blank screen with no explanation). A rejected `invoke` while the UI is up and
 * reporting it is not a crash, and must not blow the workbench away.
 */
export function installCrashHandlers(container: HTMLElement): void {
  const paint = (message: string, detail: string | null): void => {
    if (container.childElementCount > 0) return;

    const panel = document.createElement("div");
    Object.assign(panel.style, PANEL);
    panel.setAttribute("role", "alert");

    const title = document.createElement("h1");
    Object.assign(title.style, TITLE);
    title.textContent = HEADLINE;
    panel.append(title);

    const body = document.createElement("p");
    Object.assign(body.style, MESSAGE);
    body.textContent = message;
    panel.append(body);

    if (detail !== null && detail !== "") {
      const pre = document.createElement("pre");
      Object.assign(pre.style, DETAIL);
      pre.textContent = detail;
      panel.append(pre);
    }

    container.append(panel);
  };

  window.addEventListener("error", (event: ErrorEvent) => {
    const error = event.error != null ? asError(event.error) : null;
    paint(error?.message ?? event.message, error?.stack ?? null);
  });

  window.addEventListener("unhandledrejection", (event: PromiseRejectionEvent) => {
    const error = asError(event.reason);
    paint(error.message, error.stack ?? null);
  });
}
