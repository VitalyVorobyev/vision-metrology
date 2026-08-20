/**
 * Where the browser (`httpBackend`) backend lives.
 *
 * Only ever consulted in browser mode: the desktop build talks to `vision-metrology`
 * through Tauri commands/events (`tauriBackend.ts`), never HTTP, so it has no base URL
 * to resolve at all — see `shell.ts` for how the two modes are told apart. This module
 * covers the two ways a browser session can still need a non-default URL:
 *
 *  1. **`VITE_API_BASE_URL`** — a build-time override for local development
 *     (`vite dev` against a backend on a non-default port).
 *  2. **An injected `window.__VM_LAB__.apiBaseUrl`** — a generic escape hatch for any
 *     future embedding that wants to point this same bundle at a different origin
 *     without a rebuild (e.g. serving it from behind a reverse proxy).
 *
 * Neither concerns Tauri; `resolveApiBaseUrl` is dead code on the desktop build since
 * `getBackend()` never constructs an `httpBackend` there.
 */

declare global {
  interface Window {
    __VM_LAB__?: {
      apiBaseUrl?: string;
    };
  }
}

export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export interface BaseUrlSources {
  /** Injected by an embedding shell before the page loads. */
  injected?: string | undefined;
  /** Build-time override for browser development. */
  env?: string | undefined;
}

function readSources(): BaseUrlSources {
  return {
    injected: globalThis.window?.__VM_LAB__?.apiBaseUrl,
    env: import.meta.env["VITE_API_BASE_URL"] as string | undefined,
  };
}

/** Resolve the backend base URL, without a trailing slash. */
export function resolveApiBaseUrl(sources: BaseUrlSources = readSources()): string {
  const candidate = [sources.injected, sources.env].find(
    (value) => typeof value === "string" && value.trim() !== "",
  );
  return (candidate ?? DEFAULT_API_BASE_URL).trim().replace(/\/+$/, "");
}
