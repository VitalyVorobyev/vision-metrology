/**
 * Where the backend lives.
 *
 * Ported from visual-anomaly-lab's `api/baseUrl.ts`: the same bundle is meant to run in
 * two places eventually and must find the backend in each.
 *
 *  1. **Tauri shell** (future) — the shell would inject `window.__VM_LAB__.apiBaseUrl`
 *     before the app loads, since a bundled sidecar's port is only known at runtime.
 *  2. **`vite dev` / a built bundle in a browser** — `VITE_API_BASE_URL` if set, else the
 *     documented `uv run uvicorn ... --port 8000` default.
 *
 * The shell hands the URL over as an injected global rather than through
 * `@tauri-apps/api`, so nothing under `src/` needs to import a Tauri API and the browser
 * path stays first-class.
 */

// The shape of the injected global is declared once, in `shell.ts`.
import "./shell";

export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export interface BaseUrlSources {
  /** Injected by a future Tauri shell once its sidecar has announced its port. */
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
