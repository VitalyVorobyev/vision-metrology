/**
 * Detecting the desktop shell.
 *
 * The Tauri desktop build talks to `vision-metrology` through native **commands and
 * events** (`tauriBackend.ts`), not HTTP — there is no sidecar, no injected base URL, no
 * ephemeral port to discover. Detection is therefore exactly "is this page running
 * inside a Tauri webview", answered by `@tauri-apps/api/core`'s own `isTauri()`: it
 * checks for `window.__TAURI_INTERNALS__`, which the Tauri runtime sets up before any
 * page script runs, regardless of `tauri.conf.json`'s `withGlobalTauri` setting. Nothing
 * else under `src/` imports a Tauri API directly — that is what keeps the browser path
 * (`httpBackend`) first-class and the two shells swappable behind `LabBackend`.
 */

import { isTauri } from "@tauri-apps/api/core";

export function isTauriShell(): boolean {
  return isTauri();
}
