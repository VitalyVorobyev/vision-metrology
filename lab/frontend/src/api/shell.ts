/**
 * Capabilities a future desktop shell injects.
 *
 * Nothing under `src/` imports a Tauri API directly — that is what keeps the browser
 * path first-class. A shell instead injects what it can offer onto
 * `window.__VM_LAB__` before the page loads, and `baseUrl.ts` / `backend.ts`
 * feature-detect it. Today only `apiBaseUrl` exists (the sidecar's ephemeral port);
 * more capabilities (a native file picker, IPC transport) land here as they're needed.
 */

declare global {
  interface Window {
    __VM_LAB__?: {
      apiBaseUrl?: string;
    };
  }
}

export function isTauriShell(): boolean {
  return typeof globalThis.window?.__VM_LAB__ !== "undefined";
}
