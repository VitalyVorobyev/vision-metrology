/**
 * The desktop `LabBackend`: Tauri commands + events, never HTTP.
 *
 * Every method here is a thin `invoke()` call into `lab/frontend/src-tauri`'s command
 * layer (`src/commands/*.rs`), which calls `vision-metrology` directly — there is no
 * FastAPI process on the desktop build, and nothing here imports `openapi-fetch` or
 * touches `fetch`. Response *field names* are hand-kept in step with the Rust side's
 * `types.rs` (both were written to match `lab/contract/openapi.json`'s schemas closely
 * enough to reuse `generated.ts`'s types here — see that module's own doc comment) —
 * there is no generated client for this transport the way `openapi-fetch` generates one
 * for HTTP, so a renamed Rust field is a runtime mismatch, not a type error. The
 * anti-drift gate for that is `lab/frontend/src-tauri/tests/contract_parity.rs` plus
 * this file's own `tauriBackend.test.ts` (mocked `invoke`, asserts the shape sent and
 * returned).
 *
 * ## Binary data: `imageUrl` / `rectifyCropUrl`
 *
 * `LabBackend.imageUrl`/`rectifyCropUrl` are **synchronous** — every call site expects a
 * string it can drop straight into `<img src>`. The HTTP backend satisfies that for
 * free (a URL needs no data yet, the browser fetches it lazily); a command that returns
 * raw bytes (`tauri::ipc::Response`, no base64 — see `image_data`/`rectify_crop` in
 * `lib.rs`) does not. This module bridges the gap with a blob-URL cache: the first call
 * for a given key kicks off `invoke()` in the background and returns a 1x1 placeholder
 * immediately; once the bytes arrive they are cached as an object URL and every
 * *subsequent* call for that key returns the real image synchronously. A component that
 * calls `imageUrl` once and never re-renders will keep showing the placeholder — every
 * tab in this app re-renders on its own data changing (TanStack Query, selection state,
 * …) often enough in practice that this has not needed a dedicated re-render hook; if a
 * call site ever needs a guaranteed-fresh image on first paint, give it one.
 */

import { invoke } from "@tauri-apps/api/core";

import type {
  CalibrationOut,
  DisplacementRequest,
  DisplacementResponse,
  FindRequest,
  FindResponse,
  ImageOut,
  ImageTier,
  LabBackend,
  MeasureRequest,
  MeasureResponse,
  ModelCreateRequest,
  ModelOut,
  MosaicRequest,
  MosaicResponse,
  RectifyRequest,
  RectifyResponse,
} from "./backend";

// A 1x1 transparent PNG — shown for one render cycle while the real bytes arrive.
const PLACEHOLDER_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

const blobUrlCache = new Map<string, string>();
const pending = new Set<string>();

async function fetchBlobUrl(key: string, invokeCmd: string, args: Record<string, unknown>): Promise<void> {
  if (pending.has(key)) return;
  pending.add(key);
  try {
    const bytes = await invoke<ArrayBuffer | number[]>(invokeCmd, args);
    const array = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : new Uint8Array(bytes);
    const url = URL.createObjectURL(new Blob([array], { type: "image/png" }));
    const previous = blobUrlCache.get(key);
    if (previous) URL.revokeObjectURL(previous);
    blobUrlCache.set(key, url);
  } finally {
    pending.delete(key);
  }
}

/** Synchronous cache-or-kick-off-fetch, shared by `imageUrl` and `rectifyCropUrl`. */
function cachedBlobUrl(key: string, invokeCmd: string, args: Record<string, unknown>): string {
  const cached = blobUrlCache.get(key);
  if (cached) return cached;
  void fetchBlobUrl(key, invokeCmd, args);
  return PLACEHOLDER_DATA_URL;
}

async function bytesOf(file: File): Promise<number[]> {
  return Array.from(new Uint8Array(await file.arrayBuffer()));
}

function unsupported(feature: string): never {
  throw new Error(`${feature} is not available in the desktop build yet (see lab/README.md).`);
}

export function createTauriBackend(): LabBackend {
  return {
    async health() {
      const status = await invoke<string>("health");
      return { status };
    },

    async listImages() {
      return invoke<ImageOut[]>("images_list");
    },

    async uploadImage(file: File) {
      const bytes = await bytesOf(file);
      return invoke<ImageOut>("images_upload", { filename: file.name, bytes });
    },

    imageUrl(imageId: string, tier: ImageTier) {
      return cachedBlobUrl(`image:${imageId}:${tier}`, "image_data", { imageId, tier });
    },

    async listModels() {
      return invoke<ModelOut[]>("models_list");
    },

    async teachModel(req: ModelCreateRequest) {
      return invoke<ModelOut>("models_create", { req });
    },

    async find(req: FindRequest) {
      return invoke<FindResponse>("find", { req });
    },

    async measure(req: MeasureRequest) {
      return invoke<MeasureResponse>("measure", { req });
    },

    async rectify(req: RectifyRequest) {
      const resp = await invoke<RectifyResponse>("rectify", { req });
      // The Rust side reports `crop_key` (an internal cache key), not the browser
      // backend's `crop_url` — normalize to the shape `RectifyResponse` promises so
      // `rectifyCropUrl` (below) has something to key its blob cache on. Both fields
      // carry the same `"{image_id}/{model_id}/{index}"` triple; only the field name
      // differs, so this is a rename, not a reshape.
      type RustRectifyMatch = RectifyResponse["matches"][number] & { crop_key?: string };
      return {
        ...resp,
        matches: resp.matches.map((m) => {
          const raw = m as RustRectifyMatch;
          return { ...m, crop_url: raw.crop_key ?? raw.crop_url };
        }),
      };
    },

    rectifyCropUrl(imageId: string, modelId: string, index: number) {
      return cachedBlobUrl(`crop:${imageId}:${modelId}:${index}`, "rectify_crop", {
        imageId,
        modelId,
        index,
      });
    },

    async listCalibrations() {
      return invoke<CalibrationOut[]>("calibration_list");
    },

    async uploadCalibration(file: File) {
      const bytes = await bytesOf(file);
      return invoke<CalibrationOut>("calibration_upload", { filename: file.name, bytes });
    },

    async displacement(req: DisplacementRequest) {
      return invoke<DisplacementResponse>("displacement", { req });
    },

    // Deliberately not implemented this wave — the mosaic compositor
    // (`lab/backend/src/vm_lab/routers/mosaic.py`, ~315 lines) was not ported to a
    // Tauri command; see lab/README.md's desktop section for the reasoning.
    async mosaic(_req: MosaicRequest): Promise<MosaicResponse> {
      unsupported("Bird's-eye mosaic");
    },
    mosaicImageUrl(_mosaicId: string, _feather: boolean): string {
      unsupported("Bird's-eye mosaic");
    },
    mosaicSourceIdUrl(_mosaicId: string): string {
      unsupported("Bird's-eye mosaic");
    },
  };
}
