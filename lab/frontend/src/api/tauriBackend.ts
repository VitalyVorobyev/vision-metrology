/**
 * The desktop `LabBackend`: Tauri commands + events, never HTTP.
 *
 * Every method here is a thin `invoke()` call into `lab/frontend/src-tauri`'s command
 * layer (`src/commands/*.rs`), which calls `vision-metrology` directly — there is no
 * FastAPI process on the desktop build, and nothing here imports `openapi-fetch` or
 * touches `fetch`. Response *field names* are hand-kept in step with the Rust side's
 * `types.rs` — there is no generated client for this transport the way `openapi-fetch`
 * generates one for HTTP, so a renamed Rust field is a runtime mismatch, not a type
 * error. The anti-drift gate for that is
 * `lab/frontend/src-tauri/tests/contract_parity.rs` plus this file's own
 * `tauriBackend.test.ts` (mocked `invoke`, asserts the shape sent and returned).
 *
 * ## Images are files, not IPC payloads
 *
 * Both directions used to move pixels through `invoke`. Uploads went out as
 * `Array.from(new Uint8Array(...))` — a JSON array of numbers, so a 5 MB PNG
 * became ~15–20 MB of text to serialise and parse. Tiers came back as bytes
 * that had to be wrapped in a blob URL, and because `imageUrl` was synchronous
 * the first call returned a 1×1 placeholder and hoped a later render would pick
 * up the real one; the canvas stayed blank until an unrelated state change
 * happened to re-render it.
 *
 * Now neither direction carries pixels. Opening sends a **path**; the Rust side
 * reads the file. Tiers are PNG-encoded once into the app cache directory and
 * handed back as a path, which `convertFileSrc` turns into an `asset:` URL —
 * a real URL the webview loads, caches and decodes itself.
 */

import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open as openDialog } from "@tauri-apps/plugin-dialog";

import type {
  BatchFindRequest,
  BatchFindResponse,
  BatchProgress,
  CalibrationOut,
  ContourOut,
  DirEntry,
  DisplacementRequest,
  DisplacementResponse,
  FindRequestFull,
  FindResponse,
  ImageOut,
  ImageTier,
  LabBackend,
  MeasureRequest,
  MeasureResponse,
  ModelCreateRequest,
  ModelGeometryOut,
  ModelOut,
  MosaicRequest,
  MosaicResponse,
  OpProgress,
  RectifyRequest,
  RectifyResponse,
  Roi,
  TeachPreviewRequest,
  TeachPreviewResponse,
  ThumbEvent,
} from "./backend";

/** Image extensions the native picker offers — the set the Rust side can decode. */
const IMAGE_EXTENSIONS = ["png", "bmp", "pgm"];

/**
 * Blob-URL cache for the few things that only exist in memory.
 *
 * Rectified crops and model crops are computed per request and never written to
 * disk, so they cannot go through the asset protocol. Unlike the old image
 * bridge this is `async` all the way to the caller, so there is no placeholder
 * and no missed render.
 */
const blobUrls = new Map<string, string>();

async function blobUrlFor(
  key: string,
  cmd: string,
  args: Record<string, unknown>,
): Promise<string> {
  const cached = blobUrls.get(key);
  if (cached) return cached;
  const bytes = await invoke<ArrayBuffer | number[]>(cmd, args);
  const array = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : new Uint8Array(bytes);
  const url = URL.createObjectURL(new Blob([array], { type: "image/png" }));
  blobUrls.set(key, url);
  return url;
}

/** Drop a cached crop so the next ask recomputes it. */
function invalidateBlobs(prefix: string): void {
  for (const [key, url] of blobUrls) {
    if (key.startsWith(prefix)) {
      URL.revokeObjectURL(url);
      blobUrls.delete(key);
    }
  }
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
      // Kept for drag-and-drop, where all we have is a `File`. Opening from
      // disk should go through `pickImages`/`openImagePaths` instead, which
      // moves no bytes at all.
      const bytes = await bytesOf(file);
      return invoke<ImageOut>("images_upload", { filename: file.name, bytes });
    },

    async imageUrl(imageId: string, tier: ImageTier) {
      const path = await invoke<string>("image_tier_path", { imageId, tier });
      return convertFileSrc(path);
    },

    canOpenFiles: () => true,

    async pickImages() {
      const picked = await openDialog({
        multiple: true,
        directory: false,
        filters: [{ name: "Images", extensions: IMAGE_EXTENSIONS }],
      });
      if (picked === null) return [];
      return Array.isArray(picked) ? picked : [picked];
    },

    async pickFolder() {
      const picked = await openDialog({ multiple: false, directory: true });
      return typeof picked === "string" ? picked : null;
    },

    async openImagePaths(paths: string[]) {
      return invoke<ImageOut[]>("images_open_paths", { paths });
    },

    async scanDir(dir: string, recursive: boolean) {
      return invoke<DirEntry[]>("images_scan_dir", { dir, recursive });
    },

    async prewarmThumbnails(imageIds: string[]) {
      await invoke("prewarm_thumbnails", { imageIds });
    },

    onThumbReady(cb: (e: ThumbEvent) => void) {
      const pending = listen<ThumbEvent>("lab://thumb", (e) => cb(e.payload));
      return () => {
        void pending.then((un) => un());
      };
    },

    async listModels() {
      return invoke<ModelOut[]>("models_list");
    },

    async teachModel(req: ModelCreateRequest) {
      return invoke<ModelOut>("models_create", { req });
    },

    async teachPreview(req: TeachPreviewRequest) {
      const res = await invoke<{ contours: ContourOut[]; total_points: number }>(
        "teach_preview",
        { req },
      );
      return res as TeachPreviewResponse;
    },

    async modelGeometry(modelId: string, level: number, frame: "reference" | "model") {
      return invoke<ModelGeometryOut>("model_geometry", { modelId, level, frame });
    },

    async modelCropUrl(modelId: string, rect: Roi, pxPerUnit: number) {
      const key = `model-crop:${modelId}:${rect.join(",")}:${pxPerUnit}`;
      return blobUrlFor(key, "model_crop", {
        req: { model_id: modelId, rect, px_per_unit: pxPerUnit },
      });
    },

    async find(req: FindRequestFull) {
      return invoke<FindResponse>("find", { req });
    },

    async batchFind(req: BatchFindRequest) {
      return invoke<BatchFindResponse>("batch_find", { req });
    },

    onBatchProgress(cb: (p: BatchProgress) => void) {
      // `listen` resolves to the unlisten function; callers want to unsubscribe
      // synchronously (a React effect cleanup), so hold the promise and call
      // through it. Unsubscribing before it resolves still works.
      const pending = listen<BatchProgress>("lab://batch", (e) => cb(e.payload));
      return () => {
        void pending.then((un) => un());
      };
    },

    onProgress(cb: (p: OpProgress) => void) {
      const pending = listen<OpProgress>("lab://progress", (e) => cb(e.payload));
      return () => {
        void pending.then((un) => un());
      };
    },

    async measure(req: MeasureRequest) {
      return invoke<MeasureResponse>("measure", { req });
    },

    async rectify(req: RectifyRequest) {
      // A fresh rectify replaces this model's crops on the Rust side, so any
      // blob URL we handed out for them is now stale.
      invalidateBlobs(`crop:${req.image_id}:${req.model_id}:`);
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
      // A name, not a URL — the crop lives in the Rust side's cache and has to
      // be fetched. `resolveCropUrl` below is the fetch; see its doc on
      // `LabBackend` for why the two are separate.
      return `crop:${imageId}:${modelId}:${index}`;
    },

    async resolveCropUrl(key: string) {
      if (!key.startsWith("crop:")) return key;
      const [, imageId, modelId, index] = key.split(":");
      return blobUrlFor(key, "rectify_crop", { imageId, modelId, index: Number(index) });
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
