/**
 * The transport boundary: one `LabBackend` interface, implemented today over HTTP.
 *
 * Every operation the UI needs is named here rather than called ad hoc, so a second
 * implementation — a future `tauriBackend` talking to a bundled sidecar over IPC instead
 * of loopback HTTP — can slot in behind `getBackend()` without the tabs/components
 * changing at all. Nothing outside this file (and `imageUrl`'s `<img src>` consumers)
 * should import `openapi-fetch` or call `fetch` directly.
 *
 * Types are aliases into `generated.ts`, produced from `lab/contract/openapi.json` by
 * `bun run generate:api`. A field the backend renames becomes a type error here.
 */

import createClient from "openapi-fetch";

import { resolveApiBaseUrl } from "./baseUrl";
import { isTauriShell } from "./shell";
import { createTauriBackend } from "./tauriBackend";
import type { components, paths } from "./generated";

type Schemas = components["schemas"];

/**
 * The contract's `ImageOut`, plus the desktop shell's own `path`.
 *
 * A browser upload has no path to report and the FastAPI schema has no such
 * field; an image opened from disk on the desktop does, and a folder view that
 * cannot tell you where a frame came from is missing the one thing a folder
 * view is for. Optional rather than a separate type, so every call site keeps
 * working against one shape.
 */
export type ImageOut = Schemas["ImageOut"] & { path?: string | null };
export type ImageTier = Schemas["Tier"];
/**
 * The contract's teach request, plus the desktop shell's curation fields.
 *
 * `keep_contours`/`origin`/`reference_angle` have no FastAPI counterpart for
 * the same reason `teach_preview` does not (see the desktop-only note below):
 * curating contours needs a preview step the browser shell cannot offer. All
 * three are optional and absent means the old behaviour — the whole rectangle,
 * a centroid origin, and the reference image's own axes.
 */
export type ModelCreateRequest = Schemas["ModelCreateRequest"] & {
  keep_contours?: number[] | null;
  origin?: [number, number] | null;
  reference_angle?: number;
};
export type ModelOut = Schemas["ModelOut"];
export type FindRequest = Schemas["FindRequest"];
export type FindResponse = Schemas["FindResponse"];
export type MatchOut = Schemas["MatchOut"];
export type MeasureRequest = Schemas["MeasureRequest"];
export type MeasureResponse = Schemas["MeasureResponse"];
export type MeasureObjectIn = Schemas["MeasureObjectIn"];
export type MeasureObjectResultOut = Schemas["MeasureObjectResultOut"];
export type MeasureConfigIn = Schemas["MeasureConfigIn"];
export type FitConfigIn = Schemas["FitConfigIn"];
export type FixtureIn = Schemas["FixtureIn"];
export type CaliperResultOut = Schemas["CaliperResultOut"];
export type CaliperProfileOut = Schemas["CaliperProfileOut"];
export type EdgeMarkOut = Schemas["EdgeMarkOut"];
export type OverlayPrimitiveOut = Schemas["OverlayPrimitiveOut"];
export type CropSpecIn = Schemas["CropSpecIn"];
/**
 * The contract's rectify request, plus the rest of the search that produced
 * the match list being rectified.
 *
 * `rectify` re-runs the search to place its crops, so it has to run the same
 * one the caller's visible match list came from — otherwise the instances are
 * renumbered and a crop index names a different part. Optional, defaulted, and
 * absent for the browser shell, which does not expose these knobs.
 */
export type RectifyRequest = Schemas["RectifyRequest"] & {
  angle_range?: [number, number] | null;
  scale_range?: [number, number] | null;
  refinement?: string | null;
  min_contrast?: number | null;
  tuning?: SearchTuning | null;
};
export type RectifyResponse = Schemas["RectifyResponse"];
export type RectifyMatchOut = Schemas["RectifyMatchOut"];
export type Roi = NonNullable<ModelCreateRequest["roi"]>;
export type AngleRange = NonNullable<FindRequest["angle_range"]>;
export type CalibrationOut = Schemas["CalibrationOut"];
export type PlaneIn = Schemas["PlaneIn"];
export type DisplacementRequest = Schemas["DisplacementRequest"];
export type DisplacementResponse = Schemas["DisplacementResponse"];
export type DisplacementPairOut = Schemas["DisplacementPairOut"];
export type MosaicCameraIn = Schemas["MosaicCameraIn"];
export type MosaicGridIn = Schemas["MosaicGridIn"];
export type MosaicRequest = Schemas["MosaicRequest"];
export type MosaicCameraCoverageOut = Schemas["MosaicCameraCoverageOut"];
export type MosaicResponse = Schemas["MosaicResponse"];

/* ---------------------------------------------------------------------------
 * Desktop-only shapes.
 *
 * These have no `openapi.json` counterpart on purpose. Opening a folder and
 * running a model across it are things the browser shell has no honest analogue
 * for — there is no native picker behind a web page, and no path it may read —
 * so mirroring them into FastAPI would mean writing routes with no consumer.
 * The contract and its fixtures keep covering the *shared* subset (images,
 * models, find, measure, rectify, displacement), where parity is still checked
 * exactly, by `src-tauri/tests/contract_parity.rs`. See lab/README.md.
 * ------------------------------------------------------------------------ */

/** One image file found by `scanDir`, described without decoding it. */
export interface DirEntry {
  path: string;
  name: string;
  bytes: number;
  width: number;
  height: number;
}

/** A candidate contour offered before a model is built. */
export interface ContourOut {
  id: number;
  /** `[x0, y0, x1, y1, …]` in image coordinates. */
  points: number[];
  closed: boolean;
  length: number;
  mean_strength: number;
}

export interface TeachPreviewRequest {
  image_id: string;
  roi: Roi;
  min_contrast: number;
}

export interface TeachPreviewResponse {
  contours: ContourOut[];
  total_points: number;
}

export interface ModelGeometryOut {
  model_id: string;
  level: number;
  origin: [number, number];
  reference_angle: number;
  /** `[x, y, dx, dy]` per point, flattened. */
  points: number[];
  frame: "reference" | "model";
}

/** The search-effort knobs, mirroring `matching::ShapeSearchTuning`. */
export interface SearchTuning {
  greediness?: number;
  angle_step?: number | null;
  scale_step?: number | null;
  last_level?: number;
  max_candidates?: number;
  coarse_score_factor?: number;
}

export interface BatchFindRequest {
  model_id: string;
  image_ids: string[];
  min_score: number;
  max_matches?: number | null;
  angle_range?: [number, number] | null;
  scale_range?: [number, number] | null;
  refinement?: string | null;
  min_contrast?: number | null;
  tuning?: SearchTuning | null;
}

export interface BatchFindItem {
  image_id: string;
  matches: MatchOut[];
  elapsed_ms: number;
  error: string | null;
}

export interface BatchFindResponse {
  items: BatchFindItem[];
}

/** A thumbnail finished warming. */
export interface ThumbEvent {
  image_id: string;
  done: number;
  total: number;
}

export interface BatchProgress {
  done: number;
  total: number;
  image_id: string;
  matches: number;
  best_score: number | null;
  elapsed_ms: number;
}

/** A `lab://progress` tick: an operation started, or finished with a duration. */
export interface OpProgress {
  op: string;
  stage: "started" | "finished";
  elapsed_ms: number | null;
}

/** Extra `FindRequest` fields the desktop shell accepts beyond the contract's. */
export type FindRequestFull = FindRequest & {
  scale_range?: [number, number] | null;
  refinement?: string | null;
  min_contrast?: number | null;
  tuning?: SearchTuning | null;
};

/** The operations every tab/component needs, transport-agnostic. */
export interface LabBackend {
  health(): Promise<{ status: string }>;
  listImages(): Promise<ImageOut[]>;
  uploadImage(file: File): Promise<ImageOut>;
  /**
   * The URL an `<img src>` should load for this tier.
   *
   * **Asynchronous on purpose.** Over HTTP a URL needs no data and could be
   * built synchronously; on the desktop the tier has to be rendered and cached
   * first, and there is no honest synchronous answer. The previous synchronous
   * signature was satisfied there by returning a 1×1 placeholder and swapping
   * in the real image on some later render — which is why the canvas stayed
   * blank until an unrelated state change happened to re-render it. Call this
   * through `useImageUrl`, which owns the loading state.
   */
  imageUrl(imageId: string, tier: ImageTier): Promise<string>;
  listModels(): Promise<ModelOut[]>;
  teachModel(req: ModelCreateRequest): Promise<ModelOut>;
  find(req: FindRequestFull): Promise<FindResponse>;

  // -- desktop-only (see the note above `DirEntry`) --------------------------

  /** Is this shell able to open files and folders from disk? */
  canOpenFiles(): boolean;
  /** Native picker; resolves to the chosen paths, or `[]` if cancelled. */
  pickImages(): Promise<string[]>;
  /** Native picker; resolves to the chosen folder, or `null` if cancelled. */
  pickFolder(): Promise<string | null>;
  /** Register images already on disk. No pixels cross the IPC boundary. */
  openImagePaths(paths: string[]): Promise<ImageOut[]>;
  /** List a folder's image files without decoding any of them. */
  scanDir(dir: string, recursive: boolean): Promise<DirEntry[]>;
  /** Render missing thumbnails ahead of the grid reaching them. */
  prewarmThumbnails(imageIds: string[]): Promise<void>;
  /** Subscribe to thumbnail-warming progress. Returns an unsubscribe function. */
  onThumbReady(cb: (e: ThumbEvent) => void): () => void;
  /** Candidate contours to curate before a model is built. */
  teachPreview(req: TeachPreviewRequest): Promise<TeachPreviewResponse>;
  /** A model's own points, for drawing what was actually learned. */
  modelGeometry(
    modelId: string,
    level: number,
    frame: "reference" | "model",
  ): Promise<ModelGeometryOut>;
  /** The model's reference image through `rect`, as a loadable URL. */
  modelCropUrl(modelId: string, rect: Roi, pxPerUnit: number): Promise<string>;
  /** One model over a whole set of frames. */
  batchFind(req: BatchFindRequest): Promise<BatchFindResponse>;
  /** Subscribe to per-image batch progress. Returns an unsubscribe function. */
  onBatchProgress(cb: (p: BatchProgress) => void): () => void;
  /** Subscribe to operation timings. Returns an unsubscribe function. */
  onProgress(cb: (p: OpProgress) => void): () => void;
  measure(req: MeasureRequest): Promise<MeasureResponse>;
  rectify(req: RectifyRequest): Promise<RectifyResponse>;
  /** Built, not fetched — same reasoning as `imageUrl`. The response's own
   * `crop_url` is server-relative, so this is how a caller turns a match `index`
   * into something an `<img src>` can load. */
  rectifyCropUrl(imageId: string, modelId: string, index: number): string;
  /**
   * Turn what `rectifyCropUrl` returned into something an `<img src>` loads.
   *
   * Over HTTP that is already a URL and this is the identity. On the desktop a
   * crop only exists in memory — it is computed per `rectify` call and never
   * written to disk, so unlike an image tier it cannot be served as a file and
   * has to come back over IPC as bytes. Splitting "name the crop" from "fetch
   * the crop" keeps `rectifyCropUrl` synchronous for the call sites that just
   * need an identity, without any of them importing a transport.
   */
  resolveCropUrl(key: string): Promise<string>;
  listCalibrations(): Promise<CalibrationOut[]>;
  uploadCalibration(file: File): Promise<CalibrationOut>;
  displacement(req: DisplacementRequest): Promise<DisplacementResponse>;
  mosaic(req: MosaicRequest): Promise<MosaicResponse>;
  /** Built, not fetched — same reasoning as `imageUrl`. `feather=true` switches from the
   * default no-blend priority composite to the opt-in display-only linear feather. */
  mosaicImageUrl(mosaicId: string, feather: boolean): string;
  mosaicSourceIdUrl(mosaicId: string): string;
}

/**
 * Turn an openapi-fetch result into a value or an exception.
 *
 * The client returns `{ data, error }` rather than throwing, which is right for a
 * library and wrong for a TanStack Query `queryFn`/`mutationFn` — Query decides between
 * its success and error states by whether the function threw.
 *
 * `T` is asserted rather than inferred from `result.data`: openapi-fetch's `Readable<>`
 * response-typing helper matches a fixed-length tuple with its own `T extends (infer E)[]`
 * array branch and widens it to `E[]`, so a schema tuple like `roi: [number, number,
 * number, number]` comes back typed as plain `number[]`. The JSON on the wire still has
 * exactly four elements — this only re-asserts what `generated.ts` already promised.
 */
function unwrap<T>(result: { data?: unknown; error?: unknown }, what: string): T {
  if (result.error !== undefined || result.data === undefined) {
    throw new Error(describeError(result.error, what));
  }
  return result.data as T;
}

function describeError(error: unknown, what: string): string {
  if (error && typeof error === "object" && "detail" in error) {
    const { detail } = error as { detail?: unknown };
    if (typeof detail === "string") return detail;
    // FastAPI's validation errors are a list of per-field objects; the first one is
    // almost always the one worth showing.
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0] as { msg?: unknown };
      if (typeof first?.msg === "string") return first.msg;
    }
  }
  return `The backend did not return ${what}.`;
}

function createHttpBackend(): LabBackend {
  const baseUrl = resolveApiBaseUrl();
  const client = createClient<paths>({ baseUrl });

  return {
    async health() {
      const res = await client.GET("/api/health");
      return unwrap<{ status: string }>(res, "health status");
    },

    async listImages() {
      const res = await client.GET("/api/images");
      return unwrap<ImageOut[]>(res, "the image list");
    },

    async uploadImage(file: File) {
      const body = new FormData();
      body.append("file", file);
      const res = await client.POST("/api/images", {
        // openapi-fetch's default body serializer passes `FormData` straight through
        // and lets the browser set `Content-Type` with its boundary; the generated
        // request type (`{ file: string }`) describes the multipart *schema*, not the
        // runtime payload, so it does not accept `FormData` without this cast.
        body: body as unknown as { file: string },
      });
      return unwrap<ImageOut>(res, "the uploaded image");
    },

    async imageUrl(imageId: string, tier: ImageTier) {
      // Nothing to await here — the browser fetches this lazily from the
      // `<img>`. The signature is async because the desktop shell genuinely
      // needs it to be; see `LabBackend.imageUrl`.
      return `${baseUrl}/api/images/${imageId}/${tier}`;
    },

    async listModels() {
      const res = await client.GET("/api/models");
      return unwrap<ModelOut[]>(res, "the model list");
    },

    async teachModel(req: ModelCreateRequest) {
      const res = await client.POST("/api/models", { body: req });
      return unwrap<ModelOut>(res, "the taught model");
    },

    async find(req: FindRequest) {
      const res = await client.POST("/api/find", { body: req });
      return unwrap<FindResponse>(res, "find results");
    },

    async measure(req: MeasureRequest) {
      const res = await client.POST("/api/measure", { body: req });
      return unwrap<MeasureResponse>(res, "measure results");
    },

    async rectify(req: RectifyRequest) {
      const res = await client.POST("/api/rectify", { body: req });
      return unwrap<RectifyResponse>(res, "rectify results");
    },

    rectifyCropUrl(imageId: string, modelId: string, index: number) {
      return `${baseUrl}/api/rectify/${imageId}/${modelId}/${index}`;
    },

    async resolveCropUrl(key: string) {
      return key;
    },

    async listCalibrations() {
      const res = await client.GET("/api/calibration");
      return unwrap<CalibrationOut[]>(res, "the calibration list");
    },

    async uploadCalibration(file: File) {
      const body = new FormData();
      body.append("file", file);
      const res = await client.POST("/api/calibration", {
        // Same FormData/schema mismatch as `uploadImage` above.
        body: body as unknown as { file: string },
      });
      return unwrap<CalibrationOut>(res, "the uploaded calibration");
    },

    async displacement(req: DisplacementRequest) {
      const res = await client.POST("/api/displacement", { body: req });
      return unwrap<DisplacementResponse>(res, "displacement results");
    },

    async mosaic(req: MosaicRequest) {
      const res = await client.POST("/api/mosaic", { body: req });
      return unwrap<MosaicResponse>(res, "mosaic results");
    },

    mosaicImageUrl(mosaicId: string, feather: boolean) {
      return `${baseUrl}/api/mosaic/${mosaicId}/image${feather ? "?feather=true" : ""}`;
    },

    mosaicSourceIdUrl(mosaicId: string) {
      return `${baseUrl}/api/mosaic/${mosaicId}/source_id`;
    },

    // -- desktop-only ---------------------------------------------------------
    //
    // A web page cannot open a native picker or read a path, and the contract
    // deliberately does not pretend otherwise. `canOpenFiles()` is what the UI
    // branches on, so these throw rather than return something plausible: a
    // silent empty list would look like an empty folder.

    canOpenFiles: () => false,
    pickImages: () => unsupported("Opening files from disk"),
    pickFolder: () => unsupported("Opening a folder"),
    openImagePaths: () => unsupported("Opening images by path"),
    scanDir: () => unsupported("Scanning a folder"),
    // The browser fetches thumbnails over HTTP with its own cache; there is
    // nothing to warm and nothing to report.
    prewarmThumbnails: async () => {},
    onThumbReady: () => () => {},
    teachPreview: () => unsupported("Contour preview"),
    modelGeometry: () => unsupported("Model geometry"),
    modelCropUrl: () => unsupported("Model crops"),
    batchFind: () => unsupported("Batch find"),
    // No events over HTTP; a no-op unsubscribe keeps every call site uniform.
    onBatchProgress: () => () => {},
    onProgress: () => () => {},
  };
}

function unsupported(feature: string): never {
  throw new Error(`${feature} needs the desktop app (see lab/README.md).`);
}

let cached: LabBackend | undefined;

/**
 * The backend for this session.
 *
 * `isTauriShell()` (`shell.ts`) detects the Tauri webview via `@tauri-apps/api`'s own
 * `isTauri()` — nothing injected, nothing to configure. Every call site goes through
 * `LabBackend`, so this is the only place that chooses a transport; `tabs/`,
 * `components/`, and `transforms.ts` are unaffected either way.
 */
export function getBackend(): LabBackend {
  if (!cached) {
    cached = isTauriShell() ? createTauriBackend() : createHttpBackend();
  }
  return cached;
}
