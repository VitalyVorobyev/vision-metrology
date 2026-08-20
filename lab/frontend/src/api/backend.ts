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
import type { components, paths } from "./generated";

type Schemas = components["schemas"];

export type ImageOut = Schemas["ImageOut"];
export type ImageTier = Schemas["Tier"];
export type ModelCreateRequest = Schemas["ModelCreateRequest"];
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
export type RectifyRequest = Schemas["RectifyRequest"];
export type RectifyResponse = Schemas["RectifyResponse"];
export type RectifyMatchOut = Schemas["RectifyMatchOut"];
export type Roi = NonNullable<ModelCreateRequest["roi"]>;
export type AngleRange = NonNullable<FindRequest["angle_range"]>;
export type CalibrationOut = Schemas["CalibrationOut"];
export type PlaneIn = Schemas["PlaneIn"];
export type DisplacementRequest = Schemas["DisplacementRequest"];
export type DisplacementResponse = Schemas["DisplacementResponse"];
export type DisplacementPairOut = Schemas["DisplacementPairOut"];

/** The operations every tab/component needs, transport-agnostic. */
export interface LabBackend {
  health(): Promise<{ status: string }>;
  listImages(): Promise<ImageOut[]>;
  uploadImage(file: File): Promise<ImageOut>;
  /** Built, not fetched — consumed by `<img src>` so the browser owns the request,
   * caching and decoding. */
  imageUrl(imageId: string, tier: ImageTier): string;
  listModels(): Promise<ModelOut[]>;
  teachModel(req: ModelCreateRequest): Promise<ModelOut>;
  find(req: FindRequest): Promise<FindResponse>;
  measure(req: MeasureRequest): Promise<MeasureResponse>;
  rectify(req: RectifyRequest): Promise<RectifyResponse>;
  /** Built, not fetched — same reasoning as `imageUrl`. The response's own
   * `crop_url` is server-relative, so this is how a caller turns a match `index`
   * into something an `<img src>` can load. */
  rectifyCropUrl(imageId: string, modelId: string, index: number): string;
  listCalibrations(): Promise<CalibrationOut[]>;
  uploadCalibration(file: File): Promise<CalibrationOut>;
  displacement(req: DisplacementRequest): Promise<DisplacementResponse>;
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

    imageUrl(imageId: string, tier: ImageTier) {
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
  };
}

let cached: LabBackend | undefined;

/**
 * The backend for this session.
 *
 * TODO(tauri-backend): once a desktop shell exists, detect it here (`isTauriShell()`,
 * already wired through `shell.ts`) and return a `tauriBackend` that speaks to the
 * bundled sidecar over IPC instead of loopback HTTP. Every call site already goes
 * through `LabBackend`, so that swap should not touch `tabs/`, `components/`, or
 * `transforms.ts`.
 */
export function getBackend(): LabBackend {
  if (!cached) {
    if (isTauriShell()) {
      // Not yet implemented — the shell only injects `apiBaseUrl` today, which the
      // http backend already resolves via `baseUrl.ts`. Fall through to it.
    }
    cached = createHttpBackend();
  }
  return cached;
}
