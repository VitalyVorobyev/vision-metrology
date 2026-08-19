// Thin fetch wrappers over `/api`. Vite's dev proxy (vite.config.ts) forwards `/api/*` to
// the backend at :8000, so this needs no base URL or env var — same origin in dev and in
// a built bundle served by the same host.

import type {
  FindRequest,
  FindResponse,
  ImageOut,
  MeasureRequest,
  MeasureResponse,
  ModelCreateRequest,
  ModelOut,
} from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return (await res.json()) as T;
}

export function imageTierUrl(imageId: string, tier: "thumb" | "preview" | "full"): string {
  return `/api/images/${imageId}/${tier}`;
}

export async function uploadImage(file: File): Promise<ImageOut> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/images", { method: "POST", body: form });
  return json<ImageOut>(res);
}

export async function listImages(): Promise<ImageOut[]> {
  return json<ImageOut[]>(await fetch("/api/images"));
}

export async function createModel(req: ModelCreateRequest): Promise<ModelOut> {
  const res = await fetch("/api/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return json<ModelOut>(res);
}

export async function listModels(): Promise<ModelOut[]> {
  return json<ModelOut[]>(await fetch("/api/models"));
}

export async function runFind(req: FindRequest): Promise<FindResponse> {
  const res = await fetch("/api/find", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return json<FindResponse>(res);
}

export async function runMeasure(req: MeasureRequest): Promise<MeasureResponse> {
  const res = await fetch("/api/measure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return json<MeasureResponse>(res);
}
