/**
 * Where frames come from, and where a model is judged across all of them.
 *
 * The old image rail was a 224 px column of thumbnails with an "Upload
 * PNG/BMP" button — one file at a time, through a hidden `<input type=file>`,
 * with every byte marshalled over IPC as a JSON array of numbers. A metrology
 * capture is a folder, not a file, so that shape of "open" made the interesting
 * question (how does this model behave across the set?) impossible to ask.
 *
 * Opening a folder here reads only directory entries and image headers. Nothing
 * is decoded, nothing is copied, and the frames stay where the user put them —
 * so a set of several thousand opens as fast as the filesystem can list it.
 */

import {
  Badge,
  Button,
  Callout,
  Empty,
  ErrorBox,
  Field,
  NumberInput,
  Panel,
  ProgressBar,
  ScoreHistogram,
  Section,
  Select,
  Table,
} from "@vitavision/lab-ui";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";

import { getBackend } from "../api/backend";
import type { BatchFindItem, BatchProgress, DirEntry } from "../api/backend";
import { ImageGrid } from "../components/ImageGrid";
import { AppShell } from "../shell/AppShell";
import { useLab } from "../state/LabContext";

export function LibraryPage() {
  const backend = getBackend();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { images, models, selectImage, selectedImage, selectModel } = useLab();

  const [scanned, setScanned] = useState<DirEntry[] | null>(null);
  const [folder, setFolder] = useState<string | null>(null);

  const openFolder = useMutation({
    mutationFn: async () => {
      const dir = await backend.pickFolder();
      if (dir === null) return null;
      const entries = await backend.scanDir(dir, false);
      return { dir, entries };
    },
    onSuccess: (res) => {
      if (res === null) return;
      setFolder(res.dir);
      setScanned(res.entries);
    },
  });

  const openFiles = useMutation({
    mutationFn: async () => {
      const paths = await backend.pickImages();
      if (paths.length === 0) return [];
      return backend.openImagePaths(paths);
    },
    onSuccess: (opened) => {
      void queryClient.invalidateQueries({ queryKey: ["images"] });
      if (opened.length > 0) selectImage(opened[0]!.id);
    },
  });

  /**
   * Register the scanned folder's frames.
   *
   * Registration is still lazy about pixels — it reads a header and a content
   * hash per file — so this is the step that makes the frames addressable, not
   * the step that loads them.
   */
  const registerAll = useMutation({
    mutationFn: async () => {
      if (scanned === null) return [];
      return backend.openImagePaths(scanned.map((e) => e.path));
    },
    onSuccess: (opened) => {
      void queryClient.invalidateQueries({ queryKey: ["images"] });
      if (opened.length > 0) selectImage(opened[0]!.id);
      setScanned(null);
      // Warm the thumbnails in the background rather than making the grid pay
      // for them under the user's scroll. Fire-and-forget on purpose: the grid
      // already works without it, this only makes it smoother.
      void backend.prewarmThumbnails(opened.map((i) => i.id));
    },
  });

  const [warming, setWarming] = useState<{ done: number; total: number } | null>(null);
  useEffect(
    () =>
      backend.onThumbReady((e) => {
        setWarming(e.done >= e.total ? null : { done: e.done, total: e.total });
      }),
    [backend],
  );

  return (
    <AppShell
      fullBleed={
        images.length === 0 && scanned === null ? (
          <Empty
            action={
              backend.canOpenFiles() ? (
                <div className="flex gap-2">
                  <Button variant="primary" loading={openFolder.isPending} onClick={() => openFolder.mutate()}>
                    Open folder…
                  </Button>
                  <Button loading={openFiles.isPending} onClick={() => openFiles.mutate()}>
                    Open files…
                  </Button>
                </div>
              ) : undefined
            }
          >
            No frames yet.
          </Empty>
        ) : (
          <ImageGrid
            images={images}
            selectedId={selectedImage?.id ?? null}
            onSelect={selectImage}
            onOpen={(id) => {
              selectImage(id);
              void navigate("/recognize/teach");
            }}
          />
        )
      }
      inspector={
        <div className="flex flex-col gap-3">
          <Panel title="Frames">
            <div className="flex flex-col gap-3">
              {!backend.canOpenFiles() && (
                <Callout tone="info">
                  This is the browser build; opening files from disk needs the desktop app.
                </Callout>
              )}
              {backend.canOpenFiles() && (
                <div className="flex gap-2">
                  <Button size="sm" variant="primary" loading={openFolder.isPending} onClick={() => openFolder.mutate()}>
                    Open folder…
                  </Button>
                  <Button size="sm" loading={openFiles.isPending} onClick={() => openFiles.mutate()}>
                    Files…
                  </Button>
                </div>
              )}
              {openFolder.isError && <ErrorBox>{(openFolder.error as Error).message}</ErrorBox>}

              {scanned !== null && (
                <Section step={1} title={`${scanned.length} images in this folder`} hint={folder ?? undefined}>
                  <div className="flex flex-col gap-2">
                    <p className="text-xs text-fg-muted">
                      Nothing has been decoded yet — this is the directory listing and each file's
                      header.
                    </p>
                    <Button
                      size="sm"
                      variant="primary"
                      loading={registerAll.isPending}
                      onClick={() => registerAll.mutate()}
                    >
                      Add all {scanned.length}
                    </Button>
                  </div>
                </Section>
              )}

              {warming !== null && (
                <ProgressBar
                  fraction={warming.done / Math.max(warming.total, 1)}
                  label={`thumbnails ${warming.done}/${warming.total}`}
                />
              )}

              <p className="text-xs text-fg-subtle">
                <Badge tone="neutral">{images.length}</Badge> frames open
              </p>
            </div>
          </Panel>

          <BatchPanel
            onOpenFrame={(id) => {
              selectImage(id);
              void navigate("/recognize/verify");
            }}
            onPickModel={selectModel}
            models={models.map((m) => m.id)}
          />
        </div>
      }
    />
  );
}

/**
 * Run one model over every open frame.
 *
 * Sorted worst-first, because the useful question about a model is never "did
 * it work on a good frame" — it is where the score falls off, and how far.
 */
function BatchPanel({
  models,
  onOpenFrame,
  onPickModel,
}: {
  models: string[];
  onOpenFrame: (imageId: string) => void;
  onPickModel: (id: string) => void;
}) {
  const backend = getBackend();
  const { images } = useLab();
  const [modelId, setModelId] = useState(models[0] ?? "");
  const [minScore, setMinScore] = useState(0.5);
  const [progress, setProgress] = useState<BatchProgress | null>(null);

  useEffect(() => backend.onBatchProgress(setProgress), [backend]);
  useEffect(() => {
    if (modelId === "" && models.length > 0) setModelId(models[0]!);
  }, [models, modelId]);

  const run = useMutation({
    mutationFn: () =>
      backend.batchFind({
        model_id: modelId,
        image_ids: images.map((i) => i.id),
        min_score: minScore,
        max_matches: 1,
      }),
    onSuccess: () => setProgress(null),
  });

  const rows = useMemo(() => {
    const items = run.data?.items ?? [];
    return [...items].sort((a, b) => best(a) - best(b));
  }, [run.data]);

  const scores = useMemo(() => rows.map(best).filter((s) => s > 0), [rows]);

  return (
    <Panel title="Run across the set">
      <div className="flex flex-col gap-3">
        <Field label="Model">
          <Select
            value={modelId}
            onValueChange={(v) => {
              setModelId(v);
              onPickModel(v);
            }}
            options={models.map((id) => ({ value: id, label: id }))}
            placeholder="Choose a model…"
          />
        </Field>
        <Field label="Min score" annotation="0–1">
          <NumberInput
            min={0}
            max={1}
            step={0.05}
            value={minScore}
            onChange={(e) => setMinScore(Number(e.target.value))}
          />
        </Field>
        <Button
          size="sm"
          variant="primary"
          disabled={modelId === "" || images.length === 0}
          loading={run.isPending}
          onClick={() => run.mutate()}
        >
          Find in {images.length} frames
        </Button>

        {run.isPending && progress && (
          <ProgressBar
            fraction={progress.done / Math.max(progress.total, 1)}
            label={`${progress.done}/${progress.total} · ${progress.image_id}`}
          />
        )}
        {run.isError && <ErrorBox>{(run.error as Error).message}</ErrorBox>}

        {rows.length > 0 && (
          <>
            {scores.length > 1 && (
              <ScoreHistogram normal={scores} defect={[]} label="best score per frame" />
            )}
            <Table
              columns={[
                { key: "image", header: "frame", cell: (r: BatchFindItem) => r.image_id },
                {
                  key: "score",
                  header: "best",
                  numeric: true,
                  cell: (r: BatchFindItem) => (best(r) > 0 ? best(r).toFixed(3) : "—"),
                },
                {
                  key: "ms",
                  header: "ms",
                  numeric: true,
                  cell: (r: BatchFindItem) => r.elapsed_ms.toFixed(0),
                },
              ]}
              rows={rows}
              rowKey={(r) => r.image_id}
              onRowClick={(r) => onOpenFrame(r.image_id)}
              empty="No frames run yet."
            />
          </>
        )}
      </div>
    </Panel>
  );
}

function best(item: BatchFindItem): number {
  return item.matches.reduce((m, x) => Math.max(m, x.score), 0);
}
