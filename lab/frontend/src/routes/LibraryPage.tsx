/**
 * Where frames come from, and where a model is judged across all of them.
 *
 * A metrology capture is a folder, not a file, so the question worth asking here is how a
 * model behaves across the set. Opening a folder reads only directory entries and image
 * headers. Nothing is decoded, nothing is copied, and the frames stay where the user put
 * them, so a set of several thousand opens as fast as the filesystem can list it.
 *
 * Single files come in through workbench's `FileDrop`: its picker, or a drop anywhere on the
 * window. On the desktop both yield **paths** (the shell's native picker and drops, through
 * `LabBackend.pickImages` / `onFileDrop`), and the files are opened in place like a folder's.
 * In the browser they are `File`s, uploaded into the lab.
 */

import { ScoreHistogram } from "@vitavision/charts";
import {
  Badge,
  Button,
  Callout,
  ErrorBox,
  Field,
  NumberInput,
  Panel,
  ProgressBar,
  Section,
  Select,
  Table,
} from "@vitavision/ui";
import { FileDrop, type PathSource } from "@vitavision/workbench";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";

import { getBackend } from "../api/backend";
import type { BatchFindItem, BatchProgress, DirEntry, ImageOut } from "../api/backend";
import { ImageGrid } from "../components/ImageGrid";
import { AppShell } from "../shell/AppShell";
import { useLab } from "../state/LabContext";

export function LibraryPage() {
  const backend = getBackend();
  const desktop = backend.canOpenFiles();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { images, models, selectImage, selectedImage, selectModel } = useLab();

  const [scanned, setScanned] = useState<DirEntry[] | null>(null);
  const [folder, setFolder] = useState<string | null>(null);
  /** What the last drop or pick brought that this transport cannot open. */
  const [skipped, setSkipped] = useState<string[]>([]);

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

  const opened = (frames: ImageOut[]) => {
    void queryClient.invalidateQueries({ queryKey: ["images"] });
    if (frames.length > 0) selectImage(frames[0]!.id);
  };

  /** The desktop route: files opened where they are, by path. */
  const openPaths = useMutation({
    mutationFn: (paths: string[]) => backend.openImagePaths(paths),
    onSuccess: opened,
  });

  /** The browser route: each file uploaded, one at a time, in the order given. */
  const uploadFiles = useMutation({
    mutationFn: async (files: File[]) => {
      const frames: ImageOut[] = [];
      for (const file of files) frames.push(await backend.uploadImage(file));
      return frames;
    },
    onSuccess: opened,
  });

  /* The shell's native picker and drops, which give paths. Built once: a new source would
   * make `FileDrop` unsubscribe and subscribe again. */
  const pathSource = useMemo<PathSource | undefined>(
    () =>
      desktop
        ? { pick: () => backend.pickImages(), subscribe: (handlers) => backend.onFileDrop(handlers) }
        : undefined,
    [backend, desktop],
  );

  const accept = backend.imageAccept();
  const opening = openPaths.isPending || uploadFiles.isPending;
  const openError = openPaths.error ?? uploadFiles.error;

  /** One `FileDrop` is mounted at a time: each one listens for drops on the whole window. */
  const fileDrop = (zone: boolean) => (
    <FileDrop
      overlay={!zone}
      accept={accept}
      pathSource={pathSource}
      onPaths={(paths) => {
        setSkipped([]);
        openPaths.mutate(paths);
      }}
      onFiles={(files) => {
        setSkipped([]);
        uploadFiles.mutate(files);
      }}
      onRejectPaths={(paths) => setSkipped(paths.map(baseName))}
      onReject={(files) => setSkipped(files.map((file) => file.name))}
      disabled={opening}
      buttonLabel={zone ? "Open files…" : "Files…"}
      overlayMessage="Drop to open these frames"
      className={zone ? "w-full max-w-md" : undefined}
    >
      No frames yet. Drop images here, or
    </FileDrop>
  );

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

  const empty = images.length === 0 && scanned === null;

  return (
    <AppShell
      fullBleed={
        empty ? (
          <div className="grid h-full place-items-center p-6">{fileDrop(true)}</div>
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
              {!desktop && (
                <Callout tone="info">
                  This is the browser build: opened files are uploaded into the lab, and
                  opening a folder needs the desktop app.
                </Callout>
              )}
              {(desktop || !empty) && (
                <div className="flex gap-2">
                  {desktop && (
                    <Button size="sm" variant="primary" loading={openFolder.isPending} onClick={() => openFolder.mutate()}>
                      Open folder…
                    </Button>
                  )}
                  {/* While the library is empty the drop zone in its place is the file route. */}
                  {!empty && fileDrop(false)}
                </div>
              )}
              {openFolder.isError && <ErrorBox>{openFolder.error.message}</ErrorBox>}
              {openError && <ErrorBox>{openError.message}</ErrorBox>}
              {skipped.length > 0 && (
                <Callout tone="warning">
                  Not opened: {skipped.join(", ")}. The lab opens {listOf(accept.split(","))} files
                  {desktop ? "; a folder opens with Open folder…" : "."}
                </Callout>
              )}

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
  const { images, setBatch } = useLab();
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
    onSuccess: (res) => {
      setProgress(null);
      // Shared, so Find lists each frame's match and the frame strip marks the misses.
      setBatch({
        request: { model_id: modelId, min_score: minScore, max_matches: 1, roi: null, angle_range: null },
        items: new Map(res.items.map((item) => [item.image_id, item])),
      });
    },
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
          <NumberInput min={0} max={1} step={0.05} value={minScore} onValueChange={setMinScore} />
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
        {run.isError && <ErrorBox>{run.error.message}</ErrorBox>}

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

/** A path's last component, for naming a skipped file. */
function baseName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/** `[".png", ".bmp", ".pgm"]` → ".png, .bmp or .pgm". */
function listOf(items: string[]): string {
  return items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} or ${items.at(-1)}`;
}
