import { Empty, PageHeader, Skeleton, Tabs, type TabItem } from "@vitavision/lab-ui";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { listImages, listModels, uploadImage } from "./api/client";
import type { MatchOut, OverlayPrimitiveOut, Roi } from "./api/types";
import { CanvasStage } from "./components/CanvasStage";
import { ImageRail } from "./components/ImageRail";
import { FindTab } from "./tabs/FindTab";
import { MeasureTab } from "./tabs/MeasureTab";
import { TeachTab } from "./tabs/TeachTab";

type TabId = "teach" | "find" | "measure";

const TABS: TabItem<TabId>[] = [
  { id: "teach", label: "Teach" },
  { id: "find", label: "Find" },
  { id: "measure", label: "Measure" },
];

export function App() {
  const queryClient = useQueryClient();
  const imagesQuery = useQuery({ queryKey: ["images"], queryFn: listImages });
  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: listModels });

  const [selectedImageId, setSelectedImageId] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>("teach");
  const [roi, setRoi] = useState<Roi | null>(null);
  const [overlay, setOverlay] = useState<OverlayPrimitiveOut[]>([]);
  const [matches, setMatches] = useState<MatchOut[]>([]);

  const images = imagesQuery.data ?? [];
  const models = modelsQuery.data ?? [];
  const selectedImage = images.find((i) => i.id === selectedImageId) ?? null;

  // The first successful load selects an image so the workbench is not empty.
  useEffect(() => {
    if (!selectedImageId && images.length > 0) setSelectedImageId(images[0]?.id ?? null);
  }, [images, selectedImageId]);

  const uploadMutation = useMutation({
    mutationFn: uploadImage,
    onSuccess: (image) => {
      void queryClient.invalidateQueries({ queryKey: ["images"] });
      setSelectedImageId(image.id);
    },
  });

  const selectImage = (id: string) => {
    setSelectedImageId(id);
    setRoi(null);
    setOverlay([]);
    setMatches([]);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-line bg-surface px-4 py-3">
        <PageHeader
          title="Visual Metrology Lab"
          meta={<span>teach → find → measure → judge, over vision_metrology</span>}
        />
      </div>
      <div className="flex min-h-0 flex-1">
        <ImageRail
          images={images}
          selectedId={selectedImageId}
          onSelect={selectImage}
          onUpload={(file) => uploadMutation.mutate(file)}
          uploading={uploadMutation.isPending}
        />

        <main className="min-w-0 flex-1 p-4">
          {imagesQuery.isLoading ? (
            <Skeleton className="h-full w-full" />
          ) : !selectedImage ? (
            <Empty>Upload an image to begin.</Empty>
          ) : (
            <CanvasStage
              image={selectedImage}
              overlay={overlay}
              roiMode={tab === "teach"}
              onRoiChange={setRoi}
              roiPreview={roi}
            />
          )}
        </main>

        <aside className="w-96 shrink-0 overflow-y-auto border-l border-line bg-surface p-4">
          <Tabs items={TABS} active={tab} onSelect={setTab} label="Workbench mode" className="mb-4" />
          {!selectedImage ? (
            <Empty>Select an image first.</Empty>
          ) : tab === "teach" ? (
            <TeachTab image={selectedImage} roi={roi} onRoiCommitted={() => setRoi(null)} models={models} />
          ) : tab === "find" ? (
            <FindTab
              image={selectedImage}
              models={models}
              onMatches={(m, o) => {
                setMatches(m);
                setOverlay(o);
              }}
            />
          ) : (
            <MeasureTab image={selectedImage} models={models} onResult={setOverlay} />
          )}
          {tab === "find" && matches.length === 0 && overlay.length === 0 && (
            <p className="mt-2 text-xs text-fg-subtle">Run a search to see matches on the image.</p>
          )}
        </aside>
      </div>
    </div>
  );
}
