import { beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (...args: unknown[]) => invokeMock(...args),
  // The real one is a Tauri-runtime call; the shape it produces is all this
  // module depends on, and asserting on it is how we notice if the tier path
  // ever stops being handed to the webview as a URL.
  convertFileSrc: (path: string) => `asset://localhost/${path}`,
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn().mockResolvedValue(null),
}));

// Imported after the mocks so `tauriBackend.ts`'s own imports resolve to them.
const { createTauriBackend } = await import("./tauriBackend");

describe("createTauriBackend", () => {
  beforeEach(() => {
    invokeMock.mockReset();
  });

  it("health() maps the bare status string into the same shape httpBackend returns", async () => {
    invokeMock.mockResolvedValueOnce("ok");
    const backend = createTauriBackend();

    await expect(backend.health()).resolves.toEqual({ status: "ok" });
    expect(invokeMock).toHaveBeenCalledWith("health");
  });

  it("listImages() invokes images_list with no arguments", async () => {
    const images = [{ id: "img-1", filename: "a.png", width: 4, height: 4, sha256: "abc" }];
    invokeMock.mockResolvedValueOnce(images);
    const backend = createTauriBackend();

    await expect(backend.listImages()).resolves.toEqual(images);
    expect(invokeMock).toHaveBeenCalledWith("images_list");
  });

  it("uploadImage() reads the File into a plain byte array before invoking", async () => {
    const bytes = new Uint8Array([137, 80, 78, 71]);
    const file = new File([bytes], "disc.png", { type: "image/png" });
    invokeMock.mockResolvedValueOnce({ id: "img-1", filename: "disc.png", width: 1, height: 1, sha256: "x" });
    const backend = createTauriBackend();

    await backend.uploadImage(file);

    expect(invokeMock).toHaveBeenCalledWith("images_upload", {
      filename: "disc.png",
      bytes: Array.from(bytes),
    });
  });

  it("teachModel()/find()/measure()/rectify()/displacement() invoke the matching command with `req`", async () => {
    const backend = createTauriBackend();
    const cases: Array<[keyof typeof backend, string]> = [
      ["teachModel", "models_create"],
      ["find", "find"],
      ["measure", "measure"],
      ["displacement", "displacement"],
    ];
    for (const [method, command] of cases) {
      invokeMock.mockResolvedValueOnce({});
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      await (backend[method] as any)({ dummy: true });
      expect(invokeMock).toHaveBeenLastCalledWith(command, { req: { dummy: true } });
    }
  });

  /*
   * `teach_preview` is the one command with no OpenAPI counterpart *and* no contract
   * fixture — `lab/contract/fixtures/teach.json` pins the plain rectangle build only, and
   * `contract_parity.rs` passes `keep_contours: None` because the fixtures predate curated
   * teaching. So a renamed field on the Rust side is a runtime mismatch here, not a type
   * error, and this is the only thing standing between the contour inventory and a silently
   * empty list.
   */
  it("teachPreview() invokes teach_preview and passes ContourOut through field for field", async () => {
    invokeMock.mockResolvedValueOnce({
      contours: [
        { id: 0, points: [1, 2, 3, 4], closed: false, length: 2.83, mean_strength: 0.42 },
        { id: 1, points: [5, 6, 7, 8, 9, 10], closed: true, length: 5.66, mean_strength: 0.19 },
      ],
      total_points: 5,
    });
    const backend = createTauriBackend();

    const response = await backend.teachPreview({
      image_id: "img-1",
      roi: [24, 24, 80, 80],
      min_contrast: 0.15,
    });

    expect(invokeMock).toHaveBeenCalledWith("teach_preview", {
      req: { image_id: "img-1", roi: [24, 24, 80, 80], min_contrast: 0.15 },
    });
    expect(response.total_points).toBe(5);
    expect(response.contours).toHaveLength(2);
    // Named individually rather than compared as a blob: a field the Rust side renames
    // should fail on the field, not on a diff of two objects.
    const [first, second] = response.contours;
    expect(first?.id).toBe(0);
    expect(first?.points).toEqual([1, 2, 3, 4]);
    expect(first?.closed).toBe(false);
    expect(first?.length).toBeCloseTo(2.83, 6);
    expect(first?.mean_strength).toBeCloseTo(0.42, 6);
    expect(second?.closed).toBe(true);
  });

  it("rectify() renames the Rust side's crop_key to the LabBackend's crop_url", async () => {
    invokeMock.mockResolvedValueOnce({
      width: 10,
      height: 10,
      matches: [
        { index: 0, x: 1, y: 2, angle: 0, scale: 1, score: 0.9, support: 5, level: 0, validity: 1, crop_key: "img-1/model-1/0", width: 10, height: 10 },
      ],
    });
    const backend = createTauriBackend();

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const resp = await backend.rectify({ image_id: "img-1", model_id: "model-1", crop: { rect: [0, 0, 1, 1] } } as any);

    expect(resp.matches[0]?.crop_url).toBe("img-1/model-1/0");
  });

  it("imageUrl() renders the tier to a file and returns an asset URL for it", async () => {
    invokeMock.mockResolvedValueOnce("/cache/tiers/abc/thumb.png");
    const backend = createTauriBackend();

    const url = await backend.imageUrl("img-1", "thumb");

    expect(invokeMock).toHaveBeenCalledWith("image_tier_path", {
      imageId: "img-1",
      tier: "thumb",
    });
    // No pixels crossed the boundary — the command answered with a path, and
    // `convertFileSrc` (mocked below) turned it into something an <img> loads.
    expect(url).toBe("asset://localhost//cache/tiers/abc/thumb.png");
  });

  it("rectifyCropUrl() names a crop that resolveCropUrl then fetches", async () => {
    invokeMock.mockResolvedValueOnce([9, 9, 9]);
    const backend = createTauriBackend();

    const key = backend.rectifyCropUrl("img-1", "model-1", 0);
    expect(key).toBe("crop:img-1:model-1:0");

    const url = await backend.resolveCropUrl(key);
    expect(url.startsWith("blob:")).toBe(true);
    expect(invokeMock).toHaveBeenCalledWith("rectify_crop", {
      imageId: "img-1",
      modelId: "model-1",
      index: 0,
    });
  });

  it("scanDir() and openImagePaths() move paths, never bytes", async () => {
    invokeMock.mockResolvedValueOnce([]);
    const backend = createTauriBackend();
    await backend.scanDir("/frames", false);
    expect(invokeMock).toHaveBeenLastCalledWith("images_scan_dir", {
      dir: "/frames",
      recursive: false,
    });

    invokeMock.mockResolvedValueOnce([]);
    await backend.openImagePaths(["/frames/a.png"]);
    expect(invokeMock).toHaveBeenLastCalledWith("images_open_paths", {
      paths: ["/frames/a.png"],
    });
  });

  it("mosaic()/mosaicImageUrl()/mosaicSourceIdUrl() are deliberately unsupported this wave", async () => {
    const backend = createTauriBackend();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await expect(backend.mosaic({} as any)).rejects.toThrow(/desktop build/);
    expect(() => backend.mosaicImageUrl("mosaic-1", false)).toThrow(/desktop build/);
    expect(() => backend.mosaicSourceIdUrl("mosaic-1")).toThrow(/desktop build/);
  });
});
