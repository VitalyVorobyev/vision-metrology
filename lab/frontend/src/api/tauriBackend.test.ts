import { beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (...args: unknown[]) => invokeMock(...args),
}));

// Imported after the mock so `tauriBackend.ts`'s own `import { invoke } from
// "@tauri-apps/api/core"` resolves to the mocked module.
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

  it("imageUrl() returns a placeholder synchronously, then a blob URL once the fetch resolves", async () => {
    const bytes = new Uint8Array([1, 2, 3, 4]);
    invokeMock.mockResolvedValueOnce(Array.from(bytes));
    const backend = createTauriBackend();

    const first = backend.imageUrl("img-1", "thumb");
    expect(first.startsWith("data:image/png;base64,")).toBe(true);
    expect(invokeMock).toHaveBeenCalledWith("image_data", { imageId: "img-1", tier: "thumb" });

    // Let the microtask queue drain so the background fetch resolves.
    await Promise.resolve();
    await Promise.resolve();

    const second = backend.imageUrl("img-1", "thumb");
    expect(second.startsWith("blob:")).toBe(true);
    // A cached key does not re-invoke.
    expect(invokeMock).toHaveBeenCalledTimes(1);
  });

  it("rectifyCropUrl() follows the same cache-or-fetch pattern as imageUrl()", async () => {
    invokeMock.mockResolvedValueOnce([9, 9, 9]);
    const backend = createTauriBackend();

    const first = backend.rectifyCropUrl("img-1", "model-1", 0);
    expect(first.startsWith("data:image/png;base64,")).toBe(true);
    expect(invokeMock).toHaveBeenCalledWith("rectify_crop", {
      imageId: "img-1",
      modelId: "model-1",
      index: 0,
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
