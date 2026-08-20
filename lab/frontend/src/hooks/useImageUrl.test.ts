import { describe, expect, it, vi } from "vitest";

const imageUrlMock = vi.fn();

vi.mock("../api/backend", () => ({
  getBackend: () => ({ imageUrl: imageUrlMock }),
}));

const { renderHook, waitFor } = await import("@testing-library/react");
const { useImageUrl, useLazyImageUrl } = await import("./useImageUrl");

describe("useImageUrl", () => {
  it("reports the URL once it resolves, which is what makes its arrival a render", async () => {
    imageUrlMock.mockResolvedValueOnce("asset://tier.png");
    const { result } = renderHook(() => useImageUrl("img-1", "preview"));

    // The point of the hook: there is no placeholder to mistake for the image.
    expect(result.current.url).toBeNull();
    expect(result.current.loading).toBe(true);

    await waitFor(() => expect(result.current.url).toBe("asset://tier.png"));
    expect(result.current.loading).toBe(false);
  });

  it("surfaces a failure instead of leaving the frame blank forever", async () => {
    imageUrlMock.mockRejectedValueOnce(new Error("no such tier"));
    const { result } = renderHook(() => useImageUrl("img-1", "full"));

    await waitFor(() => expect(result.current.error).toBe("no such tier"));
    expect(result.current.loading).toBe(false);
  });

  it("asks for nothing when there is no image", () => {
    imageUrlMock.mockClear();
    const { result } = renderHook(() => useImageUrl(null, "thumb"));
    expect(result.current).toEqual({ url: null, loading: false, error: null });
    expect(imageUrlMock).not.toHaveBeenCalled();
  });
});

describe("useLazyImageUrl", () => {
  it("does not fetch until enabled — the whole point of a lazily scanned folder", async () => {
    imageUrlMock.mockClear();
    imageUrlMock.mockResolvedValue("asset://thumb.png");

    const { result, rerender } = renderHook(
      ({ on }: { on: boolean }) => useLazyImageUrl("img-1", "thumb", on),
      { initialProps: { on: false } },
    );
    expect(imageUrlMock).not.toHaveBeenCalled();
    expect(result.current.loading).toBe(false);

    rerender({ on: true });
    await waitFor(() => expect(result.current.url).toBe("asset://thumb.png"));
    expect(imageUrlMock).toHaveBeenCalledTimes(1);
  });
});
