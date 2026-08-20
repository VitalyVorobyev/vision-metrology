import { describe, expect, it } from "vitest";

import { DEFAULT_API_BASE_URL, resolveApiBaseUrl } from "./baseUrl";

describe("resolveApiBaseUrl", () => {
  it("prefers a URL injected by an embedding shell", () => {
    expect(
      resolveApiBaseUrl({ injected: "http://127.0.0.1:54321", env: "http://localhost:8000" }),
    ).toBe("http://127.0.0.1:54321");
  });

  it("falls back to the build-time environment override", () => {
    expect(resolveApiBaseUrl({ env: "http://localhost:9000" })).toBe("http://localhost:9000");
  });

  it("falls back to the documented development port", () => {
    expect(resolveApiBaseUrl({})).toBe(DEFAULT_API_BASE_URL);
  });

  it("ignores an empty injected value rather than producing a relative URL", () => {
    expect(resolveApiBaseUrl({ injected: "   ", env: "http://localhost:9000" })).toBe(
      "http://localhost:9000",
    );
  });

  it("strips trailing slashes, which would otherwise double up on every path", () => {
    expect(resolveApiBaseUrl({ injected: "http://127.0.0.1:8000//" })).toBe(
      "http://127.0.0.1:8000",
    );
  });
});
