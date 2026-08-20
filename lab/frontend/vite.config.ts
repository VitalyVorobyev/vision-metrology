import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// @vitavision/lab-ui is a published npm package, so its peer dependencies
// (react, react-dom, react-router) resolve to this project's own copies like
// any other package's. While it was a `file:` dependency it carried its own
// node_modules, which meant two React instances in one page — every hook its
// components called threw "Invalid hook call" — and both a `resolve.alias` and
// a Vitest `server.deps.inline` entry existed here to force a single copy.
// Both are gone with the cause.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5174,
    strictPort: true,
  },
  build: {
    outDir: "dist",
  },
  test: {
    environment: "happy-dom",
    globals: false,
    setupFiles: ["./src/test-setup.ts"],
  },
});
