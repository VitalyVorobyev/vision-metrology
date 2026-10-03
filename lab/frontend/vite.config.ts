import tailwindcss from "@tailwindcss/vite";
import { dom } from "@vitavision/config-vitest/dom";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The @vitavision/* packages are published to npm with react and react-dom as
// peers, so they resolve to this project's own copies: one React per page.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5174,
    strictPort: true,
  },
  build: {
    outDir: "dist",
  },
  // The shared happy-dom preset: the environment, explicit vitest imports, the Testing
  // Library teardown and `src/**/*.test.*` discovery. Only its `test` block: its plugins and
  // its source-resolution condition are for the package monorepo, and the app builds against
  // the published `dist`.
  test: { ...dom().test },
});
