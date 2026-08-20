import { fileURLToPath } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const resolveOwn = (specifier: string) =>
  fileURLToPath(new URL(`./node_modules/${specifier}`, import.meta.url));

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // @vitavision/lab-ui is a `file:` dependency, so it carries its own node_modules with
    // its own copies of react/react-dom (needed for its own build and test suite).
    // `resolve.dedupe` does not reach Vitest's SSR module graph, where these packages are
    // externalized and resolved by Node's own algorithm from the package's real
    // (symlink-resolved) path -- which finds lab-ui's copies first, and two React instances
    // in one page means every hook the package's components call throws "Invalid hook
    // call". Aliasing to this project's own copies applies before that externalization
    // decision, so both the build and the test run see exactly one of each.
    alias: {
      react: resolveOwn("react"),
      "react-dom": resolveOwn("react-dom"),
      "react-router": resolveOwn("react-router"),
    },
  },
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
    server: {
      deps: {
        // @vitavision/lab-ui resolves as a `file:` link outside this project's node_modules,
        // so Vitest's default externalization serves it (and the radix-ui / lucide-react it
        // pulls in) straight from disk via Node's own resolver -- which finds their own
        // node_modules/react rather than the `resolve.alias` above, since alias only applies
        // to imports Vite itself processes. Inlining forces them through Vite instead.
        inline: [/@vitavision\/lab-ui/, /@radix-ui\//, /lucide-react/],
      },
    },
  },
});
