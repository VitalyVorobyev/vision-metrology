// @ts-check
import { recommended, tokensOnly } from "@vitavision/config-eslint";

export default [
  { ignores: ["src-tauri/**", "dist/**", "src/api/generated.ts"] },
  ...recommended({ tsconfigRootDir: import.meta.dirname }),
  {
    rules: {
      // A leading underscore marks a parameter that an interface requires and this
      // implementation does not read.
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrorsIgnorePattern: "^_" },
      ],
      // These rules come with the React Compiler: setState inside an effect, refs read or
      // written during render, impure calls during render. Each fix changes when a screen
      // renders, and the toolchain upgrade changes no UI, so they report as warnings until
      // the screen that holds each one is reworked.
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/refs": "warn",
      "react-hooks/immutability": "warn",
      "react-hooks/purity": "warn",
      "react-hooks/preserve-manual-memoization": "warn",
    },
  },
  {
    // Fake backends in tests implement async interfaces with plain values.
    files: ["**/*.test.{ts,tsx}"],
    rules: { "@typescript-eslint/require-await": "off" },
  },
  // Gate G5.1 (the @vitavision visual language): in src/, colour comes from the @vitavision/ui design
  // tokens — no raw Tailwind palette classes, no hex literals (tests are exempt by the rule).
  tokensOnly(["src/**"]),
  {
    files: [
      // Draws when the app failed to start — possibly without its stylesheet, so no tokens.
      "src/shell/CrashScreen.tsx",
      // Canvas overlays need halo and highlight colours that stay legible on any photograph.
      // The design system has no overlay role tokens yet; until it does, these layers keep
      // their literal colours.
      "src/canvas/ContourLayer.tsx",
      "src/canvas/DatumLayer.tsx",
      "src/canvas/RoiLayer.tsx",
    ],
    rules: { "vitavision/tokens-only": "off" },
  },
];
