// Self-hosted rather than fetched: a local workbench should not depend on the network for
// its own chrome.
import "@fontsource-variable/ibm-plex-sans/wght.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "@fontsource/ibm-plex-mono/latin-600.css";

import { initTheme, TooltipProvider } from "@vitavision/lab-ui";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router";

import { App } from "./App";
import { CrashBoundary, installCrashHandlers } from "./shell/CrashScreen";
import { LAB_THEME_STORAGE_KEY } from "./shell/theme";
import "./styles.css";

const queryClient = new QueryClient();

const container = document.getElementById("root");
if (container === null) {
  throw new Error("index.html is missing its #root element.");
}

// Before the first render, so a module that throws on the way in still says so. The
// desktop build has no console: an uncaught error there is a black window and nothing
// else. See `shell/CrashScreen.tsx`.
installCrashHandlers(container);

// index.html already painted the stored choice before first paint; this subscribes so a
// choice of "system" keeps following the OS after mount. The key is passed explicitly:
// the default is the package's own, which is not the one the toggle writes.
initTheme(LAB_THEME_STORAGE_KEY);

createRoot(container).render(
  <StrictMode>
    <CrashBoundary>
      <QueryClientProvider client={queryClient}>
        {/* Radix tooltips read their delay/state from a provider and *throw* without one —
            `@vitavision/lab-ui`'s `ThemeToggle` and `InfoHint` both render one, so this is
            a context the app must mount exactly like the router below, not a nicety. It
            was missing when the shell grew a `ThemeToggle`, which took the whole tree down
            on first render. */}
        <TooltipProvider>
          {/* HashRouter even though this is a one-page app: @vitavision/lab-ui's PageHeader
              renders a react-router <Link>, which needs a router context to exist at all. */}
          <HashRouter>
            <App />
          </HashRouter>
        </TooltipProvider>
      </QueryClientProvider>
    </CrashBoundary>
  </StrictMode>,
);
