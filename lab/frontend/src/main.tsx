import { initTheme, TooltipProvider } from "@vitavision/ui";
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
        {/* `ThemeToggle`, `Tooltip` and `InfoHint` render Radix tooltips, which throw
            without this provider; a throw during render unmounts the whole root. */}
        <TooltipProvider>
          {/* Hash routing keeps every route inside one document, so the desktop shell's
              asset protocol and a plain static server both serve it without rewrites. */}
          <HashRouter>
            <App />
          </HashRouter>
        </TooltipProvider>
      </QueryClientProvider>
    </CrashBoundary>
  </StrictMode>,
);
