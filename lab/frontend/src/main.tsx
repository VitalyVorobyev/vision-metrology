// Self-hosted rather than fetched: a local workbench should not depend on the network for
// its own chrome.
import "@fontsource-variable/ibm-plex-sans/wght.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "@fontsource/ibm-plex-mono/latin-600.css";

import { initTheme } from "@vitavision/lab-ui";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router";

import { App } from "./App";
import "./styles.css";

const queryClient = new QueryClient();

const container = document.getElementById("root");
if (container === null) {
  throw new Error("index.html is missing its #root element.");
}

// index.html already painted the stored choice before first paint; this subscribes so a
// choice of "system" keeps following the OS after mount.
initTheme();

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      {/* HashRouter even though this is a one-page app: @vitavision/lab-ui's PageHeader
          renders a react-router <Link>, which needs a router context to exist at all. */}
      <HashRouter>
        <App />
      </HashRouter>
    </QueryClientProvider>
  </StrictMode>,
);
