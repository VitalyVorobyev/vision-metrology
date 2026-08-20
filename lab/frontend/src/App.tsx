/**
 * Routes, and nothing else.
 *
 * This file used to be the whole application: all the shared state, the tab
 * switch, the canvas, and the panel for six unrelated jobs. The state now lives
 * in `LabProvider`, the frame in `AppShell`, and each job in its own route —
 * which is what makes it possible for a workspace to show only what its own
 * task needs.
 */

import { Navigate, Route, Routes } from "react-router";

import { AlignPage, MeasurePage } from "./routes/GaugePage";
import { FindPage } from "./routes/FindPage";
import { LibraryPage } from "./routes/LibraryPage";
import { MosaicPage, MotionPage } from "./routes/CameraPage";
import { TeachPage } from "./routes/TeachPage";
import { VerifyPage } from "./routes/VerifyPage";
import { LabProvider } from "./state/LabContext";

export function App() {
  return (
    <LabProvider>
      <Routes>
        <Route path="/" element={<Navigate to="/library" replace />} />
        <Route path="/library" element={<LibraryPage />} />

        <Route path="/recognize" element={<Navigate to="/recognize/teach" replace />} />
        <Route path="/recognize/teach" element={<TeachPage />} />
        <Route path="/recognize/find" element={<FindPage />} />
        <Route path="/recognize/verify" element={<VerifyPage />} />

        <Route path="/gauge" element={<Navigate to="/gauge/measure" replace />} />
        <Route path="/gauge/measure" element={<MeasurePage />} />
        <Route path="/gauge/align" element={<AlignPage />} />

        <Route path="/camera" element={<Navigate to="/camera/motion" replace />} />
        <Route path="/camera/motion" element={<MotionPage />} />
        <Route path="/camera/mosaic" element={<MosaicPage />} />

        <Route path="*" element={<Navigate to="/library" replace />} />
      </Routes>
    </LabProvider>
  );
}
