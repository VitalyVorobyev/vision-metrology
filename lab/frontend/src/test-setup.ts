/**
 * Unmount rendered components between tests.
 *
 * Testing Library registers this automatically only when vitest's globals are enabled.
 * They are not, deliberately — an explicit `import { describe, it } from "vitest"` says
 * where the test API comes from — so the teardown is wired here instead. Without it every
 * render accumulates in the same document and queries start finding several matches for
 * what should be one element: a failure that reads as a bug in the component under test
 * rather than as leftovers from the previous one.
 *
 * This file was a stub while the suite tested only pure modules. `CrashScreen.test.tsx`
 * had already crossed that line and passed on the luck of querying uniquely.
 */

import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(cleanup);
