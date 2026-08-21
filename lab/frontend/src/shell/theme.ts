/**
 * One theme key, in one place.
 *
 * It was written in three: `AppShell` passed `"metrology-lab-theme"` to the toggle,
 * `index.html`'s pre-paint script read `"vitavision-theme"`, and `main.tsx` called
 * `initTheme()` with no key at all. So the toggle's choice was not what the no-flash script
 * read back, and a dark-mode user got a white flash on every start of the desktop app —
 * which looks exactly like a slow load.
 *
 * The value has to match the literal in `index.html`, which cannot import anything: it runs
 * before the bundle exists, and that is the whole point of it.
 */
export const LAB_THEME_STORAGE_KEY = "metrology-lab-theme";
