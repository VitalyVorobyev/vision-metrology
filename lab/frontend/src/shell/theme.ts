/**
 * One theme key, used by the toggle (`AppShell`), `initTheme` (`main.tsx`) and the
 * pre-paint script in `index.html`. If they disagree, a dark-mode user gets a light flash
 * on every start.
 *
 * The value has to match the literal in `index.html`, which cannot import anything: it runs
 * before the bundle exists, and that is the whole point of it.
 */
export const LAB_THEME_STORAGE_KEY = "metrology-lab-theme";
