// No global setup needed yet — the suite tests pure logic modules, not rendered
// components. Kept as an explicit file (rather than removing `setupFiles`) so adding a
// DOM-testing dependency later is a one-line change here, not a new vite.config edit.
export {};
