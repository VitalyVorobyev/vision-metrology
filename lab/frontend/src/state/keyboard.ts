/**
 * Keyboard helpers shared by the inventories, which listen at the window: a list is worked
 * from both the list and the image, and a binding that only fires while one element has
 * focus does nothing right after a click on the canvas.
 */

/** Whether a key press belongs to a text field (or a slider, a radio), not to the inventory. */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}
