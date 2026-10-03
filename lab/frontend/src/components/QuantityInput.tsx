/**
 * A `NumberInput` bound to a number rather than to the field's text.
 *
 * While the field has focus it shows what is being typed, so clearing it to retype is not
 * read as a zero, and a change reaches `onValueChange` only when the text is a finite number
 * inside `[min, max]`. On blur the field shows the value again.
 *
 * `@vitavision/ui`'s `NumberInput` has no numeric value API yet (its `VectorInput` does this
 * per field internally). When the package grows one, this component goes.
 */

import { NumberInput, type NumberInputProps } from "@vitavision/ui";
import { useState } from "react";

type QuantityInputProps = Omit<NumberInputProps, "value" | "defaultValue" | "onChange"> & {
  value: number;
  onValueChange: (value: number) => void;
};

export function QuantityInput({ value, onValueChange, min, max, onFocus, onBlur, ...rest }: QuantityInputProps) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <NumberInput
      {...rest}
      min={min}
      max={max}
      value={draft ?? String(value)}
      onFocus={(event) => {
        setDraft(String(value));
        onFocus?.(event);
      }}
      onChange={(event) => {
        const text = event.target.value;
        setDraft(text);
        const parsed = parseQuantity(text, min, max);
        if (parsed !== null) onValueChange(parsed);
      }}
      onBlur={(event) => {
        setDraft(null);
        onBlur?.(event);
      }}
    />
  );
}

/** The number a field's text commits, or `null` while it is empty, partial or out of range. */
export function parseQuantity(text: string, min?: number, max?: number): number | null {
  if (text.trim() === "") return null;
  const value = Number(text);
  if (!Number.isFinite(value)) return null;
  if (min !== undefined && value < min) return null;
  if (max !== undefined && value > max) return null;
  return value;
}
