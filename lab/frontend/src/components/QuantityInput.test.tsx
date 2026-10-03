import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { parseQuantity, QuantityInput } from "./QuantityInput";

describe("parseQuantity", () => {
  it("commits a finite number inside the bounds", () => {
    expect(parseQuantity("1.5")).toBe(1.5);
    expect(parseQuantity("0", 0, 1)).toBe(0);
  });

  it("commits nothing for an empty, partial or out-of-range field", () => {
    expect(parseQuantity("")).toBeNull();
    expect(parseQuantity("  ")).toBeNull();
    expect(parseQuantity("-")).toBeNull();
    expect(parseQuantity("2", 0, 1)).toBeNull();
    expect(parseQuantity("-1", 0)).toBeNull();
  });
});

describe("QuantityInput", () => {
  function Harness({ onCommit }: { onCommit: (v: number) => void }) {
    const [value, setValue] = useState(12);
    return (
      <QuantityInput
        aria-label="width"
        min={1}
        value={value}
        onValueChange={(v) => {
          setValue(v);
          onCommit(v);
        }}
      />
    );
  }

  it("reads a cleared field as being retyped, not as zero", () => {
    const onCommit = vi.fn<(v: number) => void>();
    render(<Harness onCommit={onCommit} />);
    const field = screen.getByLabelText<HTMLInputElement>("width");

    fireEvent.focus(field);
    fireEvent.change(field, { target: { value: "" } });
    expect(onCommit).not.toHaveBeenCalled();
    expect(field.value).toBe("");

    fireEvent.change(field, { target: { value: "30" } });
    expect(onCommit).toHaveBeenLastCalledWith(30);
  });

  it("shows the committed value again on blur", () => {
    render(<Harness onCommit={() => {}} />);
    const field = screen.getByLabelText<HTMLInputElement>("width");

    fireEvent.focus(field);
    fireEvent.change(field, { target: { value: "0" } }); // below min: not committed
    fireEvent.blur(field);
    expect(field.value).toBe("12");
  });
});
