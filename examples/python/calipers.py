# Run: python examples/python/calipers.py
"""Calipers: a bar's width with one strip caliper, why a caliper found nothing, and a
circle measured with a metrology model."""

import numpy as np
import vision_metrology as vm


def bar_image(width: int, height: int, x0: int, x1: int) -> np.ndarray:
    """A bright vertical bar over columns x0..x1 (inclusive) on a dark background."""
    img = np.full((height, width), 20, dtype=np.uint8)
    img[:, x0 : x1 + 1] = 200
    return img


def disc_image(width: int, height: int, cx: float, cy: float, r: float) -> np.ndarray:
    """An anti-aliased bright disc, so the true edge sits exactly at radius r."""
    ys, xs = np.mgrid[0:height, 0:width]
    cover = np.clip(r + 0.5 - np.hypot(xs - cx, ys - cy), 0.0, 1.0)
    return (20.0 + 180.0 * cover).round().astype(np.uint8)


def measure_bar() -> None:
    img = bar_image(96, 48, 30, 59)  # edges at x = 29.5 and x = 59.5
    print("Bar over columns 30..59: width 30 px")

    # The strongest rising edge, then the strongest falling edge after it.
    cfg = vm.MeasureConfig(select="in_order", sequence=["rising", "falling"])
    cal = vm.Caliper.strip((10.0, 24.0), (80.0, 24.0), half_width=5.0, config=cfg)
    rising, falling = cal.measure(img)
    width = falling.t - rising.t
    print(f"  rising edge  x = {rising.x:.3f}   (amplitude {rising.amplitude:.1f})")
    print(f"  falling edge x = {falling.x:.3f}   (amplitude {falling.amplitude:.1f})")
    print(f"  width        = {width:.3f} px")
    assert abs(rising.x - 29.5) < 0.01 and abs(falling.x - 59.5) < 0.01, (rising, falling)
    assert abs(width - 30.0) < 0.01, width

    # `explain` measures again and keeps every intermediate.
    trace = cal.explain(img)
    peak = float(np.abs(trace.response).max())
    print(
        f"  explain: {trace.samples} samples {trace.spacing:.3f} px apart, "
        f"{trace.across} lines averaged into each"
    )
    print(
        f"  explain: peak response {peak:.1f} against a threshold of {trace.threshold:.1f}, "
        f"{len(trace.candidates)} candidates, {len(trace.edges)} edges, reject = {trace.reject}"
    )

    # A strip that ends before the bar finds nothing. `measure` raises; `explain` says why.
    cal.move_to_strip((2.0, 24.0), (22.0, 24.0), half_width=5.0)
    try:
        cal.measure(img)
        raise AssertionError("expected MeasureRejected")
    except vm.MeasureRejected as e:
        print(f"  short strip: rejected, reason {e.args[0]!r}")
        assert e.args[0] == "no_edge"
    trace = cal.explain(img)
    peak = float(np.abs(trace.response).max())
    print(
        f"  explain: reject = {trace.reject!r}, peak response {peak:.1f} "
        f"against a threshold of {trace.threshold:.1f}: the window misses the edge"
    )
    assert trace.reject == "no_edge"


def measure_circle() -> None:
    cx, cy, r = 80.0, 70.0, 40.0
    img = disc_image(160, 140, cx, cy, r)
    print(f"\nDisc centred at ({cx}, {cy}), radius {r}")

    # Nominal geometry, a little off: the calipers search +-10 px around it.
    model = vm.MetrologyModel()
    model.add(vm.MetrologyObject(vm.MetrologyShape.circle((81.0, 69.0), 42.0)))

    # The identity fixture: the model is already in image coordinates.
    (result,) = model.apply(img, x=0.0, y=0.0)
    assert isinstance(result, vm.MetrologyResult), result
    c = result.circle
    print(f"  measured centre ({c.cx:.3f}, {c.cy:.3f}), radius {c.r:.3f}")
    print(f"  rms {result.rms:.4f} px, max_dev {result.max_dev:.4f} px, {result.n_used} calipers used")
    assert abs(c.cx - cx) < 0.05 and abs(c.cy - cy) < 0.05, (c.cx, c.cy)
    assert abs(c.r - r) < 0.05, c.r


def main() -> None:
    measure_bar()
    measure_circle()
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
