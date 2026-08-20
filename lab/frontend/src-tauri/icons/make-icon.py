"""Visual Metrology Lab app icon.

Family resemblance to the other vitavision labs (same rounded-square ground, same
light "part" ring), different *mark*: the anomaly lab puts a red defect blob on the
ring, this one puts a caliper across it. A metrology tool measures a part; it does
not flag one.

Palette is lab-ui's own: neutral (not blue-tinted) dark ground, `fg` for the part,
`signal` cyan for the thing the tool does. Drawn at 8x and box-downsampled, which is
the cheapest honest antialiasing and keeps the 32x32 tier readable.
"""
import os

from PIL import Image, ImageDraw

S = 512
K = 8            # supersample factor
W = S * K

GROUND = (18, 22, 26, 255)      # neutral dark, lab-ui `--surface`-ish
PART = (219, 225, 230, 255)     # lab-ui `--fg` (light theme's dark-mode fg)
SIGNAL = (59, 201, 219, 255)    # lab-ui `--signal` (dark palette)

img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# Rounded-square ground.
d.rounded_rectangle([0, 0, W - 1, W - 1], radius=int(0.22 * W), fill=GROUND)

cx = cy = W // 2
r = int(0.285 * W)
ring = int(0.062 * W)

# The part: a ring, same motif the sibling app uses.
d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=PART, width=ring)

# The caliper: two jaws closing on the part's horizontal diameter, plus the
# dimension line between them. This is the whole idea of the app in one glyph —
# a known geometry measured across a known span.
jaw_w = int(0.058 * W)
jaw_h = int(0.34 * W)
gap = int(0.022 * W)          # jaw faces sit just outside the ring
x_left = cx - r - ring // 2 - gap - jaw_w
x_right = cx + r + ring // 2 + gap

for x0 in (x_left, x_right):
    d.rounded_rectangle(
        [x0, cy - jaw_h // 2, x0 + jaw_w, cy + jaw_h // 2],
        radius=int(0.010 * W),
        fill=SIGNAL,
    )

# Dimension line across the measured span, with arrowheads into each jaw face.
line_w = int(0.036 * W)
d.line([x_left + jaw_w, cy, x_right, cy], fill=SIGNAL, width=line_w)
head = int(0.072 * W)
for x_tip, sign in ((x_left + jaw_w, 1), (x_right, -1)):
    d.polygon(
        [
            (x_tip, cy),
            (x_tip + sign * head, cy - head // 2),
            (x_tip + sign * head, cy + head // 2),
        ],
        fill=SIGNAL,
    )

out = img.resize((S, S), Image.LANCZOS)
here = os.path.dirname(os.path.abspath(__file__))
out.save(os.path.join(here, "icon.png"))
out.resize((128, 128), Image.LANCZOS).save(os.path.join(here, "128x128.png"))
out.resize((32, 32), Image.LANCZOS).save(os.path.join(here, "32x32.png"))
print(f"wrote 512/128/32 into {here}")
