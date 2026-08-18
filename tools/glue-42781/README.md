# 42781 glue-dispensing analysis

Exploratory analysis scripts for the 42781 dataset: a sequence of grayscale BMP
frames, each composed of three vertically stacked camera strips.

These are **development tooling**, not library demos — they use numpy/scipy/matplotlib
and do not exercise the `vision_metrology` bindings. For runnable library examples see
[`examples/python/`](../../examples/python).

The dataset is not distributed with this repository; pass its location explicitly.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r tools/glue-42781/requirements.txt
```

## Scripts

| Script | What it does | Figures |
|---|---|---|
| `explore.py` | Frame gallery, mean brightness over time, time-space maps, per-strip temporal median/std | `explore_*.png` |
| `motion.py` | Per-strip motion by phase correlation between consecutive frames, median-smoothed, plus PCA dominant direction | `motion_*.png` |
| `detect.py` | Nozzle localization, rear-camera identification, glue detection in the rear strip | `detect_*.png` |

```bash
python tools/glue-42781/explore.py --data-dir /path/to/42781
python tools/glue-42781/motion.py  --data-dir /path/to/42781 --smooth-window 5
python tools/glue-42781/detect.py  --data-dir /path/to/42781 --out-dir /tmp/figs
```

Figures default to `<data-dir>/output/`.
