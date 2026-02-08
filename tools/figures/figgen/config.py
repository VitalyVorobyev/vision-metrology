from __future__ import annotations

from pathlib import Path

import numpy as np

SEED = 20260208
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "docs" / "fig"
