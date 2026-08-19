"""`vision_metrology` — re-exports the compiled PyO3 extension module.

maturin places the compiled extension at `vision_metrology.vision_metrology`
(a submodule of this package, named after `module-name` in `pyproject.toml`)
rather than merging it into this `__init__`. This file is the bridge; the
type checker's view of the surface is `__init__.pyi`, not this wildcard
import (PEP 561 stub-package precedence).
"""

from .vision_metrology import *  # noqa: F401,F403
from .vision_metrology import __doc__  # noqa: F401
