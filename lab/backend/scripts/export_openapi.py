"""Dump the FastAPI app's OpenAPI schema to ``lab/contract/openapi.json``.

That file is the wire contract between `vm_lab` and the frontend's typed client
(`lab/frontend/src/api/generated.ts`, produced from it by `openapi-typescript`). It is
committed so `tsc` and CI work without a running backend — regenerate it whenever a route
or schema changes and commit the diff; the diff *is* the API contract changing.

Run from anywhere:

    uv run --directory lab/backend python scripts/export_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_BACKEND_SRC))

from vm_lab.app import create_app  # noqa: E402 - needs sys.path set up first

_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "contract" / "openapi.json"


def main() -> None:
    app = create_app()
    schema = app.openapi()
    _CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONTRACT_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {_CONTRACT_PATH}")


if __name__ == "__main__":
    main()
