# Figure Generator

This folder contains deterministic Python scripts used only for documentation
figures. It is not a runtime dependency of Rust crates in this workspace.

## Setup

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_all.py
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_all.py
```

Generated artifacts are written to `docs/fig/`.

## Structure

- `generate_all.py`: thin entry point.
- `figgen/gallery.py`: orchestration for all figures.
- `figgen/*_fig.py`: feature-specific generators.
- `figgen/common.py`: shared signal/image utilities.
