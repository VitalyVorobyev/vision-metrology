# Fixture Generator

Deterministic fixture generator for the gallery pipeline.

Outputs:
- `docs/fig/fixtures/<case>/input.png`
- `docs/fig/fixtures/<case>/truth.json`

## Usage

```bash
python tools/fixtures/gen_fixtures.py
```

Optional local environment:

```bash
cd tools/fixtures
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python gen_fixtures.py
```
