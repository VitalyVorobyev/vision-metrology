# Gallery Runner

Runs the full external-fixture gallery pipeline:

1. Generate fixtures under `docs/fig/fixtures/`
2. Execute Rust `vm_gallery` to produce raw outputs under `docs/fig/raw/`
3. Plot final images under `docs/fig/`

## Usage

```bash
python tools/gallery/run_all.py
```

## Interactive Viewer

Build and open all final figures in an interactive matplotlib window:

```bash
python tools/gallery/show_figures.py
```
: Default mode is `browse` (one figure on one canvas, interactive next/prev).

Show one required use case only (single-canvas):

```bash
python tools/gallery/show_figures.py --case edgels2d
```

Browse mode with keyboard navigation:

```bash
python tools/gallery/show_figures.py --mode browse
```
: Controls: `right/n/space` next, `left/p/backspace` previous, `q/esc` quit.

Show existing figures only (skip rebuild):

```bash
python tools/gallery/show_figures.py --no-build
```

Export a contact sheet without opening a window:

```bash
python tools/gallery/show_figures.py --no-show --export docs/fig/gallery_sheet.png
```

Optional local environment:

```bash
cd tools/gallery
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```
