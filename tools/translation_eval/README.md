# Private sequence registration evaluation

Development tooling for the general-purpose Rust `corr::TranslationEstimator`. It
reads the existing rtvt-pano pixel manifest, hash-pinned BMP frames, annotation SQLite
store and cached analysis/ribbon models. It does not import the application's mutable
store, change annotations, or write into either repository or the source dataset.

Dependencies: the installed `vision_metrology` bindings, numpy, Pillow, scipy and pytest
(for adapter checks). Build bindings with `maturin develop --release` first. Registration
and precise local measurements run in Rust; Python handles dataset adaptation, masked
normalized DoG preprocessing (sigma 1/4 px), reports and overlays.

```bash
python tools/translation_eval/run.py --data-root /path/to/rtvt1 \
  --out-dir /tmp/translation-development --propagate
python tools/translation_eval/run.py --data-root /path/to/rtvt1 \
  --out-dir /tmp/translation-held-out --split held-out
python tools/translation_eval/run.py --data-root /path/to/rtvt1 \
  --out-dir /tmp/translation-seed --sequence 15074 --seed 150 --propagate
python -m pytest tools/translation_eval/test_adapter.py
VM_RTVT_DATA=/path/to/rtvt1 python -m pytest tools/translation_eval/test_adapter.py
```

Development: 15074, 15077, 15082, 15516. Held out: 15090, 15521. Defaults are identical in
both splits; held-out mode evaluates registration without choosing seed geometry.
`--stride N` samples adjacent pairs at that interval (it does not register N-frame jumps).
Full evaluation is the default. `--max-frames` bounds each propagation direction.

`report.json` includes every selected pair, failures, support, regional consistency,
photometric residuals, forward/backward closure, and comparisons to cached raw motion.
Cached estimates are algorithm output, not independent motion ground truth. Estimator
latency includes Python conversion; preparation latency includes I/O, masks, registered
blank subtraction and filtering. Native-only timings live in the Criterion corr bench.

The motion mask combines saved foreground support, available reviewed masks, aligned
blank-difference support, nozzle and unlit masks. Foreground is dilated by three pixels
before exclusion. Blank correspondence comes from saved registration metadata, including
fallbacks where vendor path stamps freeze; equal frame numbers are never correspondence.
Difference masks provide foreground support, not a precise optical boundary reference.

Propagation prefers reviewed ribbon geometry when present, otherwise a generated green
cached ribbon, with explicit provenance. The saved masks can be reviewed without the
cached geometry being reviewed. One immutable seed is transported independently into
future and past frames. Both walls, centreline and sections move with cumulative substrate
translation. `BeadTracker` measures the visible prior; individual wall calipers search
±4 px on original images. Those measurements never update registration or the seed.
Rejected sections remain explicit. Propagation ends at registration failure, camera
handover, loss of the visible segment, sequence end or its frame budget.
Wall calipers require exactly one candidate of the expected bright-bead polarity;
multiple candidates are reported as ambiguous. The tracker uses a 0.1 pair margin.

PNG overlays show the transported walls in amber and independently measured wall points
in green. Source pixels, geometry and detailed reports remain private local artifacts;
only aggregate numbers belong in repository documentation. Whole-bead extension,
reference fusion, panoramas and inspection decisions are outside this tool's scope.
