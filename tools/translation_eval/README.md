# Private sequence registration evaluation

Development tooling for the general-purpose Rust `corr::TranslationEstimator`. It
reads the existing private sequence dataset pixel manifest, hash-pinned BMP frames, annotation SQLite
store and cached analysis/ribbon models. It does not import the application's mutable
store, change annotations, or write into either repository or the source dataset.

Dependencies: the installed `vision_metrology` bindings, numpy, Pillow, scipy and pytest
(for adapter checks). Build bindings with `maturin develop --release` first. Registration
and precise local measurements run in Rust; Python handles dataset adaptation, masked
normalized DoG preprocessing (sigma 1/4 px), reports and overlays.

```bash
python tools/translation_eval/run.py --data-root /path/to/sequence-root \
  --out-dir /tmp/translation-development --propagate
python tools/translation_eval/run.py --data-root /path/to/sequence-root \
  --out-dir /tmp/translation-held-out --split held-out
python tools/translation_eval/run.py --data-root /path/to/sequence-root \
  --out-dir /tmp/translation-seed --sequence 15074 --seed 150 --propagate
python -m pytest tools/translation_eval/test_adapter.py
VM_SEQUENCE_DATA=/path/to/sequence-root python -m pytest tools/translation_eval/test_adapter.py
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

## Similarity, reciprocal sequences and local review

`similarity_run.py` compares translation with local native similarity registration and
attempts an independent seed anchor every eight supported steps. Camera handover,
registration loss, rejected reverse registration or corner closure above 0.25 px starts
a new segment. The native outcome and sequence support policy are recorded separately.
Direct anchor constraints also require reciprocal support; unavailable anchors never
update a pose. Both uncorrected composition and anchored trajectories remain in reports.

```bash
python tools/translation_eval/similarity_run.py --raw-data data/42781 \
  --valid-roi 5 5 100 85 --out-dir /tmp/registration-42781
python tools/translation_eval/similarity_run.py --data-root /path/to/sequence-root \
  --sequence 15074 --out-dir /tmp/registration-15074 \
  --propagation-report /tmp/translation-seed/report.json
python -m pytest tools/translation_eval/test_anchors.py
```

42781's 141 raw BMPs each stack three camera strips. It has no reviewed assignments,
masks, seed ribbon or blank-pair metadata. The tool processes each strip independently;
the explicitly selected texture ROI is caller evidence, not a reviewed mask. No bead
segment or ground truth is invented for it. Native-image overlays blend the mapped
previous image with the current image; failed poses are clearly diagnostic.

`index.html` is a standalone interactive viewer: camera/model selectors, frame scrubbing,
playback, blending, masks, mapped grids, outcomes, closure and trajectories. Optional
existing propagation reports add amber transported seed walls and green independently
measured points from the translation demonstration, with geometry provenance retained.
This overlay is distinct from similarity registration and never updates its pose.
All pixels are embedded locally; there are no external scripts or image requests.
Open the HTML directly, or serve only its output directory on loopback:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory /tmp/registration-15074
```

Similarity's additional parameters and stricter full-corner regional comparisons can
lower coverage substantially on these strips. This stage supplies a bounded operator and
a review experiment, not a replacement for translation or a validated production sequence
pipeline. Native latency and actual failure/coverage numbers are in `docs/performance.md`.
