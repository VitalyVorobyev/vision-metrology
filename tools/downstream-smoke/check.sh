#!/usr/bin/env bash
# Resolves the smoke crate from scratch the way a downstream crate would, applies
# the one-line unification recipe from invariant 13, and fails unless the graph
# then holds a single nalgebra and the cross-crate round trip runs.
set -euo pipefail
cd "$(dirname "$0")"

nalgebra_versions() {
  cargo tree -e normal --prefix none --format '{p}' |
    awk '$1 == "nalgebra" { sub(/^v/, "", $2); print $2 }' | sort -u -V
}

cargo generate-lockfile

# Cargo resolves each semver-incompatible requirement on its own, so a fresh lock
# takes the newest nalgebra our range allows next to calibration-rs's 0.34. The
# downstream moves ours down onto theirs, once, and its lockfile keeps it.
found=$(nalgebra_versions)
echo "fresh resolution: $(echo "$found" | tr '\n' ' ')"
if [ "$(echo "$found" | wc -l)" -gt 1 ]; then
  low=$(echo "$found" | head -n 1)
  high=$(echo "$found" | tail -n 1)
  cargo update -p "nalgebra@$high" --precise "$low"
fi

found=$(nalgebra_versions)
if [ "$(echo "$found" | wc -l)" -ne 1 ]; then
  echo "error: more than one nalgebra in the graph: $(echo "$found" | tr '\n' ' ')" >&2
  exit 1
fi
echo "unified on nalgebra $found"
cargo tree -i nalgebra
cargo run --quiet
