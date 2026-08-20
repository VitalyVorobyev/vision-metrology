#!/usr/bin/env python3
"""Guard the invariant numbering in docs/system-design.md.

The numbered invariants are cited *by number* from source files, doc comments and the
other persistent-context documents (`invariant 4`, `invariants 2 and 3`, ...). Nothing in
the compiler checks that a cited number still means what the citer thought it meant, so
the numbering is append-only by convention — and this script is what turns that
convention into a gate.

It checks two things:

  1. The numbered list under `## Invariants` in docs/system-design.md is contiguous and
     starts at 1 (so nobody silently drops one, which would renumber everything after it).
  2. Every `invariant N` / `invariants N` citation anywhere in the repository resolves to
     an invariant that exists.

Retiring an invariant is still allowed — keep its number and mark the entry
`**(retired)**`, saying what replaced it. That keeps the list contiguous and every old
citation resolvable.

Exit status 0 on success, 1 on any violation. No dependencies beyond the standard library.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "docs" / "system-design.md"

# Where citations may live. Everything else (build outputs, node_modules, target/) is skipped.
SEARCH_ROOTS = ("crates", "docs", "lab", "tools", ".github")
SEARCH_FILES = ("README.md", "AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", "CHANGELOG.md")
SEARCH_SUFFIXES = {".rs", ".md", ".py", ".toml", ".ts", ".tsx", ".yml", ".yaml"}
SKIP_DIRS = {"target", "node_modules", ".git", "dist", "build", ".venv", "venv", "__pycache__", "gen"}

# "invariant 4", "invariants 2", "Invariant 17" — the form actually used in this repo.
CITATION = re.compile(r"\binvariants?\s+(\d+)", re.IGNORECASE)
# A top-level numbered item in the invariants list: "12. **Determinism.** ..."
ITEM = re.compile(r"^(\d+)\.\s+\S")


def parse_invariants(text: str) -> list[int]:
    """Numbers of the top-level items in the `## Invariants` section, in file order."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == "## Invariants"), None)
    if start is None:
        sys.exit(f"{SPEC}: no '## Invariants' section found")
    numbers = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        m = ITEM.match(line)
        if m:
            numbers.append(int(m.group(1)))
    return numbers


def files_to_scan():
    for name in SEARCH_FILES:
        p = REPO / name
        if p.is_file():
            yield p
    for root in SEARCH_ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in SEARCH_SUFFIXES:
                continue
            if any(part in SKIP_DIRS for part in p.relative_to(REPO).parts):
                continue
            yield p


def main() -> int:
    if not SPEC.is_file():
        sys.exit(f"missing {SPEC}")
    text = SPEC.read_text(encoding="utf-8")
    numbers = parse_invariants(text)

    problems: list[str] = []

    if not numbers:
        problems.append("docs/system-design.md: the '## Invariants' section has no numbered items")
    else:
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            problems.append(
                "docs/system-design.md: invariant numbering is not contiguous from 1.\n"
                f"    found:    {numbers}\n"
                f"    expected: {expected}\n"
                "    Numbers are append-only: retire an invariant in place (keep its number,\n"
                "    mark it **(retired)**) rather than deleting or renumbering."
            )

    known = set(numbers)
    highest = max(numbers) if numbers else 0
    for path in files_to_scan():
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(content.splitlines(), 1):
            for m in CITATION.finditer(line):
                n = int(m.group(1))
                if n not in known:
                    rel = path.relative_to(REPO)
                    problems.append(
                        f"{rel}:{lineno}: cites '{m.group(0)}', but only invariants "
                        f"1-{highest} exist in docs/system-design.md"
                    )

    if problems:
        print("invariant check FAILED:\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"invariant check OK: {len(numbers)} invariants, all citations resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
