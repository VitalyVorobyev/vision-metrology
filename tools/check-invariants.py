#!/usr/bin/env python3
"""Guard the invariant numbering and the documentation boundaries.

The numbered invariants in docs/dev/system-design.md are cited *by number* from source
files, doc comments and the developer documents (`invariant 4`, `invariants 2 and 3`, ...).
Nothing in the compiler checks that a cited number still exists, so the numbering is
append-only by convention, and this script turns that convention into a gate. It also
guards the boundary between user-facing and developer-facing text.

Checks, over every file tracked by git:

  1. The numbered list under `## Invariants` in docs/dev/system-design.md is contiguous
     and starts at 1 (so nobody silently drops one, which would renumber everything after
     it).
  2. Every number in an invariant citation resolves: `invariant 4`, `invariants 2 and 3`,
     `invariants 17, 18`, `invariants 2–4`.
  3. No dangling plan labels outside CHANGELOG.md and docs/dev/roadmap.md: `roadmap W7`,
     `roadmap B5`, "the roadmap plan", bare `W7`-style labels in comments and prose,
     `decision 9g`, `backlog item R3`, track labels (`Track 4`, `Track M`), and plan
     "waves". These name schemes that live only in a past session's plan, so a reader
     cannot resolve them.
  4. No mention of an external project name (FORBIDDEN_NAME), in any tracked file.
  5. User-facing text does not link into docs/dev/: the README files, the top-level guides
     under docs/, CHANGELOG.md, lab/README.md, the Python stubs, and rustdoc (`//!`, `///`)
     in the crates' library sources.
  6. Invariant 14's size cap, over the `.rs` files under `crates/*/src/` and
     `lab/frontend/src-tauri/src/`. A file's count is its non-blank lines that are not
     `//` comments (`//`, `///`, `//!`), up to its first `#[cfg(test)]` line. A file over
     SIZE_CAP must be listed in docs/dev/backlog.md's size-cap item (the "Code health"
     bullet naming the size cap, with one backticked path per offender), and a listed
     file must still be over the cap.

Retiring an invariant is still allowed — keep its number and mark the entry
`**(retired)**`, saying what replaced it. That keeps the list contiguous and every old
citation resolvable.

Exit status 0 on success, 1 on any violation. No dependencies beyond the standard library
and git.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent
SPEC_REL = "docs/dev/system-design.md"
SPEC = REPO / SPEC_REL
SELF_REL = "tools/check-invariants.py"

TEXT_SUFFIXES = {".rs", ".md", ".py", ".pyi", ".toml", ".ts", ".tsx", ".js", ".yml", ".yaml"}
SKIP_PARTS = {"node_modules", ".venv", "venv", "dist", "target", "gen", "__pycache__"}

# "invariant 4", "invariants 2 and 3", "Invariants 17, 18", "invariants 2–4".
CITATION = re.compile(
    r"\binvariants?\s+(\d+(?:\s*(?:,|and|or|&|–|-|to)\s*\d+)*)", re.IGNORECASE
)
# A top-level numbered item in the invariants list: "12. **Determinism.** ..."
ITEM = re.compile(r"^(\d+)\.\s+\S")

# Plan labels that only resolve inside a past session's plan.
LABELS = [
    (re.compile(r"\broadmap\s+W\d+\b", re.IGNORECASE), "roadmap wave label"),
    (re.compile(r"\broadmap\s+[A-Z]\d+(?:\.\d+)?\b"), "roadmap item label"),
    (re.compile(r"\broadmap(?:'s)?\s+(?:plan|decision)s?\b", re.IGNORECASE), "roadmap plan reference"),
    (re.compile(r"\bdecision\s+\d+[a-z]?\b", re.IGNORECASE), "plan decision label"),
    (re.compile(r"\bbacklog\s+(?:item\s+)?\*{0,2}R\d+\b", re.IGNORECASE), "backlog item label"),
    (re.compile(r"\bTrack\s+[A-Z0-9]+\b"), "track label"),
    (re.compile(r"\bwaves?\b", re.IGNORECASE), "plan wave"),
]
# Bare wave labels ("W6", "W7"): checked in comments and prose only, never in code.
BARE_WAVE = re.compile(r"\bW\d{1,2}\b")
# Physical waves are not plan waves.
WAVE_OK = re.compile(r"\b(?:sine|cosine|square|triangle|sawtooth|standing|plane)\s+waves?\b", re.IGNORECASE)
LABEL_EXEMPT = {"CHANGELOG.md", "docs/dev/roadmap.md", SELF_REL}

# Written with a character class so that this file does not itself contain the name.
FORBIDDEN_NAME = re.compile(r"rt[v]t", re.IGNORECASE)

DEV_LINK = re.compile(r"docs/dev/|\]\((?:\./)?dev/")
RUSTDOC = re.compile(r"^\s*//[/!]")
COMMENT = re.compile(r"^\s*(?://|#|\*|/\*|<!--)")

# Invariant 14: code lines per source file, counted up to the test module.
SIZE_CAP = 600
BACKLOG_REL = "docs/dev/backlog.md"
TEST_MODULE = re.compile(r"^\s*#\[cfg\(test\)\]")
TAURI_SRC = "lab/frontend/src-tauri/src/"
BACKTICKED_RS = re.compile(r"`([^`\s]+\.rs)`")


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True
    ).stdout.decode("utf-8")
    files = []
    for rel in out.split("\0"):
        if not rel:
            continue
        if SKIP_PARTS.intersection(PurePosixPath(rel).parts):
            continue
        files.append(rel)
    return files


def is_user_facing(rel: str) -> bool:
    p = PurePosixPath(rel)
    if rel in {"README.md", "CHANGELOG.md", "lab/README.md"}:
        return True
    if len(p.parts) == 3 and p.parts[0] == "crates" and p.name == "README.md":
        return True
    if len(p.parts) == 2 and p.parts[0] == "docs" and p.suffix == ".md":
        return True
    return p.suffix == ".pyi"


def is_library_source(rel: str) -> bool:
    p = PurePosixPath(rel)
    return len(p.parts) > 3 and p.parts[0] == "crates" and p.parts[2] == "src" and p.suffix == ".rs"


def is_size_capped(rel: str) -> bool:
    return is_library_source(rel) or (rel.startswith(TAURI_SRC) and rel.endswith(".rs"))


def code_lines(text: str) -> int:
    """Non-blank, non-`//`-comment lines before the first `#[cfg(test)]` line."""
    count = 0
    for line in text.splitlines():
        if TEST_MODULE.match(line):
            break
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            count += 1
    return count


def size_cap_allowed(text: str) -> list[str]:
    """The paths listed in backlog.md's size-cap item: the top-level bullet under
    `## Code health` that names the size cap, through its nested lines."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == "## Code health"), None)
    if start is None:
        return []
    paths: list[str] = []
    inside = False
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if line.startswith("- "):
            inside = "size cap" in line.lower()
        if inside:
            paths.extend(BACKTICKED_RS.findall(line))
    return paths


def parse_invariants(text: str) -> list[int]:
    """Numbers of the top-level items in the `## Invariants` section, in file order."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == "## Invariants"), None)
    if start is None:
        sys.exit(f"{SPEC_REL}: no '## Invariants' section found")
    numbers = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        m = ITEM.match(line)
        if m:
            numbers.append(int(m.group(1)))
    return numbers


def cited_numbers(group: str) -> list[int]:
    """Every invariant number in a citation's number list, with ranges expanded."""
    numbers: list[int] = []
    for part in re.split(r"\s*(?:,|and|or|&)\s*", group):
        rng = re.fullmatch(r"(\d+)\s*(?:–|-|to)\s*(\d+)", part)
        if rng:
            lo, hi = int(rng.group(1)), int(rng.group(2))
            numbers.extend(range(lo, hi + 1) if lo <= hi else [lo, hi])
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def main() -> int:
    if not SPEC.is_file():
        sys.exit(f"missing {SPEC_REL}")
    numbers = parse_invariants(SPEC.read_text(encoding="utf-8"))

    numbering: list[str] = []
    citations: list[str] = []
    labels: list[str] = []
    names: list[str] = []
    links: list[str] = []
    sizes: list[str] = []
    counts: dict[str, int] = {}

    if not numbers:
        numbering.append(f"{SPEC_REL}: the '## Invariants' section has no numbered items")
    else:
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            numbering.append(
                f"{SPEC_REL}: invariant numbering is not contiguous from 1.\n"
                f"      found:    {numbers}\n"
                f"      expected: {expected}\n"
                "      Numbers are append-only: retire an invariant in place (keep its number,\n"
                "      mark it **(retired)**) rather than deleting or renumbering."
            )
    known = set(numbers)
    highest = max(numbers) if numbers else 0

    for rel in tracked_files():
        path = REPO / rel
        try:
            if not path.is_file() or path.is_symlink():
                continue
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        if is_size_capped(rel):
            counts[rel] = code_lines(content)

        suffix = PurePosixPath(rel).suffix
        text_file = suffix in TEXT_SUFFIXES
        prose = suffix == ".md"
        user_facing = is_user_facing(rel)
        library_source = is_library_source(rel)
        check_labels = text_file and rel not in LABEL_EXEMPT

        for lineno, line in enumerate(content.splitlines(), 1):
            where = f"{rel}:{lineno}"

            if FORBIDDEN_NAME.search(line):
                names.append(f"{where}: names the external project (regex {FORBIDDEN_NAME.pattern})")

            if not text_file:
                continue

            for m in CITATION.finditer(line):
                for n in cited_numbers(m.group(1)):
                    if n not in known:
                        citations.append(
                            f"{where}: cites '{m.group(0)}', but only invariants "
                            f"1-{highest} exist in {SPEC_REL}"
                        )

            commentary = prose or bool(COMMENT.match(line))
            if check_labels:
                for pattern, what in LABELS:
                    for m in pattern.finditer(line):
                        if what == "plan wave" and (WAVE_OK.search(line) or not commentary):
                            continue
                        labels.append(f"{where}: {what} '{m.group(0)}'")
                if commentary or suffix in {".py", ".pyi"}:
                    for m in BARE_WAVE.finditer(line):
                        labels.append(f"{where}: wave label '{m.group(0)}'")

            if DEV_LINK.search(line) and (
                user_facing or (library_source and RUSTDOC.match(line))
            ):
                links.append(f"{where}: user-facing text links into docs/dev/")

    backlog = REPO / BACKLOG_REL
    allowed = size_cap_allowed(backlog.read_text(encoding="utf-8")) if backlog.is_file() else []
    over = {rel: n for rel, n in sorted(counts.items()) if n > SIZE_CAP}
    for rel, n in over.items():
        if rel not in allowed:
            sizes.append(
                f"{rel}: {n} code lines, over the cap of {SIZE_CAP}: "
                f"split it, or list it in {BACKLOG_REL}"
            )
    for rel in allowed:
        if rel not in over:
            n = counts.get(rel)
            what = (
                "not a tracked source file it checks"
                if n is None
                else f"{n} code lines, at or under the cap of {SIZE_CAP}"
            )
            sizes.append(f"{BACKLOG_REL}: lists {rel} ({what}): remove it from the list")

    sections = [
        ("invariant numbering", numbering),
        ("invariant citations", citations),
        ("dangling plan labels", labels),
        ("external project name", names),
        ("user-facing links into docs/dev/", links),
        (f"size cap of {SIZE_CAP} code lines (invariant 14)", sizes),
    ]
    failed = [(title, items) for title, items in sections if items]
    if failed:
        print("documentation check FAILED:\n", file=sys.stderr)
        for title, items in failed:
            print(f"{title} ({len(items)}):", file=sys.stderr)
            for item in items:
                print(f"  - {item}", file=sys.stderr)
            print(file=sys.stderr)
        return 1

    print(f"documentation check OK: {len(numbers)} invariants, all citations resolve")
    for rel, n in over.items():
        print(f"  over the size cap, listed in {BACKLOG_REL}: {rel} ({n} code lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
