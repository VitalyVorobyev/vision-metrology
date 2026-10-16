"""The README quick starts, run as written, plus an import check of the installed package.

The snippets are read from the README files at run time rather than copied here, so this
test cannot drift from them. A snippet runs when it is self-contained, that is, when it
starts with an `import`; the others show fragments with placeholder names. A `print`
whose line ends in a comment of numbers (`print(a, b)  # 19.5 40.5`) must print those
numbers.

It needs only numpy and the installed wheel, so the wheel smoke job runs it as a plain
script in a clean virtual environment:

    python crates/vm-python/tests/test_readme_quickstart.py

pytest collects the same checks in the bindings job.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
READMES = [REPO / "README.md", REPO / "crates" / "vm-python" / "README.md"]

_BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
_EXPECTED = re.compile(r"^\s*print\(.*\)\s*#\s*(-?[\d.]+(?:\s+-?[\d.]+)*)\s*$")


def snippets() -> list[tuple[str, str]]:
    """Every self-contained ```python block in the READMEs, labelled `file:line`."""
    out = []
    for path in READMES:
        text = path.read_text(encoding="utf-8")
        for m in _BLOCK.finditer(text):
            code = m.group(1)
            if code.lstrip().startswith(("import ", "from ")):
                line = text.count("\n", 0, m.start()) + 2
                out.append((f"{path.relative_to(REPO)}:{line}", code))
    return out


def run_snippet(label: str, code: str) -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, label, "exec"), {"__name__": "__readme__"})
    printed = buf.getvalue().splitlines()
    prints = [ln for ln in code.splitlines() if ln.lstrip().startswith("print(")]
    expected = [(i, _EXPECTED.match(ln)) for i, ln in enumerate(prints)]
    if not any(m for _, m in expected):
        return
    assert len(printed) == len(prints), f"{label}: {len(printed)} lines for {len(prints)} prints"
    for i, m in expected:
        if m is None:
            continue
        want = [float(v) for v in m.group(1).split()]
        got = [float(v) for v in printed[i].split()]
        assert len(got) == len(want) and all(
            math.isclose(g, w, abs_tol=1e-4) for g, w in zip(got, want)
        ), f"{label}: printed {printed[i]!r}, the README says {m.group(1)!r}"


def test_readme_snippets_run() -> None:
    found = snippets()
    # One in the top-level README, two in the bindings README.
    assert len(found) >= 3, [label for label, _ in found]
    for label, code in found:
        run_snippet(label, code)


def test_installed_package() -> None:
    import vision_metrology as vm
    from vision_metrology import experimental

    pkg = Path(vm.__file__).resolve().parent
    assert (pkg / "py.typed").is_file(), "the wheel must ship py.typed"
    assert (pkg / "__init__.pyi").is_file(), "the wheel must ship the stubs"
    assert hasattr(experimental, "SimilarityEstimator")

    # The wheel's version comes from Cargo.toml (`dynamic = ["version"]`).
    cargo = (REPO / "Cargo.toml").read_text(encoding="utf-8")
    m = re.search(r'^\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"', cargo, re.M | re.S)
    assert m, "no [workspace.package] version in Cargo.toml"
    assert importlib.metadata.version("vision-metrology") == m.group(1)


if __name__ == "__main__":
    import vision_metrology

    print(f"vision_metrology from {Path(vision_metrology.__file__).parent}")
    test_installed_package()
    test_readme_snippets_run()
    print("ok: " + ", ".join(label for label, _ in snippets()))
    sys.exit(0)
