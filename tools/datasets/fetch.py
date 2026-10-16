#!/usr/bin/env python3
"""Fetch a registered public dataset into the git-ignored `data/<id>/`, and verify it.

    python3 tools/datasets/fetch.py --list            # every entry, with its status
    python3 tools/datasets/fetch.py damsegment        # download if missing, verify, extract
    python3 tools/datasets/fetch.py damsegment --check   # verify what is there, download nothing
    python3 tools/datasets/fetch.py hpatches --data-root /elsewhere/data

The registry is `registry.json` beside this script. Only a `verified` entry with a direct
download is fetched. Any other entry prints its source, its licence as stated and the
manual steps, then exits with status 2, so a script that expects the data stops there.

Each file is checked against the registry's byte size and SHA-256. A file that is present
but does not match is never overwritten: the script reports it and exits with status 1.
Archives are extracted into the entry's `extract_to` folder, unless it already exists,
with every member path checked to stay inside it.

Standard library only (Python 3.9+), so it runs before any environment is set up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import urllib.request
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
REGISTRY = HERE / "registry.json"
CHUNK = 1 << 20


def load_registry(path: pathlib.Path = REGISTRY) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["datasets"]


def find(entries: list[dict], key: str) -> dict:
    key = key.lower()
    for e in entries:
        if key == e["id"] or key in (a.lower() for a in [e["name"], *e.get("aliases", [])]):
            return e
    sys.exit(f"no dataset '{key}' in {REGISTRY.name}; --list shows them")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def download(url: str, dest: pathlib.Path, size: int) -> None:
    """Stream `url` to `dest` through a `.part` file, renamed only when complete."""
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "vision-metrology-datasets/1"})
    print(f"  downloading {dest.name} ({human(size)}) from {url}")
    done, shown = 0, -1
    with urllib.request.urlopen(req) as resp, part.open("wb") as out:
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            out.write(block)
            done += len(block)
            pct = int(100 * done / size) if size else 0
            if pct // 10 != shown:
                shown = pct // 10
                print(f"    {pct:3d}%  {human(done)}", flush=True)
    part.replace(dest)


def verify(path: pathlib.Path, spec: dict) -> list[str]:
    """The problems with `path` against its registry `spec`; empty when it matches."""
    if not path.is_file():
        return [f"{path}: missing"]
    size = path.stat().st_size
    if size != spec["size"]:
        return [f"{path}: {size} bytes, the registry says {spec['size']}"]
    digest = sha256(path)
    if digest != spec["sha256"]:
        return [f"{path}: sha256 {digest}, the registry says {spec['sha256']}"]
    return []


def safe_extract(archive: pathlib.Path, dest: pathlib.Path) -> None:
    """Extract a zip or tar archive into `dest`, refusing members that leave it."""
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    root = tmp.resolve()

    def inside(name: str) -> None:
        target = (root / name).resolve()
        if target != root and root not in target.parents:
            sys.exit(f"{archive}: member '{name}' would leave {dest}; refusing to extract")

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                inside(info.filename)
            z.extractall(tmp)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                inside(m.name)
                if m.issym() or m.islnk():
                    sys.exit(f"{archive}: member '{m.name}' is a link; refusing to extract")
            t.extractall(tmp)
    else:
        sys.exit(f"{archive}: not a zip or tar archive")
    tmp.replace(dest)


def manual_steps(e: dict) -> int:
    lic = e["licence"]
    print(f"{e['name']} ({e['id']}): status '{e['status']}'. Not fetched automatically.")
    print(f"  source:   {e['source']}")
    print(f"  licence:  {lic.get('spdx') or lic.get('text')}  {lic.get('url', '')}".rstrip())
    print(f"  publish:  {e['aggregates']}")
    print(f"  route:    {e['route']}")
    if e.get("notes"):
        print(f"  notes:    {e['notes']}")
    return 2


def fetch(e: dict, data_root: pathlib.Path, check_only: bool) -> int:
    files = e.get("files", [])
    if e["status"] != "verified" or not files:
        return manual_steps(e)
    folder = data_root / e["id"]
    if not check_only:
        folder.mkdir(parents=True, exist_ok=True)
    print(f"{e['name']} -> {folder}")
    problems: list[str] = []
    for spec in files:
        path = folder / spec["name"]
        if not path.exists() and not check_only:
            download(spec["url"], path, spec["size"])
        found = verify(path, spec)
        if found:
            problems.extend(found)
            continue
        print(f"  ok  {spec['name']}  sha256 {spec['sha256'][:16]}...  {human(spec['size'])}")
        target = spec.get("extract_to")
        if target and not check_only:
            dest = folder / target
            if dest.exists():
                print(f"  {target}/ exists; not re-extracted")
            else:
                print(f"  extracting into {target}/")
                safe_extract(path, dest)
    for p in problems:
        print(f"  FAIL {p}", file=sys.stderr)
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("id", nargs="?", help="the dataset id, name or alias")
    parser.add_argument("--list", action="store_true", help="list the registry and exit")
    parser.add_argument("--check", action="store_true", help="verify only; download nothing")
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=REPO / "data",
        help="the folder that holds data/<id>/ (default: data/ in this checkout)",
    )
    args = parser.parse_args()
    entries = load_registry()
    if args.list:
        for e in entries:
            print(f"{e['id']:22s} {e['status']:16s} {e['name']}")
        return 0
    if not args.id:
        parser.error("give a dataset id, or --list")
    return fetch(find(entries, args.id), args.data_root, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
