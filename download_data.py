#!/usr/bin/env python3
"""Copy the released data bundle into the data directory ($PN_DATA_DIR, default ./data), checking every file.

The bundle is a folder with a MANIFEST file, one line per file: its path inside the bundle (and the data directory),
its size in bytes and its sha256, separated by tabs. LOCATION is that folder: a local path, a base URL
(LOCATION/MANIFEST, LOCATION/<path>) or a Hugging Face dataset's page. A file already in the data directory with the right checksum is skipped.

    uv run download_data.py LOCATION [--only PREFIX ...]
"""
import argparse
import hashlib
import re
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from persona_neurons.config import data_dir


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files_address(location: str) -> str:
    """The address a dataset's files are served from, given its Hugging Face page; any other location is left as it is."""
    page = location.rstrip("/")
    return f"{page}/resolve/main" if re.fullmatch(r"https://huggingface\.co/datasets/[^/]+/[^/]+", page) else location


def is_url(location: str) -> bool:
    return location.startswith(("http://", "https://"))


def fetch(location: str, path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if is_url(location):
        with urllib.request.urlopen(f"{location.rstrip('/')}/{path}") as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    else:
        shutil.copyfile(Path(location) / path, tmp)
    tmp.replace(dest)


def read_manifest(location: str) -> list[tuple[str, int, str]]:
    if is_url(location):
        with urllib.request.urlopen(f"{location.rstrip('/')}/MANIFEST") as r:
            text = r.read().decode()
    else:
        text = (Path(location) / "MANIFEST").read_text()
    rows = []
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            path, size, digest = line.split("\t")
            rows.append((path, int(size), digest))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("location", help="the bundle's folder: a local path, a base URL, or a Hugging Face dataset's page")
    ap.add_argument("--only", nargs="+", help="only the files whose path starts with one of these")
    args = ap.parse_args(argv)
    out = data_dir()
    args.location = files_address(args.location)
    rows = read_manifest(args.location)
    if args.only:
        rows = [r for r in rows if r[0].startswith(tuple(args.only))]
    total = sum(size for _, size, _ in rows)
    print(f"{len(rows)} files, {total / 1e9:.2f} GB, into {out}")
    bad = []
    for path, size, digest in rows:
        dest = out / path
        if dest.exists() and dest.stat().st_size == size and sha256(dest) == digest:
            continue
        fetch(args.location, path, dest)
        if dest.stat().st_size != size or sha256(dest) != digest:
            bad.append(path)
            print(f"checksum mismatch: {path}")
        else:
            print(f"ok {path}")
    if bad:
        print(f"{len(bad)} files did not match the manifest")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
