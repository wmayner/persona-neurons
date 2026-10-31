"""download_data.py: copies a bundle's files into the data directory and checks them against the MANIFEST."""
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_bundle(root: Path, files: dict[str, bytes]) -> None:
    lines = []
    for path, content in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(content)
        lines.append(f"{path}\t{len(content)}\t{hashlib.sha256(content).hexdigest()}")
    (root / "MANIFEST").write_text("\n".join(lines) + "\n")


def run(bundle, data, *extra):
    return subprocess.run([sys.executable, str(ROOT / "download_data.py"), str(bundle), *extra], capture_output=True,
                          text=True, cwd=ROOT, env={**os.environ, "PN_DATA_DIR": str(data)})


def test_copies_and_checks(tmp_path):
    bundle, data = tmp_path / "bundle", tmp_path / "data"
    make_bundle(bundle, {"split/arrays.npz": b"arrays", "vectors/all/pirate.pt": b"vector"})
    result = run(bundle, data)
    assert result.returncode == 0, result.stderr
    assert (data / "split" / "arrays.npz").read_bytes() == b"arrays"
    result = run(bundle, data, "--only", "vectors/")
    assert result.returncode == 0 and "ok" not in result.stdout  # already there: skipped

    (bundle / "split" / "arrays.npz").write_bytes(b"changed")  # the bundle no longer matches its manifest
    (data / "split" / "arrays.npz").unlink()
    result = run(bundle, data)
    assert result.returncode == 1 and "checksum mismatch: split/arrays.npz" in result.stdout


def test_a_dataset_page_becomes_the_address_of_its_files():
    spec = importlib.util.spec_from_file_location("download_data", ROOT / "download_data.py")
    download_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(download_data)
    base = "https://huggingface.co/datasets/a/b"
    assert download_data.files_address(base) == f"{base}/resolve/main"
    assert download_data.files_address(base + "/") == f"{base}/resolve/main"
    assert download_data.files_address(f"{base}/resolve/main") == f"{base}/resolve/main"
    assert download_data.files_address("some/folder") == "some/folder"
