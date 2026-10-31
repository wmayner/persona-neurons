"""The Assistant Axis repository (safety-research/assistant-axis) at a fixed commit. It has no licence, so it is
fetched at run time and never copied here. Its package is imported from the checkout, not installed, because
installing it pulls in vLLM 0.13.0."""
import subprocess
import sys
from pathlib import Path

from persona_neurons.config import UPSTREAM_COMMIT, UPSTREAM_URL


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True, check=True).stdout.strip()


def fetch_upstream(dest: Path) -> Path:
    """Clone the repository into dest, or update an existing clone, and check out UPSTREAM_COMMIT. Refuses a checkout
    whose tracked files have been changed, since the authors' code is used unchanged."""
    dest = Path(dest)
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", UPSTREAM_URL, str(dest)], check=True)
    elif subprocess.run(["git", "-C", str(dest), "cat-file", "-e", f"{UPSTREAM_COMMIT}^{{commit}}"],
                        capture_output=True).returncode != 0:
        _git(dest, "fetch", "--quiet", "origin")
    if _git(dest, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"{dest} has changes to the repository's files; the authors' code must be used unchanged")
    _git(dest, "-c", "advice.detachedHead=false", "checkout", "--quiet", UPSTREAM_COMMIT)
    head = _git(dest, "rev-parse", "HEAD")
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"{dest} is at {head}, not {UPSTREAM_COMMIT}")
    return dest


def use_upstream(checkout: Path) -> None:
    """Put the checkout first on sys.path, so that `import assistant_axis` loads the authors' package from it."""
    path = str(Path(checkout).resolve())
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def model_snapshot(model_id: str, revision: str, allow_patterns: list[str] | None = None) -> Path:
    """The local snapshot folder of model_id at revision, downloaded into the Hugging Face cache if needed. Its path
    (.../models--Qwen--Qwen3-8B/snapshots/<revision>) contains "qwen", so the authors' Qwen checks hold for it.
    allow_patterns limits the download, for tests that need only the config and tokenizer."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, revision=revision, allow_patterns=allow_patterns))
