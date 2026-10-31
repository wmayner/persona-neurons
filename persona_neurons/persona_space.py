"""Persona space as Lu et al. (2026) compute it: PCA on the mean-centred role vectors at one layer, with the authors'
`MeanScaler` and `compute_pca`. A loading is the cosine of a mean-centred vector with a principal component (paper section
2.2), and each component's sign is set so that the default vector loads positively. Needs the authors' repository on
`sys.path` (`persona_neurons.upstream.use_upstream`).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PUBLISHED_REPO = "lu-christina/assistant-axis-vectors"
PUBLISHED_REVISION = "3b3b788432ad33e3a28d9ff08e88a530c0740814"
PUBLISHED_MODEL, PUBLISHED_LAYER, PUBLISHED_N_ROLES = "qwen-3-32b", 32, 275


@dataclass
class PCAResult:
    roles: list[str]
    components: np.ndarray   # (n_components, d_model), unit rows
    loadings: np.ndarray     # (n_roles, n_components)


def _cosines(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    c = np.atleast_2d(x) - mean
    return (c / np.linalg.norm(c, axis=1, keepdims=True)) @ components.T


def pca_roles(vectors: dict[str, torch.Tensor], layer: int) -> PCAResult:
    """PCA at `layer`. vectors: name to (n_layers, d_model); the name containing "default" stays out of the PCA, as in the
    authors' `pipeline/5_axis.py`."""
    from assistant_axis import MeanScaler, compute_pca

    roles = sorted(k for k in vectors if "default" not in k)
    defaults = [k for k in vectors if "default" in k]
    if len(defaults) != 1:
        raise ValueError(f"need exactly one default vector, not {defaults}")
    x = torch.stack([vectors[r][layer] for r in roles]).float()
    _, _, _, pca, scaler = compute_pca(x, layer=None, scaler=MeanScaler(), verbose=False)
    mean = np.asarray(scaler.mean)
    components = pca.components_ / np.linalg.norm(pca.components_, axis=1, keepdims=True)
    default = vectors[defaults[0]]
    default = (default[layer] if default.ndim == 2 else default).float().numpy()
    components = components * np.where(_cosines(default, mean, components)[0] < 0, -1.0, 1.0)[:, None]
    return PCAResult(roles, components, _cosines(x.numpy(), mean, components))


def load_vectors(vectors_dir: Path) -> dict[str, torch.Tensor]:
    """The authors' vectors/*.pt files: role (or "default") to its (n_layers, d_model) vector."""
    out = {}
    for f in sorted(Path(vectors_dir).glob("*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=False)
        out[data.get("role", f.stem)] = data["vector"]
    return out


def download_published(dest: Path) -> dict[str, torch.Tensor]:
    """The authors' published Qwen3-32B role vectors and default vector, downloaded into dest."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(
        PUBLISHED_REPO, repo_type="dataset", revision=PUBLISHED_REVISION, local_dir=dest,
        allow_patterns=[f"{PUBLISHED_MODEL}/role_vectors/*.pt", f"{PUBLISHED_MODEL}/default_vector.pt"], max_workers=2))
    d = root / PUBLISHED_MODEL
    files = sorted((d / "role_vectors").glob("*.pt"))
    # When the Hub cannot be reached, snapshot_download returns the local folder as it is, however incomplete
    if len(files) != PUBLISHED_N_ROLES or not (d / "default_vector.pt").exists():
        raise RuntimeError(f"{d} is incomplete ({len(files)} of {PUBLISHED_N_ROLES} role vectors)")
    vectors = {p.stem: torch.load(p, map_location="cpu", weights_only=False) for p in files}
    vectors["default"] = torch.load(d / "default_vector.pt", map_location="cpu", weights_only=False)
    return vectors
