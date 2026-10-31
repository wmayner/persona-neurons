"""Each role's persona vector at layer 18 split exactly into the contributions of the MLP neurons of layers 0 to 18,
and of each layer's principal components of the neuron activations.

The persona vector is v_r = (mean residual over r's replies scored 2 or 3) − (mean residual over the default replies),
at the target layer, with massive-activation tokens left out (`persona_neurons.activations`). Averaging is linear, so
v_r = the embedding's difference in mean + the attention layers' differences + Σ over MLP neurons (l, n) of layers 0 to
the target of Δa_r[l, n] · W_down_l[:, n], where Δa_r is the difference in mean activation. Neuron (l, n) contributes

    c_r[l, n] = Δa_r[l, n] · (W_down_l[:, n] · v̂_r)

to the length |v_r|; a set of neurons contributes the share Σ c / |v_r|.

Layer l's principal components are the eigenvectors of the covariance of per-reply mean activations over a subsample of
replies of every role and the default (48 per role, every score). In that basis (columns u_j), component j contributes
(u_j · Δa_r[l]) (u_j · W_down_lᵀ v̂_r); a layer's components contribute in total what its neurons contribute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROLEPLAY_SCORES = ("2", "3")


@dataclass
class RoleData:
    roles: list[str]
    layers: list[int]
    delta_a: np.ndarray
    """(n_roles, len(layers), n_neurons) float32: the role's mean activation (positive plus negative part) minus the
    default's."""
    delta_attn: np.ndarray
    """(n_roles, len(layers), d_model): the same for each attention layer's output."""
    delta_embed: np.ndarray
    """(n_roles, d_model): the same for the embedding."""
    v: np.ndarray
    """(n_roles, d_model) float64: the steering vector, the role's mean residual at the target layer minus the
    default's."""
    n_default: int
    n_roleplay: dict[str, int]
    sub_a: np.ndarray | None = None
    """(n_replies, len(layers), n_neurons) float16: the activations of every reply in the subsample, default first."""


def _means(bucket) -> dict[str, np.ndarray]:
    n = bucket["count"]
    m = {k: bucket[k].numpy() / n for k in ("pos", "neg", "attn", "embed", "resid")}
    m["a"] = m["pos"] + m["neg"]
    return m


def _merge(buckets: list[dict]) -> dict:
    """The sums and count of several score buckets added together."""
    return {k: sum(b[k] for b in buckets) for k in buckets[0]}


def _subsample(s: dict, layers: list[int]) -> list[np.ndarray]:
    """a = pos + neg at `layers` of each subsample reply, in the order of their sorted keys, float16."""
    return [s["subsample"][k].float().sum(0)[layers].numpy().astype(np.float16) for k in sorted(s["subsample"])]


def load(stats_dir: Path, roles: list[str], target: int, layers=None, subsample: bool = False,
         default_role: str = "default") -> RoleData:
    """The persona vectors and their parts from step 3's `{role}.pt` (sums with massive-activation tokens left out);
    with `subsample`, also the subsample at `layers`."""
    layers = list(range(target + 1)) if layers is None else list(layers)
    s = torch.load(Path(stats_dir) / f"{default_role}.pt", map_location="cpu", weights_only=False)
    if s["scored"]:
        raise ValueError(f"{default_role} is a scored role; the default replies are not judged")
    d = _means(s["sums"]["missing"])
    n_default = s["sums"]["missing"]["count"]
    sub = _subsample(s, layers) if subsample else None
    da, dattn, demb, v, n_rp = [], [], [], [], {}
    for role in roles:
        s = torch.load(Path(stats_dir) / f"{role}.pt", map_location="cpu", weights_only=False)
        b = _merge([s["sums"][k] for k in ROLEPLAY_SCORES])
        if b["count"] == 0:
            raise ValueError(f"{role} has no replies scored {ROLEPLAY_SCORES}")
        m = _means(b)
        n_rp[role] = b["count"]
        da.append((m["a"][layers] - d["a"][layers]).astype(np.float32))
        dattn.append(m["attn"][layers] - d["attn"][layers])
        demb.append(m["embed"] - d["embed"])
        v.append(m["resid"][target] - d["resid"][target])
        if sub is not None:
            sub += _subsample(s, layers)
    return RoleData(list(roles), layers, np.stack(da), np.stack(dattn), np.stack(demb), np.stack(v).astype(np.float64),
                    n_default, n_rp, None if sub is None else np.stack(sub))


def down_proj_tensors(model_dir: Path, last_layer: int):
    """Yield (layer, W_down (d_model, n_neurons) float64) for layers 0 to last_layer from the safetensors files."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        where = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(model_dir / "model.safetensors", "pt") as f:
            where = {k: "model.safetensors" for k in f.keys()}
    names = sorted((int(k.split(".")[2]), k) for k in where if k.endswith("mlp.down_proj.weight"))
    for layer, name in names:
        if layer > last_layer:
            break
        with safe_open(model_dir / where[name], "pt") as f:
            yield layer, f.get_tensor(name).to(torch.float64).numpy()


def projections(w_down: list[np.ndarray], v: np.ndarray) -> np.ndarray:
    """p[r, l, n] = W_down_l[:, n] · v̂_r, (n_roles, n_layers, n_neurons) float32."""
    unit = v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.stack([(w.T @ unit.T).T for w in w_down], axis=1).astype(np.float32)


def rebuild(d: RoleData, w_down: list[np.ndarray]) -> np.ndarray:
    """Each role's vector rebuilt from its parts: embedding, attention layers and W_down-mapped neuron differences."""
    out = d.delta_embed + d.delta_attn.sum(1)
    for layer, w in enumerate(w_down):
        out = out + (w @ d.delta_a[:, layer].astype(np.float64).T).T
    return out


def contributions(delta: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """Neuron (l, n) contributes delta[l, n] * proj[l, n]: its activation difference times its output direction's
    projection on the unit persona vector."""
    return delta * proj


def top_curve(c: np.ndarray, norm: float, k_max: int) -> np.ndarray:
    """The share (%) of norm contributed by the k entries of c with the largest |c|, for k = 1 to k_max."""
    c = np.asarray(c, dtype=np.float64).ravel()
    part = np.argpartition(-np.abs(c), k_max - 1)[:k_max]
    top = c[part[np.argsort(-np.abs(c[part]), kind="stable")]]
    return 100 * np.cumsum(top) / norm


def top_neurons(c: np.ndarray, norm: float, k: int) -> pd.DataFrame:
    """The k neurons with the largest |contribution| in c (layers × neurons): rank, layer, neuron, contribution,
    share (% of norm)."""
    flat = c.ravel()
    k = min(k, flat.size)
    idx = np.argsort(-np.abs(flat), kind="stable")[:k]
    layer, neuron = np.unravel_index(idx, c.shape)
    return pd.DataFrame({"rank": np.arange(1, k + 1), "layer": layer, "neuron": neuron, "contribution": flat[idx],
                         "share": 100 * flat[idx] / norm})


def pca_basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The principal components of the rows of x (replies × neurons): the covariance's eigenvalues, largest first,
    and its eigenvectors as columns. With fewer replies than neurons, only the components with nonzero variance."""
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean(0)
    n, d = x.shape
    if n - 1 >= d:
        vals, v = np.linalg.eigh(xc.T @ xc / (n - 1))
        return vals[::-1], v[:, ::-1]
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    keep = s > s.max() * max(n, d) * np.finfo(np.float64).eps
    return s[keep] ** 2 / (n - 1), vt[keep].T


def pc_contributions(delta: np.ndarray, proj: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Each principal component's (columns of v, orthonormal) contribution (v_j · delta)(v_j · proj), for one layer:
    delta and proj are (n_neurons,) or (n_neurons, n_roles)."""
    x, y = np.asarray(delta, np.float64), np.asarray(proj, np.float64)
    return (v.T @ x) * (v.T @ y)


def layer_basis(stats_dir: Path, roles: list[str], layer: int, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Layer `layer`'s principal-component basis as the split step computes it, from the subsample of the default and
    every role in `roles`: (eigenvalues, eigenvectors as columns)."""
    d = load(stats_dir, roles, target=target, layers=[layer], subsample=True)
    return pca_basis(d.sub_a[:, 0].astype(np.float32))


def roles_with_vectors(vectors_dir: Path) -> list[str]:
    """Every role with a vector in vectors_dir (step 4's vectors/all/), except the default, sorted."""
    names = [torch.load(f, map_location="cpu", weights_only=False).get("role", f.stem)
             for f in sorted(Path(vectors_dir).glob("*.pt"))]
    return sorted(r for r in names if "default" not in r)
